# %% [markdown]
# # Anexo: paralelo -- N workers sobre el mismo estudio
#
# Todo lo anterior corrió con `n_jobs=1`: un trial a la vez, en un solo proceso. Aquí
# varios procesos (workers) leen y escriben el mismo estudio de Optuna al mismo tiempo,
# localmente sobre un archivo Journal o contra PostgreSQL desde Docker (`compose.yaml`,
# perfil `paralelo`), y compiten por correr trials hasta que entre todos suman
# `TALLER_N_TOTAL`.
#
# Tres modos, un solo archivo:
# - **worker** (default): lo que corre cada proceso, sea uno lanzado por `--lanzar` o un
#   contenedor `worker` de Docker.
# - **`--lanzar N`**: modo local. Arranca N workers sobre el mismo Journal, espera a
#   todos y reporta lo que pasó.
# - **`--preparar`**: crea (o reabre) el estudio una sola vez y termina, sin optimizar.
#   Lo usa el servicio `preparar` de `compose.yaml` antes de escalar los workers.
#
# Cada worker entrena con `taller_hpo.problemas.juguete` en la tarjeta de video. Con
# varios workers sobre la misma tarjeta, cada proceso reserva su propio contexto CUDA y
# compite por los mismos núcleos. La memoria de los N workers la mide el proceso padre de
# `--lanzar`: muestrea `memory.used` de nvidia-smi mientras corren, y el pico menos la
# línea base de antes de lanzar es la huella de los N procesos juntos (en WSL, nvidia-smi
# no reporta la memoria de cada proceso por separado). Cada worker imprime además en su
# log lo que reservó el asignador de memoria de torch y una estimación de su contexto
# CUDA, que con varios workers arrancando a la vez sale inflada.
#
# El protocolo (sampler, pruner, espacio, métrica) es idéntico para todos los workers; si
# no lo fuera, `crear_estudio` lo rechaza al reabrir. Solo cambia la semilla del sampler,
# que queda fuera del protocolo a propósito: cada worker debe proponer configuraciones
# distintas. `juguete.PROTOCOLO_MOTOR` sí entra: registra que PyTorch entrenó en la
# tarjeta de video, sin el índice de CUDA.
#
# ## Por qué existe `--preparar`
#
# Contra un PostgreSQL recién creado, dos workers que llaman a `crear_estudio` a la vez
# compiten por crear el esquema (`CREATE TYPE studydirection` dentro de
# `RDBStorage.__init__`), y uno revienta con un `IntegrityError`. Por eso `compose.yaml`
# corre el servicio `preparar` una sola vez antes de escalar `worker`: con el esquema y el
# estudio ya creados, cada worker solo agrega sus trials.
#
# Correr:  uv run pasos/anexo_paralelo.py --lanzar 3
# Con WSL y el repo en /mnt/c o /mnt/d, el Journal falla con varios workers (ver
# `_verificar_journal_no_9p`): usa un sistema de archivos Linux, p. ej.
# TALLER_ESTUDIOS=~/estudios_taller uv run pasos/anexo_paralelo.py --lanzar 3. Sin una
# tarjeta de video NVIDIA con CUDA, el anexo se detiene con un mensaje claro.
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo (total de trials entre
# todos los workers, no por worker)

# %%
import argparse
import dataclasses
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import optuna
from matplotlib.figure import Figure
from sqlalchemy.engine import make_url

from taller_hpo.estadistica import Pareado, diferencia_pareada
from taller_hpo.estudio import abrir_storage, crear_estudio, optimizar
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.semillas import semillas_de
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
# La ruta de este script se resuelve una vez al importar: en un montaje 9p de WSL,
# resolverla toca el disco de Windows, y ese costo caería dentro del cronómetro de
# `_spawn_n_workers`.
_SCRIPT = str(Path(__file__).resolve())

# Capturado al terminar de importar este módulo. Con el `time.monotonic()` que
# `_correr_worker` imprime al final (VIDA_PROCESO_JSON) da una cota inferior de la vida
# del proceso: incluye el import de torch, el contexto CUDA y el entrenamiento, pero no
# el arranque del intérprete. `monotonic` y no `time.time()`: el reloj de pared salta
# cuando el sistema sincroniza la hora (en WSL2 pasa durante una corrida), y en Linux
# CLOCK_MONOTONIC es el mismo para todos los procesos, así que el padre y el worker
# pueden compararse.
_INICIO_PROCESO = time.monotonic()

# Total de trials entre todos los workers, no por worker. "humo" respeta el techo de 20 s
# de las pruebas y "taller", el de 3 minutos en la máquina de referencia. Con "humo" el
# arranque de cada proceso (imports y contexto CUDA) domina sobre el entrenamiento; con
# "taller", al revés. Los tiempos medidos están en la tabla del README.
PRESUPUESTOS = {"humo": 6, "taller": 90, "completo": 300}


def _leer_presupuesto() -> tuple[str, int]:
    presupuesto = os.environ.get("PRESUPUESTO", "taller")
    if presupuesto not in PRESUPUESTOS:
        # El mismo mensaje que en los pasos numerados.
        raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
    return presupuesto, PRESUPUESTOS[presupuesto]


PRESUPUESTO, N_TOTAL_DEFECTO = _leer_presupuesto()

NOMBRE_ESTUDIO_DEFECTO = "anexo_paralelo"
ARCHIVO_JOURNAL_DEFECTO = "anexo_paralelo.log"

# CRN (números aleatorios comunes): todos los trials de todos los workers entrenan con
# los mismos datos, orden e init, así las diferencias entre configuraciones no se
# confunden con ruido de semilla. Una sola Semillas fija, no una por trial.
SEMILLA_BUSQUEDA = 20_260_925

# TPESampler(seed=...) llega hasta np.random.RandomState.seed(), que exige
# 0 <= seed < 2**32 (~4.29e9). 1_000_000 + 1000*worker + n_previos se queda muy
# por debajo aun con cientos de workers y miles de trials previos.
BASE_SEMILLA_SAMPLER = 1_000_000


# %% [markdown]
# ## Rutas y entorno compartidos por los tres modos


# %%
def _estudios_dir() -> Path:
    return Path(os.environ.get("TALLER_ESTUDIOS", RAIZ / "estudios"))


def _resultados_dir() -> Path:
    resultados = Path(os.environ.get("TALLER_RESULTADOS", RAIZ / "resultados"))
    resultados.mkdir(parents=True, exist_ok=True)
    return resultados


def _destino() -> str:
    return os.environ.get("TALLER_DESTINO") or str(_estudios_dir() / ARCHIVO_JOURNAL_DEFECTO)


def _destino_redactado(destino: str) -> str:
    """`destino` listo para imprimir: en una URL de Postgres oculta la contraseña; una
    ruta de Journal se devuelve tal cual. `TALLER_PG_PASSWORD` nunca debe llegar a un
    log."""
    if "://" in destino:
        return make_url(destino).render_as_string(hide_password=True)
    return destino


def _nombre_estudio() -> str:
    return os.environ.get("TALLER_ESTUDIO", NOMBRE_ESTUDIO_DEFECTO)


_TIPOS_FS_9P = frozenset({"9p", "v9fs", "drvfs"})


def _tipo_fs_de(ruta: Path) -> str | None:
    """Tipo de sistema de archivos del montaje que contiene `ruta`, leído de
    `/proc/self/mountinfo`, o `None` fuera de Linux.

    Se queda con el punto de montaje más largo que sea prefijo de la ruta resuelta, el
    mismo criterio del kernel con montajes anidados.
    """
    try:
        texto = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    resuelta = str(ruta.resolve())
    mejor_punto = ""
    mejor_tipo: str | None = None
    for linea in texto.splitlines():
        izquierda, separador, derecha = linea.partition(" - ")
        if not separador:
            continue
        campos_izq = izquierda.split()
        campos_der = derecha.split()
        if len(campos_izq) < 5 or not campos_der:
            continue
        punto_montaje = campos_izq[4]
        coincide = resuelta == punto_montaje or resuelta.startswith(
            punto_montaje if punto_montaje.endswith("/") else punto_montaje + "/"
        )
        if coincide and len(punto_montaje) >= len(mejor_punto):
            mejor_punto = punto_montaje
            mejor_tipo = campos_der[0]
    return mejor_tipo


def _verificar_journal_no_9p(destino: str) -> None:
    """`SystemExit` antes de tocar el storage si `destino` es un Journal sobre un montaje
    9p/v9fs/DrvFs de WSL (`/mnt/c`, `/mnt/d`, ...).

    El Journal de Optuna asume que varios procesos pueden abrir y escribir el mismo
    archivo a la vez, y eso no se sostiene sobre el montaje que WSL2 usa para las
    unidades de Windows: con `--lanzar 3`, la mitad de las corridas en `/mnt/c` fallaron
    con `OSError: [Errno 61] No data available`; en `/tmp` (ext4), ninguna. Con un solo
    worker el problema no siempre aparece.
    """
    if "://" in destino:
        return
    tipo = _tipo_fs_de(Path(destino).parent)
    if tipo in _TIPOS_FS_9P:
        raise SystemExit(
            f"el Journal compartido por N procesos falla en montajes 9p/DrvFs de WSL "
            f"({tipo!r} en {destino!r}, p. ej. /mnt/c, /mnt/d): usa TALLER_ESTUDIOS en el "
            "sistema de archivos de Linux (p. ej. ~/estudios_taller) o "
            "TALLER_DESTINO=postgresql+psycopg://..."
        )


def _memoria_gpu_usada_mib() -> float | None:
    """`memory.used` de toda la tarjeta de video en este instante (nvidia-smi, MiB), o
    `None` si nvidia-smi no está disponible o falla.

    Es global, no por proceso: solo sirve para restar un antes y un después alrededor de
    un evento puntual, y con la tarjeta compartida ese delta es ruidoso.
    """
    try:
        resultado = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if resultado.returncode != 0:
        return None
    try:
        return float(resultado.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _exigir_tarjeta_de_video() -> None:
    """`SystemExit` con `taller_hpo.tarjeta_de_video.MENSAJE_SIN_TARJETA` si no hay una
    tarjeta de video NVIDIA con CUDA disponible.

    `_preparar_estudio` y `_correr_worker` la llaman antes de importar
    `taller_hpo.problemas.juguete`, para salir con una línea clara en vez de un traceback.
    `_lanzar` la llama en el proceso padre, para que el aviso llegue a la pantalla y no
    quede solo en el log de cada worker. `torch.cuda.is_available()` no crea el contexto
    CUDA.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(MENSAJE_SIN_TARJETA)


def _medir_contexto_mib(dispositivo: str) -> float:
    """nvidia-smi antes y después del primer tensor que este proceso crea en
    `dispositivo` (más `torch.cuda.synchronize()`), o `nan` si nvidia-smi no respondió.

    Separada de `_correr_worker` para correrla sola en un proceso nuevo: así una prueba
    puede repetirla y mirar la mediana, porque con la tarjeta compartida una sola lectura
    puede salir corta o negativa. Importa torch adentro, como el resto del archivo."""
    import torch

    memoria_antes = _memoria_gpu_usada_mib()
    torch.zeros(1, device=dispositivo)
    torch.cuda.synchronize()
    memoria_despues = _memoria_gpu_usada_mib()
    if memoria_antes is None or memoria_despues is None:
        return float("nan")
    return memoria_despues - memoria_antes


def _huella_mib(pico_memoria_mib: float, memoria_base_mib: float, n: int) -> tuple[float, float]:
    """Huella de memoria de los `n` workers juntos: el pico que midió el proceso padre
    menos la línea base de antes de lanzar, y ese total entre `n` por worker."""
    huella_total_mib = pico_memoria_mib - memoria_base_mib
    huella_por_worker_mib = huella_total_mib / n
    return huella_total_mib, huella_por_worker_mib


def _n_total() -> int:
    crudo = os.environ.get("TALLER_N_TOTAL")
    if crudo is None:
        return N_TOTAL_DEFECTO
    try:
        valor = int(crudo)
    except ValueError as error:
        raise SystemExit(f"TALLER_N_TOTAL debe ser un entero, no {crudo!r}") from error
    if valor < 1:
        raise SystemExit(f"TALLER_N_TOTAL debe ser >= 1, no {valor}")
    return valor


def _resolver_worker() -> int:
    """TALLER_WORKER si está definida; si no, un id derivado del hostname.

    `--lanzar` fija TALLER_WORKER=0..N-1 en cada proceso. Las réplicas del servicio
    `worker` de compose.yaml no tienen un contador compartido, pero Docker les da un
    hostname distinto: su crc32 alcanza para que dos workers no compartan semilla.
    """
    crudo = os.environ.get("TALLER_WORKER")
    if crudo is not None:
        try:
            return int(crudo)
        except ValueError as error:
            raise SystemExit(f"TALLER_WORKER debe ser un entero, no {crudo!r}") from error
    hostname = socket.gethostname()
    derivado = zlib.crc32(hostname.encode("utf-8")) % 1_000_000
    print(f"TALLER_WORKER no está definida: derivando {derivado} del hostname {hostname!r}")
    return derivado


# %% [markdown]
# ## El protocolo y la fábrica de sampler: una sola fuente de kwargs
#
# `_SAMPLER_KWARGS` es la única fuente de los kwargs de muestreo: la fábrica que
# construye el `TPESampler` (`_fabrica_sampler`) y el protocolo que `crear_estudio`
# guarda y compara al reabrir (`_protocolo`) leen de aquí, así no pueden desviarse entre
# sí. `SAMPLER_REAL_JSON`, leído del sampler ya construido (ver `_correr_worker`),
# comprueba que el `TPESampler` de verdad los recibió.

# %%
_SAMPLER_KWARGS: dict[str, object] = {
    "multivariate": True,
    "constant_liar": True,
    "n_startup_trials": 10,
    "n_ei_candidates": 24,
    "group": False,
}


def _fabrica_sampler(
    worker: int, semilla_capturada: dict[str, object] | None = None
) -> Callable[[int], optuna.samplers.TPESampler]:
    """Fábrica `(n_previos) -> TPESampler` para `worker`.

    Sembrada con `BASE_SEMILLA_SAMPLER + 1000*worker + n_previos`: dos workers nunca
    comparten semilla, y reanudar no repite la secuencia anterior. Los demás kwargs salen
    de `_SAMPLER_KWARGS`. Si se pasa `semilla_capturada` (un dict), la fábrica escribe ahí
    los kwargs completos que le pasa a `TPESampler`, para que `_correr_worker` los
    reporte.
    """

    def fabrica(n_previos: int) -> optuna.samplers.TPESampler:
        semilla = BASE_SEMILLA_SAMPLER + 1000 * worker + n_previos
        kwargs: dict[str, object] = {"seed": semilla, **_SAMPLER_KWARGS}
        if semilla_capturada is not None:
            semilla_capturada["semilla"] = semilla
            semilla_capturada["n_previos"] = n_previos
            semilla_capturada["kwargs"] = dict(kwargs)
        return optuna.samplers.TPESampler(**kwargs)

    return fabrica


# Única fuente de los rangos del espacio: `_espacio` y `_protocolo` leen de este dict,
# así que cambiar un rango cambia lo que se muestrea y el protocolo a la vez, y reabrir
# con un rango distinto falla en `crear_estudio`.
_ESPACIO_RANGOS: dict[str, list[object]] = {
    "learning_rate_init": [1e-4, 3e-1, "log"],
    "alpha": [1e-6, 1e-1, "log"],
    "ancho": [16, 128, "step16"],
}


def _protocolo(protocolo_motor: Mapping[str, object]) -> dict[str, object]:
    """El protocolo completo que `crear_estudio` guarda y compara al reabrir.

    `protocolo_motor` es `juguete.PROTOCOLO_MOTOR`, pasado como parámetro para no
    importar torch en el proceso que llama. `politica_semillas` registra las semillas CRN
    de la búsqueda y `semilla_sampler` la fórmula de `_fabrica_sampler`: si alguna cambia,
    el estudio rechaza reabrirse en vez de mezclar trials incompatibles.
    """
    return {
        "sampler": "tpe",
        **_SAMPLER_KWARGS,
        "pruner": "nop",
        "metrica": "exactitud_val",
        "atributos": ["exactitud_verdadera", "segundos"],
        "espacio": _ESPACIO_RANGOS,
        "semilla_sampler": {
            "base": BASE_SEMILLA_SAMPLER,
            "formula": "base + 1000*worker + n_previos",
        },
        "politica_semillas": {
            "tipo": "CRN",
            "busqueda": dataclasses.asdict(_semillas_busqueda()),
        },
        **protocolo_motor,
    }


def _espacio(trial: optuna.Trial) -> dict[str, object]:
    """Único lugar que llama a `trial.suggest_*`, con los rangos de `_ESPACIO_RANGOS`."""
    valores: dict[str, object] = {}
    for nombre, (bajo, alto, modo) in _ESPACIO_RANGOS.items():
        if modo == "log":
            valores[nombre] = trial.suggest_float(nombre, bajo, alto, log=True)
        elif modo.startswith("step"):
            valores[nombre] = trial.suggest_int(nombre, bajo, alto, step=int(modo[len("step") :]))
        else:
            raise ValueError(f"modo de rango desconocido para {nombre!r}: {modo!r}")
    return valores


def _semillas_busqueda():
    return semillas_de(SEMILLA_BUSQUEDA, 1)[0]


def _params_por_worker_tras_arranque(
    trials: Iterable[optuna.trial.FrozenTrial], n_arranque: int
) -> dict[object, list[dict]]:
    """Agrupa `trials` por `user_attrs["worker"]`, en orden de `number`, solo con los que
    ya pasaron el arranque aleatorio del sampler (`t.number > n_arranque`): antes de eso,
    la primera config de cada worker coincide por construcción."""
    agrupado: dict[object, list[dict]] = {}
    for t in sorted(trials, key=lambda t: t.number):
        if t.number > n_arranque:
            agrupado.setdefault(t.user_attrs.get("worker"), []).append(t.params)
    return agrupado


def _frase_contencion(pareado: Pareado) -> str:
    """La frase de la respuesta 4 (`--comparar`) para cada veredicto de
    `estadistica.diferencia_pareada`. Dice "en promedio" porque el intervalo es sobre la
    media de la diferencia, y reporta `pareado.n` configuraciones: la unidad pareada es
    la configuración, no la semilla."""
    intervalo_confianza = f"[{pareado.ic_bajo:+.3f}, {pareado.ic_alto:+.3f}]"
    if pareado.veredicto == "mejor":
        return (
            f"en promedio, cada trial costó {pareado.diferencia_media:.3f} s más caro bajo "
            f"contención frente a uno aislado (intervalo de confianza {intervalo_confianza}, "
            f"n={pareado.n} configuraciones pareadas)"
        )
    if pareado.veredicto == "peor":
        return (
            "en promedio, cada trial bajo contención salió "
            f"{abs(pareado.diferencia_media):.3f} s más rápido que uno aislado "
            f"(intervalo de confianza {intervalo_confianza}, n={pareado.n} configuraciones "
            "pareadas)"
        )
    if pareado.veredicto == "equivalente":
        return (
            "equivalente a un trial aislado dentro del 10% de la mediana aislada "
            f"({pareado.diferencia_media:+.3f} s {intervalo_confianza}, n={pareado.n} "
            "configuraciones pareadas)"
        )
    return (
        f"no concluyente con estas {pareado.n} configuraciones "
        f"({pareado.diferencia_media:+.3f} s {intervalo_confianza})"
    )


def _contencion(pares: Iterable[Mapping[str, object]]) -> Pareado:
    """Compara `segundos_solo` contra `segundos_contencion` de `pares` con
    `estadistica.diferencia_pareada`, `delta=0.1 * mediana(segundos_solo)` y
    `direccion="minimizar"`, porque `segundos` es un costo. Separada de `_lanzar` para
    probarla con pares sintéticos."""
    segundos_solo = {p["trial_number"]: p["segundos_solo"] for p in pares}
    segundos_contencion = {p["trial_number"]: p["segundos_contencion"] for p in pares}
    mediana_solo = statistics.median(segundos_solo.values())
    return diferencia_pareada(
        segundos_solo, segundos_contencion, delta=0.1 * mediana_solo, direccion="minimizar"
    )


def _construir_objetivo(worker: int):
    # Import perezoso: importar el juguete carga torch, y esta función solo corre dentro
    # de un worker que va a entrenar, nunca en el proceso padre de `--lanzar`, que no debe
    # reservar un contexto CUDA extra.
    from taller_hpo.problemas import juguete

    objetivo = en_proceso(
        juguete.entrenar,
        _espacio,
        metricas=["exactitud_val"],
        semillas=_semillas_busqueda(),
        atributos=["exactitud_verdadera", "segundos"],
    )

    def objetivo_con_worker(trial: optuna.Trial) -> float:
        # user_attrs["worker"]: así --lanzar cuenta cuántos trials aportó cada proceso.
        trial.set_user_attr("worker", worker)
        return objetivo(trial)

    return objetivo_con_worker


# %% [markdown]
# ## Modo `--preparar`: crea el estudio una vez, sin optimizar


# %%
def _preparar_estudio() -> None:
    """Crea o reabre el estudio una sola vez y termina (ver por qué existe `--preparar` al
    principio del archivo)."""
    _exigir_tarjeta_de_video()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from taller_hpo.problemas import juguete  # perezoso: ver _construir_objetivo

    # flush=True: --lanzar redirige el stdout de cada worker a un archivo con búfer de
    # bloque y lo sondea para reimprimir esta línea; sin flush no la vería hasta el final.
    print(f"Motor: {juguete.DESCRIPCION}", flush=True)

    destino = _destino()
    nombre = _nombre_estudio()
    study = crear_estudio(
        nombre,
        destino=destino,
        direcciones=["maximize"],
        # La fábrica y no un sampler ya sembrado: re-preparar un estudio con trials
        # emitiría el aviso de "sampler con semilla fija reabierto".
        sampler=_fabrica_sampler(worker=0),
        pruner=optuna.pruners.NopPruner(),
        protocolo=_protocolo(juguete.PROTOCOLO_MOTOR),
    )
    n_trials = len(study.get_trials(deepcopy=False))
    print(
        f"estudio {nombre!r} preparado en {_destino_redactado(destino)!r} "
        f"({n_trials} trial(s) existentes)"
    )


# %% [markdown]
# ## Modo worker (default)


# %%
def _correr_worker() -> None:
    _exigir_tarjeta_de_video()
    # El log INFO de Optuna imprime una línea por trial y taparía las respuestas numeradas.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from taller_hpo.problemas import juguete  # perezoso: ver _construir_objetivo

    # Imprime el motor al arrancar, con flush=True por el mismo motivo que en
    # `_preparar_estudio`.
    print(f"Motor: {juguete.DESCRIPCION}", flush=True)

    worker = _resolver_worker()
    destino = _destino()
    _verificar_journal_no_9p(destino)
    nombre = _nombre_estudio()
    n_total = _n_total()

    # Desglose interno de memoria de este worker; la medición de los N workers juntos es
    # del proceso padre. Se mide alrededor del primer tensor en el dispositivo y no del
    # import del juguete, que agrega solo 0-2 MiB: el contexto real aparece con la primera
    # asignación. `empty_cache()` libera el tensor de sondeo antes de reiniciar el pico,
    # así `max_memory_reserved()` cuenta solo lo que reservó el entrenamiento. Con la
    # tarjeta compartida, una sola lectura de `contexto_mib` no es un número limpio.
    import torch

    contexto_mib = _medir_contexto_mib(juguete.DISPOSITIVO)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    semilla_capturada: dict[str, object] = {}
    study = crear_estudio(
        nombre,
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler(worker, semilla_capturada),
        pruner=optuna.pruners.NopPruner(),
        protocolo=_protocolo(juguete.PROTOCOLO_MOTOR),
    )
    if "semilla" in semilla_capturada:
        print("SEMILLA_SAMPLER_JSON " + json.dumps({"worker": worker, **semilla_capturada}))

    # Comprobación independiente de _SAMPLER_KWARGS, leída del sampler ya construido. Los
    # atributos privados (_multivariate, _constant_liar, _n_startup_trials,
    # _n_ei_candidates, _group) son los de TPESampler.__init__ en Optuna 5.
    sampler_real = {clave: getattr(study.sampler, "_" + clave) for clave in _SAMPLER_KWARGS}
    print("SAMPLER_REAL_JSON " + json.dumps({"worker": worker, **sampler_real}))

    # Sin recuperar_huerfanos: solo vale en modo secuencial. Aquí un trial RUNNING casi
    # siempre es de otro worker que sigue corriendo.

    inicio = time.perf_counter()
    # Divergencia (learning rate grande, pesos no finitos) puede pasar con
    # learning_rate_init hasta 3e-1. Ese trial, o uno con MetricaNoFinita, se marca FAIL
    # en vez de tirar la búsqueda entera del worker.
    optimizar(
        study,
        _construir_objetivo(worker),
        n_total=n_total,
        atrapar=(juguete.Divergencia, MetricaNoFinita),
    )
    segundos = time.perf_counter() - inicio

    # max_memory_reserved: lo que reservó el asignador de torch durante optimizar(). Es un
    # desglose interno; la huella de los N workers la mide el padre (respuesta 6).
    asignador_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)
    total_mib = contexto_mib + asignador_mib
    print(
        f"memoria de la tarjeta de video, worker {worker} (desglose interno): "
        f"contexto(nvidia-smi, antes/después del primer tensor en "
        f"{juguete.DISPOSITIVO})~={contexto_mib:.1f} MiB + "
        f"asignador(torch.cuda.max_memory_reserved)={asignador_mib:.1f} MiB = "
        f"total~={total_mib:.1f} MiB (nvidia-smi mide TODA la tarjeta de video, no solo "
        "este proceso: con otros procesos activos el delta es ruidoso -- ver la huella "
        "medida por el proceso padre para el número que sí suma entre workers)"
    )
    print(
        "MEMORIA_GPU_WORKER_JSON "
        + json.dumps(
            {
                "worker": worker,
                "contexto_mib": contexto_mib,
                "asignador_mib": asignador_mib,
                "total_mib": total_mib,
            }
        )
    )

    estados_terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    terminados = study.get_trials(deepcopy=False, states=estados_terminados)
    propios = sorted(
        (t for t in terminados if t.user_attrs.get("worker") == worker), key=lambda t: t.number
    )
    print(
        f"worker {worker}: aportó {len(propios)} trial(s) propio(s); estudio con "
        f"{len(terminados)} terminados (meta n_total={n_total}) en {segundos:.2f} s"
    )
    # Con --comparar, el worker aislado imprime sus propios trials leídos del estudio, y
    # _lanzar cruza cada par de CONTENCION_JSON contra esta lista.
    print(
        "TRIALS_PROPIOS_JSON "
        + json.dumps(
            {
                "worker": worker,
                "trials": [
                    {"number": t.number, "params": t.params, "segundos": t.user_attrs["segundos"]}
                    for t in propios
                ],
            }
        )
    )
    # Vida de este proceso desde que importó este módulo: una prueba la compara con el
    # `reloj_1` que mide el padre, que incluye el arranque del proceso.
    print(
        "VIDA_PROCESO_JSON "
        + json.dumps({"worker": worker, "inicio": _INICIO_PROCESO, "fin": time.monotonic()})
    )


# %% [markdown]
# ## Modo local: `--lanzar N`


# Intervalo de muestreo del pico de memoria y del log del worker 0: bastante fino para no
# perder un pico corto, bastante grueso para no gastar el padre en sondeos.
_INTERVALO_MUESTREO_S = 0.2


def _ultima_linea_no_vacia(log_ruta: Path) -> str:
    """La última línea no vacía de `log_ruta`, o `"(log vacío)"`. `_spawn_n_workers` la
    agrega al error de cada worker fallido, para que el motivo se vea sin abrir el log."""
    lineas = [linea for linea in log_ruta.read_text(encoding="utf-8").splitlines() if linea.strip()]
    return lineas[-1] if lineas else "(log vacío)"


def _spawn_n_workers(
    n: int,
    entorno_base: Mapping[str, str],
    estudios: Path,
    prefijo_log: str,
    *,
    imprimir_motor: bool = False,
) -> tuple[float, list[tuple[int, int, Path]], float | None]:
    """Arranca `n` procesos worker sobre `entorno_base` (solo cambia TALLER_WORKER=0..n-1),
    espera a todos y devuelve
    `(segundos_de_reloj, [(worker, código_salida, ruta_log), ...], pico_memoria_gpu_mib)`.

    Mientras espera, muestrea `memory.used` de nvidia-smi y se queda con el máximo (`None`
    si nvidia-smi no está disponible). Con `imprimir_motor=True` también sondea el log del
    worker 0 e imprime su línea `"Motor: "` en cuanto aparece.

    RuntimeError si alguno sale con código distinto de 0, con el log y la última línea de
    cada worker fallido.
    """
    script = _SCRIPT
    inicio = time.perf_counter()
    procesos = []
    for worker in range(n):
        entorno = {**entorno_base, "TALLER_WORKER": str(worker)}
        log_ruta = estudios / f"{prefijo_log}_worker_{worker}.log.txt"
        log_f = open(log_ruta, "w", encoding="utf-8")
        proceso = subprocess.Popen(
            [sys.executable, script], env=entorno, stdout=log_f, stderr=subprocess.STDOUT
        )
        procesos.append((worker, proceso, log_f, log_ruta))

    pico_memoria_mib = _memoria_gpu_usada_mib()
    log_worker_0 = procesos[0][3]
    motor_impreso = not imprimir_motor
    while True:
        actual = _memoria_gpu_usada_mib()
        if actual is not None and (pico_memoria_mib is None or actual > pico_memoria_mib):
            pico_memoria_mib = actual
        if not motor_impreso and log_worker_0.exists():
            linea_motor = next(
                (
                    linea
                    for linea in log_worker_0.read_text(encoding="utf-8").splitlines()
                    if linea.startswith("Motor: ")
                ),
                None,
            )
            if linea_motor is not None:
                print(linea_motor)
                motor_impreso = True
        if all(proceso.poll() is not None for _, proceso, _, _ in procesos):
            break
        time.sleep(_INTERVALO_MUESTREO_S)

    if not motor_impreso and log_worker_0.exists():
        # Si el worker 0 terminó último, el sondeo pudo no ver su línea "Motor: "; al
        # salir, el proceso ya vació su búfer, así que una última lectura la encuentra.
        linea_motor = next(
            (
                linea
                for linea in log_worker_0.read_text(encoding="utf-8").splitlines()
                if linea.startswith("Motor: ")
            ),
            None,
        )
        if linea_motor is not None:
            print(linea_motor)
            motor_impreso = True

    resultados = []
    for worker, proceso, log_f, log_ruta in procesos:
        codigo = proceso.wait()
        log_f.close()
        resultados.append((worker, codigo, log_ruta))
    segundos_reloj = time.perf_counter() - inicio

    fallidos = [(w, c, r) for w, c, r in resultados if c != 0]
    if fallidos:
        detalle = "; ".join(
            f"worker {w} salió con código {c} (log: {r}): {_ultima_linea_no_vacia(r)}"
            for w, c, r in fallidos
        )
        raise RuntimeError(f"{len(fallidos)} de {n} worker(s) fallaron: {detalle}")

    return segundos_reloj, resultados, pico_memoria_mib


def _memoria_gpu_por_worker(resultados: list[tuple[int, int, Path]]) -> dict[int, dict[str, float]]:
    """Lee `MEMORIA_GPU_WORKER_JSON {...}` del log de cada worker (una línea
    por worker, siempre: ver `_correr_worker`) y arma
    `{worker: {"contexto_mib", "asignador_mib", "total_mib"}}`."""
    memoria: dict[int, dict[str, float]] = {}
    for worker, _codigo, log_ruta in resultados:
        for linea in log_ruta.read_text(encoding="utf-8").splitlines():
            if linea.startswith("MEMORIA_GPU_WORKER_JSON "):
                datos = json.loads(linea.removeprefix("MEMORIA_GPU_WORKER_JSON "))
                memoria[worker] = {
                    "contexto_mib": datos["contexto_mib"],
                    "asignador_mib": datos["asignador_mib"],
                    "total_mib": datos["total_mib"],
                }
                break
    return memoria


def _semilla_sampler_por_worker(resultados: list[tuple[int, int, Path]]) -> dict[int, dict]:
    """Lee `SEMILLA_SAMPLER_JSON {...}` del log de cada worker y arma
    `{worker: {"semilla", "n_previos", "kwargs"}}`. `kwargs` son los que el worker de
    verdad le pasó a `TPESampler`."""
    semillas: dict[int, dict] = {}
    for worker, _codigo, log_ruta in resultados:
        for linea in log_ruta.read_text(encoding="utf-8").splitlines():
            if linea.startswith("SEMILLA_SAMPLER_JSON "):
                datos = json.loads(linea.removeprefix("SEMILLA_SAMPLER_JSON "))
                semillas[worker] = {
                    "semilla": datos["semilla"],
                    "n_previos": datos["n_previos"],
                    "kwargs": datos.get("kwargs"),
                }
                break
    return semillas


def _trials_previos(destino: str, nombre: str) -> set[int]:
    """Números de trial COMPLETE/PRUNED que el estudio `nombre` ya tenía en `destino`
    antes de esta llamada a `--lanzar`, o vacío si el estudio no existe todavía.

    Sirve para reportar solo los trials de esta sesión: sin esto, una segunda corrida sobre
    un estudio completo reportaría los números de la anterior.
    """
    storage = abrir_storage(destino)
    try:
        study = optuna.load_study(study_name=nombre, storage=storage)
    except KeyError:
        return set()
    estados_terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return {t.number for t in study.get_trials(deepcopy=False, states=estados_terminados)}


def _trials_corriendo(destino: str, nombre: str) -> list[int]:
    """Números de trial RUNNING que el estudio `nombre` ya tenía en `destino` antes de
    esta llamada a `--lanzar`, o vacío si el estudio no existe todavía.

    `_lanzar` la llama antes de lanzar ningún worker: en ese momento nada más toca el
    estudio, así que un RUNNING es un huérfano de una corrida anterior cortada.
    """
    storage = abrir_storage(destino)
    try:
        study = optuna.load_study(study_name=nombre, storage=storage)
    except KeyError:
        return []
    return [
        t.number
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    ]


def _lanzar(n: int, *, comparar: bool) -> None:
    if n < 1:
        raise SystemExit(f"--lanzar debe ser >= 1, no {n}")

    # La verificación del montaje va antes que la de la tarjeta: no toca la tarjeta y su
    # aviso es más fácil de resolver.
    destino = _destino()
    _verificar_journal_no_9p(destino)

    # El padre también exige la tarjeta antes de tocar el estudio: sin ella, el mensaje de
    # cada worker quedaría enterrado en su log.
    _exigir_tarjeta_de_video()

    # El log INFO de Optuna taparía las respuestas numeradas del proceso padre.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    nombre = _nombre_estudio()
    n_total = _n_total()
    estudios = _estudios_dir()
    estudios.mkdir(parents=True, exist_ok=True)

    # Antes de lanzar nada: qué trials ya tenía el estudio. Si alguien corre el paso dos
    # veces sobre el mismo TALLER_ESTUDIOS, los workers no entrenan nada nuevo y las
    # respuestas reportarían la corrida anterior.
    numeros_previos = _trials_previos(destino, nombre)

    # También antes de lanzar nada: huérfanos RUNNING de una corrida cortada, con un
    # mensaje que da las dos salidas.
    huerfanos_previos = _trials_corriendo(destino, nombre)
    if huerfanos_previos:
        raise RuntimeError(
            f"el estudio {nombre!r} en {_destino_redactado(destino)!r} ya tenía "
            f"{len(huerfanos_previos)} trial(s) RUNNING (números {huerfanos_previos}) ANTES "
            "de lanzar ningún worker de esta sesión -- huérfanos de una corrida cortada (p. "
            "ej. un kill -9 externo a un worker). Recupéralos en modo secuencial con "
            "taller_hpo.estudio.recuperar_huerfanos, o usa un TALLER_ESTUDIO nuevo."
        )

    # Los N workers reciben el mismo TALLER_DESTINO/TALLER_ESTUDIO/TALLER_N_TOTAL, ya
    # resueltos, y un TALLER_WORKER 0..N-1 explícito.
    entorno_base = {
        **os.environ,
        "TALLER_DESTINO": destino,
        "TALLER_ESTUDIO": nombre,
        "TALLER_N_TOTAL": str(n_total),
    }

    # Ruido de nvidia-smi antes de lanzar nada: varias muestras dicen si memory.used ya
    # variaba por otros procesos de la máquina. La ventana es más corta que la corrida, y
    # la respuesta 6 lo dice al imprimirla.
    _N_MUESTRAS_RUIDO = 5
    _muestras_ruido = []
    for _ in range(_N_MUESTRAS_RUIDO):
        _m = _memoria_gpu_usada_mib()
        if _m is not None:
            _muestras_ruido.append(_m)
        time.sleep(_INTERVALO_MUESTREO_S)
    ruido_ambiental_mib: float | None = None
    if len(_muestras_ruido) >= 2:
        ruido_ambiental_mib = max(_muestras_ruido) - min(_muestras_ruido)

    # Línea base de memoria antes de lanzar: el pico menos esta base es la huella de los n
    # workers juntos (respuesta 6).
    memoria_base_mib = _memoria_gpu_usada_mib()
    segundos_reloj, resultados, pico_memoria_mib = _spawn_n_workers(
        n, entorno_base, estudios, "anexo_paralelo", imprimir_motor=True
    )

    storage = abrir_storage(destino)
    study = optuna.load_study(study_name=nombre, storage=storage)
    estados_terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    terminados = study.get_trials(deepcopy=False, states=estados_terminados)
    corriendo = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))

    if corriendo:
        # Aquí los N workers ya terminaron: un RUNNING es un huérfano real, por ejemplo de
        # un kill -9 externo.
        numeros = [t.number for t in corriendo]
        raise RuntimeError(
            f"{len(corriendo)} trial(s) quedaron RUNNING tras juntar los {n} workers "
            f"(números {numeros}); con todos los procesos ya terminados, es un huérfano real."
        )

    # Todo lo que se reporta abajo es sobre `nuevos`, los trials de esta sesión, nunca
    # sobre el estudio completo. `restante` es lo que faltaba para n_total al arrancar: 0
    # si el estudio ya estaba completo.
    nuevos = [t for t in terminados if t.number not in numeros_previos]
    restante = max(0, n_total - len(numeros_previos))

    por_worker = Counter(t.user_attrs.get("worker") for t in nuevos)
    tiempo_secuencial = sum(t.user_attrs.get("segundos", 0.0) for t in nuevos)
    inicio_referencia = min((t.datetime_start for t in nuevos), default=None)

    resultados_dir = _resultados_dir()
    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    for worker in range(n):
        propios = [t for t in nuevos if t.user_attrs.get("worker") == worker]
        xs = [(t.datetime_complete - inicio_referencia).total_seconds() for t in propios]
        ys = [t.number for t in propios]
        ax.scatter(xs, ys, label=f"worker {worker}", marker="o")
    ax.set_xlabel("segundos (fin de cada trial) desde el primer trial NUEVO de esta sesión")
    ax.set_ylabel("número de trial")
    ax.set_title(f"Anexo paralelo: {n} workers, {len(nuevos)} trials nuevos (n_total={n_total})")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    figura = resultados_dir / "anexo_paralelo.png"
    fig.savefig(figura, dpi=120)

    if numeros_previos:
        # restante == 0: el estudio ya estaba completo y el assert de abajo exige 0 trials
        # nuevos, así que el rango esperado es [0, 0].
        rango_nuevos = "[0, 0]" if restante == 0 else f"[{restante}, {restante + n - 1}]"
        print(
            f"1. Trials totales en el estudio: {len(terminados)} ({len(nuevos)} nuevos de "
            f"esta sesión -- REANUDACIÓN: ya había {len(numeros_previos)} de (una) corrida "
            f"anterior). Con {n} workers y lo que faltaba para n_total={n_total} "
            f"({restante}), rango esperado de trials NUEVOS: {rango_nuevos}"
        )
    else:
        print(
            f"1. Trials totales: {len(terminados)} (rango esperado con {n} workers y "
            f"n_total={n_total}: [{n_total}, {n_total + n - 1}])"
        )
    assert len(terminados) >= n_total, f"{len(terminados)} no alcanza n_total={n_total}"
    if restante == 0:
        assert len(nuevos) == 0, (
            f"el estudio ya tenía {len(numeros_previos)} trials (>= n_total={n_total}) "
            f"antes de esta sesión: no debieron correr trials nuevos, pero corrieron "
            f"{len(nuevos)}"
        )
    else:
        assert restante <= len(nuevos) <= restante + n - 1, (
            f"{len(nuevos)} trials nuevos fuera de [{restante}, {restante + n - 1}]"
        )
    print(f"2. Trials por worker (nuevos de esta sesión): {dict(sorted(por_worker.items()))}")
    semillas_sampler = _semilla_sampler_por_worker(resultados)
    if semillas_sampler:
        # Semillas distintas por worker (ver _fabrica_sampler): sin ellas, todos
        # propondrían la misma primera configuración.
        print(f"   Semilla de sampler por worker: {semillas_sampler}")
        print("SEMILLAS_SAMPLER_JSON " + json.dumps(semillas_sampler))
    print(
        f"3. Verificación: {len(corriendo)} trial(s) quedaron RUNNING sin terminar tras "
        f"juntar los {n} workers (medido sobre TODO el estudio). También es una garantía "
        "de diseño: optimizar() corre con n_jobs=1, así que un worker solo suelta su "
        "trial cuando study.tell() ya lo cerró, nunca a medias -- si algo hubiera "
        "quedado RUNNING, el RuntimeError de arriba ya habría interrumpido este paso "
        "antes de llegar aquí."
    )

    huella_1_worker_mib: float | None = None
    costo_gpu_1_proceso_mib: float | None = None
    if comparar and restante == 0:
        # Reanudación completa: esta sesión no entrenó nada, y una aceleración contra una
        # corrida secuencial no significaría nada.
        print(
            f"4. Tiempo de reloj con {n} worker(s): {segundos_reloj:.3f} s para "
            f"{len(nuevos)} trials nuevos -- REANUDACIÓN: el estudio ya tenía "
            f"n_total={n_total} trials completos antes de esta sesión, así que ningún "
            "worker entrenó nada nuevo. El paso no mide una aceleración aquí: usa un "
            "TALLER_ESTUDIO nuevo, o un TALLER_N_TOTAL más alto, para medirla de verdad."
        )
    elif comparar:
        # Línea base secuencial real: una corrida aparte con un solo worker entrena las
        # mismas configuraciones de esta sesión, encoladas con enqueue_trial en el mismo
        # orden. n_comparacion = len(nuevos), no n_total, para comparar la misma carga de
        # trabajo. El nombre lleva el sufijo time.time_ns() para no reabrir un estudio
        # auxiliar de otra corrida.
        nuevos_ordenados = sorted(nuevos, key=lambda t: t.number)
        n_comparacion = len(nuevos_ordenados)
        nombre_solo = f"{nombre}_solo_comparacion_{time.time_ns()}"
        # Mismo motor que el estudio principal, leído de su protocolo (importar el juguete
        # en el padre crearía un contexto CUDA extra). crear_estudio corre aquí, antes de
        # lanzar el worker, para poder encolar las configuraciones.
        protocolo_motor_actual = {
            k: study.user_attrs["protocolo"][k] for k in ("motor", "dispositivo")
        }
        study_solo = crear_estudio(
            nombre_solo,
            destino=destino,
            direcciones=["maximize"],
            sampler=_fabrica_sampler(worker=0),
            pruner=optuna.pruners.NopPruner(),
            protocolo=_protocolo(protocolo_motor_actual),
        )
        for t in nuevos_ordenados:
            study_solo.enqueue_trial(t.params)
        entorno_solo = {
            **entorno_base,
            "TALLER_ESTUDIO": nombre_solo,
            "TALLER_N_TOTAL": str(n_comparacion),
        }
        memoria_base_solo_mib = _memoria_gpu_usada_mib()
        # Instante del padre con time.monotonic(), el mismo reloj que `_INICIO_PROCESO`.
        # Con `fin_reloj_solo` encierra la misma llamada que mide `reloj_1`, y la vida del
        # worker debe caer dentro de esta ventana.
        inicio_reloj_solo = time.monotonic()
        reloj_1, resultados_solo, pico_memoria_solo_mib = _spawn_n_workers(
            1, entorno_solo, estudios, "anexo_paralelo_solo"
        )
        fin_reloj_solo = time.monotonic()
        if memoria_base_solo_mib is not None and pico_memoria_solo_mib is not None:
            huella_1_worker_mib, _ = _huella_mib(pico_memoria_solo_mib, memoria_base_solo_mib, 1)
        # Costo de un proceso aislado para la respuesta 6: contexto CUDA y asignador de
        # torch que reportó el worker solo, sin contención de otros workers.
        memoria_solo_por_worker = _memoria_gpu_por_worker(resultados_solo)
        if 0 in memoria_solo_por_worker:
            costo_gpu_1_proceso_mib = (
                memoria_solo_por_worker[0]["contexto_mib"]
                + memoria_solo_por_worker[0]["asignador_mib"]
            )
        aceleracion = reloj_1 / segundos_reloj if segundos_reloj > 0 else float("nan")
        print(
            f"4. Tiempo de reloj con {n} worker(s): {segundos_reloj:.3f} s para "
            f"{len(nuevos)} trials nuevos. Tiempo de reloj con 1 worker entrenando las "
            f"MISMAS {n_comparacion} configuraciones (encoladas en el mismo orden con "
            f"enqueue_trial, no vueltas a muestrear), medido en una corrida APARTE: "
            f"{reloj_1:.3f} s. Aceleración observada en esta corrida: {aceleracion:.3f}x "
            "(reloj_1 / reloj_N, UNA corrida de cada lado: no es una estimación con "
            "intervalo de confianza)."
        )
        # Se lee el estudio auxiliar antes de borrarlo: debe tener exactamente
        # n_comparacion trials, uno por config encolada.
        terminados_solo = sorted(
            study_solo.get_trials(deepcopy=False, states=estados_terminados),
            key=lambda t: t.number,
        )
        if len(terminados_solo) != n_comparacion:
            raise RuntimeError(
                f"la corrida de comparación con 1 worker produjo {len(terminados_solo)} "
                f"trial(s) y debía producir exactamente {n_comparacion} (una config encolada "
                "por trial nuevo de esta sesión)"
            )
        # Costo bajo contención frente a aislado, emparejado por `trial_number` del
        # estudio principal. El worker aislado imprime además `TRIALS_PROPIOS_JSON`, una
        # fuente independiente para verificar cada par. `a` = aislado, `b` = bajo
        # contención, direccion="minimizar": una diferencia positiva significa que la
        # contención cuesta más.
        pares_contencion = [
            {
                "trial_number": t.number,
                # t_solo.params, no t.params: verifica que el worker solo entrenó la
                # config que encoló enqueue_trial.
                "params": t_solo.params,
                "segundos_solo": t_solo.user_attrs["segundos"],
                "segundos_contencion": t.user_attrs["segundos"],
            }
            for t, t_solo in zip(nuevos_ordenados, terminados_solo, strict=True)
        ]
        if n_comparacion < 2:
            # Con 1 sola config no hay con qué armar un intervalo pareado.
            pareado_contencion = None
            print(
                "   Costo por trial bajo contención frente a uno aislado: no concluyente "
                "(1 sola configuración nueva en esta sesión, no alcanza para una prueba "
                "pareada)."
            )
        else:
            pareado_contencion = _contencion(pares_contencion)
            frase_contencion = _frase_contencion(pareado_contencion)
            print(f"   Costo por trial bajo contención frente a uno aislado: {frase_contencion}.")
            print(
                "   Nota: la corrida aislada siempre se mide DESPUÉS de la corrida principal "
                "(cualquier cambio de carga de la máquina entre las dos entra en esta "
                "diferencia) y la corrida principal paga un arranque de CUDA por worker "
                "mientras la aislada paga uno solo."
            )
            print(
                "CONTENCION_JSON "
                + json.dumps(
                    {
                        **dataclasses.asdict(pareado_contencion),
                        "params_solo": [p["params"] for p in pares_contencion],
                        "pares": pares_contencion,
                    }
                )
            )
        datos_aceleracion = {
            "reloj_1": reloj_1,
            "reloj_n": segundos_reloj,
            "aceleracion": aceleracion,
            "estudio_comparacion": nombre_solo,
            "n_trials_comparacion": n_comparacion,
            "tiempo_secuencial_bajo_contencion": tiempo_secuencial,
            "inicio_reloj_solo": inicio_reloj_solo,
            "fin_reloj_solo": fin_reloj_solo,
        }
        print("ACELERACION_JSON " + json.dumps(datos_aceleracion))
        # Se borra el estudio auxiliar para no acumular estudios que nadie vuelve a leer.
        optuna.delete_study(study_name=nombre_solo, storage=storage)
    else:
        promedio_trial = segundos_reloj / len(nuevos) if nuevos else float("nan")
        print(
            f"4. Tiempo de reloj con {n} worker(s): {segundos_reloj:.3f} s para "
            f"{len(nuevos)} trials nuevos ({promedio_trial:.3f} s/trial en promedio de "
            f"reloj, bajo contención). La suma de 'segundos' de cada trial bajo "
            f"contención ({tiempo_secuencial:.3f} s) puede SOBRESTIMAR el costo real de "
            "un trial aislado (varios procesos compitiendo por la misma tarjeta de video); "
            "esto no lo mide, así que no es una aceleración. Usa "
            "--comparar, o "
            "PRESUPUESTO=taller|completo (lo activan por defecto), para medir la "
            "aceleración real contra una corrida secuencial."
        )

    secuencia = [t.user_attrs.get("worker") for t in sorted(nuevos, key=lambda t: t.number)]
    print(f"5. Secuencia de workers por número de trial (nuevos de esta sesión): {secuencia}")
    print("SECUENCIA_JSON " + json.dumps(secuencia))
    print(
        "   El orden de llegada entre trials de distintos workers no está garantizado: "
        "depende de cuándo cada worker propone y termina su trial (incluye el arranque del "
        "proceso) y de la carrera de n_previos al abrir el estudio -- con n>=2 workers esa "
        "carrera existe también en un estudio FRESCO, no solo al reanudar uno existente."
    )
    # Reanudación parcial: la semilla efectiva de cada worker depende de quién abrió el
    # estudio primero (ver _fabrica_sampler), así que repetir "con las mismas semillas"
    # no sería cierto.
    reanudacion_parcial = bool(numeros_previos) and restante > 0
    if comparar and n >= 2 and restante == 0:
        # Reanudación completa: no hay orden que repetir.
        print(
            "   Repetición de orden/reproducibilidad omitida: REANUDACIÓN completa, esta "
            "sesión no entrenó ningún trial nuevo con qué compararla (usa un TALLER_ESTUDIO "
            "nuevo, o un TALLER_N_TOTAL más alto, para medirla de verdad)."
        )
    elif comparar and n >= 2 and reanudacion_parcial:
        print(
            "   Repetición de orden/reproducibilidad omitida: REANUDACIÓN parcial (ya había "
            f"{len(numeros_previos)} trial(s) antes de esta sesión) -- la semilla efectiva de "
            "cada worker depende de n_previos, que depende de quién abrió el estudio primero: "
            "repetirla correría otra carrera distinta, con otro n_previos, no la misma "
            "semilla que usó esta sesión (usa un TALLER_ESTUDIO nuevo para medir "
            "reproducibilidad de verdad)."
        )
    elif comparar and n >= 2:
        # Solo con un estudio fresco. Aun así n_previos no es 0 garantizado: con varios
        # workers, el que abre más tarde puede ver trials de otro, así que la semilla
        # efectiva se mide, no se asume. Se repite con el mismo n_total: con n_total=n,
        # todos los trials caerían en el arranque aleatorio y coincidirían por
        # construcción. Es una observación de dos corridas, no una prueba estadística. El
        # nombre lleva sufijo, como `nombre_solo`.
        nombre_rep = f"{nombre}_orden_repetido_{time.time_ns()}"
        entorno_rep = {**entorno_base, "TALLER_ESTUDIO": nombre_rep, "TALLER_N_TOTAL": str(n_total)}
        _, resultados_rep, _ = _spawn_n_workers(n, entorno_rep, estudios, "anexo_paralelo_orden")
        study_rep = optuna.load_study(study_name=nombre_rep, storage=storage)
        terminados_rep = study_rep.get_trials(deepcopy=False, states=estados_terminados)
        secuencia_rep = [
            t.user_attrs.get("worker") for t in sorted(terminados_rep, key=lambda t: t.number)
        ]
        # Conteo por worker independiente de `secuencia_rep`, para cruzarlo en una prueba.
        conteo_por_worker_repetido = dict(
            Counter(t.user_attrs.get("worker") for t in terminados_rep)
        )

        # Las secuencias solo se comparan si tienen la misma longitud: un prefijo
        # arbitrario podría "coincidir" sin que las corridas se parezcan.
        if len(secuencia) != len(secuencia_rep):
            coincide = None
            veredicto = (
                f"longitudes distintas ({len(secuencia)} vs {len(secuencia_rep)}): no "
                "comparables directamente"
            )
        else:
            coincide = secuencia == secuencia_rep
            veredicto = (
                "coincidió con la corrida de arriba (pudo ser casualidad: nada lo garantiza, "
                "ver la razón arriba)."
                if coincide
                else "cambió respecto a la corrida de arriba, como puede pasar."
            )
        print(f"   Secuencia repetida: {secuencia_rep} -- {veredicto}")

        # Semillas efectivas de la repetición, leídas de sus logs: "con las mismas
        # semillas" solo vale para los workers donde coinciden.
        semillas_rep = _semilla_sampler_por_worker(resultados_rep)
        coincidencias_semilla = sum(
            1
            for w, datos in semillas_sampler.items()
            if w in semillas_rep and semillas_rep[w]["semilla"] == datos["semilla"]
        )
        _razon_semillas = (
            " (n_previos depende de quién abrió el estudio primero: otra fuente de no "
            "reproducibilidad, además de cuándo el SO agenda cada proceso)."
            if coincidencias_semilla < n
            else " (las dos corridas vieron el mismo n_previos por worker en esta ocasión)."
        )
        print(
            f"   Semillas efectivas de sampler iguales en {coincidencias_semilla} de {n} "
            f"worker(s) entre la corrida original y la repetición{_razon_semillas}"
        )

        # Lo que importa es si las configuraciones propuestas cambian entre corridas. Se
        # compara por worker (la k-ésima config de cada uno), solo entre workers con la
        # misma semilla efectiva y solo después del arranque aleatorio del sampler.
        n_arranque = _SAMPLER_KWARGS["n_startup_trials"]

        params_originales = _params_por_worker_tras_arranque(nuevos, n_arranque)
        params_repetidos = _params_por_worker_tras_arranque(terminados_rep, n_arranque)
        comparados = 0
        coinciden_params = 0
        for worker_id, propios in params_originales.items():
            datos_orig = semillas_sampler.get(worker_id)
            datos_rep = semillas_rep.get(worker_id)
            if not datos_orig or not datos_rep or datos_orig["semilla"] != datos_rep["semilla"]:
                continue
            for a, b in zip(propios, params_repetidos.get(worker_id, []), strict=False):
                comparados += 1
                if a == b:
                    coinciden_params += 1
        # Mejor valor de esta sesión (solo `nuevos`, no study.best_value, que en una
        # reanudación mezcla sesiones anteriores) contra el mejor de la repetición.
        mejor_original = max((t.value for t in nuevos), default=float("nan"))
        mejor_repetido = max((t.value for t in terminados_rep), default=float("nan"))
        # comparados == 0: ningún worker repitió su semilla efectiva, así que no hubo nada
        # que comparar y "0 de 0" confundiría.
        frase_config = (
            "no comparable: ningún worker repitió su semilla efectiva en la repetición"
            if comparados == 0
            else f"{coinciden_params} de {comparados}"
        )
        print(
            "   Configuraciones que coinciden (mismo worker, mismo orden dentro de ese "
            f"worker, trial > n_startup_trials={n_arranque}, solo entre workers con la "
            f"MISMA semilla efectiva en las dos corridas): {frase_config}. Mejor valor de "
            f"esta sesión ({len(nuevos)} trials nuevos, n_total={n_total}): "
            f"{mejor_original:.4f}; mejor valor de la repetición ({len(terminados_rep)} "
            f"trials, mismo n_total={n_total}): {mejor_repetido:.4f}."
        )
        print(
            "ORDEN_JSON "
            + json.dumps(
                {
                    "secuencia_original": secuencia,
                    "secuencia_repetida": secuencia_rep,
                    "conteo_por_worker_repetido": {
                        str(k): v for k, v in conteo_por_worker_repetido.items()
                    },
                    "coincide": coincide,
                    "estudio_repetido": nombre_rep,
                    "semillas_originales": {
                        str(k): v["semilla"] for k, v in semillas_sampler.items()
                    },
                    "semillas_repetidas": {str(k): v["semilla"] for k, v in semillas_rep.items()},
                    "params_originales": {str(k): v for k, v in params_originales.items()},
                    "params_repetidos": {str(k): v for k, v in params_repetidos.items()},
                    "comparados": comparados,
                    "coinciden_params": coinciden_params,
                }
            )
        )
        optuna.delete_study(study_name=nombre_rep, storage=storage)
    elif comparar:
        print("   (--comparar con n=1: no hay orden que comparar entre workers)")
    else:
        # La causa es la falta de --comparar, no el presupuesto: con --no-comparar esta
        # rama corre con cualquier presupuesto.
        print(
            f"   Esta corrida no se repitió (sin --comparar, PRESUPUESTO={PRESUPUESTO}): es "
            "una sola observación, no una medición de reproducibilidad."
        )

    memoria_por_worker = _memoria_gpu_por_worker(resultados)
    huella_total_mib: float | None = None
    huella_por_worker_mib: float | None = None
    if memoria_base_mib is not None and pico_memoria_mib is not None:
        huella_total_mib, huella_por_worker_mib = _huella_mib(pico_memoria_mib, memoria_base_mib, n)
    # Solo el asignador de torch por worker: el contexto CUDA medido con los n procesos
    # arrancando juntos sale inflado. El costo de un proceso aislado se da aparte.
    solo_asignador = {w: d["asignador_mib"] for w, d in memoria_por_worker.items()}
    # Costo de un proceso aislado (lo que reportó el worker de --comparar) frente a
    # huella_1_worker_mib, medida por el padre en esa misma corrida. Se imprimen los dos y
    # su diferencia, sin afirmar que coinciden: con una sola corrida no hay prueba.
    if costo_gpu_1_proceso_mib is not None and huella_1_worker_mib is not None:
        diferencia_1_proceso_mib = huella_1_worker_mib - costo_gpu_1_proceso_mib
        detalle_1_proceso = (
            " Costo estimado de UN proceso aislado (contexto CUDA + asignador de torch "
            f"del worker de la corrida --comparar de 1 worker): "
            f"{costo_gpu_1_proceso_mib:.1f} MiB; huella pico-menos-base medida por el "
            f"padre para esa misma corrida: {huella_1_worker_mib:.1f} MiB (diferencia "
            f"{diferencia_1_proceso_mib:+.1f} MiB -- miden cosas relacionadas pero no "
            "idénticas: nvidia-smi ve TODA la tarjeta de video de tu computadora, el "
            "desglose interno solo este proceso)."
        )
    elif costo_gpu_1_proceso_mib is not None:
        detalle_1_proceso = (
            " Costo estimado de UN proceso aislado (contexto CUDA + asignador de torch "
            f"del worker de la corrida --comparar de 1 worker): "
            f"{costo_gpu_1_proceso_mib:.1f} MiB."
        )
    else:
        detalle_1_proceso = ""
    detalle_ruido = (
        f" Ruido ambiental de memory.used (rango máximo-mínimo de {_N_MUESTRAS_RUIDO} "
        f"muestras, unos {(_N_MUESTRAS_RUIDO - 1) * _INTERVALO_MUESTREO_S:.1f} s antes de "
        "lanzar -- más corto que esta corrida, así que es una cota inferior del ruido "
        f"real, no la ventana completa): {ruido_ambiental_mib:.1f} MiB."
        if ruido_ambiental_mib is not None
        else ""
    )
    if huella_total_mib is not None:
        print(
            "6. Memoria de la tarjeta de video: huella medida por el proceso PADRE "
            "(nvidia-smi, pico durante la corrida menos la línea base justo antes de "
            "lanzar; memory.used es de TODA la tarjeta de video e incluye a otros "
            "procesos de tu computadora además de estos workers) "
            f"~= {huella_total_mib:.1f} MiB "
            f"para los {n} worker(s) juntos (~{huella_por_worker_mib:.1f} MiB/worker)."
            f"{detalle_ruido} Desglose por worker (asignador de torch, por proceso): "
            f"{solo_asignador}.{detalle_1_proceso}"
        )
    else:
        print(
            "6. Memoria de la tarjeta de video por worker (nvidia-smi no disponible "
            "para medir la huella del proceso padre; desglose por worker, asignador de "
            f"torch): {solo_asignador}"
        )
    print(
        "MEMORIA_JSON "
        + json.dumps(
            {
                "huella_total_mib": huella_total_mib,
                "huella_por_worker_mib": huella_por_worker_mib,
                "huella_1_worker_mib": huella_1_worker_mib,
                "costo_gpu_1_proceso_mib": costo_gpu_1_proceso_mib,
                "ruido_ambiental_mib": ruido_ambiental_mib,
                "por_worker": {w: {"asignador_mib": v} for w, v in solo_asignador.items()},
            }
        )
    )

    print(f"\nfigura: {figura}")


# %%
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Anexo paralelo: N workers de Optuna sobre el mismo estudio."
    )
    parser.add_argument(
        "--lanzar",
        type=int,
        metavar="N",
        default=None,
        help="modo local: arranca N procesos worker sobre el mismo Journal y espera a todos",
    )
    parser.add_argument(
        "--comparar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "con --lanzar: además corre un estudio aparte con 1 worker y el mismo "
            "n_total para medir la aceleración real (reloj_1 / reloj_N) y repite un "
            "lanzamiento corto para observar si el orden de llegada es estable. "
            "Default: activo con PRESUPUESTO=taller|completo, inactivo con "
            "PRESUPUESTO=humo"
        ),
    )
    parser.add_argument(
        "--preparar",
        action="store_true",
        help=(
            "crea o reabre el estudio UNA vez y termina, sin optimizar. Pensado para "
            "el servicio 'preparar' de compose.yaml (ver el docstring del módulo)"
        ),
    )
    args = parser.parse_args()

    if args.preparar and args.lanzar is not None:
        raise SystemExit("--preparar y --lanzar son incompatibles")
    if args.comparar is not None and args.lanzar is None:
        raise SystemExit("--comparar solo aplica junto con --lanzar")

    if args.preparar:
        _preparar_estudio()
    elif args.lanzar is not None:
        comparar = args.comparar if args.comparar is not None else PRESUPUESTO != "humo"
        _lanzar(args.lanzar, comparar=comparar)
    else:
        _correr_worker()


# Guardado tras `if __name__ == "__main__":`, a diferencia de los pasos numerados: las
# pruebas importan sus funciones puras sin ejecutar ningún modo.
if __name__ == "__main__":
    _main()
