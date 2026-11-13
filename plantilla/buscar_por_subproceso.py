"""La ruta completa por subproceso: `train_ejemplo.py` corrido como proceso aparte, con
`taller_hpo.estudio` y `taller_hpo.subproceso`.

Por cada trial, `taller_hpo.subproceso.por_subproceso` escribe un JSON de entrada, lanza
`train_ejemplo.py` como proceso completo con `--config` y `--salida`, y lee el JSON de
salida. Esta ruta hace falta cuando tu modelo vive en otro entorno: `TALLER_PYTHON`
apunta al python de ese venv, y por defecto usa `sys.executable`. No hay poda, porque tu
modelo no corre en el proceso de Optuna.

Usa el mismo `espacio`, las mismas `DISTRIBUCIONES` y las mismas semillas que
`buscar_en_proceso.py`, así las dos rutas hacen la misma búsqueda.

Necesita una tarjeta de video NVIDIA con CUDA donde corra `train_ejemplo.py`
(`uv sync --locked`).

Correr:  uv run plantilla/buscar_por_subproceso.py
Presupuesto:  PRESUPUESTO=humo | taller (default) | completo
Otro intérprete:  TALLER_PYTHON=/ruta/a/otro/venv/bin/python \\
                  uv run plantilla/buscar_por_subproceso.py
Dispositivo fijo:  TALLER_DISPOSITIVO=cuda(:N) elige qué tarjeta de video usa el
                   subproceso. Con `auto`, o sin definirla, se detecta cuál usar. En los
                   dos casos el script comprueba, antes de crear el estudio, que el
                   intérprete que va a entrenar vea una tarjeta de video.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import optuna

from taller_hpo.estudio import (
    abrir_storage,
    crear_estudio,
    optimizar,
    recuperar_huerfanos,
    sembrar_linea_base,
)
from taller_hpo.objetivo import MetricaNoFinita
from taller_hpo.semillas import Semillas
from taller_hpo.subproceso import por_subproceso
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
TRAIN_EJEMPLO = Path(__file__).resolve().parent / "train_ejemplo.py"

PRESUPUESTOS = {"humo": 2, "taller": 15, "completo": 40}

SEMILLA_BASE = 0
# Idénticas a buscar_en_proceso.py: ver el docstring del módulo.
SEMILLAS_BUSQUEDA = Semillas(datos=SEMILLA_BASE, orden=SEMILLA_BASE + 1, init=SEMILLA_BASE + 2)

# Un trial que revienta (RuntimeError), se cuelga (TimeoutError) o devuelve una métrica
# no finita (MetricaNoFinita) marca FAIL y la búsqueda sigue. KeyError queda fuera a
# propósito: una clave mal escrita es un bug y debe detener la búsqueda.
ATRAPAR: tuple[type[BaseException], ...] = (RuntimeError, TimeoutError, MetricaNoFinita)


def espacio(trial: optuna.Trial) -> dict[str, object]:
    """El ÚNICO lugar con `suggest_*`: idéntico a buscar_en_proceso.py, para que
    las dos rutas exploren el mismo espacio con la misma secuencia de llamadas."""
    return {
        "ancho": trial.suggest_int("ancho", 8, 64),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
    }


DISTRIBUCIONES = {
    "ancho": optuna.distributions.IntDistribution(8, 64),
    "learning_rate_init": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": optuna.distributions.FloatDistribution(1e-6, 1e-1, log=True),
}

# Copia de los DEFAULTS de train_ejemplo.py que quedan fijos. Aquí no se importa ese
# script, así que si cambias train_ejemplo.py, actualiza también esta copia (una prueba
# las compara). "dispositivo" no va aquí: se resuelve al correr, con `_resolver_dispositivo`.
_FIJOS_PROBLEMA = {
    "n_train": 300,
    "n_val": 100,
    "ruido": 0.25,
    "n_ruido": 65,
    "n_capas": 1,
    "activacion": "relu",
    "batch_size": 32,
    "epocas": 15,
}
_LINEA_BASE = {"ancho": 16, "learning_rate_init": 1e-2, "alpha": 1e-4}


# El taller entrena siempre en la tarjeta de video (ver el docstring del
# módulo): los valores válidos son "auto" y "cuda" (con ":N" opcional).
_PATRON_DISPOSITIVO = re.compile(r"auto|cuda(:\d+)?")


def _detectar_dispositivo(python: str) -> str:
    """Detecta la tarjeta de video importando torch con el mismo intérprete que va a
    entrenar, nunca con el proceso padre, que puede ser otro entorno. Sin tarjeta de
    video visible para ese intérprete, se detiene con un `RuntimeError` explícito."""
    try:
        resultado = subprocess.run(
            [python, "-c", "import torch; print('cuda' if torch.cuda.is_available() else '')"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as error:
        raise ValueError(
            f"el taller no pudo lanzar el intérprete {python!r} para detectar la tarjeta de "
            f"video: {error}. TALLER_PYTHON debe apuntar a un python que exista y que puedas "
            "ejecutar."
        ) from error
    if resultado.returncode != 0:
        raise ValueError(
            f"el taller no pudo detectar la tarjeta de video: el intérprete {python!r} no pudo "
            f"importar torch (código {resultado.returncode}). Instala las dependencias en "
            f"ESE entorno (uv sync --locked, con --python {python} si es otro entorno) o "
            "apunta TALLER_PYTHON a uno que ya las tenga."
        )
    salida = resultado.stdout.strip()
    if salida != "cuda":
        raise RuntimeError(
            f"{MENSAJE_SIN_TARJETA} (para el intérprete {python!r}); si tu modelo vive en "
            "otro entorno que sí la ve, apunta TALLER_PYTHON al python de ese entorno "
            f"(stderr: {resultado.stderr.strip()!r})"
        )
    return salida


def _resolver_dispositivo(
    dispositivo: str | None, python: str, *, ya_comprobado: bool = False
) -> str:
    """Resuelve el dispositivo real que entrena el subproceso, con su índice intacto.

    Con `dispositivo=None` o `'auto'`, toma `TALLER_DISPOSITIVO` o `'auto'`. Siempre
    comprueba con `_detectar_dispositivo` que el intérprete vea una tarjeta de video,
    salvo con `ya_comprobado=True`, que solo usan `main()` y el paso 07 cuando ya hicieron
    esa comprobación. Un índice que no existe falla después, en el primer trial.
    """
    if dispositivo is None or dispositivo == "auto":
        desde_env = os.environ.get("TALLER_DISPOSITIVO")
        dispositivo = desde_env if desde_env not in (None, "auto") else "auto"
    if not _PATRON_DISPOSITIVO.fullmatch(dispositivo):
        raise ValueError(
            "dispositivo (o TALLER_DISPOSITIVO) debe ser 'auto' o 'cuda' (opcionalmente con "
            f"':N'), no {dispositivo!r}"
        )
    if not ya_comprobado:
        _detectar_dispositivo(python)
    return "cuda" if dispositivo == "auto" else dispositivo


def _normalizar_dispositivo_protocolo(dispositivo: str) -> str:
    """El protocolo del estudio nunca lleva el índice de la tarjeta de video, así cambiar
    de `'cuda:0'` a `'cuda:1'` no rompe la reapertura."""
    return dispositivo.split(":", 1)[0]


def buscar(
    *,
    n_total: int,
    destino: str | os.PathLike,
    directorio_trials: str | os.PathLike,
    train_path: Path = TRAIN_EJEMPLO,
    nombre: str = "buscar_por_subproceso",
    timeout_s: float = 60.0,
    dispositivo: str | None = None,
    max_fallos: int = 10,
    _dispositivo_ya_comprobado: bool = False,
) -> optuna.Study:
    """Corre o reanuda la búsqueda completa por subproceso y devuelve el `Study`.

    `max_fallos` pasa tal cual a `optimizar()`. `_dispositivo_ya_comprobado` es de uso
    interno de `main()` y del paso 07, cuando ya comprobaron la tarjeta de video.
    """
    python_otro_entorno = os.environ.get("TALLER_PYTHON", sys.executable)
    comando = [python_otro_entorno, str(train_path)]
    dispositivo_real = _resolver_dispositivo(
        dispositivo, python_otro_entorno, ya_comprobado=_dispositivo_ya_comprobado
    )
    fijos = {**_FIJOS_PROBLEMA, "dispositivo": dispositivo_real}

    study = crear_estudio(
        nombre,
        destino=destino,
        direcciones=["maximize"],
        sampler=lambda n: optuna.samplers.TPESampler(seed=SEMILLA_BASE + n, n_startup_trials=2),
        pruner=optuna.pruners.NopPruner(),
        # La misma forma de protocolo que pasos/02_primer_estudio.py.
        protocolo={
            "sampler": "TPESampler",
            "sampler_kwargs": {"seed_base": SEMILLA_BASE, "n_startup_trials": 2},
            "pruner": "NopPruner",
            "politica_semillas": {
                "tipo": "CRN",
                "semillas": dataclasses.asdict(SEMILLAS_BUSQUEDA),
            },
            "metrica": "exactitud_val",
            "fijos": {**fijos, "dispositivo": _normalizar_dispositivo_protocolo(dispositivo_real)},
        },
    )
    recuperar_huerfanos(study)
    sembrar_linea_base(study, _LINEA_BASE, DISTRIBUCIONES)

    objetivo = por_subproceso(
        comando,
        espacio,
        metricas=["exactitud_val"],
        semillas=SEMILLAS_BUSQUEDA,
        timeout_s=timeout_s,
        directorio=directorio_trials,
        fijos=fijos,
    )
    optimizar(study, objetivo, n_total=n_total, atrapar=ATRAPAR, max_fallos=max_fallos)
    return study


def _pluralizar(n: int, singular: str, plural: str) -> str:
    """`singular` si `n == 1`, si no `plural`."""
    return singular if n == 1 else plural


_ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def _trials_terminados(destino: str | os.PathLike, nombre: str) -> int:
    """Cuántos trials COMPLETE o PRUNED tiene ya el estudio en `destino`, o 0 si no existe."""
    try:
        existente = optuna.load_study(study_name=nombre, storage=abrir_storage(destino))
    except KeyError:
        return 0
    return len(existente.get_trials(deepcopy=False, states=_ESTADOS_TERMINADOS))


def main() -> None:
    presupuesto = os.environ.get("PRESUPUESTO", "taller")
    if presupuesto not in PRESUPUESTOS:
        raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
    n_total = PRESUPUESTOS[presupuesto]

    estudios = Path(os.environ.get("TALLER_ESTUDIOS", RAIZ / "estudios"))
    destino = estudios / "buscar_por_subproceso.log"
    directorio_trials = estudios / "buscar_por_subproceso_trials"
    nombre = "buscar_por_subproceso"

    # Resuelve el dispositivo antes de abrir el Journal y de arrancar el reloj: lanzar un
    # intérprete que importa torch es costo de arranque, no de ningún trial, y sin
    # tarjeta de video el script se detiene sin dejar archivos.
    python_otro_entorno = os.environ.get("TALLER_PYTHON", sys.executable)
    dispositivo_resuelto = _resolver_dispositivo(None, python_otro_entorno)
    antes = _trials_terminados(destino, nombre)
    inicio = time.perf_counter()
    study = buscar(
        n_total=n_total,
        destino=destino,
        directorio_trials=directorio_trials,
        nombre=nombre,
        dispositivo=dispositivo_resuelto,
        _dispositivo_ya_comprobado=True,
    )
    segundos = time.perf_counter() - inicio
    nuevos = _trials_terminados(destino, nombre) - antes

    if nuevos == 0:
        print(
            f"por_subproceso: el estudio ya tenía {antes} trials terminados en {destino} "
            f"(pediste {n_total}): nada que correr en esta llamada (bórralo o pide más "
            "trials para ver progreso)"
        )
    else:
        print(
            f"por_subproceso: {nuevos} {_pluralizar(nuevos, 'trial nuevo', 'trials nuevos')} en "
            f"{segundos:.2f} s ({segundos / nuevos:.3f} s/trial)"
        )

    mejor = study.best_trial
    print(f"mejor exactitud_val: {mejor.value:.3f} con {mejor.params}")
    print(
        "advertencia: ese valor está inflado por selección (maldición del ganador) -- "
        "no lo reportes tal cual. Re-evalúalo con semillas nuevas antes de "
        "confiar en él (ver paso 06, honestidad)."
    )


if __name__ == "__main__":
    main()
