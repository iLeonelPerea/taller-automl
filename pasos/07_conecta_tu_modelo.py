# %% [markdown]
# # Paso 07: conecta tu modelo
#
# Todo lo anterior corrió sobre el juguete del repo. Aquí conectas la
# plantilla (`plantilla/`) contra un `train.py` ajeno de verdad
# (`plantilla/train_ejemplo.py`, dos lunas con un MLP de PyTorch) y comparas
# las dos rutas que puede tomar tu modelo.
#
# Tres preguntas:
# 1. ¿`train_ejemplo.py` cumple el contrato? ¿Qué detecta `check_importable.py`
#    en uno que NO lo cumple?
# 2. En proceso y por subproceso: ¿la MISMA config da la MISMA métrica por las
#    dos rutas?
# 3. ¿Cuánto cuesta cada ruta, en segundos por trial? ¿Cuándo conviene cada una?
#
# El paso entrena en la tarjeta de video, igual que `plantilla/train_ejemplo.py`, y se
# detiene con un mensaje claro si no hay una tarjeta NVIDIA con CUDA visible.
#
# Correr:  uv run pasos/07_conecta_tu_modelo.py
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import dataclasses
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import optuna

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure

from taller_hpo import estadistica
from taller_hpo.objetivo import en_proceso
from taller_hpo.semillas import Semillas
from taller_hpo.subproceso import escribir_json_atomico, por_subproceso

# El log INFO de Optuna imprime una línea por trial y taparía las respuestas numeradas.
optuna.logging.set_verbosity(optuna.logging.WARNING)

PRESUPUESTOS = {
    # humo usa n_busqueda=1: cada trial por_subproceso arranca un intérprete e importa
    # torch, y ese arranque es lo que más pesa en el presupuesto de 20 s. Con un solo
    # trial, la línea base sembrada, el paso igual recorre las tres secciones. Tampoco mide
    # el sobrecosto pareado (repeticiones_sobrecosto=0), y lo dice al imprimir.
    "humo": {
        "n_busqueda": 1,
        "timeout_subproceso": 30.0,
        "repeticiones_sobrecosto": 0,
    },
    "taller": {
        "n_busqueda": 12,
        "timeout_subproceso": 60.0,
        "repeticiones_sobrecosto": 3,
    },
    "completo": {
        "n_busqueda": 30,
        "timeout_subproceso": 120.0,
        "repeticiones_sobrecosto": 5,
    },
}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
CFG = PRESUPUESTOS[presupuesto]

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = Path(os.environ.get("TALLER_RESULTADOS", RAIZ / "resultados"))

PLANTILLA = RAIZ / "plantilla"
TRAIN_EJEMPLO = PLANTILLA / "train_ejemplo.py"

# El import va después de validar PRESUPUESTO: un PRESUPUESTO inválido sale con su propio
# mensaje aunque no haya tarjeta de video. Este módulo solo se usa para imprimir la
# descripción del juguete del repo; `train_ejemplo.py` resuelve su propio dispositivo.
juguete = importlib.import_module("taller_hpo.problemas.juguete")
# TALLER_RESULTADOS se crea después de comprobar la tarjeta, para no dejar un directorio
# vacío si el paso se detiene antes.
RESULTADOS.mkdir(parents=True, exist_ok=True)
print(f"Motor resuelto para el juguete de este repo: {juguete.DESCRIPCION}")

# Dispositivo de PyTorch para train_ejemplo.py: siempre la tarjeta de video, que el import
# de `juguete` ya comprobó. TALLER_DISPOSITIVO no cambia esta constante: elige la tarjeta
# de `plantilla/buscar_por_subproceso.py` cuando lo corres suelto.
DISPOSITIVO_TORCH = "cuda"
print(f"Dispositivo de PyTorch para las secciones 1-3 de este paso: {DISPOSITIVO_TORCH}")


def _pluralizar(n: int, singular: str, plural: str) -> str:
    """`singular` si `n == 1`, si no `plural`: evita imprimir "1 trials"."""
    return singular if n == 1 else plural


def _importar(ruta: Path, nombre: str) -> types.ModuleType:
    """Importa un .py por ruta, fuera de sys.path, sin instalarlo como paquete (el mismo
    truco que usa check_importable.py)."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"el paso no pudo cargar {ruta!r} como módulo")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


check_importable = _importar(PLANTILLA / "check_importable.py", "_check_importable_paso07")
buscar_en_proceso = _importar(PLANTILLA / "buscar_en_proceso.py", "_buscar_en_proceso_paso07")
buscar_por_subproceso = _importar(
    PLANTILLA / "buscar_por_subproceso.py", "_buscar_por_subproceso_paso07"
)
train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_paso07")

# %% [markdown]
# ## 1. `check_importable.py`: ¿tu `train.py` cumple el contrato?
#
# `check_importable.py` aprueba `train_ejemplo.py`, que cumple el contrato de
# `plantilla/README.md` (DEFAULTS/get_config con `'dispositivo'`, `entrenar`,
# `main` con `--config`/`--salida`, y la guarda `if __name__ == "__main__":`),
# y rechaza un script roto a propósito: un `ArgumentParser` a nivel de módulo,
# el error más común al adaptar un `train.py` existente (ver el docstring de
# `check_importable.py`).

# %%
problema_bueno = check_importable.revisar(str(TRAIN_EJEMPLO), timeout_s=10.0)
print(f"train_ejemplo.py aprueba check_importable: {problema_bueno is None}")

ROTO = """
import argparse

_parser = argparse.ArgumentParser()
_args = _parser.parse_args()


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""
with tempfile.TemporaryDirectory() as directorio_tmp:
    ruta_rota = Path(directorio_tmp) / "train_roto.py"
    ruta_rota.write_text(ROTO, encoding="utf-8")
    problema_roto = check_importable.revisar(str(ruta_rota), timeout_s=10.0)
print(f"train_roto.py aprueba check_importable: {problema_roto is None}")
print(f"diagnóstico: {problema_roto}")

# Los prints de arriba solo repiten el diagnóstico; estos asserts fallan si
# check_importable aprobara o rechazara cualquier cosa.
assert problema_bueno is None, problema_bueno
assert problema_roto is not None and "parse_args" in problema_roto, problema_roto

# %% [markdown]
# ## 2. En proceso vs por subproceso: ¿misma config, misma métrica?
#
# La misma config y las mismas semillas, por las dos rutas. `en_proceso` llama a
# `train_ejemplo.entrenar` directo; `por_subproceso` lo corre como proceso completo con
# `--config`/`--salida`. Si serializar a JSON, reconstruir `Semillas` o leer la salida
# alterara el resultado, esto lo mostraría. El paso compara el dict completo que devuelve
# `entrenar` (`exactitud_val` y `epocas`, vía `atributos`), no solo la métrica que
# optimiza Optuna.

# %%
CONFIG_COMPARACION = {"n_train": 150, "n_val": 100, "epocas": 6, "ancho": 12}
SEMILLAS_COMPARACION = Semillas(datos=1, orden=2, init=3)
# TALLER_PYTHON: el mismo intérprete que usa por_subproceso, para no comparar intérpretes
# distintos si alguien corre su modelo en otro entorno.
PYTHON_COMPARACION = os.environ.get("TALLER_PYTHON", sys.executable)

objetivo_directo = en_proceso(
    train_ejemplo.entrenar,
    lambda trial: {},
    metricas=["exactitud_val"],
    semillas=SEMILLAS_COMPARACION,
    fijos=CONFIG_COMPARACION,
    atributos=["epocas"],
)
# Sin NopPruner, Optuna 5 usa MedianPruner y en_proceso podría dejar este único trial
# PRUNED en vez de COMPLETE.
study_directo = optuna.create_study(
    sampler=optuna.samplers.RandomSampler(seed=0), pruner=optuna.pruners.NopPruner()
)
study_directo.optimize(objetivo_directo, n_trials=1)
trial_directo = study_directo.trials[0]
valor_en_proceso = trial_directo.value
epocas_en_proceso = trial_directo.user_attrs["epocas"]

with tempfile.TemporaryDirectory() as directorio_comparacion:
    objetivo_subproceso = por_subproceso(
        [PYTHON_COMPARACION, str(TRAIN_EJEMPLO)],
        lambda trial: {},
        metricas=["exactitud_val"],
        semillas=SEMILLAS_COMPARACION,
        timeout_s=CFG["timeout_subproceso"],
        directorio=Path(directorio_comparacion),
        fijos=CONFIG_COMPARACION,
        atributos=["epocas"],
    )
    study_subproceso = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=0), pruner=optuna.pruners.NopPruner()
    )
    study_subproceso.optimize(objetivo_subproceso, n_trials=1)
    trial_subproceso = study_subproceso.trials[0]
    valor_por_subproceso = trial_subproceso.value
    epocas_por_subproceso = trial_subproceso.user_attrs["epocas"]

# Las dos rutas ejecutan el mismo entrenar() con la misma config y semillas: deben
# coincidir exacto. Si esto falla, algo se altera al serializar o reconstruir.
assert valor_en_proceso == valor_por_subproceso, (
    f"en_proceso={valor_en_proceso!r} vs por_subproceso={valor_por_subproceso!r}: "
    "misma config, misma semilla, deberían coincidir exacto"
)
assert epocas_en_proceso == epocas_por_subproceso, (
    f"epocas en_proceso={epocas_en_proceso!r} vs por_subproceso={epocas_por_subproceso!r}"
)
print(f"en_proceso exactitud_val: {valor_en_proceso:.6f}, epocas: {epocas_en_proceso:.0f}")
print(
    f"por_subproceso exactitud_val: {valor_por_subproceso:.6f}, epocas: {epocas_por_subproceso:.0f}"
)
print(
    f"diferencia: {abs(valor_en_proceso - valor_por_subproceso):.2e}, exactamente la misma métrica."
)

# %% [markdown]
# ## 3. Costo de cada ruta: segundos por trial
#
# La búsqueda completa (línea base sembrada, `TPESampler`, `optimizar`) de
# `plantilla/buscar_en_proceso.py` y `plantilla/buscar_por_subproceso.py`, con el mismo
# `n_total`. Cada búsqueda va en un directorio temporal nuevo, nunca en `TALLER_ESTUDIOS`:
# reanudar un estudio guardado no entrenaría nada nuevo y el costo por trial saldría
# falso. Por eso el paso también cuenta los trials que produjo esta llamada y lo afirma con
# `assert`.
#
# `por_subproceso` paga un intérprete de Python completo (con su import de numpy y torch)
# por cada trial; `en_proceso` no. Además, solo `en_proceso` puede podar:
# `trial.report`/`should_prune` viven en el mismo proceso que Optuna.

# %%
n_busqueda = CFG["n_busqueda"]
ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def _trials_de_esta_llamada(study: optuna.Study, antes: int) -> int:
    """Trials de cualquier estado, FAIL incluido, que `study` ganó desde que se midió
    `antes`. Es un conteo, no una medición de tiempo: no depende del reloj de pared, que en
    WSL2 puede retroceder."""
    return len(study.trials) - antes


with (
    tempfile.TemporaryDirectory() as directorio_en_proceso,
    tempfile.TemporaryDirectory() as directorio_por_subproceso,
):
    destino_en_proceso = Path(directorio_en_proceso) / "en_proceso.log"
    destino_por_subproceso = Path(directorio_por_subproceso) / "por_subproceso.log"
    # El directorio temporal es nuevo, así que `destino` todavía no existe. Si existiera, la
    # búsqueda reanudaría trials de otra corrida.
    assert not destino_en_proceso.exists()
    assert not destino_por_subproceso.exists()
    antes_en_proceso = 0
    antes_por_subproceso = 0

    inicio = time.perf_counter()
    study_en_proceso = buscar_en_proceso.buscar(n_total=n_busqueda, destino=destino_en_proceso)
    segundos_en_proceso = time.perf_counter() - inicio

    # El dispositivo se resuelve antes de arrancar el reloj, para no contar como costo de
    # trial el intérprete que comprueba la tarjeta. Las dos rutas comparan en "cuda", sin
    # importar TALLER_DISPOSITIVO. Con el mismo intérprete, el import de `juguete` ya
    # comprobó la tarjeta en este proceso y `ya_comprobado` evita repetirlo;
    # `_dispositivo_ya_comprobado=True` evita que `buscar()` la compruebe otra vez dentro
    # del intervalo medido.
    dispositivo_por_subproceso = buscar_por_subproceso._resolver_dispositivo(
        "cuda", PYTHON_COMPARACION, ya_comprobado=(PYTHON_COMPARACION == sys.executable)
    )
    print(f"Dispositivo de por_subproceso para la sección 3: {dispositivo_por_subproceso}")
    inicio = time.perf_counter()
    study_por_subproceso = buscar_por_subproceso.buscar(
        n_total=n_busqueda,
        destino=destino_por_subproceso,
        directorio_trials=Path(directorio_por_subproceso) / "trials",
        timeout_s=CFG["timeout_subproceso"],
        dispositivo=dispositivo_por_subproceso,
        _dispositivo_ya_comprobado=True,
    )
    segundos_por_subproceso = time.perf_counter() - inicio

    trials_ep = sorted(
        study_en_proceso.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS),
        key=lambda t: t.number,
    )
    trials_ps = sorted(
        study_por_subproceso.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS),
        key=lambda t: t.number,
    )
    nuevos_en_proceso = len(trials_ep)
    nuevos_por_subproceso = len(trials_ps)
    # Directorio nuevo en cada corrida: los dos deben valer n_busqueda. Si no, algo dejó de
    # correr trials nuevos y dividir el tiempo entre ellos mentiría.
    assert nuevos_en_proceso == n_busqueda, (nuevos_en_proceso, n_busqueda)
    assert nuevos_por_subproceso == n_busqueda, (nuevos_por_subproceso, n_busqueda)
    # Además, en un Journal nuevo todos los trials son de esta llamada, también un FAIL de
    # reintento que los asserts de arriba no cuentan.
    assert _trials_de_esta_llamada(study_en_proceso, antes_en_proceso) == n_busqueda, (
        len(study_en_proceso.trials),
        antes_en_proceso,
        n_busqueda,
    )
    assert _trials_de_esta_llamada(study_por_subproceso, antes_por_subproceso) == n_busqueda, (
        len(study_por_subproceso.trials),
        antes_por_subproceso,
        n_busqueda,
    )

    # Las dos rutas comparten sampler, espacio y semillas: deben dar, trial a trial, las
    # mismas configs y la misma métrica. Es la comprobación de la sección 2, sobre toda la
    # búsqueda.
    for t_ep, t_ps in zip(trials_ep, trials_ps, strict=True):
        assert t_ep.params == t_ps.params, (t_ep.number, t_ep.params, t_ps.params)
        assert t_ep.user_attrs["config"] == t_ps.user_attrs["config"], t_ep.number
        assert t_ep.value == t_ps.value, (t_ep.number, t_ep.value, t_ps.value)

costo_en_proceso = segundos_en_proceso / nuevos_en_proceso
costo_por_subproceso = segundos_por_subproceso / nuevos_por_subproceso
razon_costo = costo_por_subproceso / costo_en_proceso
sobrecosto_fijo = costo_por_subproceso - costo_en_proceso

print(
    f"en_proceso: {nuevos_en_proceso} "
    f"{_pluralizar(nuevos_en_proceso, 'trial nuevo', 'trials nuevos')} en "
    f"{segundos_en_proceso:.2f} s ({costo_en_proceso:.3f} s/trial)"
)
print(
    f"por_subproceso: {nuevos_por_subproceso} "
    f"{_pluralizar(nuevos_por_subproceso, 'trial nuevo', 'trials nuevos')} en "
    f"{segundos_por_subproceso:.2f} s ({costo_por_subproceso:.3f} s/trial, "
    f"{razon_costo:.1f}x más lento por trial)"
)
_articulo_configs = "la" if n_busqueda == 1 else "las"
_palabra_configs = _pluralizar(n_busqueda, "config", "configs")
_verbo_dar = _pluralizar(n_busqueda, "da", "dan")
print(
    f"{_articulo_configs} {n_busqueda} {_palabra_configs} de las dos rutas {_verbo_dar} la MISMA "
    "métrica, trial a trial."
)


# Medición pareada del sobrecosto fijo: dividir el reloj de toda la búsqueda entre
# n_busqueda da una sola medición, sin intervalo de confianza. Usa la config de la línea
# base (train_ejemplo.DEFAULTS), no CONFIG_COMPARACION. En humo no cabe en el presupuesto
# y el paso lo dice.
def _medir_una_cli() -> float:
    """Una corrida por línea de comandos de `train_ejemplo.py` con
    `train_ejemplo.DEFAULTS`, sin promediar."""
    entrada_json = {
        "config": dict(train_ejemplo.DEFAULTS),
        "semillas": dataclasses.asdict(SEMILLAS_COMPARACION),
    }
    with tempfile.TemporaryDirectory() as d:
        entrada = Path(d) / "entrada.json"
        salida = Path(d) / "salida.json"
        escribir_json_atomico(entrada, entrada_json)
        comando = [
            PYTHON_COMPARACION,
            str(TRAIN_EJEMPLO),
            "--config",
            str(entrada),
            "--salida",
            str(salida),
        ]
        inicio = time.perf_counter()
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=CFG["timeout_subproceso"],
        )
        transcurrido = time.perf_counter() - inicio
    assert resultado.returncode == 0, resultado.stderr
    return transcurrido


def _medir_una_en_proceso() -> float:
    """Una corrida en proceso de `train_ejemplo.entrenar` con la misma config y semilla
    que `_medir_una_cli`: el par de la resta pareada."""
    inicio = time.perf_counter()
    train_ejemplo.entrenar(dict(train_ejemplo.DEFAULTS), SEMILLAS_COMPARACION)
    return time.perf_counter() - inicio


def _medir_sobrecosto_intercalado(repeticiones: int) -> tuple[list[float], list[float]]:
    """`repeticiones` rondas más una de calentamiento, con las dos series intercaladas: el
    script por línea de comandos y la llamada en proceso. El orden alterna por ronda (ABBA)
    para que la carga de otros procesos o las cachés no sesguen siempre al mismo lado. La
    ronda de calentamiento no entra a las listas, que se devuelven crudas para la prueba
    pareada."""
    tiempos_cli: list[float] = []
    tiempos_en_proceso: list[float] = []
    for ronda in range(repeticiones + 1):
        cli_primero = ronda % 2 == 0
        if cli_primero:
            t_cli = _medir_una_cli()
            t_en_proceso = _medir_una_en_proceso()
        else:
            t_en_proceso = _medir_una_en_proceso()
            t_cli = _medir_una_cli()
        if ronda > 0:
            tiempos_cli.append(t_cli)
            tiempos_en_proceso.append(t_en_proceso)
    return tiempos_cli, tiempos_en_proceso


def _texto_sobrecosto_fijo(
    tiempos_cli: list[float], tiempos_en_proceso: list[float], repeticiones: int
) -> tuple[float | None, str]:
    """Aísla el sobrecosto fijo de `por_subproceso` frente a `en_proceso` con una prueba
    pareada ronda a ronda, con la misma config."""
    mediana_referencia = statistics.median(tiempos_en_proceso)
    p = estadistica.diferencia_pareada(
        dict(enumerate(tiempos_cli)),
        dict(enumerate(tiempos_en_proceso)),
        delta=0.10 * mediana_referencia,
        direccion="maximizar",
    )
    detalle = (
        f"línea de comandos - en_proceso = {p.diferencia_media:+.3f} s "
        f"[{p.ic_bajo:+.3f}, {p.ic_alto:+.3f}] ({repeticiones} rondas intercaladas, misma "
        "config -- línea base, train_ejemplo.DEFAULTS)"
    )
    if p.veredicto == "mejor":
        return p.diferencia_media, (
            f"sobrecosto fijo medido con intervalo de confianza: "
            f"~{p.diferencia_media:.3f} s: {detalle}."
        )
    if p.veredicto == "equivalente":
        return 0.0, f"sobrecosto fijo equivalente a 0 dentro del 10% de margen: {detalle}."
    return None, (
        f"sobrecosto fijo medido con intervalo de confianza: "
        f"no concluyente con esta carga: {detalle}."
    )


if CFG["repeticiones_sobrecosto"] == 0:
    sobrecosto_medido = None
    sobrecosto_fijo_reportado = sobrecosto_fijo
    nota_sobrecosto = (
        "el sobrecosto de arriba es 1 sola medición (presupuesto humo), sin intervalo de confianza."
    )
    print(
        "En humo, este paso no mide el sobrecosto fijo con una prueba pareada: no cabe en el "
        "presupuesto de 20 s junto con las secciones 1-2 completas (ver taller o completo)."
    )
else:
    tiempos_cli_sobrecosto, tiempos_en_proceso_sobrecosto = _medir_sobrecosto_intercalado(
        CFG["repeticiones_sobrecosto"]
    )
    sobrecosto_medido, texto_sobrecosto_medido = _texto_sobrecosto_fijo(
        tiempos_cli_sobrecosto, tiempos_en_proceso_sobrecosto, CFG["repeticiones_sobrecosto"]
    )
    print(texto_sobrecosto_medido)
    if sobrecosto_medido is not None:
        sobrecosto_fijo_reportado = sobrecosto_medido
        nota_sobrecosto = (
            "el sobrecosto de arriba es la diferencia pareada (con intervalo de confianza) de "
            f"{CFG['repeticiones_sobrecosto']} rondas intercaladas, no la resta de dos "
            "agregados de toda la búsqueda con configs distintas entre sí."
        )
    else:
        sobrecosto_fijo_reportado = sobrecosto_fijo
        nota_sobrecosto = (
            "el sobrecosto medido con intervalo de confianza no salió concluyente con esta "
            "carga; el de arriba es la resta de dos agregados de toda la búsqueda, sin "
            "intervalo de confianza propio."
        )

# %% [markdown]
# ## Figura: costo de cada ruta

# %%
fig = Figure(figsize=(5, 3.5))
ax_rutas = fig.subplots()

ax_rutas.bar(
    ["en_proceso", "por_subproceso"],
    [costo_en_proceso, costo_por_subproceso],
    color=["#4C72B0", "#DD8452"],
)
ax_rutas.set_ylabel("segundos / trial")
_palabra_trials_rutas = _pluralizar(n_busqueda, "trial", "trials")
ax_rutas.set_title(
    f"Costo por ruta ({n_busqueda} {_palabra_trials_rutas} c/u, promedio, train_ejemplo.py)"
)

fig.tight_layout()
figura = RESULTADOS / "07_conecta_tu_modelo.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")

# %% [markdown]
# ## Las tres respuestas

# %%
print()
print(
    f"1. check_importable aprueba train_ejemplo.py (problema={problema_bueno!r}) y "
    f"rechaza el script roto con el diagnóstico correcto (problema={problema_roto!r})."
)
print(
    f"2. Misma config, misma semilla: en_proceso da exactitud_val={valor_en_proceso:.6f} "
    f"({epocas_en_proceso:.0f} épocas) y por_subproceso da "
    f"exactitud_val={valor_por_subproceso:.6f} ({epocas_por_subproceso:.0f} épocas) -- "
    f"diferencia {abs(valor_en_proceso - valor_por_subproceso):.2e}: el dict completo coincide."
)
# Costo relativo = sobrecosto/T: un entrenamiento de T segundos tarda T+sobrecosto por
# subproceso. sobrecosto/(T+sobrecosto) mediría otra cosa, la fracción del trial que es
# sobrecosto. `sobrecosto_fijo_reportado` sale de la medición pareada, o de la resta de
# agregados en humo o cuando la pareada no es concluyente.
porcentaje_juguete = 100 * sobrecosto_fijo_reportado / costo_en_proceso
porcentaje_t60 = 100 * sobrecosto_fijo_reportado / 60
print(
    f"3. Con {n_busqueda} {_pluralizar(n_busqueda, 'trial nuevo', 'trials nuevos')} cada "
    f"uno: en_proceso cuesta {costo_en_proceso:.3f} s/trial, por_subproceso "
    f"{costo_por_subproceso:.3f} s/trial ({razon_costo:.1f}x más lento, sobrecosto fijo "
    f"{sobrecosto_fijo_reportado:.3f} s/trial). {nota_sobrecosto} El sobrecosto "
    "hace que un entrenamiento de T segundos tarde T+sobrecosto, es decir 100*sobrecosto/T % "
    f"más: para T={costo_en_proceso:.3f} s (este juguete) es {porcentaje_juguete:.0f}% más; "
    f"para T=60 s (un modelo real) sería solo {porcentaje_t60:.1f}% más. "
    f"{_articulo_configs.capitalize()} {n_busqueda} {_palabra_configs} de las dos rutas "
    f"{_pluralizar(n_busqueda, 'dio', 'dieron')} la MISMA métrica (assert, trial a trial). Usa "
    "por_subproceso solo si de verdad necesitas aislar el entorno: además del costo fijo, SOLO "
    "en_proceso puede podar (trial.report/should_prune viven en el mismo proceso que Optuna)."
)
