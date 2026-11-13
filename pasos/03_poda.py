# %% [markdown]
# # Paso 03: poda contra early stopping
#
# El paso 02 ya corrió un estudio con método (TPE, storage, línea base). Aquí la
# pregunta es otra: si puedes cortar el mismo trial a mitad de camino cuando
# pinta mal, ¿cuánto cómputo ahorras? ¿Ese ahorro mata configuraciones que
# habrían mejorado tarde? ¿Y en qué se diferencia de un early stopping manual,
# el que cualquiera escribe sin tocar Optuna?
#
# Cinco preguntas:
# 1. ¿Cuánto cómputo (épocas) ahorra podar, estudio por estudio?
# 2. ¿Qué tan buena es la mejor configuración de cada estudio (validación y verdad)?
# 3. De los trials que Median podó, ¿alguno habría superado al ganador si lo
#    hubieras dejado terminar? (el "ganador tardío")
# 4. Un early stopping manual (con paciencia, sin Optuna) ¿ahorra épocas? ¿Eso es
#    lo mismo que podar?
# 5. ¿Por qué Optuna no poda en multiobjetivo?
#
# Correr:  uv run pasos/03_poda.py  (necesita una tarjeta de video NVIDIA con
#          CUDA y el controlador instalado; sin ella, el paso se detiene con
#          un mensaje claro)
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import dataclasses
import os
import statistics
from collections.abc import Sequence
from pathlib import Path

import optuna

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure

from taller_hpo import estudio
from taller_hpo.estadistica import diferencia_pareada, ic_proporcion, resumen
from taller_hpo.objetivo import en_proceso
from taller_hpo.problemas.juguete import (
    DEFAULTS,
    DESCRIPCION,
    PROTOCOLO_MOTOR,
    Divergencia,
    entrenar,
)
from taller_hpo.semillas import Semillas, semillas_de

# El log INFO de Optuna taparía las respuestas numeradas; el paso lo baja a WARNING.
optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"Motor: {DESCRIPCION}\n")

PRESUPUESTOS = {"humo": 8, "taller": 40, "completo": 100}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
N_TOTAL = PRESUPUESTOS[presupuesto]

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)
ESTUDIOS = Path(
    os.environ.get("TALLER_ESTUDIOS", Path(__file__).resolve().parent.parent / "estudios")
)
ESTUDIOS.mkdir(parents=True, exist_ok=True)

EPOCAS = DEFAULTS["epocas"]  # 30: también el max_resource de Hyperband (una época = un recurso).

# %% [markdown]
# ## El espacio: el mismo del paso 02, sin el placebo
#
# Este paso busca las mismas perillas que el paso 02, con los mismos rangos, salvo el
# placebo: aquí no mides importancias, así que cada trial gastado en él sería un trial
# menos para comparar pruners. `activacion` queda en su default, y el oráculo usa el
# `n_oraculo` completo de `DEFAULTS`.


# %%
def espacio(trial: optuna.Trial) -> dict:
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 3),
        "ancho": trial.suggest_int("ancho", 8, 128, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


# Una sola Semillas para todos los trials de los tres estudios (CRN), igual que el paso
# 02: con una semilla distinta por trial, cada uno vería su propio conjunto de
# validación y MedianPruner compararía curvas medidas sobre datos distintos.
SEMILLA_BUSQUEDA = semillas_de(20_260_301, 1)[0]

# La misma fábrica de sampler para los tres estudios: cada uno arranca vacío, así que el
# TPE parte de la misma semilla y la única diferencia entre estudios es el pruner. En
# humo, con menos trials que `n_startup_trials`, los tres piden exactamente los mismos
# hiperparámetros. `constant_liar=False` como en el paso 02: con `n_jobs=1` no hay trials
# concurrentes.
SAMPLER_KWARGS = {
    "seed_base": 20_260_302,
    "seed": "seed_base + n_previos",
    "n_startup_trials": 10,
    "multivariate": True,
    "constant_liar": False,
}


def fabrica_sampler(n_previos: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=SAMPLER_KWARGS["seed_base"] + n_previos,
        n_startup_trials=SAMPLER_KWARGS["n_startup_trials"],
        multivariate=SAMPLER_KWARGS["multivariate"],
        constant_liar=SAMPLER_KWARGS["constant_liar"],
    )


POLITICA_SEMILLAS = {
    "tipo": "CRN",
    "base": 20_260_301,
    "n": 1,
    "semillas": dataclasses.asdict(SEMILLA_BUSQUEDA),
}

# %% [markdown]
# ## Tres estudios, mismo objective, distinto pruner
#
# `NopPruner` nunca poda: es la línea base de "cuánto cuesta no podar". `MedianPruner`
# poda contra la mediana de los trials completos en el mismo paso. `HyperbandPruner`
# reparte el presupuesto en "brackets" con `SuccessiveHalvingPruner` (`min_resource=3`,
# `max_resource=epocas` y `reduction_factor=3`: en cada ronda sobrevive 1 de cada 3).
#
# El paso atrapa `Divergencia` (pesos no finitos, típico con un learning_rate_init alto):
# `optimizar` la cuenta como FAIL y sigue con el siguiente trial. No corras este paso dos
# veces a la vez sobre el mismo `TALLER_ESTUDIOS`: `recuperar_huerfanos` de uno marcaría
# FAIL los trials vivos del otro.

# %%
objetivo = en_proceso(
    entrenar,
    espacio,
    metricas=["exactitud_val"],
    semillas=SEMILLA_BUSQUEDA,
    atributos=["exactitud_verdadera", "epocas"],
)

# Kwargs de cada pruner para el protocolo persistido: los pruners los aceptan directo como
# `**kwargs`, así que este dict es la única fuente de verdad.
PRUNER_KWARGS = {
    "nop": {},
    "median": {"n_startup_trials": 5, "n_warmup_steps": 5},
    "hyperband": {"min_resource": 3, "max_resource": EPOCAS, "reduction_factor": 3},
}
_CLASES_PRUNER: dict[str, type[optuna.pruners.BasePruner]] = {
    "nop": optuna.pruners.NopPruner,
    "median": optuna.pruners.MedianPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
}
PRUNERS = {nombre: _CLASES_PRUNER[nombre](**kwargs) for nombre, kwargs in PRUNER_KWARGS.items()}

estudios: dict[str, optuna.Study] = {}
for nombre_pruner, pruner in PRUNERS.items():
    nombre_estudio = f"03_poda_{nombre_pruner}"
    study = estudio.crear_estudio(
        nombre_estudio,
        # El presupuesto entra al nombre del archivo, así correr humo y luego taller no
        # reabre el mismo estudio con otro n_total.
        destino=ESTUDIOS / f"{nombre_estudio}_{presupuesto}.log",
        direcciones=["maximize"],
        sampler=fabrica_sampler,
        pruner=pruner,
        protocolo={
            "paso": "03_poda",
            "pruner": nombre_pruner,
            "pruner_kwargs": PRUNER_KWARGS[nombre_pruner],
            "sampler": "TPESampler",
            "sampler_kwargs": SAMPLER_KWARGS,
            "politica_semillas": POLITICA_SEMILLAS,
            "metrica": "exactitud_val",
            "n_total": N_TOTAL,
            # PROTOCOLO_MOTOR entra al protocolo: registra que PyTorch entrenó en la
            # tarjeta de video, sin el índice.
            **PROTOCOLO_MOTOR,
        },
    )
    estudio.recuperar_huerfanos(study)
    estudio.optimizar(study, objetivo, n_total=N_TOTAL, atrapar=(Divergencia,))
    estudios[nombre_pruner] = study

ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

# %% [markdown]
# ## Pregunta 1 y 2: cómputo ahorrado y calidad del ganador


# %%
def epocas_totales(study: optuna.Study) -> float:
    """Suma de épocas de los trials terminados (COMPLETE + PRUNED). Los trials FAIL no
    guardan `epocas`, así que no suman."""
    trials = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    return sum(t.user_attrs["epocas"] for t in trials)


epocas_por_estudio = {nombre: epocas_totales(s) for nombre, s in estudios.items()}
epocas_nop = epocas_por_estudio["nop"]
ahorro_pct = {
    nombre: 100 * (1 - valor / epocas_nop) for nombre, valor in epocas_por_estudio.items()
}

mejores = {
    nombre: (s.best_trial.value, s.best_trial.user_attrs["exactitud_verdadera"])
    for nombre, s in estudios.items()
}

print(f"Presupuesto: {presupuesto} (n_total={N_TOTAL} trials por estudio)\n")
for nombre in PRUNERS:
    val, verdad = mejores[nombre]
    print(
        f"{nombre:>9}: {epocas_por_estudio[nombre]:>6.0f} épocas totales "
        f"(ahorro {ahorro_pct[nombre]:5.1f}%) | mejor validación {val:.3f} "
        f"| verdadera de ese mejor {verdad:.3f}"
    )

print()
ahorradas_median = epocas_nop - epocas_por_estudio["median"]
ahorradas_hyperband = epocas_nop - epocas_por_estudio["hyperband"]
print(
    f"1. Sin podar (nop), los {N_TOTAL} trials consumieron {epocas_nop:.0f} épocas. "
    f"Median usó {epocas_por_estudio['median']:.0f}: ahorró {ahorro_pct['median']:.1f}% "
    f"({ahorradas_median:.0f} épocas). Hyperband usó {epocas_por_estudio['hyperband']:.0f}: "
    f"ahorró {ahorro_pct['hyperband']:.1f}% ({ahorradas_hyperband:.0f} épocas)."
)
val_nop, verdad_nop = mejores["nop"]
val_median, verdad_median = mejores["median"]
val_hb, verdad_hb = mejores["hyperband"]
delta_verdad_median = verdad_median - verdad_nop
delta_verdad_hb = verdad_hb - verdad_nop
print(
    f"2. Mejor validación: nop {val_nop:.3f}, median {val_median:.3f}, hyperband {val_hb:.3f}. "
    f"Su exactitud verdadera: nop {verdad_nop:.3f}, median {verdad_median:.3f} "
    f"({delta_verdad_median:+.3f} vs. nop), hyperband {verdad_hb:.3f} "
    f"({delta_verdad_hb:+.3f} vs. nop). Una sola realización por pruner (n=1): no alcanza para "
    "afirmar que podar 'no empeora' al ganador en general -- hacen falta réplicas: repetir los "
    "tres estudios con varias semillas de sampler y de datos y compararlos pareados (el método "
    "del paso 06)."
)

# %% [markdown]
# ## Pregunta 3: el contrafactual, ¿Median mató a un ganador tardío?
#
# Cada trial que Median podó guarda su config y sus semillas en `user_attrs`.
# Re-entrenarlo sin poda contesta lo que el estudio real no puede: ¿qué hubiera dado si lo
# dejabas terminar? Con la misma semilla para todos los trials, el conjunto de validación
# es el mismo en todo el paso.
#
# El veredicto sale de un solo número: quién habría ganado Median sin podar, el mejor por
# validación entre los trials completos y los podados re-entrenados (desempatando por el
# número de trial más bajo, como `Study.best_trial`). Si es un trial podado, Median sí
# cambió a quién elige la búsqueda, y el oráculo dice si ese cambio fue para bien o para
# mal. Las tres categorías por trial que imprime el paso solo describen cuántos podados
# quedaron de cada lado.
#
# El contrafactual compara solo estas configuraciones: sin podar, TPE habría visto otras
# curvas y propuesto otras configuraciones. La búsqueda de verdad sin poda es el estudio
# `nop`.

# %%
podados_median = estudios["median"].get_trials(
    deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)
)
ganador_median = estudios["median"].best_trial
mejor_val_median, mejor_verdad_median = mejores["median"]

# Cota de error Monte Carlo del oráculo con `n_oraculo=200_000` (ver
# `paridad.riesgo_verdadero`). Usar la suma de las dos cotas como margen de empate solo
# puede declarar un empate de más, nunca de menos.
COTA_ERROR_ORACULO = 0.0012
EMPATE_ORACULO = 2 * COTA_ERROR_ORACULO

reentrenados: list[tuple[int, dict]] = []
for trial in podados_median:
    config = trial.user_attrs["config"]
    s = trial.user_attrs["semillas"]
    semillas_trial = Semillas(datos=s["datos"], orden=s["orden"], init=s["init"])
    # Un podado que diverge después de su época de poda queda fuera de las categorías,
    # igual que un trial FAIL.
    try:
        completo = entrenar(config, semillas_trial)
    except Divergencia:
        print(f"   re-entrenado trial {trial.number}: diverge sin podar (excluido)")
        continue
    print(
        f"   re-entrenado trial {trial.number}: {completo['epocas']:.0f} épocas completas "
        f"(sin podar), validación {completo['exactitud_val']:.3f}, "
        f"verdadera {completo['exactitud_verdadera']:.3f}"
    )
    reentrenados.append((trial.number, completo))

# Las tres categorías son excluyentes y solo describen: el margen EMPATE_ORACULO evita
# contar como "superó la verdad" una diferencia que es ruido del oráculo. El veredicto lo
# decide `ganador_sin_podar`.
tardios_reales = [
    (n, c)
    for n, c in reentrenados
    if c["exactitud_val"] > mejor_val_median
    and c["exactitud_verdadera"] > mejor_verdad_median + EMPATE_ORACULO
]
solo_verdad = [
    (n, c)
    for n, c in reentrenados
    if c["exactitud_verdadera"] > mejor_verdad_median + EMPATE_ORACULO
    and c["exactitud_val"] <= mejor_val_median
]
solo_val = [
    (n, c)
    for n, c in reentrenados
    if c["exactitud_val"] > mejor_val_median
    and c["exactitud_verdadera"] <= mejor_verdad_median + EMPATE_ORACULO
]
numeros_solo_verdad = sorted(n for n, _ in solo_verdad)

# "Sin podar": el mejor por validación entre los COMPLETE reales y los podados
# re-entrenados, desempatando por el número de trial más bajo. Es la única fuente del
# veredicto.
completos_median = estudios["median"].get_trials(
    deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
)
candidatos_sin_podar: list[tuple[int, float, float]] = sorted(
    [(t.number, t.value, t.user_attrs["exactitud_verdadera"]) for t in completos_median]
    + [(n, c["exactitud_val"], c["exactitud_verdadera"]) for n, c in reentrenados]
)
ganador_sin_podar = max(candidatos_sin_podar, key=lambda c: c[1])
numero_sin_podar, val_sin_podar, verdad_sin_podar = ganador_sin_podar

# Si el mejor por validación es un trial podado, la poda sí cambió a quién habría elegido
# la búsqueda, y el oráculo decide si fue para bien o para mal.
podado_gano_sin_podar = numero_sin_podar != ganador_median.number
diferencia_verdad_sin_podar = verdad_sin_podar - mejor_verdad_median
if podado_gano_sin_podar and abs(diferencia_verdad_sin_podar) <= EMPATE_ORACULO:
    # Las dos verdades traen cada una hasta COTA_ERROR_ORACULO de error, así que una
    # diferencia dentro de esa suma no distingue cuál es mayor.
    veredicto_tardio = (
        f"Entre estas mismas configuraciones, sin podar la búsqueda habría elegido el trial "
        f"{numero_sin_podar} por validación, pero su verdad ({verdad_sin_podar:.3f}) y la del "
        f"ganador real ({mejor_verdad_median:.3f}) empatan dentro del error del oráculo (cota "
        f"{EMPATE_ORACULO:.4f}): no puedes decidir si podar cambió algo para bien o para mal."
    )
elif podado_gano_sin_podar and diferencia_verdad_sin_podar > 0.0:
    veredicto_tardio = (
        f"Sí mató un ganador tardío: entre estas mismas configuraciones, sin podar la "
        f"búsqueda habría elegido el trial {numero_sin_podar} por validación, y el oráculo "
        f"confirma que era mejor (verdad {verdad_sin_podar:.3f} contra "
        f"{mejor_verdad_median:.3f} del ganador real)."
    )
elif podado_gano_sin_podar:
    veredicto_tardio = (
        f"Entre estas mismas configuraciones, sin podar la búsqueda habría elegido el trial "
        f"{numero_sin_podar} por validación, pero el oráculo dice que es PEOR (verdad "
        f"{verdad_sin_podar:.3f} contra {mejor_verdad_median:.3f} del ganador real): podar no "
        "le costó el mejor modelo -- esa ventaja de validación era suerte de este conjunto."
    )
elif numeros_solo_verdad:
    veredicto_tardio = (
        f"El oráculo prefiere a los trials {numeros_solo_verdad} sobre el ganador real, más "
        f"allá del empate del oráculo (verdad por encima de "
        f"{mejor_verdad_median + EMPATE_ORACULO:.3f}), pero entre estas mismas configuraciones "
        "la búsqueda, sin podar, igual habría elegido al mismo ganador "
        f"(trial {ganador_median.number}): podar no cambió a quién elige la búsqueda."
    )
else:
    veredicto_tardio = (
        "Ningún ganador tardío en esta corrida: podar no le costó el mejor modelo entre estas "
        "configuraciones."
    )

print(
    f"3. Median podó {len(podados_median)} trials; re-entrenados completos, "
    f"{len(tardios_reales) + len(solo_val)} habrían superado la mejor VALIDACIÓN "
    f"({mejor_val_median:.3f}) y {len(tardios_reales) + len(solo_verdad)} la verdad del "
    f"ganador de Median ({mejor_verdad_median:.3f}). Entre estas mismas configuraciones, sin "
    f"podar, el ganador de Median habría sido el trial {numero_sin_podar} (validación "
    f"{val_sin_podar:.3f}, verdad {verdad_sin_podar:.3f}) frente al ganador real, trial "
    f"{ganador_median.number} (validación {mejor_val_median:.3f}, verdad "
    f"{mejor_verdad_median:.3f}). {veredicto_tardio} Nota: esta es UNA sola realización de "
    "Median (n=1), no alcanza para estimar con qué frecuencia poda mata ganadores tardíos en "
    "general -- hacen falta réplicas: repetir el estudio con varias semillas de sampler y de "
    "datos y compararlo pareado (el método del paso 06)."
)

# %% [markdown]
# ## Pregunta 4: early stopping manual, ahorra épocas pero no compara entre trials
#
# Esto no es poda: es una función `reportar` escrita a mano, sin Optuna, que para un
# entrenamiento cuando la validación no mejora en `paciencia` épocas seguidas. Solo ve los
# valores de ese entrenamiento, así que ahorra épocas pero no puede decidir "este trial va
# peor que la mediana de los otros", que es lo que hace un pruner.
#
# El paso lo mide con el presupuesto real del problema (`EPOCAS`) sobre varias semillas
# nuevas: en cuántas corta la paciencia, en qué época y qué le costó en verdad frente a
# dejarlo terminar. Clasifica cada corte con el intervalo de Wilson de su validación
# ("bajo el azar", "no se distingue del azar" o "despegado") y mide el costo con una
# prueba pareada con un delta fijado de antemano. También clasifica igual el último
# reporte de cada podado de Median.


# %%
def early_stopping_por_paciencia(paciencia: int, margen: float = 1e-4):
    """Para un entrenamiento cuando la validación no mejora en `paciencia` épocas.

    Cada llamada a `entrenar` arma su propio estado desde cero: no compara contra otros
    trials.
    """
    estado = {"mejor": -float("inf"), "espera": 0}

    def reportar(epoca: int, valor: float) -> bool:
        if valor > estado["mejor"] + margen:
            estado["mejor"] = valor
            estado["espera"] = 0
        else:
            estado["espera"] += 1
        return estado["espera"] >= paciencia

    return reportar


def categoria_por_ic(ic_bajo: float, ic_alto: float) -> str:
    """Clasifica un intervalo de Wilson en 'bajo' (`ic_alto < 0.5`), 'no_distingue' (el
    intervalo contiene 0.5, extremos incluidos) o 'despegado' (`ic_bajo > 0.5`)."""
    if ic_alto < 0.5:
        return "bajo"
    if ic_bajo > 0.5:
        return "despegado"
    return "no_distingue"


def conclusion_por_ic(ics: Sequence[tuple[float, float]], cortes: int) -> str:
    """Frase de conclusión de la pregunta 4 a partir de los intervalos de Wilson de los
    cortes, cada uno clasificado con `categoria_por_ic`."""
    if cortes != len(ics):
        raise ValueError(f"cortes ({cortes}) no coincide con len(ics) ({len(ics)})")
    if cortes == 0:
        return "sin cortes no hay con qué correr una prueba pareada."
    if cortes == 1:
        return (
            "Con un solo corte (n=1) no puedes generalizar la clasificación ni correr una "
            "prueba pareada (hacen falta >= 2 semillas cortadas)."
        )
    categorias = [categoria_por_ic(ic_bajo, ic_alto) for ic_bajo, ic_alto in ics]
    n_bajo = categorias.count("bajo")
    n_despegado = categorias.count("despegado")
    n_no_distingue = categorias.count("no_distingue")
    if n_no_distingue == cortes:
        return (
            f"Los {cortes} cortes no se distinguen del azar (su intervalo de confianza de "
            "Wilson contiene 0.5): la paciencia cortó mientras la validación no se distinguía "
            "del azar."
        )
    if n_no_distingue == 0:
        return (
            f"Ninguno de los {cortes} cortes se queda indistinguible del azar: {n_despegado} ya "
            f"se habían despegado (intervalo de confianza entero sobre 0.5) y {n_bajo} quedan "
            "por debajo del azar (intervalo entero bajo 0.5)."
        )
    return (
        f"De los {cortes} cortes, {n_no_distingue} no se distinguen del azar, {n_despegado} ya "
        f"se habían despegado y {n_bajo} quedan por debajo del azar: una mezcla, no una meseta "
        "uniforme."
    )


def conclusion_despegue(ic_completo: tuple[float, float], veredicto: str) -> str:
    """Frase que decide si la paciencia cortó antes de que el modelo se despegara del azar,
    cuando todos los cortes caen en 'no_distingue'.

    Cita solo la evidencia que se cumplió en esta corrida: `ic_completo`, el intervalo de
    la validación del entrenamiento completo, y `veredicto`, el de la prueba pareada que el
    paso imprime a continuación. Solo un veredicto 'peor' cuenta como evidencia.
    """
    ic_bajo, ic_alto = ic_completo
    evidencias = []
    if ic_bajo > 0.5:
        evidencias.append(
            "la validación del completo tiene intervalo 95% "
            f"[{ic_bajo:.3f}, {ic_alto:.3f}], entero sobre 0.5"
        )
    if veredicto == "peor":
        evidencias.append(
            "la prueba pareada que aparece a continuación tiene veredicto 'peor' para la parada"
        )
    if evidencias:
        return (
            f" {' y, además, '.join(evidencias)}: la paciencia cortó dentro de la meseta "
            "inicial, ANTES de que el modelo se despegara del azar."
        )
    return (
        " Esta corrida no alcanza para decir si eso fue antes o después de que el modelo se "
        "despegara del azar: ni el intervalo 95% de validación del completo queda entero "
        "sobre 0.5 en estas semillas ni la prueba pareada que aparece a continuación dice 'peor'."
    )


PACIENCIA = 5
N_VAL = DEFAULTS["n_val"]  # cuenta de validación: base para el intervalo de cada corte.
# Delta fijado A PRIORI (un punto porcentual de exactitud verdadera): la
# escala en la que "perder eso al cortar" importa para este problema, no un
# valor elegido después de ver el resultado de la corrida.
DELTA_PAREADA_PACIENCIA = 0.01
# humo=4: con estas semillas cortan 2 de las 4, así que humo también recorre la
# clasificación agregada y la prueba pareada.
N_SEMILLAS_PACIENCIA = {"humo": 4, "taller": 10, "completo": 10}[presupuesto]
semillas_paciencia = semillas_de(20_260_304, N_SEMILLAS_PACIENCIA, datos=SEMILLA_BUSQUEDA.datos)

# Clasifica el último reporte de cada podado de Median con el mismo intervalo de Wilson que
# los cortes de la paciencia.
podas_median_por_categoria = {"bajo": 0, "no_distingue": 0, "despegado": 0}
for trial in podados_median:
    epoca_del_reporte = max(trial.intermediate_values)
    val_en_ese_reporte = trial.intermediate_values[epoca_del_reporte]
    ic_del_reporte = ic_proporcion(round(val_en_ese_reporte * N_VAL), N_VAL)
    podas_median_por_categoria[categoria_por_ic(*ic_del_reporte)] += 1
zona_median = (
    f"De los {len(podados_median)} trials que podó Median: "
    f"{podas_median_por_categoria['no_distingue']} con último reporte indistinguible del azar "
    "(la misma zona de indiferencia que clasifican estos cortes de paciencia), "
    f"{podas_median_por_categoria['despegado']} ya despegados y "
    f"{podas_median_por_categoria['bajo']} por debajo del azar."
)

epocas_paro: list[float] = []
val_parada: list[float] = []
verdad_parada: list[float] = []
verdad_completo: list[float] = []
val_completo: list[float] = []
cambios_verdad: list[float] = []  # parada - completo: negativo = perdió verdad al cortar
for s in semillas_paciencia:
    completo_s = entrenar({}, s)
    parada_s = entrenar({}, s, early_stopping_por_paciencia(PACIENCIA))
    epocas_paro.append(parada_s["epocas"])
    val_parada.append(parada_s["exactitud_val"])
    verdad_parada.append(parada_s["exactitud_verdadera"])
    verdad_completo.append(completo_s["exactitud_verdadera"])
    val_completo.append(completo_s["exactitud_val"])
    cambios_verdad.append(parada_s["exactitud_verdadera"] - completo_s["exactitud_verdadera"])

cortadas = [i for i, e in enumerate(epocas_paro) if e < EPOCAS]
cortes = len(cortadas)
cambio_verdad_promedio = statistics.mean(cambios_verdad)

# Ahorro comparable al de la pregunta 1: las semillas que no cortaron suman 0.
epocas_posibles_paciencia = N_SEMILLAS_PACIENCIA * EPOCAS
epocas_ahorradas_paciencia = sum(EPOCAS - e for e in epocas_paro)
ahorro_pct_paciencia = 100 * epocas_ahorradas_paciencia / epocas_posibles_paciencia

if cortes:
    # Clasifica cada corte por el intervalo de Wilson de su validación, nunca con una media
    # cruda contra un umbral fijo.
    detalles_corte = sorted(
        (int(epocas_paro[i]), val_parada[i], ic_proporcion(round(val_parada[i] * N_VAL), N_VAL))
        for i in cortadas
    )
    ics_corte = [ic for _, _, ic in detalles_corte]
    val_corte_prom = statistics.mean(val_parada[i] for i in cortadas)
    verdad_corte_prom = statistics.mean(verdad_parada[i] for i in cortadas)
    verdad_completo_corte_prom = statistics.mean(verdad_completo[i] for i in cortadas)
    perdida_corte = verdad_completo_corte_prom - verdad_corte_prom
    lista_cortes = ", ".join(
        f"época {e} val {v:.3f} "
        f"({'intervalo de confianza 95%' if i == 0 else 'intervalo 95%'} "
        f"[{ic[0]:.3f}, {ic[1]:.3f}])"
        for i, (e, v, ic) in enumerate(detalles_corte)
    )

    conclusion_azar = conclusion_por_ic(ics_corte, cortes)

    if cortes >= 2:
        pareada_paciencia = diferencia_pareada(
            {i: verdad_parada[i] for i in cortadas},
            {i: verdad_completo[i] for i in cortadas},
            delta=DELTA_PAREADA_PACIENCIA,
        )
        texto_pareada = (
            f" Prueba pareada (parada contra completo, delta={DELTA_PAREADA_PACIENCIA}): "
            f"diferencia media {pareada_paciencia.diferencia_media:+.3f} (intervalo "
            f"{100 * pareada_paciencia.confianza:.0f}% [{pareada_paciencia.ic_bajo:.3f}, "
            f"{pareada_paciencia.ic_alto:.3f}]), veredicto: {pareada_paciencia.veredicto}."
        )
        # Ver `conclusion_despegue`: solo cita la evidencia que se cumplió.
        if all(categoria_por_ic(*ic) == "no_distingue" for ic in ics_corte):
            r_completo_corte = resumen([val_completo[i] for i in cortadas])
            conclusion_azar += conclusion_despegue(
                (r_completo_corte.ic_bajo, r_completo_corte.ic_alto), pareada_paciencia.veredicto
            )
    else:
        # cortes == 1: `conclusion_azar` ya explicó por qué no alcanza para una
        # prueba pareada -- repetirlo aquí sería la misma frase dos veces.
        texto_pareada = ""

    detalle_cortes = (
        f"En las que cortó ({lista_cortes}), validación media {val_corte_prom:.3f} y verdad "
        f"media {verdad_corte_prom:.3f} contra {verdad_completo_corte_prom:.3f} del completo "
        f"(pérdida de {100 * perdida_corte:.1f} puntos porcentuales de exactitud verdadera, "
        f"completo - parada). {conclusion_azar}{texto_pareada}"
    )
    frase_ahorro = (
        f"Ahorró {epocas_ahorradas_paciencia:.0f} de {epocas_posibles_paciencia:.0f} épocas "
        f"posibles ({ahorro_pct_paciencia:.1f}%)."
    )
else:
    detalle_cortes = (
        f"Con el presupuesto real ({EPOCAS:.0f} épocas) la paciencia no alcanzó a activarse en "
        f"estas semillas: {conclusion_por_ic([], 0)}"
    )
    frase_ahorro = (
        f"En estas semillas no ahorró épocas (0 de {epocas_posibles_paciencia:.0f} épocas "
        "posibles)."
    )

print(
    f"4. Con paciencia={PACIENCIA} y el presupuesto real del problema ({EPOCAS:.0f} épocas) "
    f"sobre {N_SEMILLAS_PACIENCIA} semillas nuevas, cortó en {cortes} de "
    f"{N_SEMILLAS_PACIENCIA}. {detalle_cortes} Cambio en verdad promedio "
    f"sobre las {N_SEMILLAS_PACIENCIA} semillas (parada - completo): "
    f"{cambio_verdad_promedio:+.3f}. {frase_ahorro} {zona_median}"
)

# %% [markdown]
# ## Pregunta 5: por qué la poda no existe en multiobjetivo
#
# Con más de una métrica, `en_proceso` pasa `reportar=None` y el entrenamiento siempre
# corre completo. La razón está en Optuna: `Trial.report` y `Trial.should_prune` levantan
# `NotImplementedError` cuando el estudio tiene más de una dirección. "¿Va peor que la
# mediana?" no tiene una sola respuesta con dos objetivos que pueden empatar en uno y
# perder en el otro.

# %%
falla_multiobjetivo = None
try:
    estudio_multi = optuna.create_study(
        directions=["maximize", "maximize"], sampler=optuna.samplers.RandomSampler(seed=0)
    )
    trial_multi = estudio_multi.ask()
    trial_multi.report(0.5, 0)
except NotImplementedError as error:
    falla_multiobjetivo = error

print(
    f"5. Trial.report en un estudio multiobjetivo levanta "
    f"{type(falla_multiobjetivo).__name__ if falla_multiobjetivo else 'NADA (inesperado)'}: "
    f"{falla_multiobjetivo}"
)
if falla_multiobjetivo is None:
    raise AssertionError(
        "Trial.report en multiobjetivo debía levantar NotImplementedError (optuna 5.0.0)"
    )

# %% [markdown]
# ## Figura: curvas de aprendizaje del estudio Median, podados vs. completos

# %%
fig = Figure(figsize=(7, 4))
ax = fig.subplots()
trials_median = estudios["median"].get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
etiquetas_usadas = set()
for trial in trials_median:
    if not trial.intermediate_values:
        continue
    pasos = sorted(trial.intermediate_values)
    valores = [trial.intermediate_values[p] for p in pasos]
    podado = trial.state == optuna.trial.TrialState.PRUNED
    color = "tab:red" if podado else "tab:blue"
    etiqueta = "podado" if podado else "completo"
    ax.plot(
        pasos,
        valores,
        color=color,
        alpha=0.5,
        linewidth=1.2,
        marker="x" if podado else None,
        markevery=[-1] if podado else None,
        label=etiqueta if etiqueta not in etiquetas_usadas else None,
    )
    etiquetas_usadas.add(etiqueta)
ax.set_xlabel("época")
ax.set_ylabel("exactitud de validación (intermedia)")
ax.set_title(f"Paso 03: curvas de aprendizaje, estudio Median ({presupuesto})")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
figura = RESULTADOS / "03_poda.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
