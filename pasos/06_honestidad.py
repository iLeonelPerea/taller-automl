# %% [markdown]
# # Paso 06: honestidad -- la maldición del ganador
#
# El paso 00 mostró que la validación miente un poco, incluso sin buscar nada.
# Cuando SÍ buscas -- pruebas `N_BUSQUEDA` configuraciones y te quedas con
# la que dio el mejor `exactitud_val` -- esa mentira deja de ser aleatoria: el
# valor del ganador es el MÁXIMO de varias mediciones ruidosas, y el máximo de
# un ruido con media cero tiene media positiva. Eso es la maldición del
# ganador, y aquí la mides en vez de solo advertirla.
#
# Seis preguntas:
# 1. ¿Cuánto infló la búsqueda el valor del ganador? (contra una re-evaluación)
# 2. ¿Cuánto miente ese mismo valor contra el oráculo?
# 3. Re-evaluado, ¿el ganador le sigue ganando a la línea base?
# 4. ¿Cuántas semillas harían falta para estar seguros de eso?
# 5. Si repites la búsqueda completa muchas veces, ¿la inflación es
#    sistemática? ¿Coincide con lo que predice la teoría?
# 6. Con la disciplina de un conjunto de prueba sellado (congelas el
#    ganador ANTES de mirarlo y lo abres una sola vez): ¿qué tan bien se
#    sostiene?
#
# Correr:  uv run pasos/06_honestidad.py  (necesita una tarjeta de video NVIDIA
#          con CUDA y el controlador instalado; sin ella, el paso se detiene
#          con un mensaje claro)
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import dataclasses
import math
import os
import statistics as st
from pathlib import Path

import numpy as np
import optuna

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState
from scipy import stats as scipy_stats

from taller_hpo import estadistica, estudio
from taller_hpo.honestidad import (
    PruebaSellada,
    inflacion_ganador,
    mejores_de,
    plan_semillas,
    reevaluar,
)
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.problemas import juguete
from taller_hpo.semillas import semillas_de

# El log INFO de Optuna imprime una línea por trial y taparía las respuestas numeradas.
optuna.logging.set_verbosity(optuna.logging.WARNING)

PRESUPUESTOS = {
    "humo": {"n_busqueda": 8, "n_eval": 3, "r": 2},
    # El costo lo dominan n_busqueda*(R+1) entrenamientos: la búsqueda del ganador más
    # las R búsquedas repetidas de la pregunta 5. n_eval fija la precisión de las
    # respuestas 1, 3 y 4.
    "taller": {"n_busqueda": 30, "n_eval": 10, "r": 6},
    "completo": {"n_busqueda": 80, "n_eval": 20, "r": 20},
}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
N_BUSQUEDA = PRESUPUESTOS[presupuesto]["n_busqueda"]
N_EVAL = PRESUPUESTOS[presupuesto]["n_eval"]
R = PRESUPUESTOS[presupuesto]["r"]

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)
ESTUDIOS = Path(
    os.environ.get("TALLER_ESTUDIOS", Path(__file__).resolve().parent.parent / "estudios")
)

# Primero, en qué tarjeta de video entrena esta corrida.
print(f"Motor: {juguete.DESCRIPCION}\n")

# %% [markdown]
# ## El espacio de búsqueda: los mismos seis hiperparámetros del paso 00
#
# La línea base (los defaults tal cual) entra sembrada en el estudio, así que la
# búsqueda siempre puede compararse contra "lo que ya tenías".
#
# Este espacio difiere del de los pasos 02 y 03 en cinco hiperparámetros: `n_capas` (1-2
# en vez de 1-3), `ancho` (16-128 lineal en vez de 8-128 logarítmico), `alpha` (hasta
# 1e-1 en vez de 1.0), `batch_size` (entero 16-128 en vez de 32, 64 o 128) y, además,
# busca `activacion`. Solo `learning_rate_init` coincide.


# %%
def _espacio(trial: optuna.Trial) -> dict:
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "ancho": trial.suggest_int("ancho", 16, 128),
        "activacion": trial.suggest_categorical("activacion", ["relu", "tanh", "logistic"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 128),
    }


DISTRIBUCIONES = {
    "n_capas": IntDistribution(1, 2),
    "ancho": IntDistribution(16, 128),
    "activacion": CategoricalDistribution(["relu", "tanh", "logistic"]),
    "learning_rate_init": FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": FloatDistribution(1e-6, 1e-1, log=True),
    "batch_size": IntDistribution(16, 128),
}
LINEA_BASE_PARAMS = {k: juguete.DEFAULTS[k] for k in juguete.HIPERPARAMETROS}

# %% [markdown]
# ## El plan de semillas: números aleatorios comunes en la búsqueda, frescas para re-evaluar
#
# Toda la búsqueda usa UNA sola `Semillas` (`plan.busqueda`): así, cuando el
# sampler compara dos configuraciones, la diferencia es la configuración y no
# la suerte de la semilla (números aleatorios comunes). Re-evaluar con esas
# MISMAS semillas repetiría exactamente la suerte que el sampler ya explotó;
# `plan.evaluacion` trae `N_EVAL` semillas frescas, con los mismos datos.

# %%
SEMILLA_PLAN = 20_260_601
SEMILLA_TPE = 20_260_602
SEMILLA_REPETICIONES = 20_260_603

plan = plan_semillas(SEMILLA_PLAN, N_EVAL)

# %% [markdown]
# ## La búsqueda: TPE con fábrica de sampler (reanudar no repite) y línea base sembrada
#
# `juguete.PROTOCOLO_MOTOR` entra al protocolo del estudio: registra que PyTorch entrenó
# en la tarjeta de video, sin el índice, así otra tarjeta puede sumar trials al mismo
# estudio.

# %%
objetivo_ganador = en_proceso(
    juguete.entrenar,
    _espacio,
    metricas=["exactitud_val"],
    semillas=plan.busqueda,
    atributos=["exactitud_verdadera"],
)
estudio_ganador = estudio.crear_estudio(
    "06_honestidad_ganador",
    destino=ESTUDIOS / f"06_honestidad_ganador_{presupuesto}.log",
    direcciones=["maximize"],
    sampler=lambda n_previos: optuna.samplers.TPESampler(
        seed=SEMILLA_TPE + n_previos, n_startup_trials=5, multivariate=True
    ),
    pruner=optuna.pruners.NopPruner(),
    protocolo={
        "metrica": "exactitud_val",
        "sampler": "tpe",
        # Todos los kwargs del sampler explícitos, como en el paso 02.
        "sampler_kwargs": {
            "seed": "SEMILLA_TPE + n_previos",
            "seed_base": SEMILLA_TPE,
            "n_startup_trials": 5,
            "multivariate": True,
        },
        "pruner": "NopPruner",
        # Política de semillas CRN: reabrir con otra SEMILLA_PLAN cambiaría en silencio
        # el conjunto de validación de todos los trials ya corridos.
        "politica_semillas": {
            "tipo": "CRN",
            "plan_base": SEMILLA_PLAN,
            "busqueda": dataclasses.asdict(plan.busqueda),
        },
        "presupuesto": presupuesto,
        "n_busqueda": N_BUSQUEDA,
        **juguete.PROTOCOLO_MOTOR,
    },
)
estudio.recuperar_huerfanos(estudio_ganador)
estudio.sembrar_linea_base(estudio_ganador, LINEA_BASE_PARAMS, DISTRIBUCIONES)
estudio.optimizar(
    estudio_ganador,
    objetivo_ganador,
    n_total=N_BUSQUEDA,
    atrapar=(juguete.Divergencia, MetricaNoFinita),
)

# %% [markdown]
# ## El ganador, la línea base, y su re-evaluación pareada
#
# `mejores_de` saca el top-3 (excluyendo la línea base) y la línea base del
# estudio ya corrido. `reevaluar` corre esas 4 configuraciones otra vez, fuera
# del estudio, con las MISMAS `plan.evaluacion` para las 4 -- así la
# comparación entre el ganador y la línea base es pareada semilla por semilla.

# %%
seleccion = mejores_de(estudio_ganador, 3, incluir_linea_base=True)
reevaluados = reevaluar(
    {etiqueta: info["params"] for etiqueta, info in seleccion.items()},
    juguete.entrenar,
    plan.evaluacion,
)
ganador = seleccion["top_1"]
trial_ganador = next(t for t in estudio_ganador.trials if t.number == ganador["trial"])
val_ganador_reeval = [m["exactitud_val"] for m in reevaluados["top_1"]]
val_base_reeval = [m["exactitud_val"] for m in reevaluados["linea_base"]]
# El oráculo de la re-evaluación ya viene en `reevaluados`: sirve para revelar, nunca
# para elegir.
verdad_ganador_reeval = [m["exactitud_verdadera"] for m in reevaluados["top_1"]]
verdad_base_reeval = [m["exactitud_verdadera"] for m in reevaluados["linea_base"]]

# %% [markdown]
# ## Preguntas 1-4: inflación, oráculo, ¿de verdad gana?, y cuántas semillas harían falta


# %%
def _veredicto_ic(bajo: float, alto: float, *, no_concluyente: str = "no concluyente") -> str:
    if bajo > 0.0:
        return "positiva"
    if alto < 0.0:
        return "negativa"
    return no_concluyente


def _distingue_de_una_semilla(d: float, pareado) -> bool:
    """¿`d`, la diferencia observada en la semilla de la búsqueda, se distingue de las
    `pareado.n` diferencias re-evaluadas?

    Sin selección, la semilla de la búsqueda es intercambiable con las de evaluación, así
    que la vara es un intervalo de predicción para una observación nueva,
    `de_dif * sqrt(1 + 1/n)` con la t de `n - 1` grados de libertad. El intervalo de la
    media declararía selección de más.
    """
    residuo = d - pareado.diferencia_media
    t_pred = float(scipy_stats.t.ppf(0.975, pareado.n - 1))
    margen = t_pred * pareado.de_dif * math.sqrt(1.0 + 1.0 / pareado.n)
    return abs(residuo) > margen


def _alcanzo_potencia(n: int, n_bajo: int, n_alto: int) -> str:
    """¿`n` semillas alcanzan la potencia planeada para detectar delta=0.01, dado el
    intervalo chi-cuadrado [`n_bajo`, `n_alto`] del n necesario? Si `n` cae dentro, la
    respuesta es "no concluyente".
    """
    if n >= n_alto:
        return "sí, las semillas alcanzaron la potencia planeada"
    if n < n_bajo:
        return "no, las semillas no alcanzaron la potencia planeada"
    return f"no concluyente (el n necesario va de {n_bajo} a {n_alto})"


def _texto_brecha(ic_bajo: float, ic_alto: float, *, cola: str = "") -> str:
    """¿La brecha val-verdad, la suerte del conjunto de validación que re-evaluar con los
    mismos datos no ve, se distingue de cero? `cola` solo se agrega si se distingue.
    """
    if ic_bajo <= 0.0 <= ic_alto:
        return "no concluyente: con estas semillas la brecha no se distingue de cero"
    return (
        "distinta de cero: esa parte (la suerte del conjunto de validación) no la ve "
        f"re-evaluar con los mismos datos{cola}"
    )


def _comparacion_teoria(
    ic_bajo: float, ic_alto: float, referencia_baja: float, referencia_alta: float, r: int
) -> str:
    """¿El intervalo de la inflación medida en `r` repeticiones coincide con la cota
    teórica, que también trae su propio intervalo chi-cuadrado? Dice "sí" o "no" solo
    cuando el intervalo medido queda entero a un lado del de la cota.
    """
    if ic_alto < 0.0:
        return (
            "no: la inflación es negativa (intervalo por debajo de 0), contra lo que predice "
            "la teoría"
        )
    if ic_alto < referencia_baja:
        cola = f"; con R={r} la inflación tampoco se distingue de 0" if ic_bajo <= 0.0 else ""
        return (
            "sí: la media medida queda por debajo de la cota, consistente con que mezclar "
            "configuraciones con verdades distintas (la búsqueda compara configuraciones con "
            "verdades distintas, no N configuraciones de la misma verdad) solo puede empujar la "
            "inflación por debajo de la cota, nunca por encima" + cola
        )
    if ic_bajo > referencia_alta:
        return (
            "no: la media medida queda POR ENCIMA de la cota -- mezclar configuraciones con "
            "verdades distintas NO puede ser la causa (esa mezcla solo empuja la inflación hacia "
            "abajo); las causas posibles son ruido de validación heterocedástico entre "
            "configuraciones (que la cota iid no captura), sigma estimado en la búsqueda del "
            "ganador con pocos grados de libertad, R chico (intervalo ancho con pocas "
            "repeticiones) o colas no normales"
        )
    if ic_bajo <= 0.0:
        return (
            f"no concluyente con R={r}: el intervalo medido contiene el 0 y se solapa con el "
            "de la cota"
        )
    return f"no concluyente con R={r}: el intervalo medido y el de la cota se solapan"


inflacion = inflacion_ganador(ganador["valor"], val_ganador_reeval, direccion="maximizar")
veredicto_ganador = _veredicto_ic(inflacion.ic_bajo, inflacion.ic_alto)
# Control: la línea base nunca fue elegida por la búsqueda, así que su propio residuo es
# la vara para distinguir selección de la varianza de una semilla cualquiera.
inflacion_base = inflacion_ganador(
    seleccion["linea_base"]["valor"], val_base_reeval, direccion="maximizar"
)
veredicto_base = _veredicto_ic(inflacion_base.ic_bajo, inflacion_base.ic_alto)
resumen_ganador_reeval = estadistica.resumen(val_ganador_reeval)
pareado = estadistica.diferencia_pareada(
    dict(enumerate(val_ganador_reeval)),
    dict(enumerate(val_base_reeval)),
    delta=0.01,
    direccion="maximizar",
)
# `_distingue_de_una_semilla` trata a la semilla de la búsqueda como una semilla nueva
# cualquiera. `residuo_busqueda` y `margen_una_semilla` solo se imprimen: muestran la
# magnitud que compara la función.
d_valor_busqueda = ganador["valor"] - seleccion["linea_base"]["valor"]
residuo_busqueda = d_valor_busqueda - pareado.diferencia_media
margen_una_semilla = (
    float(scipy_stats.t.ppf(0.975, pareado.n - 1))
    * pareado.de_dif
    * math.sqrt(1.0 + 1.0 / pareado.n)
)
separa_control = _distingue_de_una_semilla(d_valor_busqueda, pareado)
# Revelación del oráculo para las respuestas 1 y 3: re-evaluar con los mismos datos quita
# la suerte de orden e init, pero no la del conjunto de validación. Para esa hace falta
# la prueba sellada de la pregunta 6.
resumen_verdad_ganador = estadistica.resumen(verdad_ganador_reeval)
pareado_verdad = estadistica.diferencia_pareada(
    dict(enumerate(verdad_ganador_reeval)),
    dict(enumerate(verdad_base_reeval)),
    delta=0.01,
    direccion="maximizar",
)
# de_dif es una estimación con pocas semillas: la respuesta 4 reporta un rango, con su
# intervalo chi-cuadrado propagado por `semillas_necesarias`. Con de_dif == 0.0 no hay
# varianza que estimar.
ALFA_DEDIF = 0.05
if pareado.de_dif > 0.0:
    gl_dedif = pareado.n - 1
    de_dif_bajo = pareado.de_dif * math.sqrt(
        gl_dedif / scipy_stats.chi2.ppf(1 - ALFA_DEDIF / 2, gl_dedif)
    )
    de_dif_alto = pareado.de_dif * math.sqrt(
        gl_dedif / scipy_stats.chi2.ppf(ALFA_DEDIF / 2, gl_dedif)
    )
    n_delta = estadistica.semillas_necesarias(pareado.de_dif, delta=0.01)
    n_delta_bajo = estadistica.semillas_necesarias(de_dif_bajo, delta=0.01)
    n_delta_alto = estadistica.semillas_necesarias(de_dif_alto, delta=0.01)
else:
    de_dif_bajo = de_dif_alto = 0.0
    n_delta = n_delta_bajo = n_delta_alto = None

# %% [markdown]
# ## Pregunta 5: la búsqueda completa, repetida R veces con otras semillas
#
# Cada repetición corre una búsqueda TPE completa de `N_BUSQUEDA` trials con su propia
# semilla, en memoria.
#
# La inflación cruda de cada repetición (el valor del ganador menos su verdad) mezcla el
# desvío común del conjunto de validación de esa búsqueda con el efecto de elegir el
# máximo. Restarle el desvío medio de todos los trials de la búsqueda, incluido el
# ganador, aísla el segundo. Si los desvíos son ruido iid de media cero, la cota es
# `sigma * maximo_esperado_normal(N_BUSQUEDA)`, con `sigma` estimada de la búsqueda del
# ganador. La inflación esperada queda por debajo de esa cota, y la respuesta 5 lo decide
# con su propio intervalo.

# %%
completos_ganador = estudio_ganador.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
ruido = [t.value - t.user_attrs["exactitud_verdadera"] for t in completos_ganador]
sigma = float(np.std(ruido, ddof=1))
maximo_esperado = estadistica.maximo_esperado_normal(N_BUSQUEDA)
referencia = sigma * maximo_esperado
# sigma es una estimación: su intervalo chi-cuadrado se propaga a la referencia antes de
# comparar (`_comparacion_teoria`).
gl_sigma = len(ruido) - 1
sigma_bajo = sigma * math.sqrt(gl_sigma / scipy_stats.chi2.ppf(1 - ALFA_DEDIF / 2, gl_sigma))
sigma_alto = sigma * math.sqrt(gl_sigma / scipy_stats.chi2.ppf(ALFA_DEDIF / 2, gl_sigma))
referencia_baja = sigma_bajo * maximo_esperado
referencia_alta = sigma_alto * maximo_esperado

semillas_repeticiones = semillas_de(SEMILLA_REPETICIONES, R)
crudas_repeticiones: list[float] = []
inflaciones_repeticiones: list[float] = []
for r, semilla_r in enumerate(semillas_repeticiones):
    objetivo_r = en_proceso(
        juguete.entrenar,
        _espacio,
        metricas=["exactitud_val"],
        semillas=semilla_r,
        atributos=["exactitud_verdadera"],
    )
    estudio_r = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=SEMILLA_TPE + 1000 * (r + 1), n_startup_trials=5, multivariate=True
        ),
        pruner=optuna.pruners.NopPruner(),
    )
    estudio.optimizar(
        estudio_r, objetivo_r, n_total=N_BUSQUEDA, atrapar=(juguete.Divergencia, MetricaNoFinita)
    )
    completos_r = estudio_r.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    desvios_r = {t.number: t.value - t.user_attrs["exactitud_verdadera"] for t in completos_r}
    ganador_r = mejores_de(estudio_r, 1, incluir_linea_base=False)["top_1"]
    cruda_r = desvios_r[ganador_r["trial"]]
    crudas_repeticiones.append(cruda_r)
    # Media de todos los desvíos de esta repetición, incluido el ganador (ver la celda de
    # arriba).
    inflaciones_repeticiones.append(cruda_r - st.mean(desvios_r.values()))

resumen_crudas = estadistica.resumen(crudas_repeticiones)
resumen_inflaciones = estadistica.resumen(inflaciones_repeticiones)
veredicto_inflacion = _veredicto_ic(
    resumen_inflaciones.ic_bajo,
    resumen_inflaciones.ic_alto,
    no_concluyente=f"no concluyente con R={R}",
)

# %% [markdown]
# ## Pregunta 6: la prueba sellada
#
# El paso congela al ganador (`sellar`) antes de mirar la prueba. `abrir` entrena la misma
# configuración con `plan.busqueda`, la semilla que ya produjo ese modelo en la búsqueda,
# ahora con `n_prueba=5000`: un conjunto de prueba con su propia semilla fija
# (`juguete.SEMILLA_PRUEBA`), independiente de los datos de entrenamiento y de validación.
# El paso la abre una sola vez.

# %%
prueba = PruebaSellada(
    evaluar=lambda params: juguete.entrenar({**params, "n_prueba": 5000}, plan.busqueda, None)[
        "exactitud_prueba"
    ]
)
prueba.sellar(ganador["params"])
exactitud_prueba = prueba.abrir(ganador["params"])

# %% [markdown]
# ## Las seis respuestas

# %%
print(f"Presupuesto: {presupuesto} (búsqueda={N_BUSQUEDA}, re-evaluación={N_EVAL}, R={R})\n")

distingue_control = "SÍ se distingue" if separa_control else "NO se distingue"
# Brecha val-verdad de la re-evaluación del ganador, semilla por semilla, para que
# `resumen` le dé su propio intervalo y la frase pueda decir "no concluyente".
brecha_reeval_verdad_valores = [
    v - t for v, t in zip(val_ganador_reeval, verdad_ganador_reeval, strict=True)
]
resumen_brecha_reeval_verdad = estadistica.resumen(brecha_reeval_verdad_valores)
brecha_reeval_verdad = 100 * resumen_brecha_reeval_verdad.media
texto_brecha_reeval_verdad = _texto_brecha(
    resumen_brecha_reeval_verdad.ic_bajo, resumen_brecha_reeval_verdad.ic_alto
)
print(
    f"1. Diferencia búsqueda - re-evaluación: la búsqueda vio {inflacion.valor_busqueda:.3f}, la "
    f"re-evaluación ({N_EVAL} semillas) da {inflacion.media_reevaluada:.3f} "
    f"[{resumen_ganador_reeval.ic_bajo:.3f}, {resumen_ganador_reeval.ic_alto:.3f}]: "
    f"{100 * inflacion.inflacion:+.1f} puntos "
    f"(intervalo de confianza [{100 * inflacion.ic_bajo:+.1f}, {100 * inflacion.ic_alto:+.1f}]) "
    f"-- {veredicto_ganador}. Control (línea base, que nadie eligió): se desvía de su propia "
    "re-evaluación "
    f"{100 * inflacion_base.inflacion:+.1f} puntos "
    f"(intervalo [{100 * inflacion_base.ic_bajo:+.1f}, "
    f"{100 * inflacion_base.ic_alto:+.1f}]) -- {veredicto_base}; la inflación del ganador "
    f"{distingue_control} de ese control: diferencia de inflaciones {100 * residuo_busqueda:+.1f} "
    f"puntos contra ±{100 * margen_una_semilla:.1f} puntos que da UNA semilla cualquiera sin "
    f"selección (intervalo de predicción t, {pareado.n - 1} "
    f"{'grado' if pareado.n - 1 == 1 else 'grados'} de libertad). "
    "Revelación del oráculo (sin usarla para elegir): en "
    f"verdad, el ganador re-evaluado promedia {resumen_verdad_ganador.media:.3f} "
    f"(intervalo [{resumen_verdad_ganador.ic_bajo:.3f}, {resumen_verdad_ganador.ic_alto:.3f}]); la "
    f"re-evaluación (mismos datos) queda {brecha_reeval_verdad:+.1f} puntos respecto a la "
    f"verdad (intervalo [{100 * resumen_brecha_reeval_verdad.ic_bajo:+.1f}, "
    f"{100 * resumen_brecha_reeval_verdad.ic_alto:+.1f}]) -- {texto_brecha_reeval_verdad}."
)
exactitud_verdadera_ganador = trial_ganador.user_attrs["exactitud_verdadera"]
print(
    f"2. Contra el oráculo: la validación del ganador durante la búsqueda fue "
    f"{ganador['valor']:.3f}, su exactitud verdadera era {exactitud_verdadera_ganador:.3f} "
    f"-- la validación ya mentía "
    f"{100 * (ganador['valor'] - exactitud_verdadera_ganador):+.1f} puntos antes de "
    "re-evaluar nada."
)
# Brecha val-verdad de la comparación pareada, también semilla por semilla.
brecha_val_verdad_valores = [
    (gv - bv) - (gt - bt)
    for gv, bv, gt, bt in zip(
        val_ganador_reeval, val_base_reeval, verdad_ganador_reeval, verdad_base_reeval, strict=True
    )
]
resumen_brecha_val_verdad = estadistica.resumen(brecha_val_verdad_valores)
brecha_val_verdad = 100 * resumen_brecha_val_verdad.media
# La cola "para verla del todo..." va solo en la rama "distinta de cero".
texto_brecha_val_verdad = _texto_brecha(
    resumen_brecha_val_verdad.ic_bajo,
    resumen_brecha_val_verdad.ic_alto,
    cola="; para verla del todo hace falta un conjunto nuevo (pregunta 6)",
)
print(
    f"3. Re-evaluado, el ganador ({st.mean(val_ganador_reeval):.3f}) contra la línea base "
    f"({st.mean(val_base_reeval):.3f}): diferencia {100 * pareado.diferencia_media:+.1f} "
    "puntos (intervalo de confianza "
    f"[{100 * pareado.ic_bajo:+.1f}, {100 * pareado.ic_alto:+.1f}]), "
    f"veredicto '{pareado.veredicto}' con delta=0.01 y {pareado.n} semillas. En verdad "
    "(revelación del oráculo), la misma comparación pareada da "
    f"{100 * pareado_verdad.diferencia_media:+.1f} puntos (brecha val-verdad: "
    f"{brecha_val_verdad:+.1f} puntos, intervalo [{100 * resumen_brecha_val_verdad.ic_bajo:+.1f}, "
    f"{100 * resumen_brecha_val_verdad.ic_alto:+.1f}]) -- {texto_brecha_val_verdad}."
)
diferencia_abs = abs(pareado.diferencia_media)
if pareado.de_dif == 0.0:
    detalle_4 = (
        "las diferencias pareadas de esta re-evaluación son idénticas (su desviación estándar "
        "es 0.0): no hay "
        "varianza con la que dimensionar cuántas semillas harían falta"
    )
elif diferencia_abs > 0.0:
    # n_obs es potencia post hoc, solo como referencia de cuán chica es la diferencia
    # observada frente a delta. El veredicto sale del intervalo de `pareado`.
    n_obs = estadistica.semillas_necesarias(pareado.de_dif, delta=diferencia_abs)
    # _alcanzo_potencia da tres ramas: decidir solo con el punto central exageraría la
    # precisión.
    alcanzo_potencia = _alcanzo_potencia(pareado.n, n_delta_bajo, n_delta_alto)
    detalle_4 = (
        f"la diferencia observada ({pareado.diferencia_media:.3f}, "
        f"{diferencia_abs / 0.01:.0f} veces delta) por sí sola necesitaría {n_obs} semillas "
        "para detectarse con potencia 80%, pero planear con la diferencia observada es "
        "circular -- ya pasó por la selección y es ruidosa; planea con delta, y con "
        f"{pareado.n} semillas: {alcanzo_potencia} (veredicto '{pareado.veredicto}')"
    )
else:
    detalle_4 = (
        "la diferencia observada fue exactamente 0.0: ninguna cantidad de semillas la "
        "distingue de cero"
    )
if pareado.de_dif > 0.0:
    print(
        f"4. Con una desviación estándar de las diferencias de {pareado.de_dif:.4f} (con solo "
        f"{pareado.n} semillas de evaluación, intervalo de confianza "
        f"{100 * (1 - ALFA_DEDIF):.0f}% de esa desviación [{de_dif_bajo:.4f}, "
        f"{de_dif_alto:.4f}]), "
        "detectar una diferencia del tamaño de delta=0.01 (la mínima que importa) con potencia "
        f"80% y alfa 0.05 exige {n_delta} semillas con el punto central (entre {n_delta_bajo} y "
        f"{n_delta_alto} según esa misma incertidumbre; hubo {pareado.n}); {detalle_4}."
    )
else:
    print(
        f"4. Con una desviación estándar de las diferencias de 0.0000 (las {pareado.n} "
        "diferencias pareadas de esta re-evaluación son "
        f"idénticas): {detalle_4}."
    )
referencia_puntos = 100 * referencia
referencia_baja_puntos = 100 * referencia_baja
referencia_alta_puntos = 100 * referencia_alta
# Los dos lados en puntos (*100): resumen_inflaciones está en fracción (0-1).
ic_alto_infl_puntos = 100 * resumen_inflaciones.ic_alto
ic_bajo_infl_puntos = 100 * resumen_inflaciones.ic_bajo
comparacion_teoria = _comparacion_teoria(
    ic_bajo_infl_puntos, ic_alto_infl_puntos, referencia_baja_puntos, referencia_alta_puntos, R
)
print(
    f"5. En {R} búsquedas completas repetidas (otras semillas), la parte de la inflación "
    f"debida a elegir el máximo promedia {100 * resumen_inflaciones.media:+.2f} puntos "
    "(intervalo de confianza "
    f"[{100 * resumen_inflaciones.ic_bajo:+.2f}, {100 * resumen_inflaciones.ic_alto:+.2f}]) "
    f"-- {veredicto_inflacion}. La inflación cruda (sin aislar el desvío común del conjunto de "
    f"validación) promedia {100 * resumen_crudas.media:+.2f} puntos "
    f"(intervalo [{100 * resumen_crudas.ic_bajo:+.2f}, {100 * resumen_crudas.ic_alto:+.2f}]). "
    f"Referencia teórica (cota superior si los {N_BUSQUEDA} desvíos val-verdad son iid con la "
    "misma sigma; con la misma verdad sería igualdad; sigma medido en los trials de la "
    f"búsqueda del ganador): sigma·E[max de {N_BUSQUEDA} normales] = {sigma:.4f} * "
    f"{maximo_esperado:.3f} = {referencia:.4f} ({referencia_puntos:.2f} puntos; intervalo "
    f"{100 * (1 - ALFA_DEDIF):.0f}% de la cota, propagando el intervalo de sigma: "
    f"[{referencia_baja_puntos:.2f}, {referencia_alta_puntos:.2f}]). ¿Coincide con la teoría? "
    f"{comparacion_teoria}."
)
ic_prueba_bajo, ic_prueba_alto = estadistica.ic_proporcion(round(exactitud_prueba * 5000), 5000)
contiene_verdad = ic_prueba_bajo <= exactitud_verdadera_ganador <= ic_prueba_alto
separa_validacion = not (ic_prueba_bajo <= ganador["valor"] <= ic_prueba_alto)
print(
    f"6. Prueba sellada (n=5000, abierta una sola vez): exactitud_prueba={exactitud_prueba:.3f} "
    f"(intervalo de confianza 95% [{ic_prueba_bajo:.3f}, {ic_prueba_alto:.3f}]) vs. verdad="
    f"{exactitud_verdadera_ganador:.3f} vs. validación={ganador['valor']:.3f} -- el intervalo "
    f"{'contiene' if contiene_verdad else 'NO contiene'} la verdad y "
    f"{'separa' if separa_validacion else 'NO separa'} la prueba de la validación."
)

# %% [markdown]
# ## Figura: la inflación por repetición, y el ganador contra la línea base re-evaluados

# %%
fig = Figure(figsize=(10, 4))
ax_izq, ax_der = fig.subplots(1, 2)

ax_izq.scatter(range(1, R + 1), inflaciones_repeticiones, marker="o", label="inflación medida")
ax_izq.axhline(0.0, color="black", linewidth=0.8)
ax_izq.axhline(referencia, linestyle="--", color="tab:red", label=f"referencia {referencia:.3f}")
ax_izq.axhline(
    resumen_inflaciones.media,
    linestyle=":",
    color="tab:blue",
    label=f"media {resumen_inflaciones.media:.3f}",
)
ax_izq.set_xlabel("repetición")
ax_izq.set_ylabel("inflación (elegir el máximo, vs. oráculo)")
ax_izq.set_title(f"Inflación en {R} búsquedas repetidas")
ax_izq.legend(loc="best", fontsize=7)

etiquetas = list(seleccion)
for i, etiqueta in enumerate(etiquetas):
    valores = [m["exactitud_val"] for m in reevaluados[etiqueta]]
    jitter = np.linspace(-0.08, 0.08, len(valores))
    ax_der.scatter(
        i + jitter,
        valores,
        marker="o",
        color="tab:blue",
        s=15,
        alpha=0.7,
        label="re-evaluación" if i == 0 else None,
    )
    ax_der.scatter(
        [i],
        [seleccion[etiqueta]["valor"]],
        marker="x",
        color="tab:red",
        s=60,
        label="valor de la búsqueda" if i == 0 else None,
    )
ax_der.set_xticks(range(len(etiquetas)))
ax_der.set_xticklabels(etiquetas, rotation=20)
ax_der.set_ylabel("exactitud_val")
ax_der.set_title("Ganador y línea base: búsqueda vs. re-evaluación")
ax_der.legend(loc="best", fontsize=7)

fig.tight_layout()
figura = RESULTADOS / "06_honestidad.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
