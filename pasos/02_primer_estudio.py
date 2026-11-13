# %% [markdown]
# # Paso 02: primer estudio con método
#
# El paso 00 midió tu línea base y el 01 mostró que random gana a grid en
# alta dimensión. Aquí juntas todo en un estudio real de Optuna: sampler
# escrito explícito, storage en disco, la línea base sembrada DENTRO del
# estudio (no aparte, como en el paso 00) y reanudar entre dos sesiones sin
# repetir trabajo.
#
# Cinco preguntas:
# 1. ¿Qué tan lejos está la línea base del mejor encontrado por validación?
# 2. Ese mejor por validación, ¿dice la verdad? Re-entrenando la misma config ganadora,
#    y la línea base como control, con semillas nuevas sobre los mismos datos: ¿cuánto
#    de la ventaja del ganador es la config, cuánto es la semilla que vio la búsqueda y
#    cuánto de eso ves sin oráculo?
# 3. ¿Reanudar en la segunda sesión repitió alguna configuración?
# 4. ¿Qué perillas importan de verdad, contra un placebo que el modelo no lee?
# 5. ¿Cómo abres el dashboard para explorar el estudio a mano?
#
# Correr:  uv run pasos/02_primer_estudio.py  (necesita una tarjeta de video
#          NVIDIA con CUDA y el controlador instalado; sin ella, el paso se
#          detiene con un mensaje claro)
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
from scipy import stats as scipy_stats

from taller_hpo import estadistica, estudio, honestidad
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.problemas import juguete, paridad
from taller_hpo.semillas import semillas_de

# El log INFO de Optuna imprime una línea por trial y tapa las respuestas
# numeradas de abajo; con WARNING solo ves avisos de verdad.
optuna.logging.set_verbosity(optuna.logging.WARNING)

PRESUPUESTOS = {"humo": 10, "taller": 60, "completo": 150}
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
DESTINO = ESTUDIOS / f"02_primer_estudio_{presupuesto}.log"
NOMBRE_ESTUDIO = "02_primer_estudio"

# El paso entrena el juguete siempre a través de `juguete` e imprime en qué tarjeta de
# video corre.
print(f"Motor: {juguete.DESCRIPCION}\n")

ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

# %% [markdown]
# ## El espacio: el único lugar con `suggest_*`
#
# Cinco perillas reales más `placebo`, un control negativo que `entrenar` NUNCA
# lee (ver `taller_hpo.problemas.juguete`): si una búsqueda le asigna
# importancia, esa importancia es puro ruido y sirve de piso para las demás.

# %%
CLAVES_ESPACIO = ["learning_rate_init", "alpha", "ancho", "n_capas", "batch_size", "placebo"]

DISTRIBUCIONES = {
    "learning_rate_init": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": optuna.distributions.FloatDistribution(1e-6, 1.0, log=True),
    "ancho": optuna.distributions.IntDistribution(8, 128, log=True),
    "n_capas": optuna.distributions.IntDistribution(1, 3),
    "batch_size": optuna.distributions.CategoricalDistribution([32, 64, 128]),
    "placebo": optuna.distributions.FloatDistribution(0.0, 1.0),
}

# La línea base: los DEFAULTS del juguete para esas mismas claves, como si nunca la
# ajustaras.
LINEA_BASE = {clave: juguete.DEFAULTS[clave] for clave in CLAVES_ESPACIO}

# El problema, fijo durante la búsqueda, y su techo de Bayes: la respuesta 2 los usa para
# revelar con el oráculo si este conjunto de validación es fácil o difícil.
PROBLEMA = {clave: juguete.DEFAULTS[clave] for clave in ("d", "a", "k_ruido", "rotar")}
TECHO = 1.0 - paridad.error_bayes(PROBLEMA["d"], PROBLEMA["a"])


def espacio(trial: optuna.Trial) -> dict:
    return {
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1.0, log=True),
        "ancho": trial.suggest_int("ancho", 8, 128, log=True),
        "n_capas": trial.suggest_int("n_capas", 1, 3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "placebo": trial.suggest_float("placebo", 0.0, 1.0),
    }


# %% [markdown]
# ## El protocolo: sampler escrito, política de semillas, storage
#
# `crear_estudio` guarda este `protocolo` en `study.user_attrs`: reabrir con uno distinto
# es un ValueError, no un estudio mezclado en silencio. La política de semillas es CRN
# (números aleatorios comunes): todas las configs se miden con la misma semilla, así la
# comparación es pareada. Eso no quita la suerte de cada config con esa semilla, y el
# ganador probablemente la tuvo buena (la respuesta 2 mide cuánta). Optuna 5 cambió los
# defaults del sampler, así que el paso escribe todos sus kwargs.

# %%
# Base fija de la fábrica de sampler: cada reapertura le suma los trials ya corridos, así
# reanudar no repite la secuencia de la sesión anterior.
SEMILLA_BASE_SAMPLER = 314_159
N_STARTUP_TRIALS = 10  # el default de optuna.samplers.TPESampler, escrito explícito.


# Registra la semilla que de verdad recibió cada llamada a la fábrica, para imprimirla
# tal cual.
semillas_sampler: list[int] = []


def fabrica_sampler(n_previos: int) -> optuna.samplers.TPESampler:
    semilla = SEMILLA_BASE_SAMPLER + n_previos
    semillas_sampler.append(semilla)
    return optuna.samplers.TPESampler(
        seed=semilla,
        n_startup_trials=N_STARTUP_TRIALS,
        multivariate=True,
        constant_liar=False,
    )


# Semillas comunes (CRN): la misma para cada trial. `base=0` entra al protocolo, así
# reabrir con otra base es un error y no un cambio silencioso del conjunto de validación.
SEMILLA_UNICA = semillas_de(0, 1)[0]

PROTOCOLO = {
    "sampler": "TPESampler",
    "sampler_kwargs": {
        "seed_base": SEMILLA_BASE_SAMPLER,
        "seed": "seed_base + n_previos",
        "n_startup_trials": N_STARTUP_TRIALS,
        "multivariate": True,
        "constant_liar": False,
    },
    "pruner": "NopPruner",
    "politica_semillas": {
        "tipo": "CRN",
        "base": 0,
        "n": 1,
        "semillas": dataclasses.asdict(SEMILLA_UNICA),
    },
    "metrica": "exactitud_val",
    # PROTOCOLO_MOTOR entra al protocolo: registra que PyTorch entrenó en la tarjeta de
    # video, sin el índice, así otra tarjeta puede seguir el mismo estudio.
    **juguete.PROTOCOLO_MOTOR,
}

objetivo = en_proceso(
    juguete.entrenar,
    espacio,
    metricas=["exactitud_val"],
    semillas=SEMILLA_UNICA,
    atributos=["exactitud_verdadera", "n_parametros", "segundos"],
)

# %% [markdown]
# ## Dos sesiones, como si fuera otro día
#
# La primera sesión corre la mitad del presupuesto y "cierra". La segunda vuelve a llamar
# `crear_estudio` con la misma fábrica y el mismo protocolo, como harías al retomar el
# script otro día, y completa hasta `N_TOTAL`. Si `DESTINO` ya tiene los trials, no hay
# nada que hacer. Antes de sembrar la línea base, el paso recupera los huérfanos de una
# corrida cortada, porque JournalStorage no tiene heartbeat.

# %%
mitad = N_TOTAL // 2

# Cuenta las llamadas de esta ejecución, aunque las celdas se re-ejecuten en el mismo
# proceso.
llamadas_sampler_previas = len(semillas_sampler)

sesion_1 = estudio.crear_estudio(
    NOMBRE_ESTUDIO,
    destino=DESTINO,
    direcciones=["maximize"],
    sampler=fabrica_sampler,
    pruner=optuna.pruners.NopPruner(),
    protocolo=PROTOCOLO,
)
n_previos_1 = len(sesion_1.get_trials(deepcopy=False))
print(
    f"Sesión 1: {'creada' if n_previos_1 == 0 else 'reabierta'} con {n_previos_1} trials "
    f"previos, semilla del sampler {semillas_sampler[-1]}."
)
recuperados = estudio.recuperar_huerfanos(sesion_1)
print(f"Huérfanos recuperados al abrir la sesión 1: {len(recuperados)}.")
sembrada = estudio.sembrar_linea_base(sesion_1, LINEA_BASE, DISTRIBUCIONES)
print(f"Línea base sembrada en esta corrida: {sembrada} (False = ya estaba de una corrida previa)")
estudio.optimizar(sesion_1, objetivo, n_total=mitad, atrapar=(juguete.Divergencia, MetricaNoFinita))
n_tras_1 = len(sesion_1.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS))
print(f"Sesión 1 completa: {n_tras_1} trials terminados (COMPLETE + PRUNED).")

# "Otro día": el paso abandona `sesion_1` y reabre el estudio desde cero, con la fábrica
# calculando la semilla del sampler a partir de los trials que YA existen.
sesion_2 = estudio.crear_estudio(
    NOMBRE_ESTUDIO,
    destino=DESTINO,
    direcciones=["maximize"],
    sampler=fabrica_sampler,
    pruner=optuna.pruners.NopPruner(),
    protocolo=PROTOCOLO,
)
# `crear_estudio` llama a la fábrica una vez por sesión: si las dos sesiones se fundieran
# en una, la cuenta daría 1 en vez de 2.
llamadas_sampler_en_esta_corrida = len(semillas_sampler) - llamadas_sampler_previas
if llamadas_sampler_en_esta_corrida != 2:
    raise RuntimeError(
        "fabrica_sampler debía recibir 2 llamadas en esta ejecución (una por sesión) y recibió "
        f"{llamadas_sampler_en_esta_corrida}: revisa que la sesión 2 reabra con su propia llamada "
        "a crear_estudio en vez de reusar el objeto Study de la sesión 1"
    )
n_previos_2 = len(sesion_2.get_trials(deepcopy=False))
print(
    f"Sesión 2: reabierta con {n_previos_2} trials previos, semilla del sampler "
    f"{semillas_sampler[-1]}."
)
estudio.optimizar(
    sesion_2, objetivo, n_total=N_TOTAL, atrapar=(juguete.Divergencia, MetricaNoFinita)
)
n_tras_2 = len(sesion_2.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS))
print(f"Sesión 2 completa: {n_tras_2} trials terminados en total.\n")

# %% [markdown]
# ## Las cinco respuestas

# %%
trials = sorted(
    sesion_2.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS),
    key=lambda t: t.number,
)

# Guarda de consistencia: `espacio()` y DISTRIBUCIONES deben tener los mismos rangos.
if not all(
    t.distributions == DISTRIBUCIONES for t in trials if t.user_attrs.get("rol") != "linea_base"
):
    raise RuntimeError(
        "DISTRIBUCIONES quedó desincronizado con espacio(): revisa que ambos definan los mismos "
        "rangos"
    )

candidatos_base = [t for t in trials if t.user_attrs.get("rol") == "linea_base"]
if not candidatos_base:
    raise RuntimeError(
        "no hay ningún trial COMPLETE/PRUNED con user_attrs['rol'] == 'linea_base': "
        "revisa que sembrar_linea_base se haya llamado y que recuperar_huerfanos haya corrido "
        "antes de optimizar (si no, una línea base cortada a la mitad se queda RUNNING para "
        "siempre y nunca llega a completarse)"
    )
linea_base_trial = candidatos_base[0]
mejor = sesion_2.best_trial

base_verdad = linea_base_trial.user_attrs["exactitud_verdadera"]
mejor_verdad = mejor.user_attrs["exactitud_verdadera"]

# Varios trials pueden empatar en validación: el paso reporta el empate en vez de
# esconderlo detrás del primero.
empatados = sorted(t.number for t in trials if t.number != mejor.number and t.value == mejor.value)
sufijo_empate = f" (empatado con los trials {empatados})" if empatados else ""

print(
    f"1. Línea base: validación {linea_base_trial.value:.3f} (verdadera {base_verdad:.3f}). "
    f"Mejor por validación (trial {mejor.number}){sufijo_empate}: validación {mejor.value:.3f} "
    f"(verdadera {mejor_verdad:.3f}). Diferencia de validación: "
    f"{100 * (mejor.value - linea_base_trial.value):+.1f} puntos. Mejora verdadera: "
    f"{100 * (mejor_verdad - base_verdad):+.1f} puntos."
)

# %% [markdown]
# ## Respuesta 2: config, semilla y lo que ya ves sin oráculo
#
# El valor que Optuna reportó para el ganador mezcla tres cosas: cuánto vale su config en
# promedio, la suerte de la semilla que le tocó y haber sido elegida como el máximo.
# Re-entrenar la misma config con semillas nuevas aísla la primera. El residuo que queda
# no es, por sí solo, evidencia de selección: el paso lo juzga contra un intervalo de
# predicción de una observación nueva, no contra el error estándar de la media, y lo
# compara con el mismo residuo de la línea base, el control que nadie eligió.

# %%
N_REEVAL_POR_PRESUPUESTO = {"humo": 2, "taller": 8, "completo": 16}
N_REEVAL = N_REEVAL_POR_PRESUPUESTO[presupuesto]
# Semillas de orden e init nuevas y disjuntas de las de la búsqueda, sobre los mismos
# datos.
SEMILLA_BASE_REEVAL = 271_828
semillas_reeval = semillas_de(SEMILLA_BASE_REEVAL, N_REEVAL, datos=SEMILLA_UNICA.datos)

# La línea base se reevalúa con las mismas semillas que el ganador: nadie la eligió, así
# que separa "esta config varía mucho" de "esta config fue seleccionada".
reevaluaciones = honestidad.reevaluar(
    {"ganador": dict(mejor.params), "linea_base": dict(linea_base_trial.params)},
    juguete.entrenar,
    semillas_reeval,
)


def _gaps(metricas: list[dict[str, float]]) -> list[float]:
    return [m["exactitud_val"] - m["exactitud_verdadera"] for m in metricas]


resumen_ganador = estadistica.resumen(_gaps(reevaluaciones["ganador"]))
resumen_base = estadistica.resumen(_gaps(reevaluaciones["linea_base"]))

inflado = mejor.value - mejor_verdad
inflado_base = linea_base_trial.value - base_verdad
suerte_ganador = resumen_ganador.media

# El residuo: cuánto se aleja el gap que vio la búsqueda del promedio de la misma config
# con semillas nuevas. `z` lo mide en desviaciones de una semilla, no en errores
# estándar de la media.
e_ganador = inflado - suerte_ganador
e_base = inflado_base - resumen_base.media


def _z_semilla(residuo: float, de: float) -> float:
    if de > 0.0:
        return residuo / de
    return 0.0 if residuo == 0.0 else math.copysign(math.inf, residuo)


def _umbral_z(n_reeval: float) -> float:
    """Umbral de un intervalo de predicción de una observación nueva,
    `t.ppf(0.95, n-1) * sqrt(1 + 1/n)`. Es de una cola porque la maldición del ganador
    predice el signo, y sale grande con pocas reevaluaciones.
    """
    return float(scipy_stats.t.ppf(0.95, n_reeval - 1) * math.sqrt(1 + 1 / n_reeval))


UMBRAL_Z = _umbral_z(N_REEVAL)
z_ganador = _z_semilla(e_ganador, resumen_ganador.de)
z_base = _z_semilla(e_base, resumen_base.de)

# Lo que un tesista sin oráculo sí puede calcular: la validación del ganador contra el
# promedio de validación de sus re-evaluaciones.
visible_sin_oraculo = mejor.value - st.mean(m["exactitud_val"] for m in reevaluaciones["ganador"])

# Lo que sigue invisible sin oráculo: si este conjunto de validación es fácil o difícil
# para cualquier modelo bueno. Lo revela el clasificador de Bayes.
_n_train = juguete.DEFAULTS["n_train"]
_x_split, _y_split = paridad.generar(
    _n_train + juguete.DEFAULTS["n_val"], semilla=SEMILLA_UNICA.datos, **PROBLEMA
)
_y_val = _y_split[_n_train:]
_p_bayes = paridad.posterior(
    _x_split[_n_train:], d=PROBLEMA["d"], a=PROBLEMA["a"], rotar=PROBLEMA["rotar"]
)
acc_bayes = float(np.mean((_p_bayes >= 0.5).astype(int) == _y_val))
# Cuánto gana el clasificador de Bayes sobre el techo en este conjunto, como fracción.
ganancia_bayes = acc_bayes - TECHO

# Revelación del oráculo: la brecha val-verdad de un trial cualquiera y cuánto depende de
# la calidad del modelo.
otros = [t for t in trials if t.number != mejor.number]
brecha_tipica = st.median(t.value - t.user_attrs["exactitud_verdadera"] for t in otros)
verdad_otros = [t.user_attrs["exactitud_verdadera"] for t in otros]
brechas_otros = [t.value - t.user_attrs["exactitud_verdadera"] for t in otros]
rho_spearman, p_spearman = scipy_stats.spearmanr(verdad_otros, brechas_otros)


def _frase_inflado(valor: float) -> str:
    if valor >= 0:
        return "la validación sobreestima la verdad"
    return "la validación subestima la verdad"


def _frase_sesgo_config(resumen: estadistica.Resumen) -> str:
    """Dice de qué lado se ve la brecha val-verdad de esta config si su intervalo de
    confianza al 95% no cruza cero; si lo cruza, no alcanza para decir la dirección."""
    valor = 100 * resumen.media
    ic_bajo = 100 * resumen.ic_bajo
    ic_alto = 100 * resumen.ic_alto
    if resumen.ic_bajo > 0.0 or resumen.ic_alto < 0.0:
        direccion = "mejor" if resumen.media >= 0 else "peor"
        return (
            f"con ESTOS datos de validación, esta config se ve en promedio {valor:+.1f} pts "
            f"{direccion} que su verdad (intervalo de confianza 95% "
            f"[{ic_bajo:+.1f}, {ic_alto:+.1f}]): esa parte no "
            "depende de la semilla de orden/init -- re-evaluar con los mismos datos no la "
            "quita -- es el conjunto de validación, no la config"
        )
    return (
        f"con {resumen.n} semillas, la brecha media de esta config ({valor:+.1f} pts, "
        f"intervalo de confianza 95% [{ic_bajo:+.1f}, {ic_alto:+.1f}]) no se distingue de "
        "cero: no alcanza para atribuirla al conjunto de validación"
    )


def _frase_bayes_gana(resumen: estadistica.Resumen, ganancia_bayes: float) -> str:
    """Atribuye parte de la brecha del ganador al conjunto solo si esa brecha ya se
    distinguió de cero y el clasificador de Bayes se mueve en el mismo sentido."""
    bayes = 100 * ganancia_bayes
    atribuible = resumen.ic_bajo > 0.0 or resumen.ic_alto < 0.0
    mismo_signo = (resumen.media >= 0) == (ganancia_bayes >= 0)
    if atribuible and mismo_signo:
        return (
            f"de esos {100 * resumen.media:+.1f} pts, el clasificador de Bayes también gana "
            f"{bayes:+.1f} en este conjunto"
        )
    return f"en este mismo conjunto el clasificador de Bayes gana {bayes:+.1f} pts"


def _frase_z(z_gan: float, z_b: float, e_b: float, umbral: float) -> str:
    if abs(z_gan) <= umbral:
        return (
            "no se distingue de la suerte de una semilla cualquiera (la línea base, que "
            f"nadie eligió, se desvía {100 * e_b:+.1f} pts de su propio promedio)"
        )
    if z_gan > umbral and abs(z_b) <= umbral:
        return (
            "el ganador tuvo una semilla de búsqueda más afortunada de lo que su config da "
            "en promedio y la línea base no: es lo esperable de la maldición del ganador, y "
            "con UN ganador es una anécdota (el paso 06 repite la búsqueda para medirla como "
            "distribución)"
        )
    if z_gan < -umbral:
        return (
            "el ganador tuvo una semilla de búsqueda MENOS afortunada de lo que su config da "
            "en promedio: la búsqueda lo escogió pese a esa mala suerte, no por ella"
        )
    return (
        "la línea base, que nadie eligió, también quedó fuera de su intervalo "
        f"(|z|={abs(z_b):.2f} > {umbral:.2f}, "
        f"{'en el mismo sentido' if z_b > 0 else 'en sentido contrario'}): si un control no "
        "elegido también se sale, el intervalo no basta para atribuir la del ganador a la "
        "selección"
    )


def _frase_dificultad_conjunto(acc: float, techo: float) -> str:
    if acc >= techo:
        return (
            f"este conjunto de validación es fácil para los modelos buenos (el clasificador "
            f"de Bayes acierta {acc:.3f} en él, por encima del techo de {techo:.3f} sobre "
            "toda la distribución)"
        )
    return (
        f"este conjunto de validación es más duro que el promedio (el clasificador de Bayes "
        f"acierta {acc:.3f} en él, por debajo del techo de {techo:.3f} sobre toda la "
        "distribución)"
    )


def _frase_spearman(rho: float, p: float, n: int) -> str:
    if p < 0.05 and rho > 0:
        return (
            f"la brecha crece con la verdad (Spearman(verdad, brecha)={rho:.3f}, p={p:.1e} "
            f"con {n} trials): esa mediana subestima la de un modelo tan bueno como el ganador"
        )
    return (
        f"con {n} trials no hay relación detectable (Spearman(verdad, brecha)={rho:.3f}, "
        f"p={p:.1e}): la mediana es solo una referencia gruesa"
    )


print(
    f"2. El ganador dice {mejor.value:.3f} y su verdad es {mejor_verdad:.3f} "
    f"({100 * inflado:+.1f} pts: {_frase_inflado(inflado)}). Re-evaluando la MISMA config con "
    f"{N_REEVAL} semillas nuevas de orden/init sobre los mismos datos (método del paso 06): "
    f"{_frase_sesgo_config(resumen_ganador)}; "
    f"{_frase_bayes_gana(resumen_ganador, ganancia_bayes)}. Frente a esa misma reevaluación, "
    "la semilla de la búsqueda quedó "
    f"{100 * e_ganador:+.1f} pts fuera de ese promedio (z={z_ganador:+.2f}, umbral "
    f"{UMBRAL_Z:.2f} con n={N_REEVAL}) y la línea base (control, nunca elegida) quedó "
    f"{100 * e_base:+.1f} pts fuera del suyo (z={z_base:+.2f}): "
    f"{_frase_z(z_ganador, z_base, e_base, UMBRAL_Z)}. Sin oráculo, lo único visible es que la "
    f"validación del ganador difiere {100 * visible_sin_oraculo:+.1f} pts del promedio de sus "
    f"propias re-evaluaciones; {_frase_dificultad_conjunto(acc_bayes, TECHO)}. "
    f"(Revelación del oráculo: la mediana de los demás {len(otros)} trials es "
    f"{100 * brecha_tipica:+.1f} pts -- {_frase_spearman(rho_spearman, p_spearman, len(otros))}.)"
)


def _contar_repetidas(trials) -> tuple[int, int]:
    configuraciones_unicas = {tuple(sorted(t.params.items())) for t in trials}
    return len(configuraciones_unicas), len(trials) - len(configuraciones_unicas)


n_unicas, repetidas = _contar_repetidas(trials)
if repetidas == 0:
    print(
        f"3. Los {len(trials)} trials completados usan {n_unicas} "
        "configuraciones distintas: reanudar en la segunda sesión no repitió ninguna."
    )
else:
    print(
        f"3. De {len(trials)} trials, {repetidas} repiten una configuración ya vista "
        f"({n_unicas} únicas): reanudar SÍ repitió trabajo."
    )

# PED-ANOVA compara el mejor `target_quantile` contra el resto, y con pocos trials sobre
# ese cuantil la estimación es ruido. Su docstring recomienda al menos N_TOP_MINIMO.
TARGET_QUANTILE = 0.1  # default de PedAnovaImportanceEvaluator, escrito explícito.
N_TOP_MINIMO = 5

evaluador = optuna.importance.PedAnovaImportanceEvaluator(target_quantile=TARGET_QUANTILE)
importancias = estudio.importancia_con_piso(sesion_2, evaluador=evaluador, placebo="placebo")
piso = importancias["placebo"]["importancia"]
n_top = math.ceil(TARGET_QUANTILE * len(trials))
confiable = n_top >= N_TOP_MINIMO
sobre_el_piso = [p for p, v in importancias.items() if p != "placebo" and v["sobre_el_piso"]]
nota_confiabilidad = (
    "estimación con UNA realización del placebo"
    if confiable
    else (
        f"orientativo, no una conclusión: PED-ANOVA recomienda al menos {N_TOP_MINIMO} trials "
        f"sobre el cuantil (hay {n_top})"
    )
)
print(
    f"4. {len(trials)} trials, {n_top} sobre el cuantil {TARGET_QUANTILE:.0%} "
    f"(placebo={piso:.3f}): {nota_confiabilidad}. Perillas sobre el piso del placebo: "
    f"{sobre_el_piso or 'ninguna'}."
)
for parametro in CLAVES_ESPACIO:
    marca = "sobre el piso" if importancias[parametro]["sobre_el_piso"] else "en el piso"
    print(f"   {parametro}: importancia {importancias[parametro]['importancia']:.3f} ({marca})")

# Un solo cuantil no basta para decir que una perilla importa, así que el paso mide la
# robustez con PED-ANOVA en el 10%, el 20% y el 30% mejor de los trials. "q" abrevia
# `target_quantile`.
EVALUADORES_ROBUSTEZ = {
    "PED-ANOVA q=0.1": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.1),
    "PED-ANOVA q=0.2": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.2),
    "PED-ANOVA q=0.3": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.3),
}
sobre_el_piso_por_evaluador: dict[str, set[str]] = {}
for nombre_evaluador, evaluador_robustez in EVALUADORES_ROBUSTEZ.items():
    importancias_e = estudio.importancia_con_piso(
        sesion_2, evaluador=evaluador_robustez, placebo="placebo"
    )
    sobre_el_piso_por_evaluador[nombre_evaluador] = {
        p for p, v in importancias_e.items() if p != "placebo" and v["sobre_el_piso"]
    }
    print(
        f"   {nombre_evaluador}: sobre el piso "
        f"{sorted(sobre_el_piso_por_evaluador[nombre_evaluador]) or 'ninguna'}"
    )
robustas = set.intersection(*sobre_el_piso_por_evaluador.values())
# La misma reserva de la nota anterior (n_top < N_TOP_MINIMO): con pocos trials
# sobre el cuantil, esta intersección tampoco es una conclusión, sea vacía o no.
prefijo_robustez = "" if confiable else "orientativo: "
if robustas:
    print(
        f"   {prefijo_robustez}Solo {sorted(robustas)} supera al placebo con los tres cuantiles "
        "de PED-ANOVA."
    )
else:
    print(
        f"   {prefijo_robustez}Ninguna perilla supera al placebo con los tres cuantiles de "
        f"PED-ANOVA: en esta corrida ({len(trials)} trials), la importancia no es una conclusión."
    )

print(f"5. Para explorar el estudio a mano: uv run optuna-dashboard {DESTINO}")

# %%
fig = Figure(figsize=(8, 4))
ax = fig.subplots()
numeros = [t.number for t in trials]
validacion = [t.value for t in trials]
verdadera = [t.user_attrs["exactitud_verdadera"] for t in trials]
ax.plot(numeros, validacion, marker="o", label="validación", linewidth=1)
ax.plot(numeros, verdadera, marker="x", label="verdadera (oráculo)", linewidth=1)
ax.axhline(
    linea_base_trial.value,
    linestyle="--",
    color="C0",
    label=f"línea base validación {linea_base_trial.value:.3f}",
)
ax.axhline(
    base_verdad,
    linestyle=":",
    color="C1",
    label=f"línea base verdadera {base_verdad:.3f}",
)
ax.set_xlabel("trial")
ax.set_ylabel("exactitud")
ax.set_title(
    f"Paso 02: primer estudio con método ({presupuesto}, {N_TOTAL} trials, {juguete.MOTOR})"
)
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout()
figura = RESULTADOS / "02_primer_estudio.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
