# %% [markdown]
# # Paso 04: Pareto -- mejor pero más caro
#
# Hasta ahora cada paso optimizó una sola métrica. Pero "mejor modelo" casi
# siempre compite con "modelo barato": una red más ancha suele validar mejor y
# también pesa más. Este paso busca las dos cosas a la vez, sin colapsarlas en
# un solo número.
#
# Preguntas:
# 1. ¿Cuántas configuraciones no dominadas hay y qué tan lejos están entre sí
#    en exactitud y en costo?
# 2. El extremo caro del frente contra el extremo barato, re-evaluados con las
#    MISMAS semillas nuevas: ¿compra algo distinguible del ruido?
# 3. Una búsqueda estocástica eligió el frente: ¿cuánto estaba el valor que vio la
#    búsqueda de cada punto por encima de su media al re-entrenarlo con semillas nuevas,
#    frente a un control que nadie seleccionó, y cuánto de eso no se ve con los mismos
#    datos?
# 4. Un presupuesto de parámetros que de verdad restringe el espacio, ¿a cuántos trials
#    deja fuera y cuánto cuesta frente al mejor punto del frente sin filtrar?
# 5. Optimizar una sola métrica sujeta a ese presupuesto, ¿da algo distinto a quedarse
#    con el mejor punto del frente dentro del mismo presupuesto?
#
# Correr:  uv run pasos/04_pareto.py  (necesita una tarjeta de video NVIDIA con
#          CUDA y el controlador instalado; sin ella, el paso se detiene con
#          un mensaje claro)
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import os
from pathlib import Path

import numpy as np
import optuna
from matplotlib.figure import Figure
from optuna.trial import TrialState
from scipy import stats as scipy_stats

from taller_hpo import estudio
from taller_hpo.estadistica import diferencia_pareada, resumen, semillas_necesarias
from taller_hpo.honestidad import inflacion_ganador
from taller_hpo.objetivo import en_proceso
from taller_hpo.problemas import juguete
from taller_hpo.problemas.paridad import error_bayes
from taller_hpo.semillas import Semillas, semillas_de

# El juguete se entrena siempre a través de `juguete`, y un solo `catch=(Divergencia,)`
# cubre todo el módulo.
DEFAULTS = juguete.DEFAULTS
Divergencia = juguete.Divergencia
entrenar_juguete = juguete.entrenar

# El INFO de Optuna imprime una línea por trial y taparía las respuestas numeradas.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# n: trials de cada una de las dos búsquedas (Pareto y la variante con restricción).
# k_reevaluar: semillas nuevas para re-evaluar; más semillas angostan el intervalo de
# cada punto a cambio de más entrenamientos. max_frente_reevaluar: cuántos puntos
# representativos del frente se re-entrenan, salvo que el frente tenga a lo sumo uno
# más, y entonces se re-entrena completo. Después de cada comparación pareada, el paso
# imprime cuántas semillas hubieran hecho falta para detectar ±2 puntos: un "no
# concluyente" nunca quiere decir "no hay diferencia". En humo, `epocas=3` recorta el
# tiempo pero cambia qué arquitectura gana el frente, así que humo solo comprueba que el
# paso corre.
PRESUPUESTOS = {
    "humo": {"n": 12, "k_reevaluar": 3, "max_frente_reevaluar": 4, "epocas": 3},
    "taller": {"n": 60, "k_reevaluar": 9, "max_frente_reevaluar": 6, "epocas": 30},
    "completo": {"n": 150, "k_reevaluar": 9, "max_frente_reevaluar": 10, "epocas": 30},
}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
CFG = PRESUPUESTOS[presupuesto]
N_TOTAL = CFG["n"]
K_REEVALUAR = CFG["k_reevaluar"]
MAX_FRENTE_REEVALUAR = CFG["max_frente_reevaluar"]
FIJOS = {"epocas": CFG["epocas"]}

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Primero, en qué tarjeta de video entrena esta corrida.
print(f"Motor: {juguete.DESCRIPCION}")

# Este paso no reanuda entre corridas: corre sus dos búsquedas en memoria y compara
# siempre desde cero, así que no deja un journal en TALLER_ESTUDIOS. `sembrar_linea_base`
# sí sirve con un estudio en memoria y valida la línea base contra el espacio.


def _indices_representativos(n: int, m: int) -> list[int]:
    """m índices en [0, n) lo más separados posible, incluidos los extremos."""
    if m >= n:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, m)})


def _n_repr_frente(n_frente: int, maximo: int) -> int:
    """Cuántos puntos del frente re-evaluar: todos si el frente tiene a lo sumo
    `maximo + 1`, o los `maximo` representativos (con los dos extremos) si es más grande."""
    return n_frente if n_frente <= maximo + 1 else maximo


def _n_parametros_mlp(n_capas: int, ancho: int, *, entradas: int) -> int:
    """Cuenta analítica de pesos y sesgos de la MLP del juguete (salida de 1 logit), la
    misma que reporta `n_parametros`, para justificar `PRESUPUESTO_PARAMETROS` antes de
    buscar."""
    capas = [entradas, *([ancho] * n_capas), 1]
    # strict=False a propósito: capas[1:] tiene un elemento menos que capas (es la
    # ventana deslizante "capa de entrada -> siguiente capa"), no un error de longitud.
    return sum(a * b + b for a, b in zip(capas, capas[1:], strict=False))


def _reevaluar_o_divergente(
    configs: dict[str, dict],
    entrenar,
    semillas: list[Semillas],
    *,
    fijos: dict | None = None,
) -> tuple[dict[str, list[dict]], int]:
    """Como `honestidad.reevaluar`, con las mismas `semillas` para todas las `configs`,
    pero si una config diverge con una semilla, esa semilla se descarta para todas, así
    las listas siguen pareadas. Devuelve `(resultado, semillas_descartadas)`."""
    resultado: dict[str, list[dict]] = {etiqueta: [] for etiqueta in configs}
    descartadas = 0
    for semilla in semillas:
        intentos: dict[str, dict] = {}
        try:
            for etiqueta, config in configs.items():
                config_final = {**(fijos or {}), **config}
                intentos[etiqueta] = entrenar(config_final, semilla, None)
        except Divergencia:
            descartadas += 1
            continue
        for etiqueta, valor in intentos.items():
            resultado[etiqueta].append(valor)
    return resultado, descartadas


# Funciones puras de clasificación, sin Optuna ni torch, para probar sus ramas con
# valores sintéticos.
def _veredicto_relativo(ic_bajo: float, ic_alto: float) -> str:
    """Clasifica el intervalo de una cantidad relativa en 'arriba' (todo sobre 0),
    'abajo' (todo bajo 0) o 'no_distingue' (cruza el 0)."""
    if ic_bajo > 0.0:
        return "arriba"
    if ic_alto < 0.0:
        return "abajo"
    return "no_distingue"


def _frase_control_punto(veredicto: str) -> str:
    """Prosa del veredicto de `_veredicto_relativo` para un punto del frente contra el
    control, la vara correcta para decir si hubo selección."""
    return {
        "arriba": "infla más que el control",
        "abajo": "infla menos que el control",
        "no_distingue": "no se distingue del control",
    }[veredicto]


def _frase_ic_cero(veredicto: str) -> str:
    """Prosa del veredicto de `_veredicto_relativo` aplicado contra 0 (la
    inflación del propio control, o cualquier intervalo de confianza que se
    juzgue contra 0)."""
    return {
        "arriba": "se distingue de 0 por encima",
        "abajo": "se distingue de 0 por debajo",
        "no_distingue": "no se distingue de 0",
    }[veredicto]


def _frase_extremos(veredicto: str, diferencia_pct: float, delta_pct: float) -> str:
    """Prosa de `taller_hpo.estadistica.diferencia_pareada.veredicto` para la
    comparación pareada de los dos extremos del frente (respuesta 2)."""
    if veredicto == "mejor":
        return f"SÍ compra {diferencia_pct:.2f} puntos de exactitud_val"
    if veredicto == "peor":
        return (
            f"resulta PEOR tras re-evaluar: pierde {abs(diferencia_pct):.2f} puntos frente al "
            "barato"
        )
    if veredicto == "equivalente":
        return (
            f"no compra nada distinguible (diferencia {diferencia_pct:+.2f} puntos, dentro de "
            f"±{delta_pct:.0f})"
        )
    return (
        f"da un resultado no concluyente (diferencia {diferencia_pct:+.2f} puntos, no se "
        "distingue del ruido con las semillas re-evaluadas)"
    )


def _frase_delta_busqueda(
    delta_busqueda_pct: float,
    ic_bajo_pred_pct: float,
    ic_alto_pred_pct: float,
    *,
    delta_pct: float,
    k: int,
    margen_asintotico_pct: float,
) -> str:
    """Clasifica la diferencia cruda de la búsqueda entre los dos extremos contra un
    intervalo de predicción bilateral de una diferencia nueva, construido con las `k`
    re-evaluaciones pareadas. Si el piso asintótico del margen ya supera ±delta_pct, la
    rama 'no concluyente' dice que más semillas no lo resolverían."""
    if delta_busqueda_pct > ic_alto_pred_pct:
        return "la búsqueda la sobreestimaba más de lo que explica una sola corrida"
    if delta_busqueda_pct < ic_bajo_pred_pct:
        return "la búsqueda la subestimaba más de lo que explica una sola corrida"
    # 'coincide' exige que la desviación del intervalo de predicción respecto a
    # delta_busqueda_pct quede dentro de ±delta_pct.
    if (
        delta_busqueda_pct - delta_pct < ic_bajo_pred_pct
        and ic_alto_pred_pct < delta_busqueda_pct + delta_pct
    ):
        return f"coincide dentro de ±{delta_pct:.0f} con lo que explica una sola corrida"
    if margen_asintotico_pct >= delta_pct:
        return (
            f"no concluyente: una sola corrida de la búsqueda tiene su propia dispersión "
            f"(estimada en esta corrida en ±{margen_asintotico_pct:.1f} puntos de predicción, "
            "incluso con infinitas semillas de re-evaluación), mayor que "
            f"±{delta_pct:.0f} -- ninguna cantidad de semillas de re-evaluación permite decir "
            "si estaba inflada, salvo que se desviara más que eso"
        )
    return (
        f"no concluyente: con {k} semillas no puedes decir si la diferencia de la búsqueda "
        "estaba inflada, aunque más semillas de re-evaluación sí podrían resolverlo"
    )


def _frase_costo_restriccion(veredicto: str, valor_abs_pct: float, delta_pct: float, k: int) -> str:
    """Prosa del costo real de la restricción (respuesta 4), una frase por veredicto."""
    if veredicto == "mejor":
        return f"la restricción SÍ cuesta {valor_abs_pct:.2f} puntos"
    if veredicto == "peor":
        return f"la restricción GANA {valor_abs_pct:.2f} puntos"
    if veredicto == "equivalente":
        return f"la restricción no cuesta nada distinguible (±{delta_pct:.0f})"
    return f"no concluyente con {k} semillas"


def _frase_no_vista(veredicto: str) -> str:
    """Prosa del veredicto de `_veredicto_relativo` aplicado a (reevaluada - verdadera):
    la parte que ninguna re-evaluación con los mismos datos puede ver."""
    if veredicto == "arriba":
        return "la validación de este dataset SOBREESTIMA la verdad"
    if veredicto == "abajo":
        return "la validación de este dataset SUBESTIMA la verdad"
    return "no se distingue de 0: esta re-evaluación no detecta un sesgo propio de este dataset"


def _margen_prediccion(de_dif: float, n: int, *, confianza: float = 0.95) -> float:
    """Semiancho de un intervalo de predicción bilateral para una diferencia pareada
    nueva: `t.ppf(0.5 + confianza/2, n - 1) * de_dif * sqrt(1 + 1/n)`. Con `de_dif == 0`
    el margen es 0."""
    if de_dif <= 0.0:
        return 0.0
    t_critico = float(scipy_stats.t.ppf(0.5 + confianza / 2.0, df=n - 1))
    return t_critico * de_dif * float(np.sqrt(1.0 + 1.0 / n))


def _margen_prediccion_no_pareada(
    de_dif: float, de_a: float, de_b: float, n: int, *, confianza: float = 0.95
) -> float:
    """Semiancho de un intervalo de predicción para una diferencia no pareada entre dos
    trials de la búsqueda: la incertidumbre de estimar el centro (`de_dif**2 / n`) más la
    dispersión de una diferencia nueva (`de_a**2 + de_b**2`)."""
    varianza = (de_dif**2 / n) + de_a**2 + de_b**2
    if varianza <= 0.0:
        return 0.0
    t_critico = float(scipy_stats.t.ppf(0.5 + confianza / 2.0, df=n - 1))
    return t_critico * float(np.sqrt(varianza))


def _margen_asintotico_no_pareada(de_a: float, de_b: float, *, confianza: float = 0.95) -> float:
    """Piso de `_margen_prediccion_no_pareada` cuando `n` tiende a infinito: ninguna
    cantidad de semillas angosta el margen más allá de este valor."""
    varianza = de_a**2 + de_b**2
    if varianza <= 0.0:
        return 0.0
    z_critico = float(scipy_stats.norm.ppf(0.5 + confianza / 2.0))
    return z_critico * float(np.sqrt(varianza))


def _ic_relativo_control(
    d_crudo: float,
    diferencia_media: float,
    de_dif: float,
    de_punto: float,
    de_control: float,
    n: int,
    *,
    confianza: float = 0.95,
) -> tuple[float, float, float]:
    """`(media, ic_bajo, ic_alto)` de una diferencia cruda de búsqueda (`d_crudo`, fija)
    menos la media de `n` diferencias pareadas re-evaluadas. Como `d_crudo` compara dos
    trials distintos, el margen sale de `_margen_prediccion_no_pareada`."""
    media = d_crudo - diferencia_media
    margen = _margen_prediccion_no_pareada(de_dif, de_punto, de_control, n, confianza=confianza)
    return media, media - margen, media + margen


def _ic_agregado_vs_control(
    media_agregado: float,
    ee_agregado: float,
    n_agregado: int,
    inflacion_control: float,
    de_control: float,
    n_control: int,
    *,
    confianza: float = 0.95,
) -> tuple[float, float, float]:
    """`(diferencia, ic_bajo, ic_alto)` del agregado de inflaciones del frente menos la
    inflación del control, sumando la varianza de predicción del control (es una sola
    extracción) a la dispersión entre puntos."""
    varianza_pred_control = de_control**2 * (1.0 + 1.0 / n_control)
    se_combinado = float(np.sqrt(ee_agregado**2 + varianza_pred_control))
    t_critico = float(scipy_stats.t.ppf(0.5 + confianza / 2.0, df=n_agregado - 1))
    diferencia = media_agregado - inflacion_control
    return diferencia, diferencia - t_critico * se_combinado, diferencia + t_critico * se_combinado


def _nota_semillas_necesarias(de_dif: float, k_usado: int, *, delta: float) -> str:
    """Nota descriptiva para acompañar un veredicto pareado: con el `de_dif` medido,
    cuántas semillas hubieran hecho falta para detectar `delta` con potencia de 80%,
    junto a las que usó la corrida. No sirve para decidir con las semillas ya usadas."""
    if de_dif <= 0.0:
        return (
            "semillas_necesarias no aplica (la desviación estándar de las diferencias medida "
            f"es 0.0); esta corrida usó {k_usado}"
        )
    return (
        f"semillas para DETECTAR una diferencia de ±{100 * delta:.0f} con potencia 80%: "
        f"{semillas_necesarias(de_dif, delta)}; esta corrida usó {k_usado}"
    )


# %% [markdown]
# ## El espacio de búsqueda: arquitectura y optimización
#
# `n_capas` y `ancho` deciden el costo (`n_parametros`, determinista dado el
# tamaño de entrada). El resto son perillas de optimización que no afectan el
# costo pero sí la exactitud alcanzable.
#
# `ancho` va de 1 a 64 (log): con `n_capas=1` cubre de 16 parámetros (casi
# lineal) a 961 (DEFAULTS), y con `n_capas=2` llega a 5121 (verificado abajo
# con `_n_parametros_mlp`) -- así el espacio va de redes casi lineales hasta
# bastante más caras que DEFAULTS, y el frente puede mostrar el compromiso
# completo, no solo su punta barata.

# %%
ENTRADAS = DEFAULTS["d"] + DEFAULTS["k_ruido"]


def espacio(trial: optuna.Trial) -> dict:
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "ancho": trial.suggest_int("ancho", 1, 64, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "activacion": trial.suggest_categorical("activacion", ["relu", "tanh", "logistic"]),
        "batch_size": trial.suggest_int("batch_size", 16, 128, log=True),
    }


# %% [markdown]
# ## Un solo dataset, como lo tendría un tesista
#
# Las dos búsquedas fijan `datos`, porque un tesista con un dataset ya recolectado no
# puede resortear filas entre trials, y solo varían `orden` e `init` por trial. Eso se
# aparta a propósito de los números aleatorios comunes de los pasos 02 y 03: la
# respuesta 3 agrega la inflación de varios puntos del frente, y para que la dispersión
# entre puntos mida esa incertidumbre, la suerte de cada punto en la búsqueda tiene que
# ser independiente. Con datos fijos, la re-evaluación no puede medir la suerte de qué
# validación tocó.

# %%
BASE_DATOS = 20_260_920
DATOS_FIJOS = semillas_de(BASE_DATOS, 1)[0].datos

# %% [markdown]
# ## Búsqueda multiobjetivo
#
# El sampler multiobjetivo por defecto de Optuna 5 es `TPESampler`, y el paso lo escribe
# explícito, con todos sus argumentos. Es también el sampler de la variante con
# restricción, así esa comparación aísla la restricción del sampler. La búsqueda de
# Pareto siembra DEFAULTS como línea base; la de restricción no, porque DEFAULTS no cabe
# en `PRESUPUESTO_PARAMETROS`.
#
# Con dos métricas, `en_proceso` apaga la poda (Optuna no poda en multiobjetivo), y el
# paso escribe `NopPruner` explícito para no depender de un default.

# %%
BASE_BUSQUEDA_PARETO = 20_260_925


def _tpe_explicito(seed: int) -> optuna.samplers.TPESampler:
    """`TPESampler` con todos sus argumentos explícitos, en sus defaults salvo `seed`.
    Las dos búsquedas usan esta misma función."""
    return optuna.samplers.TPESampler(
        consider_prior=None,
        prior_weight=None,
        consider_magic_clip=None,
        consider_endpoints=None,
        n_startup_trials=10,
        n_ei_candidates=24,
        gamma=None,
        weights=None,
        seed=seed,
        multivariate=None,
        group=False,
        warn_independent_sampling=None,
        constant_liar=True,
        constraints_func=None,  # deprecado: la variante con restricción usa set_constraint.
    )


semillas_busqueda_pareto = semillas_de(BASE_BUSQUEDA_PARETO, N_TOTAL, datos=DATOS_FIJOS)


def semillas_pareto(trial: optuna.Trial):
    return semillas_busqueda_pareto[trial.number]


objetivo_pareto = en_proceso(
    entrenar_juguete,
    espacio,
    metricas=["exactitud_val", "n_parametros"],
    semillas=semillas_pareto,
    fijos=FIJOS,
    atributos=["exactitud_verdadera"],
)

sampler_pareto = _tpe_explicito(BASE_BUSQUEDA_PARETO)

study_pareto = optuna.create_study(
    directions=["maximize", "minimize"],
    sampler=sampler_pareto,
    pruner=optuna.pruners.NopPruner(),  # multiobjetivo no poda: en_proceso ya lo impone.
)
# Línea base sembrada con `sembrar_linea_base`, que valida `params` contra
# DISTRIBUCIONES_ESPACIO antes de encolar. DEFAULTS, tu configuración de siempre, entra a
# la misma búsqueda y queda marcada en la figura.
DISTRIBUCIONES_ESPACIO = {
    "n_capas": optuna.distributions.IntDistribution(1, 2),
    "ancho": optuna.distributions.IntDistribution(1, 64, log=True),
    "learning_rate_init": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": optuna.distributions.FloatDistribution(1e-6, 1e-1, log=True),
    "activacion": optuna.distributions.CategoricalDistribution(["relu", "tanh", "logistic"]),
    "batch_size": optuna.distributions.IntDistribution(16, 128, log=True),
}
estudio.sembrar_linea_base(
    study_pareto, {k: DEFAULTS[k] for k in juguete.HIPERPARAMETROS}, DISTRIBUCIONES_ESPACIO
)
# catch=(Divergencia,): un learning_rate_init grande puede volver los pesos no finitos,
# y ese trial queda FAIL sin cortar la búsqueda.
study_pareto.optimize(objetivo_pareto, n_trials=N_TOTAL, n_jobs=1, catch=(Divergencia,))

trials_pareto_completos = study_pareto.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
# El n_parametros que reportó cada entrenamiento debe coincidir con la cuenta analítica.
for _t in trials_pareto_completos:
    _cfg = _t.user_attrs["config"]
    assert _n_parametros_mlp(_cfg["n_capas"], _cfg["ancho"], entradas=ENTRADAS) == _t.values[1], (
        _t.number,
        _cfg,
        _t.values[1],
    )
linea_base_trial = next(
    (t for t in trials_pareto_completos if t.user_attrs.get("rol") == "linea_base"), None
)
frente = study_pareto.best_trials
frente_ordenado = sorted(frente, key=lambda t: t.values[1])  # por n_parametros, barato -> caro

# %% [markdown]
# ## ¿Por qué no una suma ponderada?
#
# `w1 * exactitud_val - w2 * n_parametros` reduciría el problema a una sola
# métrica que podrías optimizar con cualquier sampler mono-objetivo. Dos
# razones para no hacerlo:
#
# - **Depende de la escala.** `exactitud_val` vive en [0, 1] y `n_parametros`
#   en el orden de los miles: sin normalizar, `w2` tiene que absorber ese
#   factor de ~1000, y cualquier elección de `w1`/`w2` es arbitraria -- mover
#   los pesos un poco cambia qué punto del frente "gana" sin que el problema
#   real haya cambiado en nada.
# - **Esconde el compromiso.** Un solo número ("el score fue 0.71") no dice
#   si esa configuración es barata-y-mediocre o cara-y-buena, ni cuánta
#   exactitud dejaste sobre la mesa por elegir esos pesos. El frente completo
#   sí lo dice: ves todo el menú de compromisos disponibles, no solo el que
#   un par de pesos fijos hubiera elegido por adelantado.
#
# (Una tercera razón, más técnica: una suma ponderada con pesos no negativos
# solo puede alcanzar puntos de la envolvente convexa del frente. Si el frente
# tiene una parte cóncava, ningún par de pesos la encuentra.)

# %% [markdown]
# ## El presupuesto de parámetros: 100
#
# El paso elige 100 entre `ancho=6` (91 parámetros) y `ancho=7` (106) con `n_capas=1`, y
# entre `ancho=4` (81) y `ancho=5` (106) con `n_capas=2`: deja pasar redes chicas y corta
# justo antes de las medianas. Cuánto cuesta de verdad ese presupuesto lo mide la
# respuesta 4.
#
# ## El punto del frente al mismo presupuesto
#
# El paso filtra el frente por `PRESUPUESTO_PARAMETROS` y toma el de mejor
# `exactitud_val`. Ese punto y el mejor punto del frente sin filtrar entran a la misma
# re-evaluación pareada que el resto del frente, elegidos los dos por su valor de
# búsqueda.

# %%
assert _n_parametros_mlp(1, 6, entradas=ENTRADAS) == 91
assert _n_parametros_mlp(1, 7, entradas=ENTRADAS) == 106
assert _n_parametros_mlp(2, 4, entradas=ENTRADAS) == 81
assert _n_parametros_mlp(2, 5, entradas=ENTRADAS) == 106
# Tope del espacio con n_capas=2 (ver "El espacio de búsqueda" arriba): 13*64 +
# 64 (entrada -> primera capa) + 64*64 + 64 (primera -> segunda capa) + 64 + 1
# (segunda capa -> salida) = 5121.
assert _n_parametros_mlp(2, 64, entradas=ENTRADAS) == 5121

PRESUPUESTO_PARAMETROS = 100.0

candidatos_frente_presupuesto = [
    t for t in frente_ordenado if t.values[1] <= PRESUPUESTO_PARAMETROS
]
if candidatos_frente_presupuesto:
    punto_comparable = max(candidatos_frente_presupuesto, key=lambda t: t.values[0])
    frente_sin_candidato_en_presupuesto = False
else:
    # Si ningún punto del frente cabe en el presupuesto, el paso usa el más barato y
    # avisa que lo excede.
    punto_comparable = frente_ordenado[0]
    frente_sin_candidato_en_presupuesto = True
# El mejor punto del frente sin filtrar, elegido con el mismo criterio que
# punto_comparable, para aislar el costo de la restricción en la respuesta 4.
mejor_frente_total = max(frente_ordenado, key=lambda t: t.values[0])

# %% [markdown]
# ## La variante con restricción
#
# Un solo estudio que maximiza `exactitud_val` sujeto a `n_parametros <=
# PRESUPUESTO_PARAMETROS`. `constraints_func` está deprecado en Optuna 5, así que cada
# trial declara su restricción con `Trial.set_constraint`, factible cuando el valor es
# <= 0. TPESampler lee esas restricciones y manda los trials infactibles al grupo "malo"
# al modelar. El paso corre esta búsqueda antes de re-evaluar el frente, para que su
# ganador factible entre a la misma re-evaluación.

# %%
BASE_BUSQUEDA_RESTRICCION = 20_260_930
semillas_busqueda_restriccion = semillas_de(BASE_BUSQUEDA_RESTRICCION, N_TOTAL, datos=DATOS_FIJOS)


def semillas_restriccion(trial: optuna.Trial):
    return semillas_busqueda_restriccion[trial.number]


objetivo_restriccion_base = en_proceso(
    entrenar_juguete,
    espacio,
    metricas=["exactitud_val"],
    semillas=semillas_restriccion,
    fijos=FIJOS,
    atributos=["exactitud_verdadera", "n_parametros"],
)


def objetivo_restriccion(trial: optuna.Trial) -> float:
    valor = objetivo_restriccion_base(trial)
    n_parametros = trial.user_attrs["n_parametros"]
    trial.set_constraint("presupuesto_parametros", n_parametros - PRESUPUESTO_PARAMETROS)
    return valor


sampler_restriccion = _tpe_explicito(BASE_BUSQUEDA_RESTRICCION)

study_restriccion = optuna.create_study(
    direction="maximize",
    sampler=sampler_restriccion,
    pruner=optuna.pruners.NopPruner(),  # la restricción no reemplaza a la poda; van por separado.
)
study_restriccion.optimize(objetivo_restriccion, n_trials=N_TOTAL, n_jobs=1, catch=(Divergencia,))

trials_restriccion = study_restriccion.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
factibles = [t for t in trials_restriccion if all(v <= 0.0 for v in t.constraints.values())]
# study.best_trial falla si ningún trial es factible, así que el paso lo revisa antes,
# para responder en vez de lanzar una excepción.
mejor_restriccion = study_restriccion.best_trial if factibles else None
mejor_sin_restriccion = (
    max(trials_restriccion, key=lambda t: t.value) if trials_restriccion else None
)

# %% [markdown]
# ## ¿El frente también se infla?
#
# Si la selección de Pareto favorece configs con suerte en `exactitud_val`, el frente se
# inflaría: la maldición del ganador, un punto a la vez. La respuesta 3 lo mide
# re-entrenando cada punto con semillas nuevas (mismos `datos`, `orden` e `init`
# nuevos). Lo que no mide es la suerte de qué validación tocó, que con datos fijos ya no
# varía.
#
# Un intervalo de inflación contra 0 no es evidencia de selección. Por eso la línea base,
# la única config que nadie eligió, entra a la misma re-evaluación como control, y cada
# punto se juzga contra la inflación del control con un intervalo de predicción no
# pareado, porque punto y control son trials distintos con su propia semilla.
#
# Los puntos representativos, el control y el ganador factible de la restricción se
# re-evalúan con las mismas `K_REEVALUAR` semillas en una sola llamada, así todas las
# comparaciones de las respuestas 2 a 5 son pareadas.

# %%
BASE_REEVALUACION = 20_260_935_000
# Base de semillas de re-evaluación, separada de las de búsqueda y de comparación.
TECHO_BAYES = 1.0 - error_bayes(DEFAULTS["d"], DEFAULTS["a"])

# delta = 0.02 (2 puntos de exactitud_val): dos configs que difieren por menos son
# intercambiables en este taller. Se usa en las tres comparaciones pareadas del paso.
DELTA_COMPARACION = 0.02

CONTROL_ETIQUETA = "control_linea_base"
RESTRICCION_ETIQUETA = "restriccion_ganador"
# punto_comparable, mejor_frente_total y el ganador factible de la restricción entran a
# esta misma re-evaluación, para que su comparación sea pareada.
PRESUPUESTO_ETIQUETA = "frente_presupuesto"
TOTAL_ETIQUETA = "frente_total"
# Re-evalúa el frente completo o sus representantes, según `_n_repr_frente`.
n_repr_frente = _n_repr_frente(len(frente_ordenado), MAX_FRENTE_REEVALUAR)
indices_reevaluar = _indices_representativos(len(frente_ordenado), n_repr_frente)
semillas_reeval_frente = semillas_de(BASE_REEVALUACION, K_REEVALUAR, datos=DATOS_FIJOS)
# El ganador de la restricción va primero, así en cada semilla se entrena antes que el
# frente.
configs_frente_repr: dict[str, dict] = {}
if mejor_restriccion is not None:
    configs_frente_repr[RESTRICCION_ETIQUETA] = mejor_restriccion.user_attrs["config"]
configs_frente_repr.update(
    {str(indice): frente_ordenado[indice].user_attrs["config"] for indice in indices_reevaluar}
)
if linea_base_trial is not None:
    configs_frente_repr[CONTROL_ETIQUETA] = linea_base_trial.user_attrs["config"]
# etiquetas_por_indice: qué etiqueta ya entrena cada índice de frente_ordenado, para no
# entrenar dos veces el mismo punto.
etiquetas_por_indice: dict[int, str] = {indice: str(indice) for indice in indices_reevaluar}


def _etiqueta_para(trial: optuna.trial.FrozenTrial, nombre_por_defecto: str) -> str:
    """Etiqueta para re-evaluar `trial`: reusa la de un índice representativo o la que ya
    tiene, en vez de duplicar la config. Encuentra el trial por igualdad de valor."""
    indice = frente_ordenado.index(trial)
    etiqueta_existente = etiquetas_por_indice.get(indice)
    if etiqueta_existente is not None:
        return etiqueta_existente
    etiquetas_por_indice[indice] = nombre_por_defecto
    configs_frente_repr[nombre_por_defecto] = trial.user_attrs["config"]
    return nombre_por_defecto


ETIQUETA_TOTAL = _etiqueta_para(mejor_frente_total, TOTAL_ETIQUETA)
ETIQUETA_PRESUPUESTO = _etiqueta_para(punto_comparable, PRESUPUESTO_ETIQUETA)
reevaluados_frente, descartadas_frente = _reevaluar_o_divergente(
    configs_frente_repr, entrenar_juguete, semillas_reeval_frente
)

control_reeval = reevaluados_frente.get(CONTROL_ETIQUETA, [])
# El valor de búsqueda de DEFAULTS, la línea base sembrada. La respuesta 4 lo reusa para
# juzgar al ganador de la restricción contra el mismo control.
control_busqueda = linea_base_trial.values[0] if linea_base_trial is not None else None
# >= 2: lo mismo que exige taller_hpo.estadistica.resumen para un intervalo -- con
# menos, no hay con qué comparar ningún punto contra el control.
control_disponible = linea_base_trial is not None and len(control_reeval) >= 2
if control_disponible:
    control_exactitudes = [r["exactitud_val"] for r in control_reeval]
    control_resumen_val = resumen(control_exactitudes)
    control_inflacion = inflacion_ganador(
        control_busqueda, control_exactitudes, direccion="maximizar"
    )

# El costo real de la restricción (respuesta 4): el mejor punto del frente dentro del
# presupuesto contra el mejor sin filtrar, los dos elegidos por su valor de búsqueda y
# re-evaluados con las mismas semillas.
costo_restriccion_real = None
resumen_frente_total_costo = None
resumen_frente_presupuesto_costo = None
val_frente_total_reeval = dict(
    enumerate(r["exactitud_val"] for r in reevaluados_frente[ETIQUETA_TOTAL])
)
val_frente_presupuesto_reeval = dict(
    enumerate(r["exactitud_val"] for r in reevaluados_frente[ETIQUETA_PRESUPUESTO])
)
if len(val_frente_total_reeval) >= 2 and len(val_frente_presupuesto_reeval) >= 2:
    costo_restriccion_real = diferencia_pareada(
        val_frente_total_reeval,
        val_frente_presupuesto_reeval,
        delta=DELTA_COMPARACION,
        direccion="maximizar",
    )
    resumen_frente_total_costo = resumen(list(val_frente_total_reeval.values()))
    resumen_frente_presupuesto_costo = resumen(list(val_frente_presupuesto_reeval.values()))

puntos_reevaluados = []
for indice in indices_reevaluar:
    trial = frente_ordenado[indice]
    resultados_reeval = reevaluados_frente[str(indice)]
    if len(resultados_reeval) < 2:
        # Menos de 2 semillas sobrevivieron para este punto: la nota de descartadas lo
        # documenta.
        continue
    exactitudes = [r["exactitud_val"] for r in resultados_reeval]
    verdaderas = [r["exactitud_verdadera"] for r in resultados_reeval]
    # inflacion_ganador, la misma medición del paso 06, aplicada a cada punto del frente.
    # Contra 0 no es evidencia de selección: ver la comparación contra el control.
    punto = {
        "n_parametros": trial.values[1],
        "exactitud_busqueda": trial.values[0],
        "exactitud_pareada": dict(enumerate(exactitudes)),
        "resumen_val": resumen(exactitudes),
        "resumen_verdad": resumen(verdaderas),
        "inflacion": inflacion_ganador(trial.values[0], exactitudes, direccion="maximizar"),
    }
    # Contra el control: inflación del punto menos inflación del control, con el
    # intervalo no pareado de `_ic_relativo_control`, porque punto y control son trials
    # distintos de la búsqueda.
    if control_disponible and len(control_exactitudes) == len(exactitudes):
        par_control = diferencia_pareada(
            dict(enumerate(exactitudes)),
            dict(enumerate(control_exactitudes)),
            delta=DELTA_COMPARACION,
            direccion="maximizar",
        )
        d_control = trial.values[0] - control_busqueda
        rel_media, rel_ic_bajo, rel_ic_alto = _ic_relativo_control(
            d_control,
            par_control.diferencia_media,
            par_control.de_dif,
            punto["resumen_val"].de,
            control_resumen_val.de,
            par_control.n,
        )
        punto["rel_control_media"] = rel_media
        punto["rel_control_ic_bajo"] = rel_ic_bajo
        punto["rel_control_ic_alto"] = rel_ic_alto
        punto["veredicto_control"] = _veredicto_relativo(rel_ic_bajo, rel_ic_alto)
    puntos_reevaluados.append(punto)

# %% [markdown]
# ## Cuánto cuesta de verdad la restricción, y el punto del frente al mismo
# ## presupuesto contra la variante con restricción (respuestas 4 y 5)
#
# El costo real de la respuesta 4 compara el mejor punto del frente dentro del
# presupuesto contra el mejor sin filtrar, los dos de la misma búsqueda de Pareto y
# re-evaluados con las mismas semillas. Comparar el ganador de la búsqueda con
# restricción contra el de la búsqueda sin ella mezclaría el costo con la maldición del
# ganador de cada una.
#
# Aparte, el paso mide cuánto se desvió la búsqueda con restricción de su propio
# ganador, siempre contra el control y nunca contra 0. La respuesta 5 compara ese
# ganador factible contra el mejor punto del frente dentro del presupuesto: ¿el
# presupuesto encontró algo mejor que filtrar el frente después? Todas estas
# comparaciones salen de la misma re-evaluación de arriba.

# %%
val_restriccion = dict(
    enumerate(r["exactitud_val"] for r in reevaluados_frente.get(RESTRICCION_ETIQUETA, []))
)
val_control_propio = [r["exactitud_val"] for r in control_reeval]
comparacion = None
resumen_restriccion = None
resumen_frente_cmp = None
resumen_factible_propio = None
if mejor_restriccion is not None and len(val_restriccion) >= 2:
    comparacion = diferencia_pareada(
        val_restriccion,
        val_frente_presupuesto_reeval,
        delta=DELTA_COMPARACION,
        direccion="maximizar",
    )
    resumen_restriccion = resumen(list(val_restriccion.values()))
    # `val_restriccion` y `val_frente_presupuesto_reeval` tienen el mismo largo, porque
    # salen de la misma llamada a `_reevaluar_o_divergente`.
    resumen_frente_cmp = resumen_frente_presupuesto_costo
    # El ganador factible se re-evaluó una sola vez: su resumen es el mismo en las
    # respuestas 4 y 5.
    resumen_factible_propio = resumen_restriccion

# %% [markdown]
# ## Las respuestas

# %%
exactitudes_frente = [t.values[0] for t in frente_ordenado]
parametros_frente = [t.values[1] for t in frente_ordenado]

print()
print(
    f"1. El frente tiene {len(frente_ordenado)} configuraciones no dominadas de las "
    f"{len(trials_pareto_completos)} evaluadas: exactitud_val entre {min(exactitudes_frente):.3f} "
    f"y {max(exactitudes_frente):.3f}, n_parametros entre {min(parametros_frente):.0f} y "
    f"{max(parametros_frente):.0f}."
)
print(
    "frente: "
    + str(
        [
            (int(round(p)), round(e, 4))
            for p, e in zip(parametros_frente, exactitudes_frente, strict=True)
        ]
    )
)
if linea_base_trial is not None:
    en_el_frente = linea_base_trial in frente_ordenado
    veredicto_base = (
        "SÍ quedó en el frente"
        if en_el_frente
        else (
            "quedó DOMINADA (otro trial la iguala o supera en las dos métricas y la supera en "
            "al menos una)"
        )
    )
    print(
        f"   línea base (DEFAULTS, n_capas={DEFAULTS['n_capas']}, ancho={DEFAULTS['ancho']}): "
        f"{linea_base_trial.values[1]:.0f} parámetros, exactitud_val "
        f"{linea_base_trial.values[0]:.3f} -- {veredicto_base}."
    )

if len(frente_ordenado) < 2:
    print("2. El frente tiene un solo punto: no hay extremo barato y extremo caro que comparar.")
elif len(puntos_reevaluados) < 2:
    print(
        "2. Menos de dos puntos representativos del frente sobrevivieron a la re-evaluación "
        f"(Divergencia descartó {descartadas_frente} semilla(s) de las {K_REEVALUAR} pedidas): "
        "no hay comparación pareada posible entre el extremo barato y el caro."
    )
else:
    barato, caro = puntos_reevaluados[0], puntos_reevaluados[-1]
    comparacion_extremos = diferencia_pareada(
        caro["exactitud_pareada"],
        barato["exactitud_pareada"],
        delta=DELTA_COMPARACION,
        direccion="maximizar",
    )
    diferencia_pct = 100 * comparacion_extremos.diferencia_media
    frase_extremos = _frase_extremos(
        comparacion_extremos.veredicto, diferencia_pct, 100 * DELTA_COMPARACION
    )
    delta_busqueda = 100 * (caro["exactitud_busqueda"] - barato["exactitud_busqueda"])
    delta_param_extremos = caro["n_parametros"] - barato["n_parametros"]
    # delta_busqueda es una sola observación no pareada de cada extremo, así que la vara
    # es un intervalo de predicción no pareado, no el intervalo de la media re-evaluada.
    k_ext = comparacion_extremos.n
    margen_pred = _margen_prediccion_no_pareada(
        comparacion_extremos.de_dif, caro["resumen_val"].de, barato["resumen_val"].de, k_ext
    )
    ic_pred_bajo = comparacion_extremos.diferencia_media - margen_pred
    ic_pred_alto = comparacion_extremos.diferencia_media + margen_pred
    margen_asintotico = _margen_asintotico_no_pareada(
        caro["resumen_val"].de, barato["resumen_val"].de
    )
    frase_busqueda = _frase_delta_busqueda(
        delta_busqueda,
        100 * ic_pred_bajo,
        100 * ic_pred_alto,
        delta_pct=100 * DELTA_COMPARACION,
        k=k_ext,
        margen_asintotico_pct=100 * margen_asintotico,
    )
    print(
        f"2. Re-evaluado con las mismas {k_ext} semillas nuevas para los dos: el extremo más "
        f"caro del frente ({caro['n_parametros']:.0f} parámetros, reevaluado "
        f"{caro['resumen_val'].media:.3f}) contra el más barato ({barato['n_parametros']:.0f} "
        f"parámetros, reevaluado {barato['resumen_val'].media:.3f}) {frase_extremos} "
        "(intervalo de confianza "
        f"{100 * comparacion_extremos.confianza:.0f}% [{100 * comparacion_extremos.ic_bajo:+.2f}, "
        f"{100 * comparacion_extremos.ic_alto:+.2f}]). El valor de la búsqueda por sí solo, sin "
        f"re-evaluar, sugería {delta_busqueda:+.1f} puntos por {delta_param_extremos:+.0f} "
        f"parámetros más -- contra el intervalo de PREDICCIÓN de una diferencia nueva con esas "
        f"mismas {k_ext} re-evaluaciones pareadas ([{100 * ic_pred_bajo:+.2f}, "
        f"{100 * ic_pred_alto:+.2f}]): {frase_busqueda}."
    )
    nota_k_extremos = _nota_semillas_necesarias(
        comparacion_extremos.de_dif, k_ext, delta=DELTA_COMPARACION
    )
    print(f"   ({nota_k_extremos})")

if puntos_reevaluados:
    if not control_disponible:
        motivo_sin_control = (
            "no hay ningún trial COMPLETE con user_attrs['rol'] == 'linea_base' en esta búsqueda"
            if linea_base_trial is None
            else (
                f"Divergencia dejó menos de 2 semillas para el control "
                f"({len(control_reeval)} de {K_REEVALUAR} sobrevivieron)"
            )
        )
        print(
            f"3. Sin control disponible ({motivo_sin_control}): un intervalo de confianza de "
            "inflación contra 0, por sí solo, NO es evidencia de selección (ver el docstring de "
            "taller_hpo.honestidad.inflacion_ganador) -- esta corrida no tiene con qué distinguir "
            "la selección de la dispersión propia de cada config."
        )
    else:
        n_arriba = sum(1 for p in puntos_reevaluados if p.get("veredicto_control") == "arriba")
        n_abajo = sum(1 for p in puntos_reevaluados if p.get("veredicto_control") == "abajo")
        n_no = sum(1 for p in puntos_reevaluados if p.get("veredicto_control") == "no_distingue")
        # El agregado aplica resumen() a las inflaciones de los M puntos contra 0: cada
        # punto tuvo su propia semilla de búsqueda, así que la dispersión entre puntos ya
        # captura esa suerte.
        inflaciones_puntos = [p["inflacion"].inflacion for p in puntos_reevaluados]
        if len(inflaciones_puntos) >= 2:
            resumen_agregado = resumen(inflaciones_puntos)
            veredicto_agregado_absoluto = _veredicto_relativo(
                resumen_agregado.ic_bajo, resumen_agregado.ic_alto
            )
            frase_ic_cero_agregado = _frase_ic_cero(veredicto_agregado_absoluto)
            frase_absoluto = (
                f"la inflación típica del frente contra su propia búsqueda (nulo exacto, sin "
                f"mirar el control) es {100 * resumen_agregado.media:+.2f} puntos "
                "(intervalo de confianza "
                f"{100 * resumen_agregado.confianza:.0f}% [{100 * resumen_agregado.ic_bajo:+.2f}, "
                f"{100 * resumen_agregado.ic_alto:+.2f}]) -- {frase_ic_cero_agregado}"
            )
            # Suma la varianza de predicción del control a la dispersión entre puntos.
            diff_vs_control, ic_bajo_control, ic_alto_control = _ic_agregado_vs_control(
                resumen_agregado.media,
                resumen_agregado.ee,
                resumen_agregado.n,
                control_inflacion.inflacion,
                control_resumen_val.de,
                control_resumen_val.n,
            )
            veredicto_agregado_control = _veredicto_relativo(ic_bajo_control, ic_alto_control)
            if veredicto_agregado_control == "arriba":
                frase_vs_control = (
                    "contra el control, sumando su propia varianza de predicción, el frente "
                    f"infla MÁS {100 * diff_vs_control:+.2f} puntos (intervalo "
                    f"{100 * resumen_agregado.confianza:.0f}% [{100 * ic_bajo_control:+.2f}, "
                    f"{100 * ic_alto_control:+.2f}])"
                )
            elif veredicto_agregado_control == "abajo":
                frase_vs_control = (
                    "contra el control, sumando su propia varianza de predicción, el frente "
                    f"infla MENOS {100 * diff_vs_control:+.2f} puntos (intervalo "
                    f"{100 * resumen_agregado.confianza:.0f}% [{100 * ic_bajo_control:+.2f}, "
                    f"{100 * ic_alto_control:+.2f}])"
                )
            else:
                frase_vs_control = (
                    "contra el control, sumando su propia varianza de predicción, no concluyente "
                    f"(diferencia {100 * diff_vs_control:+.2f} puntos, intervalo "
                    f"{100 * resumen_agregado.confianza:.0f}% [{100 * ic_bajo_control:+.2f}, "
                    f"{100 * ic_alto_control:+.2f}])"
                )
            frase_agregado = f"{frase_absoluto}; {frase_vs_control}"
        else:
            frase_agregado = (
                "un solo punto re-evaluado: no hay con qué calcular un intervalo de confianza "
                "entre puntos"
            )
        print(
            "3. Frente re-evaluado CONTRA EL CONTROL (línea base, que nadie seleccionó), con las "
            f"mismas {K_REEVALUAR} semillas nuevas para cada punto y el control: de los "
            f"{len(puntos_reevaluados)} puntos, {n_arriba} infla(n) más que el control, {n_abajo} "
            f"menos, {n_no} no se distinguen de él; {frase_agregado}."
        )
        print(
            f"   (nota: con {len(puntos_reevaluados)} comparaciones contra el control al 95% "
            "-- la confianza que usa `resumen()` en todo este paso --, esperas que alrededor de "
            "1 de cada 20 caiga 'arriba' o 'abajo' solo por azar, incluso si ninguna config "
            "estuviera realmente sesgada -- sin corrección por comparaciones múltiples.)"
        )
        if descartadas_frente:
            print(
                f"   ({descartadas_frente} semilla(s) descartada(s) en esta re-evaluación por "
                "Divergencia: el paso las tiró para TODOS los puntos y el control, no solo el que "
                "divergió, para no romper el emparejamiento.)"
            )
        for punto in puntos_reevaluados:
            infl = punto["inflacion"]
            rv = punto["resumen_val"]
            n_params_txt = f"{punto['n_parametros']:.0f}"
            if "veredicto_control" in punto:
                detalle_control = (
                    f", vs control {_frase_control_punto(punto['veredicto_control'])} (diferencia "
                    f"de inflaciones {100 * punto['rel_control_media']:+.2f} puntos, intervalo "
                    f"[{100 * punto['rel_control_ic_bajo']:+.2f}, "
                    f"{100 * punto['rel_control_ic_alto']:+.2f}])"
                )
            else:
                detalle_control = ""
            # El margen de cada punto es un intervalo de predicción, como el del control,
            # porque compara contra una sola observación de búsqueda.
            margen_punto_propio = _margen_prediccion(rv.de, rv.n)
            ic_bajo_punto_propio = infl.inflacion - margen_punto_propio
            ic_alto_punto_propio = infl.inflacion + margen_punto_propio
            print(
                f"   - {n_params_txt} parámetros: búsqueda {punto['exactitud_busqueda']:.3f}, "
                "reevaluada "
                f"{rv.media:.3f} (intervalo de confianza [{rv.ic_bajo:.3f}, {rv.ic_alto:.3f}]), "
                f"verdadera {punto['resumen_verdad'].media:.3f}: inflación (vs 0, intervalo de "
                f"PREDICCIÓN de una observación nueva) {100 * infl.inflacion:+.2f} puntos "
                f"(intervalo [{100 * ic_bajo_punto_propio:+.2f}, "
                f"{100 * ic_alto_punto_propio:+.2f}])"
                f"{detalle_control}."
            )
        # La inflación del control también usa un intervalo de predicción para una
        # observación nueva.
        margen_control_propio = _margen_prediccion(control_resumen_val.de, control_resumen_val.n)
        ic_bajo_control_propio = control_inflacion.inflacion - margen_control_propio
        ic_alto_control_propio = control_inflacion.inflacion + margen_control_propio
        veredicto_control_solo = _veredicto_relativo(ic_bajo_control_propio, ic_alto_control_propio)
        print(
            f"   control (línea base): búsqueda {control_busqueda:.3f}, reevaluada "
            f"{control_resumen_val.media:.3f} (intervalo de confianza "
            f"[{control_resumen_val.ic_bajo:.3f}, "
            f"{control_resumen_val.ic_alto:.3f}]): inflación propia (vs 0, intervalo de "
            f"PREDICCIÓN de una observación nueva) {100 * control_inflacion.inflacion:+.2f} puntos "
            f"(intervalo [{100 * ic_bajo_control_propio:+.2f}, "
            f"{100 * ic_alto_control_propio:+.2f}]) -- "
            f"{_frase_ic_cero(veredicto_control_solo)}."
        )
    diffs_no_vistas = [
        p["resumen_val"].media - p["resumen_verdad"].media for p in puntos_reevaluados
    ]
    if len(diffs_no_vistas) >= 2:
        resumen_no_vista = resumen(diffs_no_vistas)
        veredicto_no_vista = _veredicto_relativo(resumen_no_vista.ic_bajo, resumen_no_vista.ic_alto)
        # La cola solo aplica si el intervalo se distingue de 0.
        cola_no_vista = (
            ""
            if veredicto_no_vista == "no_distingue"
            else (
                " -- eso no es algo que re-entrenar con más semillas pueda corregir, hace falta "
                "un conjunto de validación nuevo"
            )
        )
        print(
            "   La parte que ninguna re-evaluación con los MISMOS datos puede ver -- la reevaluada "
            f"contra la verdad del oráculo -- promedia {100 * resumen_no_vista.media:+.2f} puntos "
            "(intervalo de confianza "
            f"{100 * resumen_no_vista.confianza:.0f}% [{100 * resumen_no_vista.ic_bajo:+.2f}, "
            f"{100 * resumen_no_vista.ic_alto:+.2f}], rango {100 * min(diffs_no_vistas):+.2f} a "
            f"{100 * max(diffs_no_vistas):+.2f}): {_frase_no_vista(veredicto_no_vista)}"
            f"{cola_no_vista}."
        )
    else:
        print(
            "   Un solo punto re-evaluado: no hay con qué calcular un intervalo de confianza de "
            "la parte que ninguna "
            "re-evaluación con los MISMOS datos puede ver."
        )
    val_sobre_techo = [
        p["exactitud_busqueda"] for p in puntos_reevaluados if p["exactitud_busqueda"] > TECHO_BAYES
    ]
    verdad_sobre_techo = [
        p["resumen_verdad"].media
        for p in puntos_reevaluados
        if p["resumen_verdad"].media > TECHO_BAYES
    ]
    if val_sobre_techo or verdad_sobre_techo:
        partes_techo = []
        if val_sobre_techo:
            partes_techo.append(
                f"{len(val_sobre_techo)} exactitud_val de la búsqueda (una sola validación de "
                f"{DEFAULTS['n_val']} datos: puede rebasarlo por el ruido de MUESTREO de ese "
                "conjunto, no por una sola semilla con suerte)"
            )
        if verdad_sobre_techo:
            # n real que sobrevivió a la Divergencia, el mismo para todos los puntos. El
            # ruido Monte Carlo del oráculo sale de la fórmula de una proporción binomial.
            n_verdad_techo = puntos_reevaluados[0]["resumen_verdad"].n
            ruido_oraculo_pct = (
                100 * (TECHO_BAYES * (1.0 - TECHO_BAYES) / DEFAULTS["n_oraculo"]) ** 0.5
            )
            partes_techo.append(
                f"{len(verdad_sobre_techo)} exactitud_verdadera media ({n_verdad_techo} "
                "semillas: solo puede rebasarlo por el ruido MONTE CARLO del propio oráculo, "
                f"~{ruido_oraculo_pct:.2f} puntos con su n_oraculo)"
            )
        print(f"   Superan el techo de Bayes ({TECHO_BAYES:.3f}): " + "; ".join(partes_techo) + ".")
else:
    print(
        "3. Ningún punto representativo del frente sobrevivió a la re-evaluación: todas las "
        f"semillas nuevas hicieron Divergencia en las {len(indices_reevaluar)} configs elegidas."
    )

if mejor_restriccion is None:
    n_infactibles = len(trials_restriccion)
    print(
        f"4. Ningún trial fue factible con n_parametros <= {PRESUPUESTO_PARAMETROS:.0f} (0 de "
        f"{len(trials_restriccion)})."
    )
else:
    n_infactibles = len(trials_restriccion) - len(factibles)
    if ETIQUETA_TOTAL == ETIQUETA_PRESUPUESTO:
        # mejor_frente_total y punto_comparable son el mismo trial, así que la
        # comparación sería contra sí misma: el paso lo dice en vez de imprimir ceros.
        frase_costo_real = (
            "el mejor punto del frente ya cabe en el presupuesto: la restricción no le "
            "quita nada por construcción"
        )
    elif costo_restriccion_real is None:
        frase_costo_real = (
            "El paso no pudo calcular el costo real: menos de 2 semillas sobrevivieron a la "
            "Divergencia para el punto del frente sin filtrar o para el punto dentro del "
            "presupuesto, en la re-evaluación del frente"
        )
    else:
        valor_abs_costo_pct = abs(100 * costo_restriccion_real.diferencia_media)
        frase_costo_valor = _frase_costo_restriccion(
            costo_restriccion_real.veredicto,
            valor_abs_costo_pct,
            100 * DELTA_COMPARACION,
            costo_restriccion_real.n,
        )
        frase_costo_real = (
            "El costo real, con el MISMO mecanismo de selección en los dos lados (el mejor punto "
            f"del frente en total contra el mejor dentro del presupuesto, los dos re-evaluados con "
            f"las mismas {costo_restriccion_real.n} semillas nuevas): sin filtrar "
            f"{resumen_frente_total_costo.media:.3f}, dentro del presupuesto "
            f"{resumen_frente_presupuesto_costo.media:.3f} -- {frase_costo_valor} "
            "(intervalo de confianza "
            f"{100 * costo_restriccion_real.confianza:.0f}% "
            f"[{100 * costo_restriccion_real.ic_bajo:+.2f}, "
            f"{100 * costo_restriccion_real.ic_alto:+.2f}])"
        )
    if resumen_factible_propio is None:
        frase_propia = (
            "aparte, el paso no pudo medir la desviación de esta búsqueda con restricción: "
            f"Divergencia descartó {descartadas_frente} de las {K_REEVALUAR} "
            "semillas nuevas pedidas"
        )
    else:
        busqueda_factible = mejor_restriccion.value
        desviacion_pct = 100 * (busqueda_factible - resumen_factible_propio.media)
        if (
            control_busqueda is None
            or len(val_control_propio) < 2
            or (len(val_control_propio) != len(val_restriccion))
        ):
            frase_propia = (
                "aparte -- desviación realizada, por sí sola NO es evidencia de selección (hace "
                f"falta un control): búsqueda {busqueda_factible:.3f}, re-evaluada "
                f"{resumen_factible_propio.media:.3f} ({desviacion_pct:+.2f} puntos)"
            )
        else:
            par_control_restriccion = diferencia_pareada(
                val_restriccion,
                dict(enumerate(val_control_propio)),
                delta=DELTA_COMPARACION,
                direccion="maximizar",
            )
            d_control_restriccion = busqueda_factible - control_busqueda
            media_rel_r, ic_bajo_rel_r, ic_alto_rel_r = _ic_relativo_control(
                d_control_restriccion,
                par_control_restriccion.diferencia_media,
                par_control_restriccion.de_dif,
                resumen_factible_propio.de,
                resumen(val_control_propio).de,
                par_control_restriccion.n,
            )
            veredicto_rel_r = _veredicto_relativo(ic_bajo_rel_r, ic_alto_rel_r)
            frase_propia = (
                "aparte -- un fenómeno DISTINTO del costo de la restricción --, búsqueda "
                f"{busqueda_factible:.3f}, re-evaluada {resumen_factible_propio.media:.3f} "
                f"({desviacion_pct:+.2f} puntos): contra el control (línea base), "
                f"{_frase_control_punto(veredicto_rel_r)} (diferencia de inflaciones "
                f"{100 * media_rel_r:+.2f} puntos, intervalo de confianza "
                f"[{100 * ic_bajo_rel_r:+.2f}, "
                f"{100 * ic_alto_rel_r:+.2f}])"
            )
    sin_params = mejor_sin_restriccion.user_attrs["n_parametros"]
    if sin_params <= PRESUPUESTO_PARAMETROS:
        frase_ratio = "cae dentro: en esta búsqueda la restricción no le quitó el mejor trial"
    else:
        ratio_sin_restriccion = sin_params / PRESUPUESTO_PARAMETROS
        frase_ratio = f"{ratio_sin_restriccion:.1f}x el presupuesto -- queda fuera de ella"
    print(
        f"4. Con n_parametros <= {PRESUPUESTO_PARAMETROS:.0f}: {len(factibles)} de "
        f"{len(trials_restriccion)} trials completos fueron factibles ({n_infactibles} "
        "infactibles); el mejor trial de ESTA búsqueda con restricción, sin mirar la restricción "
        f"(exactitud_val {mejor_sin_restriccion.value:.3f}, {sin_params:.0f} parámetros): "
        f"{frase_ratio}. El ganador FACTIBLE tiene "
        f"{mejor_restriccion.user_attrs['n_parametros']:.0f} parámetros. {frase_costo_real}; "
        f"{frase_propia}."
    )
    if costo_restriccion_real is not None:
        nota_k_costo = _nota_semillas_necesarias(
            costo_restriccion_real.de_dif, costo_restriccion_real.n, delta=DELTA_COMPARACION
        )
        print(f"   ({nota_k_costo})")

if mejor_restriccion is None:
    print(
        "5. Sin ningún trial factible en la variante con restricción (respuesta 4), no hay con "
        "qué comparar el punto del frente al mismo presupuesto."
    )
elif comparacion is None:
    print(
        "5. El paso no pudo calcular la comparación pareada entre la variante con restricción y "
        "el punto del frente al mismo presupuesto: Divergencia descartó "
        f"{descartadas_frente} de las {K_REEVALUAR} semillas nuevas pedidas."
    )
else:
    if comparacion.veredicto == "mejor":
        frase_comparacion = f"gana por {100 * comparacion.diferencia_media:.2f} puntos"
    elif comparacion.veredicto == "peor":
        frase_comparacion = f"pierde por {100 * abs(comparacion.diferencia_media):.2f} puntos"
    elif comparacion.veredicto == "equivalente":
        frase_comparacion = (
            f"es equivalente (diferencia {100 * comparacion.diferencia_media:+.2f} puntos, "
            f"dentro de ±{100 * DELTA_COMPARACION:.0f})"
        )
    else:
        frase_comparacion = (
            f"da un resultado no concluyente (diferencia "
            f"{100 * comparacion.diferencia_media:+.2f} puntos, no se distingue del ruido con "
            f"{comparacion.n} semillas)"
        )
    nota_presupuesto = (
        " (excede el presupuesto: el frente no tuvo ningún punto dentro)"
        if frente_sin_candidato_en_presupuesto
        else ""
    )
    print(
        f"5. El mejor punto del frente con n_parametros <= {PRESUPUESTO_PARAMETROS:.0f}"
        f"{nota_presupuesto} tiene {punto_comparable.values[1]:.0f} parámetros y exactitud_val "
        f"{punto_comparable.values[0]:.3f} en la búsqueda. Re-evaluado con las mismas "
        f"{comparacion.n} semillas nuevas que la config de la restricción: restricción "
        f"{resumen_restriccion.media:.3f}, frente {resumen_frente_cmp.media:.3f} "
        "(intervalo de confianza "
        f"{100 * comparacion.confianza:.0f}% [{100 * comparacion.ic_bajo:+.2f}, "
        f"{100 * comparacion.ic_alto:+.2f}] sobre la diferencia): en ESTA réplica de cada "
        f"búsqueda, la variante con restricción {frase_comparacion} frente a buscar en dos "
        "objetivos y filtrar después. Para decir cuál de los dos MÉTODOS es mejor en general (no "
        "solo en esta corrida) harían falta varias réplicas de cada búsqueda, como en el paso 01 "
        "-- una config ganadora por búsqueda no basta."
    )
    nota_k_comparacion = _nota_semillas_necesarias(
        comparacion.de_dif, comparacion.n, delta=DELTA_COMPARACION
    )
    print(f"   ({nota_k_comparacion})")
    # Las semillas descartadas son las mismas que ya reportó la respuesta 3.

# %%
fig = Figure(figsize=(7, 5))
ax = fig.subplots()

ax.scatter(
    [t.values[1] for t in trials_pareto_completos],
    [t.values[0] for t in trials_pareto_completos],
    color="lightgray",
    marker=".",
    label=f"los {len(trials_pareto_completos)} trials",
)
ax.plot(parametros_frente, exactitudes_frente, color="tab:blue", linewidth=1, alpha=0.6)
ax.scatter(
    parametros_frente,
    exactitudes_frente,
    color="tab:blue",
    marker="o",
    label=f"frente ({len(frente_ordenado)})",
)
x_reeval = [p["n_parametros"] for p in puntos_reevaluados]
y_reeval = [p["resumen_val"].media for p in puntos_reevaluados]
ax.plot(x_reeval, y_reeval, color="tab:red", linewidth=1, linestyle="--", alpha=0.7)
ax.scatter(
    x_reeval,
    y_reeval,
    color="tab:red",
    marker="x",
    label=f"frente re-evaluado ({len(puntos_reevaluados)} pts x {K_REEVALUAR} semillas)",
)
if linea_base_trial is not None:
    ax.scatter(
        [linea_base_trial.values[1]],
        [linea_base_trial.values[0]],
        color="tab:green",
        marker="*",
        s=150,
        label="línea base (DEFAULTS)",
        zorder=5,
    )
ax.axvline(
    PRESUPUESTO_PARAMETROS,
    color="gray",
    linestyle=":",
    linewidth=1,
    alpha=0.7,
    label=f"presupuesto ({PRESUPUESTO_PARAMETROS:.0f})",
)
# Log en x: en escala lineal, los pocos puntos caros aplastarían contra el eje casi todo
# el frente.
ax.set_xscale("log")
ax.set_xlabel("n_parametros (costo, escala log)")
ax.set_ylabel("exactitud_val")
ax.set_title(f"Paso 04: Pareto exactitud vs. costo ({presupuesto}, {N_TOTAL} trials)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
figura = RESULTADOS / "04_pareto.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
