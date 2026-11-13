# %% [markdown]
# # Paso 05: buscar el modelo, no solo sus perillas
#
# Hasta el paso 04 el algoritmo era fijo (el MLP del juguete) y la búsqueda solo movía sus
# hiperparámetros. Aquí `algoritmo` también es un hiperparámetro: `logistica`, `bosque`
# (bosque aleatorio), `hgb` (boosting de histogramas) o `mlp`, cada uno con sus propios
# parámetros con prefijo. Esto es CASH (selección conjunta de algoritmo e
# hiperparámetros), y en el MLP también decide cuántas capas y qué tan anchas, una
# búsqueda de arquitectura a esta escala.
#
# El espacio es condicional: si `algoritmo == "logistica"`, los parámetros de `bosque` no
# existen para ese trial. `espacio(trial)` decide en tiempo de ejecución qué sugerir
# (define-by-run). Por eso el sampler importa: sin `group`,
# `TPESampler(multivariate=True)` solo modela en conjunto los parámetros que aparecen en
# todos los trials, aquí solo `algoritmo`. Con `group=True`, modela juntos los parámetros
# que siempre aparecen juntos, como los tres de `hgb`.
#
# Tres preguntas:
# 1. ¿Un sampler que entiende el espacio condicional (TPE con group) le gana
#    a muestrear al azar, con el mismo presupuesto de evaluaciones?
# 2. ¿Qué algoritmo eligió la búsqueda con más frecuencia?
# 3. ¿La búsqueda con TPE le gana a los baselines (knn, krr) con SUS
#    defaults, sin ninguna búsqueda?
#
# La métrica de comparación es honesta: no el `exactitud_val` que ve la
# búsqueda (ese tiende a favorecer al que más mira los datos de validación),
# sino la `exactitud_verdadera` (el oráculo) del trial que la búsqueda
# HABRÍA elegido mirando solo su validación. El paso 06 mide cuánto se infla
# la exactitud_val del ganador frente a su verdad; aquí el paso usa la verdad
# solo para comparar métodos de búsqueda entre sí, nunca para elegir nada.
#
# Correr:  uv run pasos/05_buscar_el_modelo.py  (necesita una tarjeta de video
#          NVIDIA con CUDA, con torch y XGBoost compilados con soporte de CUDA;
#          sin ella, el paso se detiene con un mensaje claro)
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import math
import os
import statistics as st
import warnings
from collections import Counter
from pathlib import Path

import optuna
import torch

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure
from optuna.trial import FrozenTrial, TrialState
from scipy.stats import binomtest
from scipy.stats import t as t_student

from taller_hpo import estadistica
from taller_hpo.estudio import optimizar
from taller_hpo.objetivo import en_proceso
from taller_hpo.problemas import cash_paridad, juguete
from taller_hpo.semillas import semillas_de

# El log INFO de Optuna imprime una línea por trial y taparía las respuestas.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# R y N controlan el tiempo; el tamaño del problema se deja igual al de
# cash_paridad.DEFAULTS (ver FIJOS, más abajo).
PRESUPUESTOS = {"humo": (2, 4), "taller": (5, 15), "completo": (15, 55)}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
R, N = PRESUPUESTOS[presupuesto]

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)
# Los estudios de esta comparación son de una sola vez, así que quedan en memoria.

# FIJOS vacío: el problema es el mismo de cash_paridad.DEFAULTS, salvo `rotar` (ver
# _frase_tamano_problema). R y N son la única palanca de presupuesto de este paso.
FIJOS = {}

# cash_paridad no expone una DESCRIPCION pública, así que el paso resuelve el dispositivo
# de la config que entrena para imprimir el nombre de tu tarjeta de video.
_CONFIG_DISPOSITIVO = {**cash_paridad.DEFAULTS, **FIJOS}
_DISPOSITIVO_CASH = cash_paridad._resolver_dispositivo(_CONFIG_DISPOSITIVO["dispositivo"])
print(
    f"cash_paridad: dispositivo={_DISPOSITIVO_CASH} "
    f"({torch.cuda.get_device_name(_DISPOSITIVO_CASH)})"
)


def _frase_tamano_problema(config: dict, defaults_juguete: dict) -> str:
    """Compara `n_train`, `n_val` y `rotar` de la config que entrena el paso contra
    `juguete.DEFAULTS`, en vez de afirmar una igualdad fija. `rotar` va aparte: no es un
    tamaño, y aquí siempre difiere para quitarles a los árboles la ventaja de cortar por
    ejes."""
    claves = ("n_train", "n_val")
    if all(config[c] == defaults_juguete[c] for c in claves):
        frase_tamano = "el mismo tamaño que el resto del taller"
    else:
        diferencias = ", ".join(
            f"{c}={config[c]} (el resto del taller usa {defaults_juguete[c]})"
            for c in claves
            if config[c] != defaults_juguete[c]
        )
        frase_tamano = f"un tamaño distinto al resto del taller: {diferencias}"
    if config["rotar"] == defaults_juguete["rotar"]:
        frase_rotar = f"rotar={config['rotar']}, igual que el resto del taller"
    else:
        frase_rotar = (
            f"rotado (rotar={config['rotar']}), a diferencia del resto del taller "
            f"(rotar={defaults_juguete['rotar']})"
        )
    return f"{frase_tamano}; {frase_rotar}"


print(
    f"Problema: paridad de d={_CONFIG_DISPOSITIVO['d']} bits, a={_CONFIG_DISPOSITIVO['a']}, "
    f"{_CONFIG_DISPOSITIVO['k_ruido']} columnas de ruido, "
    f"{_CONFIG_DISPOSITIVO['n_train']} datos de entrenamiento, "
    f"{_CONFIG_DISPOSITIVO['n_val']} de validación -- "
    f"{_frase_tamano_problema(_CONFIG_DISPOSITIVO, juguete.DEFAULTS)}."
)

# %% [markdown]
# ## El espacio condicional: define-by-run
#
# `espacio(trial)` es el único lugar que llama a `trial.suggest_*`
# (convención de `taller_hpo.objetivo.en_proceso`). Con `algoritmo="mlp"`,
# `espacio` sugiere los anchos SOLO hasta `n_capas`: con `n_capas=1`, Optuna
# ni siquiera sortea `ancho_2` ni `ancho_3` para ese trial.

# %%
_RANGOS_ARBOL = {
    "bosque_n_arboles": (20, 120),
    "bosque_profundidad_max": (2, 14),
    "bosque_min_muestras_hoja": (1, 10),
    "hgb_max_hojas": (7, 63),
    "hgb_max_iter": (20, 120),
}


def espacio(trial: optuna.Trial) -> dict:
    algoritmo = trial.suggest_categorical("algoritmo", ["logistica", "bosque", "hgb", "mlp"])
    config: dict = {"algoritmo": algoritmo}

    if algoritmo == "logistica":
        config["logistica_C"] = trial.suggest_float("logistica_C", 1e-3, 1e2, log=True)
    elif algoritmo == "bosque":
        config["bosque_n_arboles"] = trial.suggest_int(
            "bosque_n_arboles", *_RANGOS_ARBOL["bosque_n_arboles"], log=True
        )
        config["bosque_profundidad_max"] = trial.suggest_int(
            "bosque_profundidad_max", *_RANGOS_ARBOL["bosque_profundidad_max"]
        )
        config["bosque_min_muestras_hoja"] = trial.suggest_int(
            "bosque_min_muestras_hoja", *_RANGOS_ARBOL["bosque_min_muestras_hoja"]
        )
    elif algoritmo == "hgb":
        config["hgb_learning_rate"] = trial.suggest_float("hgb_learning_rate", 1e-3, 1.0, log=True)
        config["hgb_max_hojas"] = trial.suggest_int(
            "hgb_max_hojas", *_RANGOS_ARBOL["hgb_max_hojas"], log=True
        )
        config["hgb_max_iter"] = trial.suggest_int(
            "hgb_max_iter", *_RANGOS_ARBOL["hgb_max_iter"], log=True
        )
    else:  # "mlp"
        n_capas = trial.suggest_int("n_capas", 1, 3)
        config["n_capas"] = n_capas
        for i in range(1, n_capas + 1):
            config[f"ancho_{i}"] = trial.suggest_int(f"ancho_{i}", 8, 96, log=True)
        config["mlp_learning_rate_init"] = trial.suggest_float(
            "mlp_learning_rate_init", 1e-4, 5e-2, log=True
        )
        config["mlp_alpha"] = trial.suggest_float("mlp_alpha", 1e-6, 1e-1, log=True)

    return config


# %% [markdown]
# ## TPE no actúa desde el primer trial
#
# `TPESampler` arranca con `n_startup_trials=10` trials al azar, el default de Optuna 5.
# Con `PRESUPUESTO=humo` (`N=4`), TPE nunca deja de ser aleatorio, y la respuesta 1 lo
# dice.

# %% [markdown]
# ## R réplicas: TPE contra Random, mismos datos dentro de cada réplica
#
# Dentro de una réplica, TPE y Random ven exactamente los mismos datos
# (números aleatorios comunes vía una única `Semillas` por réplica, el mismo
# patrón que documenta `taller_hpo.honestidad.plan_semillas`): así la
# diferencia entre ellos es el sampler, no la suerte del dataset. Entre
# réplicas, tanto los datos como las semillas de cada sampler cambian.

# %%
BASE_DATOS = 20_260_925
BASE_TPE = 30_000_000
BASE_RANDOM = 40_000_000
N_STARTUP_TRIALS_TPE = 10  # TPESampler.n_startup_trials, default en Optuna 5.0.0.
ALGORITMOS_ORDEN = ("logistica", "bosque", "hgb", "mlp")  # orden fijo, no depende del hash seed.


def _mejor_por_validacion(study: optuna.Study) -> FrozenTrial:
    """El trial que la búsqueda misma habría elegido: el de mayor `exactitud_val`, nunca
    por el oráculo."""
    completos = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return max(completos, key=lambda t: t.value)


tpe_por_replica: dict[int, float] = {}
random_por_replica: dict[int, float] = {}
algoritmo_ganador_tpe: list[str] = []
knn_por_replica: dict[int, float] = {}
krr_por_replica: dict[int, float] = {}
segundos_por_algoritmo: dict[str, list[float]] = {a: [] for a in (*ALGORITMOS_ORDEN, "knn", "krr")}

for r, semillas_r in enumerate(semillas_de(BASE_DATOS, R)):
    objetivo_r = en_proceso(
        cash_paridad.entrenar,
        espacio,
        metricas=["exactitud_val"],
        semillas=semillas_r,
        fijos=FIJOS,
        atributos=["exactitud_verdadera", "segundos"],
    )

    with warnings.catch_warnings(record=True) as avisos:
        # "always" sobre todos los avisos, para re-emitir abajo cualquier otro aviso.
        warnings.simplefilter("always", Warning)
        sampler_tpe = optuna.samplers.TPESampler(
            seed=BASE_TPE + r,
            multivariate=True,
            group=True,
            # Explícito, para que la respuesta 1 describa lo que este sampler hace.
            n_startup_trials=N_STARTUP_TRIALS_TPE,
        )
    # Re-emite tal cual cualquier aviso que no sea el ExperimentalWarning esperado de
    # group=True, para que bajo -W error se vuelva una excepción visible.
    for aviso in avisos:
        if not issubclass(aviso.category, optuna.exceptions.ExperimentalWarning):
            warnings.warn_explicit(aviso.message, aviso.category, aviso.filename, aviso.lineno)
    if r == 0 and any(
        issubclass(a.category, optuna.exceptions.ExperimentalWarning) for a in avisos
    ):
        # group=True es experimental en Optuna 5 y es el que modela bien un espacio
        # condicional (ver el markdown de arriba).
        print(
            f"Nota: TPESampler(group=True) es experimental en Optuna {optuna.__version__}; el "
            "paso lo usa a propósito porque su espacio es condicional (define-by-run)."
        )

    study_tpe = optuna.create_study(
        sampler=sampler_tpe, pruner=optuna.pruners.NopPruner(), direction="maximize"
    )
    optimizar(study_tpe, objetivo_r, n_total=N, atrapar=(cash_paridad.Divergencia,))

    sampler_random = optuna.samplers.RandomSampler(seed=BASE_RANDOM + r)
    study_random = optuna.create_study(
        sampler=sampler_random, pruner=optuna.pruners.NopPruner(), direction="maximize"
    )
    optimizar(study_random, objetivo_r, n_total=N, atrapar=(cash_paridad.Divergencia,))

    for estudio in (study_tpe, study_random):
        for t in estudio.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            segundos_por_algoritmo[t.params["algoritmo"]].append(t.user_attrs["segundos"])

    elegido_tpe = _mejor_por_validacion(study_tpe)
    # maximo_val_tpe se recalcula aparte, como chequeo de consistencia de la línea impresa.
    completos_tpe = study_tpe.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    maximo_val_tpe = max(t.value for t in completos_tpe)
    tpe_por_replica[r] = elegido_tpe.user_attrs["exactitud_verdadera"]
    algoritmo_ganador_tpe.append(elegido_tpe.params["algoritmo"])

    elegido_random = _mejor_por_validacion(study_random)
    random_por_replica[r] = elegido_random.user_attrs["exactitud_verdadera"]

    resultado_knn = cash_paridad.entrenar({**FIJOS, "algoritmo": "knn"}, semillas_r)
    knn_por_replica[r] = resultado_knn["exactitud_verdadera"]
    segundos_por_algoritmo["knn"].append(resultado_knn["segundos"])

    resultado_krr = cash_paridad.entrenar({**FIJOS, "algoritmo": "krr"}, semillas_r)
    krr_por_replica[r] = resultado_krr["exactitud_verdadera"]
    segundos_por_algoritmo["krr"].append(resultado_krr["segundos"])

    # Línea parseable por réplica, para que las pruebas recalculen las medias sin leer la
    # prosa de las respuestas.
    print(
        f"replica {r}: dispositivo={_DISPOSITIVO_CASH} | "
        f"tpe exactitud_val(elegido)={elegido_tpe.value:.4f} "
        f"exactitud_val(maximo_estudio)={maximo_val_tpe:.4f} | "
        f"verdadera tpe={tpe_por_replica[r]:.4f} random={random_por_replica[r]:.4f} "
        f"knn={knn_por_replica[r]:.4f} krr={krr_por_replica[r]:.4f} | "
        f"algoritmo_ganador_tpe={algoritmo_ganador_tpe[r]}"
    )

medianas_segundos = {a: round(st.median(v), 4) for a, v in segundos_por_algoritmo.items() if v}
print(
    f"\nsegundos por algoritmo (mediana de ESTA corrida, ajuste + validación, sin oráculo; "
    f"dispositivo={_DISPOSITIVO_CASH}): {medianas_segundos}"
)

# %% [markdown]
# ## Las tres respuestas

# %%
DELTA_EQUIVALENCIA = 0.01  # 1 punto de exactitud: la zona de indiferencia del TOST.
comparacion = estadistica.diferencia_pareada(
    tpe_por_replica, random_por_replica, delta=DELTA_EQUIVALENCIA, direccion="maximizar"
)
comparacion_knn = estadistica.diferencia_pareada(
    tpe_por_replica, knn_por_replica, delta=DELTA_EQUIVALENCIA, direccion="maximizar"
)
comparacion_krr = estadistica.diferencia_pareada(
    tpe_por_replica, krr_por_replica, delta=DELTA_EQUIVALENCIA, direccion="maximizar"
)

media_tpe = sum(tpe_por_replica.values()) / R
media_random = sum(random_por_replica.values()) / R
media_knn = sum(knn_por_replica.values()) / R
media_krr = sum(krr_por_replica.values()) / R

# Counter con orden fijo, nunca `set(...)`: el orden de un set depende de PYTHONHASHSEED
# y podría cambiar el ganador de un empate.
conteo_algoritmos = Counter(algoritmo_ganador_tpe)
conteo_completo = {a: conteo_algoritmos.get(a, 0) for a in ALGORITMOS_ORDEN}


def _frase_arranque(n: int, n_startup: int) -> str:
    """Describe cuántos de los `n` trials completos de TPE por réplica fueron aleatorios
    antes de agotar `n_startup_trials`."""
    n_aleatorios = max(0, min(n, n_startup))
    if n_aleatorios >= n:
        return (
            f"con estas {n} evaluaciones completas por réplica, TPE arrancó aleatorio "
            f"(TPESampler.n_startup_trials={n_startup}) y nunca llegó a actuar de "
            f"forma guiada"
        )
    return (
        f"de los {n} trials completos por réplica, los primeros {n_aleatorios} fueron "
        f"aleatorios (TPESampler.n_startup_trials={n_startup}) y los "
        f"{n - n_aleatorios} restantes usaron el modelo"
    )


def _frase_conteo(conteo: dict, r: int) -> str:
    """Respuesta 2: qué algoritmo ganó más veces con TPE, con el intervalo de Wilson del
    conteo máximo. Si hay un ganador único, lo compara contra el segundo lugar con una
    prueba binomial cara a cara, nunca con el traslape de sus intervalos."""
    maximo = max(conteo.values())
    top = [a for a in conteo if conteo[a] == maximo]
    ic_bajo, ic_alto = estadistica.ic_proporcion(maximo, r)
    if len(top) == 1:
        frase = (
            f"2. El algoritmo que TPE eligió como mejor con más frecuencia fue "
            f"'{top[0]}' ({maximo} de {r} réplicas, intervalo de confianza 95% Wilson de "
            f"la proporción [{ic_bajo:.2f}, {ic_alto:.2f}]); "
            f"conteo completo (orden fijo): {conteo}."
        )
        conteo_segundo = max(v for a, v in conteo.items() if a != top[0])
        segundos = [a for a in conteo if a != top[0] and conteo[a] == conteo_segundo]
        ic2_bajo, ic2_alto = estadistica.ic_proporcion(conteo_segundo, r)
        nombres_segundo = " y ".join(f"'{a}'" for a in segundos)
        p_cara_a_cara = float(binomtest(maximo, maximo + conteo_segundo).pvalue)
        if p_cara_a_cara < 0.05:
            frase += (
                f" Cara a cara contra {nombres_segundo} ({conteo_segundo} de {r}, intervalo "
                f"95% Wilson [{ic2_bajo:.2f}, {ic2_alto:.2f}]): sí se distingue (prueba de "
                f"signos binomial, p={p_cara_a_cara:.4f})."
            )
        else:
            frase += (
                f" Cara a cara contra {nombres_segundo} ({conteo_segundo} de {r}, intervalo "
                f"95% Wilson [{ic2_bajo:.2f}, {ic2_alto:.2f}]): no se distingue con {r} "
                f"réplicas (prueba de signos binomial, p={p_cara_a_cara:.4f})."
            )
        return frase
    lista_empate = " y ".join(f"'{a}'" for a in top)
    return (
        f"2. Empate entre {lista_empate} como el algoritmo que TPE eligió como mejor con "
        f"más frecuencia ({maximo} de {r} réplicas cada uno, intervalo de confianza 95% "
        f"Wilson de la proporción [{ic_bajo:.2f}, {ic_alto:.2f}]); "
        f"conteo completo (orden fijo): {conteo}."
    )


frase_arranque = _frase_arranque(N, N_STARTUP_TRIALS_TPE)


def _frase_potencia(comparacion: estadistica.Pareado, r: int) -> str:
    """Con veredicto 'no concluyente', dice cuántas réplicas harían falta para detectar
    una diferencia real del tamaño de DELTA_EQUIVALENCIA con potencia de 80%. Con
    cualquier otro veredicto devuelve una cadena vacía.
    """
    if comparacion.veredicto != "no concluyente":
        return ""
    if comparacion.de_dif <= 0.0:
        return (
            " (la desviación estándar de las diferencias midió 0: no hay variación entre "
            "réplicas que estimar)."
        )
    n_necesario = estadistica.semillas_necesarias(comparacion.de_dif, DELTA_EQUIVALENCIA)
    return (
        f" Detectar una diferencia real del tamaño de delta={DELTA_EQUIVALENCIA} "
        f"con potencia 80% y alfa 0.05 exigiría ~{n_necesario} réplicas (estimado con "
        f"la desviación estándar de las diferencias de estas {r} réplicas, a su vez incierta "
        f"con {r - 1} {'grado' if r - 1 == 1 else 'grados'} de libertad)."
    )


frase_potencia = _frase_potencia(comparacion, R)

print()
print(
    f"1. TPE (group=True) vs Random, {R} réplicas de {N} evaluaciones cada una "
    f"({frase_arranque}): exactitud verdadera media {media_tpe:.3f} vs {media_random:.3f} "
    f"(diferencia {comparacion.diferencia_media:+.3f}, intervalo de confianza 95% "
    f"[{comparacion.ic_bajo:.3f}, {comparacion.ic_alto:.3f}], veredicto "
    f"'{comparacion.veredicto}' con delta={DELTA_EQUIVALENCIA}).{frase_potencia}"
)

print(_frase_conteo(conteo_completo, R))


def _frase_veredicto(nombre_baseline: str, comp: estadistica.Pareado) -> str:
    """El verbo de "la búsqueda con TPE" contra `nombre_baseline`, a partir del veredicto
    de `diferencia_pareada` y nunca del signo de la diferencia sola."""
    if comp.veredicto == "mejor":
        base = f"le gana a {nombre_baseline} por {comp.diferencia_media:+.3f}"
        if comp.ic_bajo < comp.delta:
            # El intervalo excluye el 0, pero su borde inferior cae dentro de la zona de
            # indiferencia: la ventaja real podría ser menor que delta.
            return base + " (el intervalo no descarta una ventaja menor que delta)"
        return base
    if comp.veredicto == "peor":
        return f"pierde contra {nombre_baseline} por {comp.diferencia_media:+.3f}"
    if comp.veredicto == "equivalente":
        return (
            f"empata con {nombre_baseline} dentro de ±{comp.delta}, diferencia "
            f"{comp.diferencia_media:+.3f}"
        )
    return (
        f"no se distingue de {nombre_baseline} con {R} réplicas, diferencia "
        f"{comp.diferencia_media:+.3f}"
    )


print(
    f"3. Baselines con sus defaults (sin ninguna búsqueda): knn "
    f"{media_knn:.3f}, krr {media_krr:.3f}. La búsqueda con TPE (exactitud verdadera "
    f"media {media_tpe:.3f}) {_frase_veredicto('knn', comparacion_knn)} "
    f"(intervalo de confianza 95% [{comparacion_knn.ic_bajo:.3f}, "
    f"{comparacion_knn.ic_alto:.3f}], veredicto '{comparacion_knn.veredicto}'), y "
    f"{_frase_veredicto('krr', comparacion_krr)} "
    f"(intervalo 95% [{comparacion_krr.ic_bajo:.3f}, {comparacion_krr.ic_alto:.3f}], "
    f"veredicto '{comparacion_krr.veredicto}')."
)

# Las tres comparaciones comparten la serie de TPE y cada una es al 95% por su cuenta. El
# paso reporta el p ajustado por Holm-Bonferroni y si el ajuste cambiaría algún "mejor" o
# "peor". Etiqueta y comparación van en una sola tupla para que no se desincronicen.
_COMPARACIONES_HOLM = (
    ("Random", comparacion),
    ("knn", comparacion_knn),
    ("krr", comparacion_krr),
)
p_holm = estadistica.holm([c.p_valor for _, c in _COMPARACIONES_HOLM])
_etiquetas_holm = tuple(nombre for nombre, _ in _COMPARACIONES_HOLM)
cambia_holm = [
    (c.veredicto in ("mejor", "peor")) != (p < 0.05)
    for (_, c), p in zip(_COMPARACIONES_HOLM, p_holm, strict=True)
]


def _frase_holm(cambia: list[bool], etiquetas: tuple[str, ...]) -> str:
    """Nota de Holm-Bonferroni: si ningún veredicto 'mejor' o 'peor' cambia lo dice, y si
    alguno cambia lista solo los afectados. 'equivalente' sale del TOST y no se ajusta."""
    if not any(cambia):
        return (
            "ajustar por Holm-Bonferroni no cambia ningún veredicto 'mejor'/'peor' con estos datos"
        )
    afectados = ", ".join(e for e, c in zip(etiquetas, cambia, strict=True) if c)
    return (
        f"al ajustar por Holm-Bonferroni el veredicto 'mejor'/'peor' cambiaría contra: {afectados}"
    )


frase_holm = _frase_holm(cambia_holm, _etiquetas_holm)
# Cada p impreso se lee en paralelo con su etiqueta (zip), nunca con índices literales.
_lineas_p_holm = [
    f"p={p:.3f} vs {etiqueta}" for etiqueta, p in zip(_etiquetas_holm, p_holm, strict=True)
]
print(
    f"   (las tres comparaciones de arriba comparten la serie de TPE y no llevan ajuste "
    f"por multiplicidad; con Holm-Bonferroni: {', '.join(_lineas_p_holm)} -- {frase_holm})"
)

# %% [markdown]
# ## Figura: por réplica, y las diferencias pareadas con su intervalo de
# ## confianza 95% y su intervalo del TOST
#
# El panel izquierdo muestra las cuatro series por réplica (TPE, Random, knn, krr),
# porque la comparación es pareada. El panel derecho dibuja dos intervalos por
# comparación: el fino y negro es el de 95%, que decide 'mejor' o 'peor' si no cruza el
# cero; el grueso y azul es el de nivel 1 - 2*alfa (90%), que decide 'equivalente' contra
# la zona sombreada ±DELTA_EQUIVALENCIA.

# %%
fig = Figure(figsize=(10, 4.6))
ax1, ax2 = fig.subplots(1, 2)
replicas = list(range(R))
ax1.scatter(
    replicas,
    [tpe_por_replica[r] for r in replicas],
    label="TPE (group=True), por validación",
    marker="o",
)
ax1.scatter(
    replicas,
    [random_por_replica[r] for r in replicas],
    label="Random, por validación",
    marker="x",
)
ax1.scatter(replicas, [knn_por_replica[r] for r in replicas], label="knn (baseline)", marker="^")
ax1.scatter(replicas, [krr_por_replica[r] for r in replicas], label="krr (baseline)", marker="s")
ax1.set_xlabel("réplica")
ax1.set_ylabel("exactitud verdadera")
ax1.set_title("Por réplica")
ax1.set_xticks(replicas)

_COMPARACIONES = (
    ("TPE - Random", comparacion),
    ("TPE - knn", comparacion_knn),
    ("TPE - krr", comparacion_krr),
)
_pos_comp = list(range(len(_COMPARACIONES)))
_medias_comp = [c.diferencia_media for _, c in _COMPARACIONES]
_yerr_comp = [
    [c.diferencia_media - c.ic_bajo for _, c in _COMPARACIONES],
    [c.ic_alto - c.diferencia_media for _, c in _COMPARACIONES],
]


def _ic_tost(comp: estadistica.Pareado) -> tuple[float, float]:
    """El intervalo de nivel 1 - 2*alfa que `diferencia_pareada` usa para decidir
    'equivalente', recalculado solo para dibujarlo en la figura."""
    if comp.de_dif <= 0.0:
        return comp.diferencia_media, comp.diferencia_media
    ee = comp.de_dif / math.sqrt(comp.n)
    alfa = 1.0 - comp.confianza
    t_critico = float(t_student.ppf(1.0 - alfa, df=comp.n - 1))
    return comp.diferencia_media - t_critico * ee, comp.diferencia_media + t_critico * ee


_ic_tost_comp = [_ic_tost(c) for _, c in _COMPARACIONES]
_yerr_tost = [
    [m - lo for m, (lo, _) in zip(_medias_comp, _ic_tost_comp, strict=True)],
    [hi - m for m, (_, hi) in zip(_medias_comp, _ic_tost_comp, strict=True)],
]
ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1)
ax2.axhspan(
    -DELTA_EQUIVALENCIA,
    DELTA_EQUIVALENCIA,
    color="tab:green",
    alpha=0.15,
    label=f"zona de equivalencia ±{DELTA_EQUIVALENCIA}",
)
# El intervalo 90% (TOST) se dibuja PRIMERO -- grueso y translúcido -- para
# que el intervalo 95%, más fino y encima, no quede tapado por él.
ax2.errorbar(
    _pos_comp,
    _medias_comp,
    yerr=_yerr_tost,
    fmt="none",
    ecolor="tab:blue",
    elinewidth=6,
    alpha=0.3,
    capsize=0,
    label="diferencia pareada (intervalo 90%, TOST: decide 'equivalente')",
)
ax2.errorbar(
    _pos_comp,
    _medias_comp,
    yerr=_yerr_comp,
    fmt="o",
    capsize=4,
    color="black",
    label="diferencia pareada (intervalo 95%: decide 'mejor'/'peor')",
)
ax2.set_xlim(-0.5, len(_pos_comp) - 0.5)
ax2.set_xticks(_pos_comp)
ax2.set_xticklabels([nombre for nombre, _ in _COMPARACIONES])
ax2.set_ylabel("diferencia pareada")
ax2.set_title("Comparaciones pareadas")

_handles1, _labels1 = ax1.get_legend_handles_labels()
_handles2, _labels2 = ax2.get_legend_handles_labels()
fig.legend(
    _handles1 + _handles2,
    _labels1 + _labels2,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=7,
)

fig.suptitle("Paso 05: buscar el modelo -- TPE vs Random vs baselines")
fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.96))
figura = RESULTADOS / "05_buscar_el_modelo.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
