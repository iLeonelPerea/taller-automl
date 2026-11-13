# %% [markdown]
# # Paso 01: grid contra random, con réplicas
#
# El resultado de Bergstra y Bengio (2012) no es una anécdota de una corrida:
# es una afirmación sobre una distribución de resultados. Aquí el paso repite
# el experimento con `R` réplicas (una semilla por réplica, un paisaje
# distinto en cada una) y agrega un tercer brazo de diseño de experimentos,
# el muestreo cuasi Monte Carlo (QMC, con la secuencia de Sobol), que tampoco
# reparte puntos por eje como la rejilla.
#
# El paisaje (`taller_hpo.problemas.analitico`) calcula el regret con una fórmula
# cerrada, sin ningún modelo que entrenar, así que el paso puede repetir el experimento
# con muchas réplicas. Es el único paso que no usa la tarjeta de video: no hay nada que
# entrenar.
#
# Cuatro preguntas:
# 1. ¿En cuántas réplicas random vence a grid?
# 2. ¿Cuál es la diferencia pareada media (grid - random) y qué dice el intervalo de confianza?
# 3. ¿Por qué? ¿Cuántos valores distintos del eje importante prueba cada brazo?
# 4. ¿Dónde queda QMC frente a los otros dos, medido igual que la pregunta 2?
#
# Correr:  uv run pasos/01_grid_vs_random.py
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import math
import os
import warnings
from pathlib import Path

import optuna

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure

from taller_hpo.estadistica import diferencia_pareada, ic_proporcion
from taller_hpo.problemas.analitico import crear_paisaje, regret_minimo_grid

# El log INFO de Optuna imprime una línea por trial y taparía las respuestas; WARNING
# deja pasar solo lo importante.
optuna.logging.set_verbosity(optuna.logging.WARNING)

PRESUPUESTOS = {"humo": 10, "taller": 100, "completo": 300}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
R = PRESUPUESTOS[presupuesto]

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)

print(
    "Este paso no usa la tarjeta de video: el paisaje analítico es una fórmula "
    "cerrada, sin ningún modelo que entrenar.\n"
)

NIVELES = (1 / 6, 1 / 2, 5 / 6)  # 3 niveles/eje -> 3**3 = 27, mismo presupuesto para los 3 brazos
N_EJES = 3
N_TRIALS = len(NIVELES) ** N_EJES  # 27

# DELTA se fija antes de correr, nunca a partir de los resultados: es el regret de errar
# la coordenada del eje importante por el 1% de su rango, 0.01**2.
DELTA = 0.01**2

# QMCSampler saca su primer trial de un RandomSampler; con la misma semilla que el brazo
# random, ese trial sería idéntico al de random. Este offset separa las dos series.
SEMILLA_QMC_INDEPENDIENTE = 10_000

# %% [markdown]
# ## El aviso de QMCSampler
#
# `QMCSampler` es experimental en Optuna 5 y avisa con `ExperimentalWarning` al
# instanciarlo. El paso atrapa ese aviso una sola vez e imprime una nota, en vez de
# silenciarlo o repetirlo en cada réplica.

# %%
with warnings.catch_warnings(record=True) as avisos_qmc:
    # "always" sobre todos los avisos, para re-emitir abajo cualquier otro aviso de
    # QMCSampler en vez de perderlo.
    warnings.simplefilter("always", Warning)
    optuna.samplers.QMCSampler(qmc_type="sobol", scramble=True, seed=0)
# Re-emite tal cual cualquier aviso que no sea el ExperimentalWarning esperado, para que
# bajo -W error se vuelva una excepción visible.
for aviso in avisos_qmc:
    if not issubclass(aviso.category, optuna.exceptions.ExperimentalWarning):
        warnings.warn_explicit(aviso.message, aviso.category, aviso.filename, aviso.lineno)
if any(issubclass(a.category, optuna.exceptions.ExperimentalWarning) for a in avisos_qmc):
    print(
        "Nota: QMCSampler es experimental en Optuna 5.0.0 (ver optuna/samplers/_qmc.py); "
        "el paso lo usa a propósito como brazo de diseño de experimentos, no de producción.\n"
    )


def objetivo(paisaje):
    """Cierra sobre `paisaje`: el objective de Optuna solo puede recibir `trial`."""

    def _fn(trial: optuna.Trial) -> float:
        params = {f"x{i}": trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(paisaje.n_ejes)}
        return paisaje.regret(params)

    return _fn


def _texto_rango(valores: list[int]) -> str:
    """'27' si todas las réplicas dieron el mismo número, '3-9' si varió."""
    bajo, alto = min(valores), max(valores)
    return str(bajo) if bajo == alto else f"{bajo}-{alto}"


# %% [markdown]
# ## Los tres brazos, réplica por réplica
#
# Cada réplica arma un paisaje nuevo (eje importante y óptimo al azar) y corre los tres
# estudios con el mismo presupuesto de 27 evaluaciones:
# - **grid**: `GridSampler` con una rejilla 3x3x3 (niveles 1/6, 1/2, 5/6 por eje).
# - **random**: `RandomSampler` sembrado, 27 trials.
# - **qmc**: `QMCSampler` (Sobol, `scramble=True`), 27 trials. 27 no es potencia de 2,
#   así que pierde algo de la baja discrepancia de Sobol, y su primer trial sale de su
#   `independent_sampler`.

# %%
mejor_grid: dict[int, float] = {}
mejor_random: dict[int, float] = {}
mejor_qmc: dict[int, float] = {}

n_trials_grid: list[int] = []
n_trials_random: list[int] = []
n_trials_qmc: list[int] = []
distintos_grid: list[int] = []
distintos_random: list[int] = []
distintos_qmc: list[int] = []
grid_alcanza_minimo = 0

for r in range(R):
    paisaje = crear_paisaje(r)
    eje = paisaje.eje_importante
    fn = objetivo(paisaje)

    espacio_grid = {f"x{i}": list(NIVELES) for i in range(paisaje.n_ejes)}
    estudio_grid = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.GridSampler(espacio_grid, seed=r)
    )
    estudio_grid.optimize(fn, n_trials=N_TRIALS)
    mejor_grid[r] = estudio_grid.best_value
    n_trials_grid.append(len(estudio_grid.trials))
    distintos_grid.append(len({t.params[f"x{eje}"] for t in estudio_grid.trials}))
    minimo_grid = regret_minimo_grid(paisaje, NIVELES)
    # La rejilla cubre cada nivel del eje importante, así que su mejor regret debe
    # coincidir con el mínimo analítico (con math.isclose, por el punto flotante).
    if math.isclose(estudio_grid.best_value, minimo_grid, rel_tol=1e-9, abs_tol=1e-12):
        grid_alcanza_minimo += 1

    estudio_random = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=r)
    )
    estudio_random.optimize(fn, n_trials=N_TRIALS)
    mejor_random[r] = estudio_random.best_value
    n_trials_random.append(len(estudio_random.trials))
    distintos_random.append(len({t.params[f"x{eje}"] for t in estudio_random.trials}))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler_qmc = optuna.samplers.QMCSampler(
            qmc_type="sobol",
            scramble=True,
            seed=r,
            independent_sampler=optuna.samplers.RandomSampler(seed=SEMILLA_QMC_INDEPENDIENTE + r),
        )
    estudio_qmc = optuna.create_study(direction="minimize", sampler=sampler_qmc)
    estudio_qmc.optimize(fn, n_trials=N_TRIALS)
    mejor_qmc[r] = estudio_qmc.best_value
    n_trials_qmc.append(len(estudio_qmc.trials))
    distintos_qmc.append(len({t.params[f"x{eje}"] for t in estudio_qmc.trials}))

# %% [markdown]
# ## Las cuatro respuestas
#
# `delta = 0.01**2 = 1e-4` es una constante del problema. "mejor" y "peor" salen del
# intervalo de confianza 95% sin mirar delta; delta solo interviene cuando ese intervalo
# cruza el cero, para separar "equivalente" (el intervalo de las dos pruebas
# unilaterales, TOST, cae dentro de `(-delta, +delta)`) de "no concluyente".

# %%
victorias_random = sum(1 for r in range(R) if mejor_random[r] < mejor_grid[r])
ic1_bajo, ic1_alto = ic_proporcion(victorias_random, R)

# Con direccion="minimizar", `diferencia_pareada` invierte el signo para que "mejor"
# signifique "a gana". Por eso `a` es random: la diferencia queda en unidades
# "grid - random" y el veredicto describe a random frente a grid.
comparacion = diferencia_pareada(mejor_random, mejor_grid, delta=DELTA, direccion="minimizar")

# Mismo patrón que `comparacion`, pero con `a` = qmc: así el veredicto
# describe a QMC frente a cada rival y `diferencia_media` queda en unidades
# "rival - qmc" (positiva cuando el rival es peor).
qmc_vs_random = diferencia_pareada(mejor_qmc, mejor_random, delta=DELTA, direccion="minimizar")
qmc_vs_grid = diferencia_pareada(mejor_qmc, mejor_grid, delta=DELTA, direccion="minimizar")

print()
print(
    f"1. Random vence a grid en {victorias_random} de {R} réplicas "
    f"({100 * victorias_random / R:.1f}%, intervalo de confianza 95% Wilson "
    f"[{100 * ic1_bajo:.1f}%, {100 * ic1_alto:.1f}%])."
)
print(
    f"2. Diferencia pareada media de regret (grid - random) = {comparacion.diferencia_media:.2e} "
    f"[{comparacion.ic_bajo:.2e}, {comparacion.ic_alto:.2e}] con delta={comparacion.delta:.1e} "
    f"(regret de errar el eje importante por {100 * math.sqrt(comparacion.delta):.0f}% "
    f"de su rango) -> random resulta "
    f"{comparacion.veredicto} frente a grid (p={comparacion.p_valor:.2g})."
)
texto_trials = _texto_rango(n_trials_grid + n_trials_random + n_trials_qmc)
# El paso solo afirma la razón si de verdad la midió: grid probó exactamente
# len(NIVELES) valores del eje importante y alcanzó el mínimo analítico en todas las
# réplicas.
razon_grid = (
    f": agota sus {len(NIVELES)} niveles del eje importante, "
    "pero nunca prueba otro valor de ese eje."
    if set(distintos_grid) == {len(NIVELES)} and grid_alcanza_minimo == R
    else "."
)
print(
    f"3. En cada réplica cada brazo corrió {texto_trials} "
    f"trials ({N_TRIALS} pedidos); valores distintos del eje importante: "
    f"grid {_texto_rango(distintos_grid)}, random {_texto_rango(distintos_random)}, "
    f"qmc {_texto_rango(distintos_qmc)}. Grid alcanzó el mínimo analítico de su propia "
    f"rejilla (regret_minimo_grid) en {grid_alcanza_minimo} de {R} réplicas{razon_grid}"
)
nota_qmc = (
    f" Con estas {R} réplicas, QMC no se distingue de random."
    if qmc_vs_random.veredicto == "no concluyente"
    else ""
)
print(
    f"4. QMC frente a random: {qmc_vs_random.veredicto} "
    f"(random - qmc = {qmc_vs_random.diferencia_media:.2e} "
    f"[{qmc_vs_random.ic_bajo:.2e}, {qmc_vs_random.ic_alto:.2e}], "
    f"p={qmc_vs_random.p_valor:.2g}); frente a grid: {qmc_vs_grid.veredicto} "
    f"(grid - qmc = {qmc_vs_grid.diferencia_media:.2e} "
    f"[{qmc_vs_grid.ic_bajo:.2e}, {qmc_vs_grid.ic_alto:.2e}], "
    f"p={qmc_vs_grid.p_valor:.2g}).{nota_qmc}"
)

# %% [markdown]
# ## Figura: regret de grid vs random por réplica, escala log

# %%
valores_grid = [mejor_grid[r] for r in range(R)]
valores_random = [mejor_random[r] for r in range(R)]
if min(valores_grid) <= 0.0 or min(valores_random) <= 0.0:
    # Un regret exactamente 0 no cabe en escala log (log(0) = -inf):
    # mejor fallar explícito que recortarlo en silencio y perder el punto.
    raise ValueError(
        "un regret exactamente 0 no cabe en escala log "
        f"(mínimo grid={min(valores_grid)!r}, mínimo random={min(valores_random)!r})"
    )
fig = Figure(figsize=(5.5, 5.5))
ax = fig.subplots()
ax.scatter(valores_grid, valores_random, alpha=0.6, edgecolors="none")
limite_bajo = 0.5 * min(valores_grid + valores_random)
limite_alto = 2.0 * max(valores_grid + valores_random)
ax.plot([limite_bajo, limite_alto], [limite_bajo, limite_alto], linestyle="--", color="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(limite_bajo, limite_alto)
ax.set_ylim(limite_bajo, limite_alto)
ax.set_xlabel("regret grid (mejor de 27)")
ax.set_ylabel("regret random (mejor de 27)")
ax.set_title(f"Paso 01: grid vs random, {R} réplicas\n(bajo la diagonal: random gana)")
fig.tight_layout()
figura = RESULTADOS / "01_grid_vs_random.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
