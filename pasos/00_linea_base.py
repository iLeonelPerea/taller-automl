# %% [markdown]
# # Paso 00: tu línea base, su ruido y el techo
#
# Antes de buscar nada: ¿cuánto vale tu configuración actual y cuánto varía?
# Aquí "tu modelo" es el MLP del juguete con sus defaults, tal como lo dejarías
# si nunca lo ajustaras, entrenado en la tarjeta de video (la línea impresa
# abajo lo confirma). Todavía no hay Optuna.
#
# Tres preguntas:
# 1. ¿Cuánto cambia el resultado solo por la semilla?
# 2. ¿La exactitud de validación dice la verdad?
# 3. ¿Cuánto falta para el mejor resultado posible?
#
# Solo puedes contestar las preguntas 2 y 3 porque el juguete tiene oráculo:
# conoces la distribución real y su error de Bayes. En tu tesis no tendrás
# oráculo, y por eso este paso muestra qué es lo que NO ves.
#
# Correr:  uv run pasos/00_linea_base.py  (necesita una tarjeta de video NVIDIA
#          con CUDA y el controlador instalado; sin ella, el paso se detiene
#          con un mensaje claro)
# Presupuesto:  PRESUPUESTO=humo | taller (default) | completo

# %%
import os
import statistics as st
from pathlib import Path

# Figure y no pyplot: guarda el PNG sin backend gráfico, igual en una terminal remota o
# en Docker.
from matplotlib.figure import Figure
from scipy import stats as scipy_stats

from taller_hpo.estadistica import resumen
from taller_hpo.problemas import juguete
from taller_hpo.problemas.paridad import error_bayes
from taller_hpo.semillas import semillas_de

PRESUPUESTOS = {"humo": 3, "taller": 10, "completo": 30}
presupuesto = os.environ.get("PRESUPUESTO", "taller")
if presupuesto not in PRESUPUESTOS:
    raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
N_SEMILLAS = PRESUPUESTOS[presupuesto]

RESULTADOS = Path(
    os.environ.get("TALLER_RESULTADOS", Path(__file__).resolve().parent.parent / "resultados")
)
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Primero, en qué tarjeta de video entrena esta corrida.
print(f"Motor: {juguete.DESCRIPCION}\n")

# %% [markdown]
# ## Tu configuración
#
# Los defaults tal cual: lo que "siempre usas".

# %%
LINEA_BASE: dict = {}
print("Línea base:", {k: juguete.DEFAULTS[k] for k in [*juguete.HIPERPARAMETROS, "epocas"]})
print(
    f"Problema: paridad de d={juguete.DEFAULTS['d']} bits, a={juguete.DEFAULTS['a']}, "
    f"{juguete.DEFAULTS['k_ruido']} columnas de ruido, "
    f"{juguete.DEFAULTS['n_train']} datos de entrenamiento, "
    f"{juguete.DEFAULTS['n_val']} de validación\n"
)

# %% [markdown]
# ## Mismo código, mismos datos de origen, solo cambian las semillas
#
# `semillas_de(base, n)` sin fijar `datos` varía las TRES fuentes de azar a la
# vez: qué datos ve el modelo, en qué orden y con qué inicialización (ver
# `taller_hpo.semillas`). Eso responde "¿cuánto varía mi resultado si repito
# el experimento desde cero?", pero mezcla el remuestreo de los datos con el
# ruido de entrenar -- un tesista con un dataset ya recolectado no puede
# variar eso. Más abajo el paso mide también la franja sin remuestrear los
# datos.

# %%
juegos = semillas_de(0, N_SEMILLAS)
corridas = []
for i, semillas in enumerate(juegos):
    m = juguete.entrenar(LINEA_BASE, semillas)
    corridas.append(m)
    print(
        f"semilla {i:>2}: validación {m['exactitud_val']:.3f} | "
        f"verdadera {m['exactitud_verdadera']:.3f} | {m['segundos']:.2f} s"
    )

# %% [markdown]
# ## La franja que de verdad ve un tesista con un dataset fijo
#
# Con `datos` fijo y solo `orden` e `init` variando (`semillas_de(..., datos=...)`), el
# paso mide lo que puede cambiar sin recolectar datos nuevos. La corrida i de abajo
# comparte `orden` e `init` con la corrida i de arriba, así que las dos franjas quedan
# emparejadas. La corrida 0 es la misma en las dos, y el paso la reusa.

# %%
datos_fijos = juegos[0].datos
juegos_fijos = semillas_de(0, N_SEMILLAS, datos=datos_fijos)
verdad_datos_fijos = [corridas[0]["exactitud_verdadera"]] + [
    juguete.entrenar(LINEA_BASE, s)["exactitud_verdadera"] for s in juegos_fijos[1:]
]

# %% [markdown]
# ## Cuál franja es más ancha, y si se mueven juntas
#
# Las dos franjas están emparejadas por orden e init (la corrida i de una
# comparte esas dos semillas con la corrida i de la otra), así que no son
# muestras independientes y una F de varianzas no aplica. Pitman-Morgan sí:
# mira la correlación entre la suma y la diferencia de cada par, cuyo signo
# dice cuál franja varía más sin asumir independencia. La correlación entre
# franjas contesta algo distinto -- si el efecto de compartir orden+init
# persiste al cambiar los datos, o sea si el resultado de una corrida se
# parece al de su pareja aunque los datos sean otros.


# %%
def veredictos_franjas(*, total: list[float], fijos: list[float]) -> tuple[str, str]:
    """¿Es más ancha la franja `fijos` (datos fijos) o `total` (las tres semillas
    variando)? ¿Se mueven juntas? Con pocas semillas, dos desviaciones estándar muestrales
    difieren bastante aunque las varianzas de fondo sean iguales, así que la función usa
    pruebas en vez de comparar las "de". Solo usa las corridas 1..N-1, porque la corrida 0
    es idéntica en las dos franjas.
    """
    n = len(total)
    pares = range(1, n)
    n_pares = len(pares)
    if n_pares < 3:
        veredicto_ancho = (
            f"no concluyente: con {n_pares} pares no alcanza para decidir cuál franja es más ancha"
        )
        veredicto_junto = (
            f"no concluyente: con {n_pares} pares no alcanza para medir la correlación"
        )
        return veredicto_ancho, veredicto_junto
    # Pitman-Morgan para varianzas pareadas: Cov(a + b, a - b) = Var(a) -
    # Var(b), así que el signo de la correlación entre la suma y la
    # diferencia de cada par dice cuál de las dos tiene más varianza, sin
    # asumir nada de antemano. a = franja de datos fijos, b = franja total.
    sumas = [fijos[i] + total[i] for i in pares]
    diferencias_pm = [fijos[i] - total[i] for i in pares]
    if len(set(diferencias_pm)) == 1:
        # Diferencias constantes: Cov(a+b, a-b) no está definida y pearsonr daría NaN.
        veredicto_ancho = (
            "no concluyente: las diferencias entre franjas son constantes, Pitman-Morgan no aplica"
        )
    else:
        pm = scipy_stats.pearsonr(sumas, diferencias_pm)
        if pm.pvalue >= 0.05:
            veredicto_ancho = (
                f"no concluyente: con {n_pares} pares los datos no distinguen cuál franja es "
                f"más ancha (Pitman-Morgan, p={pm.pvalue:.3f})"
            )
        elif pm.statistic > 0.0:
            veredicto_ancho = (
                f"la franja de datos fijos es más ancha (Pitman-Morgan, p={pm.pvalue:.3f})"
            )
        else:
            veredicto_ancho = (
                f"la franja de datos fijos es más angosta (Pitman-Morgan, p={pm.pvalue:.3f})"
            )
    # Las dos franjas comparten orden e init índice a índice: una correlación
    # que no cruza el cero dice que ese par de semillas también influye, no
    # que decida por sí solo (positiva es compatible con que los datos también
    # aporten a la diferencia).
    correlacion = scipy_stats.pearsonr([total[i] for i in pares], [fijos[i] for i in pares])
    ic_corr = correlacion.confidence_interval(confidence_level=0.95)
    if ic_corr.low > 0.0:
        veredicto_junto = (
            f"se mueven juntas (r={correlacion.statistic:.3f}, intervalo de confianza 95% "
            f"[{ic_corr.low:.3f}, {ic_corr.high:.3f}]): con el mismo orden+init el resultado "
            "se parece aunque cambien los datos"
        )
    elif ic_corr.high < 0.0:
        veredicto_junto = (
            f"se mueven en sentido contrario (r={correlacion.statistic:.3f}, intervalo de "
            f"confianza 95% [{ic_corr.low:.3f}, {ic_corr.high:.3f}]): con el mismo orden+init, "
            "cuando una franja sube la otra baja"
        )
    else:
        veredicto_junto = (
            f"no concluyente (r={correlacion.statistic:.3f}, intervalo de confianza 95% "
            f"[{ic_corr.low:.3f}, {ic_corr.high:.3f}])"
        )
    return veredicto_ancho, veredicto_junto


# %% [markdown]
# ## Las tres respuestas


# %%
def desvio_validacion(val: list[float], verdad: list[float]) -> tuple[float, float]:
    """Cuánto se desvía la exactitud de validación de la verdadera (oráculo): la media y
    el máximo del valor absoluto de la diferencia, pareadas por corrida. Con signo, una
    desviación hacia arriba y otra hacia abajo se cancelarían.
    """
    errores = [abs(v - t) for v, t in zip(val, verdad, strict=True)]
    return st.mean(errores), max(errores)


# %%
techo = 1.0 - error_bayes(juguete.DEFAULTS["d"], juguete.DEFAULTS["a"])
val = [m["exactitud_val"] for m in corridas]
verdad = [m["exactitud_verdadera"] for m in corridas]
media_error_val, max_error_val = desvio_validacion(val, verdad)
r = resumen(verdad)
r_fijos = resumen(verdad_datos_fijos)
veredicto_ancho, veredicto_junto = veredictos_franjas(total=verdad, fijos=verdad_datos_fijos)

print()
print(
    f"1. Con {r.n} juegos de semillas (datos, orden e init), la exactitud verdadera va de "
    f"{min(verdad):.3f} a {max(verdad):.3f} ({100 * (max(verdad) - min(verdad)):.1f} puntos, "
    f"de={100 * r.de:.1f} puntos) sin cambiar una línea de código. Con los mismos datos y "
    f"solo orden e init variando -- lo único que puede cambiar un tesista con un dataset ya "
    f"fijo -- el rango es de {100 * (max(verdad_datos_fijos) - min(verdad_datos_fijos)):.1f} "
    f"puntos, de={100 * r_fijos.de:.1f} puntos. ¿Cuál franja es más ancha? {veredicto_ancho}. "
    f"¿Se mueven juntas? {veredicto_junto}."
)
print(
    f"2. Con las mismas {r.n} semillas, la validación ({juguete.DEFAULTS['n_val']} datos) se "
    f"desvía de la verdad {100 * media_error_val:.1f} puntos en promedio "
    f"y {100 * max_error_val:.1f} en el peor caso."
)
print(
    f"3. El techo (1 - error de Bayes) es {techo:.3f}. Tu línea base, con {r.n} semillas, "
    f"promedia {r.media:.3f} (intervalo de confianza {100 * r.confianza:.0f}% "
    f"[{r.ic_bajo:.3f}, {r.ic_alto:.3f}]): "
    f"le faltan entre {100 * (techo - r.ic_alto):.1f} y {100 * (techo - r.ic_bajo):.1f} puntos "
    f"(intervalo {100 * r.confianza:.0f}%; {100 * (techo - r.media):.1f} en promedio)."
)

# %%
fig = Figure(figsize=(7, 3.5))
ax = fig.subplots()
posiciones = range(len(corridas))
ax.scatter(posiciones, val, label="validación (lo que ves)", marker="o")
ax.scatter(posiciones, verdad, label="verdadera (oráculo)", marker="x")
ax.axhline(techo, linestyle="--", label=f"techo de Bayes {techo:.3f}")
ax.set_xlabel("semilla")
ax.set_ylabel("exactitud")
ax.set_title("Paso 00: la misma configuración, distintas semillas")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
figura = RESULTADOS / "00_linea_base.png"
fig.savefig(figura, dpi=120)
print(f"\nfigura: {figura}")
