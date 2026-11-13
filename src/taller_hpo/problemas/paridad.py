"""Paridad gaussiana: un problema de juguete con oráculo exacto.

Cada muestra tiene d bits ocultos s_i en {-1, +1}, equiprobables, y se observa
x = a·s + z, con z ~ N(0, I). La etiqueta es la paridad: y = 1 si prod(s_i) = +1.

- El techo se conoce: el error de Bayes es (1 - (1 - 2·Φ(-a))^d) / 2.
- La posterior es exacta: P(y = 1 | x) = ½ · (1 + prod(tanh(a·x_i))).

Con la posterior se calcula el riesgo VERDADERO de cualquier clasificador sin
depender de un conjunto de prueba finito. Así mides cuánto se infla el ganador
de una búsqueda.

Para quien viene de comunicaciones: Φ(-a) es la BER de BPSK en AWGN con
amplitud a y σ = 1, y el error de Bayes es la probabilidad de un número impar
de errores de bit.

Las columnas informativas van primero. Después vienen `k_ruido` columnas N(0, 1)
que no dicen nada de y. `rotar=True` gira 45° cada par de columnas informativas:
el error de Bayes no cambia, pero la frontera deja de estar alineada con los ejes,
y los árboles pierden la ventaja de cortar por ejes.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


def _validar(d: int, a: float, k_ruido: int) -> None:
    if not isinstance(d, int) or d < 1:
        raise ValueError(f"d debe ser un entero >= 1, no {d!r}")
    if not a > 0:
        raise ValueError(f"a debe ser > 0, no {a!r}")
    if not isinstance(k_ruido, int) or k_ruido < 0:
        raise ValueError(f"k_ruido debe ser un entero >= 0, no {k_ruido!r}")


def _rotacion(d: int) -> np.ndarray:
    """Matriz ortogonal que gira 45° los pares de ejes (0,1), (2,3), ...

    Si d es impar, el último eje queda igual.
    """
    r = np.eye(d)
    c = 1.0 / math.sqrt(2.0)
    for i in range(0, d - 1, 2):
        r[i, i], r[i, i + 1] = c, -c
        r[i + 1, i], r[i + 1, i + 1] = c, c
    return r


def generar(
    n: int,
    *,
    d: int = 2,
    a: float = 1.5,
    k_ruido: int = 0,
    rotar: bool = False,
    semilla: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve X de forma (n, d + k_ruido) e y en {0, 1} de forma (n,).

    Con la misma semilla, las columnas informativas y las etiquetas no cambian
    aunque cambie `k_ruido`: el ruido se sortea al final.
    """
    _validar(d, a, k_ruido)
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n debe ser un entero >= 1, no {n!r}")

    rng = np.random.default_rng(semilla)
    s = rng.choice(np.array([-1.0, 1.0]), size=(n, d))
    x = a * s + rng.standard_normal((n, d))
    if rotar:
        x = x @ _rotacion(d).T
    if k_ruido:
        x = np.hstack([x, rng.standard_normal((n, k_ruido))])
    y = (np.prod(s, axis=1) > 0).astype(np.int64)
    return x, y


def error_bayes(d: int, a: float) -> float:
    """Error del mejor clasificador posible: (1 - (1 - 2·Φ(-a))^d) / 2."""
    _validar(d, a, 0)
    p_bit = 0.5 * math.erfc(a / math.sqrt(2.0))  # Φ(-a)
    return (1.0 - (1.0 - 2.0 * p_bit) ** d) / 2.0


def posterior(x: np.ndarray, *, d: int, a: float, rotar: bool = False) -> np.ndarray:
    """P(y = 1 | x) exacta. Usa solo las d primeras columnas: el ruido no informa."""
    _validar(d, a, 0)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < d:
        raise ValueError(f"x debe tener forma (n, >= {d}), no {x.shape}")
    informativas = x[:, :d]
    if rotar:
        informativas = informativas @ _rotacion(d)  # deshace x @ R.T porque R es ortogonal
    return 0.5 * (1.0 + np.prod(np.tanh(a * informativas), axis=1))


@functools.lru_cache(maxsize=8, typed=True)
def _muestras_oraculo(
    n: int, semilla: int, d: int, a: float, k_ruido: int, rotar: bool
) -> tuple[np.ndarray, np.ndarray]:
    """x frescas y su posterior p, cacheadas por (n, semilla, d, a, k_ruido, rotar).

    `typed=True` evita que `n=100.0` reuse la entrada de `n=100` y se salte la
    validación. Los arreglos devueltos son los de la caché, así que van de solo lectura.
    """
    x, _ = generar(n, d=d, a=a, k_ruido=k_ruido, rotar=rotar, semilla=semilla)
    p = posterior(x, d=d, a=a, rotar=rotar)
    x.flags.writeable = False
    p.flags.writeable = False
    return x, p


def riesgo_verdadero(
    predecir: Predictor,
    *,
    d: int = 2,
    a: float = 1.5,
    k_ruido: int = 0,
    rotar: bool = False,
    n: int = 200_000,
    semilla: int,
) -> float:
    """Error esperado de `predecir` sobre la distribución real, no sobre un test finito.

    Promedia P(y != h(x) | x) con la posterior exacta sobre n muestras frescas de x.
    No sortea etiquetas, así que el único error Monte Carlo es el de x: con
    n = 200 000 queda por debajo de 0.0012.
    """
    x, p = _muestras_oraculo(n, semilla, d, a, k_ruido, rotar)
    prediccion = np.asarray(predecir(x))
    if prediccion.shape != (n,):
        raise ValueError(f"predecir debe devolver forma ({n},), no {prediccion.shape}")
    if not np.isin(prediccion, (0, 1)).all():
        raise ValueError("predecir debe devolver etiquetas 0/1")
    return float(np.mean(np.where(prediccion == 1, 1.0 - p, p)))
