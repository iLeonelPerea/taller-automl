"""Semillas separadas por fuente de azar: datos, orden de los minibatches e inicialización.

Separarlas permite, por ejemplo, fijar los datos y variar solo la inicialización,
que es lo que puede hacer un tesista con un dataset fijo. Con una sola semilla
para todo, esas fuentes de variación quedan mezcladas y no se pueden separar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Semillas:
    datos: int
    orden: int
    init: int


def semillas_de(base: int, n: int, *, datos: int | None = None) -> list[Semillas]:
    """n juegos de semillas independientes derivados de `base` con numpy.SeedSequence.

    Con `datos` fijo, todos los juegos comparten los mismos datos y solo cambian el
    orden y la inicialización.
    """
    if n < 1:
        raise ValueError(f"n debe ser >= 1, no {n}")
    juegos = []
    for hijo in np.random.SeedSequence(base).spawn(n):
        s_datos, s_orden, s_init = (int(v) for v in hijo.generate_state(3))
        juegos.append(
            Semillas(datos=s_datos if datos is None else datos, orden=s_orden, init=s_init)
        )
    return juegos
