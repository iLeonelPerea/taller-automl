"""Paisaje analítico sin modelos que entrenar: por qué la rejilla pierde contra el azar.

Bergstra y Bengio (2012) muestran que, con el mismo presupuesto, una búsqueda al azar
supera a una rejilla cuando solo unas pocas perillas importan. Con 27 evaluaciones y 3
ejes, una rejilla 3x3x3 prueba solo 3 valores del eje que importa, y 27 puntos al azar
prueban 27 valores distintos de ese eje.

Aquí el regret (qué tan lejos del óptimo queda la mejor configuración probada) sale de
una fórmula cerrada, así que el experimento se repite con muchas réplicas y la
afirmación se mide como una distribución.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


def _validar_n_ejes(n_ejes: int) -> None:
    if not isinstance(n_ejes, int) or n_ejes < 1:
        raise ValueError(f"n_ejes debe ser un entero >= 1, no {n_ejes!r}")


@dataclass(frozen=True)
class Paisaje:
    """Un paisaje de regret con un único eje importante entre `n_ejes`.

    Los demás ejes son "placebo": están en el espacio de búsqueda pero el
    regret no los lee, igual que la perilla `placebo` de `mlp_paridad`.
    """

    n_ejes: int
    eje_importante: int
    optimo: float

    def __post_init__(self) -> None:
        """Valida que `n_ejes` >= 1, que `eje_importante` esté en [0, n_ejes) y que
        `optimo` esté en [0, 1]."""
        _validar_n_ejes(self.n_ejes)
        if not isinstance(self.eje_importante, int) or not 0 <= self.eje_importante < self.n_ejes:
            raise ValueError(
                f"eje_importante debe ser un entero en [0, {self.n_ejes}), "
                f"no {self.eje_importante!r}"
            )
        if not 0.0 <= self.optimo <= 1.0:
            raise ValueError(f"optimo debe estar en [0, 1], no {self.optimo!r}")

    def regret(self, params: Mapping[str, float]) -> float:
        """(x[eje_importante] - optimo)**2. Los demás ejes de `params` no afectan.

        Claves faltantes o sobrantes respecto de `espacio()`, o un valor fuera
        de [0, 1], son un error: no un 0 silencioso ni un clip.
        """
        esperadas = set(self.espacio())
        recibidas = set(params)
        if recibidas != esperadas:
            faltantes = sorted(esperadas - recibidas)
            sobrantes = sorted(recibidas - esperadas)
            detalles = []
            if faltantes:
                detalles.append(f"faltan {faltantes}")
            if sobrantes:
                detalles.append(f"sobran {sobrantes}")
            raise ValueError("params inválido: " + "; ".join(detalles))
        for clave, valor in params.items():
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{clave!r} debe estar en [0, 1], no {valor!r}")
        x_importante = params[f"x{self.eje_importante}"]
        return (x_importante - self.optimo) ** 2

    def espacio(self) -> dict[str, tuple[float, float]]:
        """{"x0": (0.0, 1.0), ..., "x{n_ejes - 1}": (0.0, 1.0)}."""
        return {f"x{i}": (0.0, 1.0) for i in range(self.n_ejes)}


def crear_paisaje(semilla: int, *, n_ejes: int = 3) -> Paisaje:
    """Un `Paisaje` con eje importante y óptimo al azar, derivados de `semilla`.

    El óptimo cambia con la semilla (uniforme en [0.05, 0.95]) para que cada réplica
    aporte algo distinto.
    """
    _validar_n_ejes(n_ejes)  # antes de rng.integers, para dar un mensaje propio
    rng = np.random.default_rng(semilla)
    eje_importante = int(rng.integers(n_ejes))
    optimo = float(rng.uniform(0.05, 0.95))
    return Paisaje(n_ejes=n_ejes, eje_importante=eje_importante, optimo=optimo)


def regret_minimo_grid(paisaje: Paisaje, niveles: Sequence[float]) -> float:
    """Mejor regret alcanzable por una rejilla con `niveles` en cada eje.

    El regret solo depende del eje importante, así que basta el mínimo de la fórmula
    sobre `niveles`. `niveles` vacío o con valores fuera de [0, 1] es un error.
    """
    niveles = list(niveles)
    if not niveles:
        raise ValueError("niveles no puede estar vacío")
    for nivel in niveles:
        if not 0.0 <= nivel <= 1.0:
            raise ValueError(f"cada nivel debe estar en [0, 1], no {nivel!r}")
    return min((nivel - paisaje.optimo) ** 2 for nivel in niveles)
