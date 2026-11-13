"""Configuración explícita: defaults + overrides validados.

Un hiperparámetro que vive como constante de módulo (`from config import LR`)
queda congelado al importar, y ninguna búsqueda lo puede variar. Aquí la
configuración es un dict que se construye en cada llamada. Además, una clave mal
escrita es un error y no un valor ignorado en silencio.
"""

from __future__ import annotations

import difflib
import numbers
from typing import Any


def get_config(defaults: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Copia de `defaults` con `overrides` aplicados y convertidos al tipo del default.

    KeyError si una clave no existe en `defaults` (con sugerencia si se parece a
    una que sí existe). TypeError si el valor no es compatible con el tipo del
    default.
    """
    desconocidas = sorted(set(overrides) - set(defaults))
    if desconocidas:
        detalles = []
        for clave in desconocidas:
            parecidas = difflib.get_close_matches(clave, list(defaults), n=1)
            sugerencia = f" (¿quisiste decir {parecidas[0]!r}?)" if parecidas else ""
            detalles.append(f"{clave!r}{sugerencia}")
        raise KeyError("claves desconocidas: " + ", ".join(detalles))

    config = dict(defaults)
    for clave, valor in overrides.items():
        config[clave] = _convertir(clave, valor, defaults[clave])
    return config


def _convertir(clave: str, valor: Any, referencia: Any) -> Any:
    def _error() -> TypeError:
        tipo = type(referencia).__name__
        return TypeError(f"{clave!r} espera {tipo}, recibió {valor!r} ({type(valor).__name__})")

    # bool va primero: en Python, bool es subclase de int.
    if isinstance(referencia, bool):
        if not isinstance(valor, bool):
            raise _error()
        return valor
    if isinstance(referencia, numbers.Integral):
        if isinstance(valor, bool):
            raise _error()
        if isinstance(valor, numbers.Integral):
            return int(valor)
        if isinstance(valor, numbers.Real) and float(valor).is_integer():
            return int(valor)
        raise _error()
    if isinstance(referencia, numbers.Real):
        if isinstance(valor, bool) or not isinstance(valor, numbers.Real):
            raise _error()
        return float(valor)
    if isinstance(referencia, str):
        if not isinstance(valor, str):
            raise _error()
        return valor
    return valor
