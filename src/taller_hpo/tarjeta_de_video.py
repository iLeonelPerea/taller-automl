"""El aviso que ves cuando tu máquina no tiene una tarjeta de video.

El taller entrena siempre en una tarjeta de video NVIDIA con CUDA, y cada módulo que la
exige toma el mensaje de aquí. Este módulo no importa torch, así que la suite puede
cargarlo en una máquina sin tarjeta de video.
"""

from __future__ import annotations

MENSAJE_SIN_TARJETA = (
    "el taller necesita una tarjeta de video NVIDIA con CUDA y el controlador "
    "instalado; aquí no hay ninguna disponible"
)
