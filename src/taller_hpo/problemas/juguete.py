"""El punto por el que los pasos del taller entrenan el juguete.

Los pasos que entrenan el MLP del juguete lo hacen a través de este módulo, sin importar
`paridad_torch` directo. El paso 01 usa `problemas.analitico` y el 05 usa
`problemas.cash_paridad`.

Sin una tarjeta de video NVIDIA con CUDA visible, importar este módulo falla con un
`RuntimeError` claro. `entrenar(config, semillas, reportar=None)` tiene el mismo contrato
que `paridad_torch.entrenar` y solo completa `"dispositivo"` cuando la config no lo trae.

`PROTOCOLO_MOTOR` es la parte del motor que entra a la huella del estudio
(`estudio.crear_estudio` rechaza reabrir un estudio con otro protocolo), y siempre vale
`{"motor": "torch", "dispositivo": "cuda"}`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from taller_hpo.problemas import paridad_torch
from taller_hpo.problemas.mlp_contrato import SEMILLA_ORACULO, SEMILLA_PRUEBA, Divergencia
from taller_hpo.semillas import Semillas
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

Reportar = Callable[[int, float], bool]

HIPERPARAMETROS: tuple[str, ...] = (
    "n_capas",
    "ancho",
    "activacion",
    "learning_rate_init",
    "alpha",
    "batch_size",
)

if not torch.cuda.is_available():
    raise RuntimeError(MENSAJE_SIN_TARJETA)

MOTOR = "torch"
DISPOSITIVO = "cuda"

DEFAULTS: dict[str, Any] = {**paridad_torch.DEFAULTS, "dispositivo": DISPOSITIVO}
DESCRIPCION = f"PyTorch {torch.__version__} en {DISPOSITIVO} ({torch.cuda.get_device_name()})"


def _protocolo_motor(motor: str, dispositivo: str) -> dict[str, Any]:
    """Arma `PROTOCOLO_MOTOR`, con cualquier `"cuda:N"` normalizado a `"cuda"`."""
    return {"motor": motor, "dispositivo": dispositivo.split(":", 1)[0]}


PROTOCOLO_MOTOR: dict[str, Any] = _protocolo_motor(MOTOR, DISPOSITIVO)


def entrenar(
    config: dict[str, Any], semillas: Semillas, reportar: Reportar | None = None
) -> dict[str, float]:
    """Entrena el MLP del juguete en la tarjeta de video. Mismo contrato que `paridad_torch`."""
    if "dispositivo" not in config:
        config = {**config, "dispositivo": DISPOSITIVO}
    return paridad_torch.entrenar(config, semillas, reportar)


__all__ = [
    "DEFAULTS",
    "DESCRIPCION",
    "DISPOSITIVO",
    "Divergencia",
    "HIPERPARAMETROS",
    "MOTOR",
    "PROTOCOLO_MOTOR",
    "SEMILLA_ORACULO",
    "SEMILLA_PRUEBA",
    "entrenar",
]
