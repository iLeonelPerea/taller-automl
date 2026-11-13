"""El contrato compartido del MLP del juguete: el problema, los hiperparámetros por
defecto y el error de divergencia.

El motor del taller (`paridad_torch`, en la tarjeta de video) y el modelo de referencia de
las pruebas (`mlp_paridad`, con scikit-learn) parten de este mismo `DEFAULTS` y de la
misma `Divergencia`. Este módulo no importa scikit-learn, así que el motor no depende de
su modelo de referencia.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    # El problema: fijo durante una búsqueda, no son hiperparámetros. Con d=3 y 10
    # columnas de ruido el techo es 0.825, la configuración base promedia 0.70 y una
    # buena configuración llega a 0.81, así que buscar paga y queda espacio para la
    # maldición del ganador.
    "d": 3,
    "a": 1.5,
    "k_ruido": 10,
    "rotar": False,
    "n_train": 2000,
    "n_val": 500,
    # Hiperparámetros.
    "n_capas": 1,
    "ancho": 64,
    "activacion": "relu",
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
    "batch_size": 64,
    "epocas": 30,
    # Control negativo: el entrenamiento NO la lee. Si una búsqueda le asigna
    # importancia, esa es la importancia que produce el puro ruido.
    "placebo": 0.0,
    # Oráculo: muestras frescas de x para el riesgo verdadero.
    "n_oraculo": 200_000,
    # Conjunto de prueba finito, lo que tendría un tesista sin oráculo (0 = sin prueba).
    # Sale de SEMILLA_PRUEBA y se abre una sola vez con honestidad.PruebaSellada.
    "n_prueba": 0,
}

# Semilla fija del oráculo: todos los modelos se evalúan sobre las mismas x, y así
# las diferencias de exactitud_verdadera no traen ruido Monte Carlo propio.
SEMILLA_ORACULO = 20_260_924
# Semilla fija del conjunto de prueba: la misma prueba para todos los modelos.
SEMILLA_PRUEBA = 20_260_925


class Divergencia(RuntimeError):
    """El entrenamiento produjo pesos no finitos (típico con learning rate enorme)."""


__all__ = ["DEFAULTS", "Divergencia", "SEMILLA_ORACULO", "SEMILLA_PRUEBA"]
