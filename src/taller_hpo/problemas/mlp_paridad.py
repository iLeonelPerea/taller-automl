"""El modelo de referencia de las pruebas: un MLP de scikit-learn sobre la paridad
gaussiana, nunca un motor del taller.

El taller entrena siempre en la tarjeta de video con `paridad_torch`. Este módulo existe
para que las pruebas comprueben contra un MLP de scikit-learn conocido que la tarjeta de
video da los resultados correctos. Cumple el mismo contrato que el taller le pide al
modelo del tesista:

    entrenar(config, semillas, reportar=None) -> dict de métricas

- `config`: dict de overrides sobre DEFAULTS; una clave desconocida es un error.
- `semillas`: Semillas(datos, orden, init), una semilla por cada fuente de azar.
- `reportar(epoca, exactitud_val) -> bool`: se llama al final de cada época; si devuelve
  True, el entrenamiento para. Así la poda vive fuera del modelo.

Métricas devueltas:
- `exactitud_val`: lo único que puede ver una búsqueda.
- `exactitud_verdadera`: la exactitud contra la distribución real, calculada con el
  oráculo. Nunca se optimiza: sirve para medir cuánto miente `exactitud_val`.
- `n_parametros`, `epocas`: costo determinista.
- `segundos`: tiempo de reloj; varía entre corridas.

Detalles de scikit-learn que fijan este diseño:
- `partial_fit` vuelve a crear su RandomState en cada llamada, así que con
  `shuffle=True` todas las épocas usarían el mismo orden. Por eso `shuffle=False` y el
  orden se permuta aquí, con su propia semilla.
- `early_stopping=True` con `partial_fit` lanza ValueError: la parada va por `reportar`.
- Si los pesos se vuelven no finitos, sklearn lanza ValueError y este módulo lo convierte
  en `Divergencia`.

Todo el cómputo corre con un solo hilo de BLAS: las matrices son chicas, y varios hilos
solo compiten por núcleos con otros procesos sin cambiar ninguna métrica.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import threadpoolctl
from sklearn.neural_network import MLPClassifier

from taller_hpo.configuracion import get_config
from taller_hpo.problemas.mlp_contrato import DEFAULTS, SEMILLA_ORACULO, SEMILLA_PRUEBA, Divergencia
from taller_hpo.problemas.paridad import generar, riesgo_verdadero
from taller_hpo.semillas import Semillas

Reportar = Callable[[int, float], bool]

__all__ = ["DEFAULTS", "Divergencia", "SEMILLA_ORACULO", "SEMILLA_PRUEBA", "entrenar"]


@functools.cache
def _controlador_hilos() -> threadpoolctl.ThreadpoolController:
    """`ThreadpoolController` cacheado, creado en la primera llamada a `entrenar`."""
    return threadpoolctl.ThreadpoolController()


def entrenar(
    config: dict[str, Any], semillas: Semillas, reportar: Reportar | None = None
) -> dict[str, float]:
    c = get_config(DEFAULTS, **config)
    problema = {k: c[k] for k in ("d", "a", "k_ruido", "rotar")}

    x, y = generar(c["n_train"] + c["n_val"], semilla=semillas.datos, **problema)
    x_tr, y_tr = x[: c["n_train"]], y[: c["n_train"]]
    x_va, y_va = x[c["n_train"] :], y[c["n_train"] :]

    modelo = MLPClassifier(
        hidden_layer_sizes=(c["ancho"],) * c["n_capas"],
        activation=c["activacion"],
        solver="adam",
        alpha=c["alpha"],
        learning_rate_init=c["learning_rate_init"],
        batch_size=c["batch_size"],
        shuffle=False,
        random_state=semillas.init,
    )
    rng_orden = np.random.default_rng(semillas.orden)
    clases = np.array([0, 1])

    # Un solo límite de hilos para todo el cómputo numérico: el límite es global al
    # proceso, así que entrar una vez por época no cambiaría nada.
    with _controlador_hilos().limit(limits=1):
        inicio = time.perf_counter()
        epocas = 0
        exactitud_val = float("nan")
        for epoca in range(c["epocas"]):
            orden = rng_orden.permutation(len(y_tr))
            try:
                modelo.partial_fit(x_tr[orden], y_tr[orden], classes=clases)
            except ValueError as error:
                if "non-finite" in str(error):
                    raise Divergencia(
                        f"pesos no finitos en la época {epoca} con learning_rate_init="
                        f"{c['learning_rate_init']}"
                    ) from error
                raise
            epocas = epoca + 1
            exactitud_val = float(modelo.score(x_va, y_va))
            if reportar is not None and reportar(epoca, exactitud_val):
                break
        segundos = time.perf_counter() - inicio

        riesgo = riesgo_verdadero(
            modelo.predict, n=c["n_oraculo"], semilla=SEMILLA_ORACULO, **problema
        )
        n_parametros = sum(w.size for w in modelo.coefs_) + sum(b.size for b in modelo.intercepts_)
        metricas = {
            "exactitud_val": exactitud_val,
            "exactitud_verdadera": 1.0 - riesgo,
            "n_parametros": float(n_parametros),
            "epocas": float(epocas),
            "segundos": segundos,
        }
        if c["n_prueba"] > 0:
            x_pr, y_pr = generar(c["n_prueba"], semilla=SEMILLA_PRUEBA, **problema)
            metricas["exactitud_prueba"] = float(modelo.score(x_pr, y_pr))
    return metricas
