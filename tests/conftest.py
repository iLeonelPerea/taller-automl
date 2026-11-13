"""Este archivo registra la marca `gpu` y, sin tarjeta de video, hace fallar solo
a las pruebas que ya la llevan.

El taller entrena siempre en una tarjeta de video NVIDIA con CUDA (ver
`docs/arquitectura.md`): toda prueba que necesita la tarjeta de video lleva
`@pytest.mark.gpu` (o un `pytestmark = pytest.mark.gpu` a nivel de módulo,
cuando el archivo entero la necesita). Sin una tarjeta de video visible, esas
pruebas FALLAN aquí con un mensaje claro -- nunca se saltan: un contenedor o
una máquina sin tarjeta de video no debe salir en verde por accidente, y
`pytest.skip` haría justo eso. La integración continua (`.github/workflows/pruebas.yml`),
que no tiene tarjeta de video, corre `pytest -m "not gpu"` para excluirlas a
propósito, en vez de dejarlas fallar una por una.
"""

from __future__ import annotations

import pytest
import torch

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: necesita una tarjeta de video NVIDIA con CUDA (falla, no se salta, sin ella)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    # get_closest_marker mira solo la marca de verdad (el @pytest.mark.gpu de
    # la prueba o el pytestmark de su módulo). "gpu" in item.keywords también
    # es verdadero cuando un id de parametrize, o el nombre de un nodo, es
    # exactamente "gpu", así que una prueba sin la marca podría fallar aquí
    # sin necesitar la tarjeta de video para nada -- justo lo que este
    # chequeo evita.
    if item.get_closest_marker("gpu") is not None and not torch.cuda.is_available():
        pytest.fail(MENSAJE_SIN_TARJETA, pytrace=False)
