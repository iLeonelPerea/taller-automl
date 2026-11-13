"""Prueba de regresión sobre la regla central de tests/conftest.py: sin
tarjeta de video, una prueba marcada `gpu` FALLA, nunca se salta, y una
prueba sin la marca no se ve afectada aunque su id de parametrize sea
exactamente "gpu".

Esta prueba copia el conftest.py real (nunca una reescritura a mano: una
copia manual podría divergir del archivo real y dejar de proteger nada)
junto con un archivo de prueba mínimo a un mini-proyecto en `tmp_path`, y
corre pytest en un subproceso propio con `CUDA_VISIBLE_DEVICES=""` -- así la
prueba funciona igual con o sin una tarjeta de video real en la máquina que
la ejecuta.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
CONFTEST_REAL = RAIZ / "tests" / "conftest.py"

_ARCHIVO_DE_PRUEBA = """
import pytest


@pytest.mark.gpu
def test_marcada_gpu():
    assert True


@pytest.mark.parametrize("valor", ["x"], ids=["gpu"])
def test_sin_marca_con_id_de_parametrize_gpu(valor):
    assert valor == "x"
"""


@pytest.fixture(scope="module")
def resultado_mini_suite(tmp_path_factory) -> subprocess.CompletedProcess:
    """Este fixture corre, una sola vez por módulo, el conftest.py real
    contra el archivo de prueba mínimo de arriba, en un mini-proyecto propio
    (no el repo: un conftest.py adicional en `tmp_path` no debe mezclarse con
    el real) y con `CUDA_VISIBLE_DEVICES=""` en el entorno del subproceso."""
    destino = tmp_path_factory.mktemp("mini_suite_gpu")
    shutil.copy(CONFTEST_REAL, destino / "conftest.py")
    (destino / "test_mini.py").write_text(_ARCHIVO_DE_PRUEBA, encoding="utf-8")
    entorno = dict(os.environ)
    entorno["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "."],
        cwd=destino,
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_prueba_marcada_gpu_falla_sin_tarjeta_de_video_en_vez_de_saltarse(resultado_mini_suite):
    """Si conftest.py cambiara `pytest.fail` por `pytest.skip`, esta prueba lo
    nota: la marcada `gpu` pasaría a salir "skipped" (código 0 para pytest) en
    vez de fallar con el mensaje claro. `pytest.fail` disparado en el setup de
    una prueba pytest lo cuenta como "error" (la fase donde ocurre, no el
    mensaje, decide esa palabra), nunca como "skipped" -- eso es lo que
    protege esta prueba, junto con el código de salida distinto de 0."""
    assert resultado_mini_suite.returncode != 0, resultado_mini_suite.stdout
    assert "1 error" in resultado_mini_suite.stdout, resultado_mini_suite.stdout
    assert "skipped" not in resultado_mini_suite.stdout, resultado_mini_suite.stdout
    assert MENSAJE_SIN_TARJETA in resultado_mini_suite.stdout, resultado_mini_suite.stdout


def test_prueba_con_id_de_parametrize_gpu_pero_sin_la_marca_pasa_normal(resultado_mini_suite):
    """Si conftest.py mirara `"gpu" in item.keywords` en vez de la marca
    real, esta prueba también fallaría sin tarjeta de video: su id de
    parametrize es exactamente "gpu", pero la prueba en sí no lleva
    `@pytest.mark.gpu` y no necesita ninguna tarjeta de video."""
    assert "1 passed" in resultado_mini_suite.stdout, resultado_mini_suite.stdout
