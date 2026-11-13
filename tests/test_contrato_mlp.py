"""El contrato compartido del MLP (`taller_hpo.problemas.mlp_contrato`) no carga
scikit-learn.

`juguete`, `paridad_torch` y `cash_paridad` importan `DEFAULTS`, `Divergencia`,
`SEMILLA_ORACULO` y `SEMILLA_PRUEBA` desde `mlp_contrato`, nunca desde `mlp_paridad` (el
modelo de referencia, el único que carga `sklearn`). Las pruebas lo verifican en un
subproceso limpio, porque otro archivo de pruebas pudo haber importado `sklearn` en este
proceso. Las tres primeras revisan el import de cada módulo; la cuarta entrena
`juguete.entrenar` y exige lo mismo, para atrapar un import perezoso dentro del
entrenamiento. Solo las que importan o entrenan `juguete` llevan la marca `gpu`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _correr(codigo: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_importar_cash_paridad_no_carga_sklearn_ni_threadpoolctl():
    codigo = (
        "import sys\n"
        "from taller_hpo.problemas import cash_paridad\n"
        "assert 'sklearn' not in sys.modules, sorted(m for m in sys.modules if 'sklearn' in m)\n"
        "assert 'threadpoolctl' not in sys.modules\n"
        "print('OK')\n"
    )
    resultado = _correr(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


def test_importar_paridad_torch_no_carga_sklearn_ni_threadpoolctl():
    codigo = (
        "import sys\n"
        "from taller_hpo.problemas import paridad_torch\n"
        "assert 'sklearn' not in sys.modules, sorted(m for m in sys.modules if 'sklearn' in m)\n"
        "assert 'threadpoolctl' not in sys.modules\n"
        "print('OK')\n"
    )
    resultado = _correr(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


@pytest.mark.gpu
def test_importar_juguete_no_carga_sklearn_ni_threadpoolctl():
    codigo = (
        "import sys\n"
        "from taller_hpo.problemas import juguete\n"
        "assert 'sklearn' not in sys.modules, sorted(m for m in sys.modules if 'sklearn' in m)\n"
        "assert 'threadpoolctl' not in sys.modules\n"
        "print('OK')\n"
    )
    resultado = _correr(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


@pytest.mark.gpu
def test_entrenar_juguete_no_carga_sklearn_ni_threadpoolctl():
    """Esta prueba entrena de verdad `juguete.entrenar`, la ruta de los pasos
    00-04 y 06 (la prueba de arriba solo revisa el IMPORT de `juguete`, no su
    entrenamiento), y exige que `entrenar` tampoco haya cargado `sklearn` ni
    `threadpoolctl` en el camino (`paridad_torch.entrenar_red`,
    `riesgo_verdadero_en_dispositivo`)."""
    codigo = (
        "import sys\n"
        "from taller_hpo.problemas import juguete\n"
        "from taller_hpo.semillas import Semillas\n"
        "config = {'epocas': 1, 'n_train': 200, 'n_val': 100, 'n_oraculo': 1000, 'ancho': 8}\n"
        "juguete.entrenar(config, Semillas(1, 2, 3))\n"
        "assert 'sklearn' not in sys.modules, sorted(m for m in sys.modules if 'sklearn' in m)\n"
        "assert 'threadpoolctl' not in sys.modules\n"
        "print('OK')\n"
    )
    resultado = _correr(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


def test_mlp_paridad_si_carga_sklearn():
    """Control negativo: el modelo de referencia SÍ carga sklearn (a propósito,
    ver su docstring) -- sin este control, un error en las tres pruebas de
    arriba (por ejemplo, comprobar la condición invertida) pasaría igual."""
    codigo = (
        "import sys\n"
        "from taller_hpo.problemas import mlp_paridad\n"
        "assert 'sklearn' in sys.modules\n"
        "print('OK')\n"
    )
    resultado = _correr(codigo)
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout
