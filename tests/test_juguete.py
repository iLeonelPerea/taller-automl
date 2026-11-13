"""`taller_hpo.problemas.juguete`, el módulo por el que los pasos entrenan el juguete.

Cada prueba que toca el módulo importado lleva `@pytest.mark.gpu`. La única sin la marca
es `test_subproceso_sin_gpu_visible_falla_al_importar`, que importa el módulo en un
subproceso con `CUDA_VISIBLE_DEVICES=""` y corre en cualquier máquina.

Sin tarjeta de video, importar `juguete` lanza `RuntimeError`. El try/except de abajo
evita que eso aborte la colección entera con `pytest -m "not gpu"`: `juguete` queda en
`None` y ninguna prueba lo toca, porque `tests/conftest.py` ya las hizo fallar por la marca.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from taller_hpo.problemas import mlp_paridad, paridad_torch
from taller_hpo.semillas import Semillas

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAPIDO = {"epocas": 5, "n_train": 200, "n_val": 100, "n_oraculo": 2_000, "ancho": 8}
S = Semillas(datos=1, orden=2, init=3)


def _sin_reloj(metricas):
    return {k: v for k, v in metricas.items() if k != "segundos"}


# --- sin tarjeta de video, la importación falla ------------------------------


def test_subproceso_sin_gpu_visible_falla_al_importar():
    entorno = dict(os.environ)
    entorno["CUDA_VISIBLE_DEVICES"] = ""
    resultado = subprocess.run(
        [sys.executable, "-c", "from taller_hpo.problemas import juguete"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0
    assert "RuntimeError" in resultado.stderr
    assert "tarjeta de video" in resultado.stderr


# --- el módulo ya importado: MOTOR/DISPOSITIVO/DEFAULTS/entrenar/re-exports ---


@pytest.mark.gpu
def test_atributos_del_modulo_tienen_la_forma_del_contrato():
    assert juguete.MOTOR == "torch"
    assert juguete.DISPOSITIVO == "cuda"
    assert isinstance(juguete.DESCRIPCION, str) and juguete.DESCRIPCION
    assert juguete.PROTOCOLO_MOTOR == {"motor": "torch", "dispositivo": "cuda"}
    assert set(juguete.HIPERPARAMETROS) <= set(juguete.DEFAULTS)
    assert set(juguete.HIPERPARAMETROS) <= set(mlp_paridad.DEFAULTS)


@pytest.mark.gpu
def test_defaults_son_los_de_paridad_torch_con_cuda_resuelto():
    assert juguete.DEFAULTS == {**paridad_torch.DEFAULTS, "dispositivo": "cuda"}
    assert juguete.DEFAULTS["dispositivo"] == "cuda"


@pytest.mark.gpu
def test_descripcion_incluye_la_version_de_torch_y_el_nombre_de_la_gpu():
    import torch

    assert juguete.DESCRIPCION == (
        f"PyTorch {torch.__version__} en cuda ({torch.cuda.get_device_name()})"
    )


@pytest.mark.gpu
def test_reexports_son_los_mismos_objetos_que_mlp_paridad():
    assert juguete.Divergencia is mlp_paridad.Divergencia
    # Enteros grandes (fuera del rango cacheado de CPython): "is" solo da True
    # si de verdad es el mismo objeto reexportado, no una coincidencia de valor.
    assert juguete.SEMILLA_ORACULO is mlp_paridad.SEMILLA_ORACULO
    assert juguete.SEMILLA_PRUEBA is mlp_paridad.SEMILLA_PRUEBA


@pytest.mark.gpu
def test_entrenar_delega_exactamente_igual_que_llamar_a_paridad_torch():
    resultado = _sin_reloj(juguete.entrenar(RAPIDO, S))
    directo = paridad_torch.entrenar({**RAPIDO, "dispositivo": "cuda"}, S)
    assert resultado == _sin_reloj(directo)


@pytest.mark.gpu
def test_entrenar_agrega_dispositivo_solo_si_falta_en_la_config(monkeypatch):
    # Un espía que registra qué "dispositivo" recibe paridad_torch.entrenar
    # distingue si juguete.entrenar pisó el que ya traía la config o lo
    # respetó -- comparar métricas no alcanza aquí (la misma red da la misma
    # exactitud en "cuda" con o sin índice explícito).
    recibido = []

    def espia(config, semillas, reportar=None):
        recibido.append(config["dispositivo"])
        return {}

    monkeypatch.setattr(paridad_torch, "entrenar", espia)
    propio = "dispositivo-de-la-config"
    juguete.entrenar({**RAPIDO, "dispositivo": propio}, S)
    juguete.entrenar(RAPIDO, S)
    assert recibido == [propio, "cuda"]


def _cortar_en(epoca_corte, registro):
    def reportar(epoca, valor):
        registro.append((epoca, valor))
        return epoca == epoca_corte

    return reportar


@pytest.mark.gpu
def test_entrenar_pasa_reportar_a_paridad_torch():
    # Si `entrenar` dejara de pasar `reportar`, paridad_torch directo no
    # cortaría en la época 2 y sus métricas dejarían de coincidir.
    via_juguete, directo = [], []
    resultado = _sin_reloj(juguete.entrenar(RAPIDO, S, _cortar_en(2, via_juguete)))
    esperado = _sin_reloj(
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "cuda"}, S, _cortar_en(2, directo))
    )
    assert [e for e, _ in via_juguete] == [0, 1, 2]
    assert resultado["epocas"] == 3
    assert via_juguete == directo
    assert resultado == esperado


# --- PROTOCOLO_MOTOR: función pura ------------------------------------------


@pytest.mark.gpu
def test_protocolo_motor_normaliza_el_indice_de_gpu():
    # _protocolo_motor es pura: esta prueba la ejercita con un "cuda:N" cualquiera,
    # sin depender de qué tarjeta de video (ni de cuántas) tenga la máquina que
    # corre la prueba.
    assert juguete._protocolo_motor("torch", "cuda:1") == {"motor": "torch", "dispositivo": "cuda"}
    assert juguete._protocolo_motor("torch", "cuda") == {"motor": "torch", "dispositivo": "cuda"}
