"""`taller_hpo.problemas.paridad_torch` es un espejo de `mlp_paridad` en la tarjeta de video.

Cada prueba que entrena o construye una red lleva la marca `gpu`. Las que no tocan la
tarjeta (simulan su ausencia con `monkeypatch`, validan antes de construir nada o prueban
una función pura como `_gen_batches`/`_lr_schedule_epoca`) van sin la marca. La mayoría
entrena con la configuración chica `RAPIDO`; las que necesitan que la red aprenda parten de
los DEFAULTS del taller.
"""

import inspect
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

from taller_hpo.problemas import mlp_paridad, paridad, paridad_torch
from taller_hpo.problemas.mlp_paridad import SEMILLA_ORACULO, Divergencia
from taller_hpo.problemas.paridad import error_bayes
from taller_hpo.semillas import Semillas, semillas_de
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

DEFAULTS = paridad_torch.DEFAULTS

RAPIDO = {
    "epocas": 5,
    "n_train": 200,
    "n_val": 100,
    "n_oraculo": 2_000,
    "ancho": 8,
    "dispositivo": "cuda",
}
S = Semillas(datos=1, orden=2, init=3)


@pytest.fixture(scope="module")
def _cuda_precalentado():
    """Precalienta CUDA (contexto, cuBLAS) una vez por módulo.

    La primera llamada a entrenar() en "cuda" de todo el proceso paga la
    inicialización de CUDA, mucho más cara que un entrenamiento chico; como
    fixture de módulo el costo aparece una sola vez, como "setup".
    """
    paridad_torch.entrenar(
        {
            "dispositivo": "cuda",
            "n_train": 10,
            "n_val": 10,
            "n_oraculo": 10,
            "epocas": 1,
            "ancho": 2,
        },
        Semillas(0, 0, 0),
    )


def _sin_reloj(metricas):
    return {k: v for k, v in metricas.items() if k != "segundos"}


# --- Contrato básico (semillas, reportar, placebo, hiperparámetros) ---------------


@pytest.mark.gpu
def test_misma_semilla_mismo_resultado_en_cuda():
    assert _sin_reloj(paridad_torch.entrenar(RAPIDO, S)) == _sin_reloj(
        paridad_torch.entrenar(RAPIDO, S)
    )


@pytest.mark.gpu
def test_cada_fuente_de_azar_cambia_el_resultado():
    base = _sin_reloj(paridad_torch.entrenar(RAPIDO, S))
    for otra in (Semillas(9, 2, 3), Semillas(1, 9, 3), Semillas(1, 2, 9)):
        assert _sin_reloj(paridad_torch.entrenar(RAPIDO, otra)) != base, otra


@pytest.mark.gpu
def test_reportar_recibe_cada_epoca_y_puede_parar():
    llamadas = []

    def reportar(epoca, exactitud_val):
        llamadas.append((epoca, exactitud_val))
        return epoca == 2

    metricas = paridad_torch.entrenar(RAPIDO, S, reportar)
    assert [e for e, _ in llamadas] == [0, 1, 2]
    assert all(0.0 <= v <= 1.0 for _, v in llamadas)
    assert metricas["epocas"] == 3
    assert metricas["exactitud_val"] == llamadas[-1][1]


@pytest.mark.gpu
def test_placebo_no_cambia_nada():
    assert _sin_reloj(paridad_torch.entrenar({**RAPIDO, "placebo": 0.0}, S)) == _sin_reloj(
        paridad_torch.entrenar({**RAPIDO, "placebo": 123.0}, S)
    )


# DEFAULTS completos salvo dispositivo/n_oraculo: RAPIDO no aprende (~0.5) y 10 épocas
# tampoco, así que hace falta la config completa para tener una base que sí aprenda.
APRENDE = {"n_oraculo": 5_000, "dispositivo": "cuda"}


@pytest.fixture(scope="module")
def base_que_aprende():
    base = _sin_reloj(paridad_torch.entrenar(APRENDE, S))
    assert base["exactitud_val"] > 0.6, base  # precondicion: si no aprende, comparar no sirve
    return base


@pytest.mark.parametrize(
    ("clave", "valor"),
    [("learning_rate_init", 1e-2), ("alpha", 1e-1), ("batch_size", 32), ("activacion", "tanh")],
)
@pytest.mark.gpu
def test_cada_hiperparametro_llega_al_modelo(base_que_aprende, clave, valor):
    """El placebo no debe cambiar el resultado; cada perilla real sí debe hacerlo.

    Sin esto, una perilla ignorada (bug de plomería: no llega de config a la
    red, al optimizador o al bucle de minibatches) queda indistinguible del
    placebo, y una búsqueda de HPO le asignaría importancia de ruido sin
    ningún aviso.
    """
    assert _sin_reloj(paridad_torch.entrenar({**APRENDE, clave: valor}, S)) != base_que_aprende


@pytest.mark.gpu
def test_n_parametros_de_una_arquitectura_conocida():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    metricas = paridad_torch.entrenar({**RAPIDO, "ancho": 8, "n_capas": 1}, S)
    assert metricas["n_parametros"] == entradas * 8 + 8 + 8 * 1 + 1


@pytest.mark.gpu
def test_n_parametros_con_dos_capas():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    metricas = paridad_torch.entrenar({**RAPIDO, "ancho": 8, "n_capas": 2}, S)
    assert metricas["n_parametros"] == entradas * 8 + 8 + 8 * 8 + 8 + 8 * 1 + 1


@pytest.mark.gpu
def test_red_respeta_n_capas_y_activacion():
    tipos = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "logistic": torch.nn.Sigmoid,
        "identity": torch.nn.Identity,
    }
    for nombre, clase in tipos.items():
        red = paridad_torch._construir_red(
            (4, 4), nombre, 3, dispositivo="cuda", dtype=torch.float64
        )
        assert [type(m) for m in red] == [
            torch.nn.Linear,
            clase,
            torch.nn.Linear,
            clase,
            torch.nn.Linear,
        ]


@pytest.mark.gpu
def test_nadie_supera_el_techo_de_bayes():
    techo = 1.0 - error_bayes(DEFAULTS["d"], DEFAULTS["a"])
    for s in semillas_de(5, 3):
        metricas = paridad_torch.entrenar(
            {"epocas": 10, "n_oraculo": 50_000, "dispositivo": "cuda"}, s
        )
        # 0.006: error Monte Carlo del oráculo con 50 000 muestras (< 3 EE), igual que en
        # tests/test_mlp_paridad.py.
        assert metricas["exactitud_verdadera"] <= techo + 0.006


@pytest.mark.gpu
def test_oraculo_acotado_por_los_dos_lados():
    """Cota inferior del oráculo: la convención pide probar en las dos direcciones.

    Con DEFAULTS completos (30 épocas) la semilla S sí aprende, así que sirve
    de cota inferior real: un valor invertido (exactitud_verdadera = riesgo en
    vez de 1 - riesgo) o una predicción con el signo cambiado harían fallar
    esta prueba.
    """
    techo = 1.0 - error_bayes(DEFAULTS["d"], DEFAULTS["a"])
    metricas = paridad_torch.entrenar({"n_oraculo": 50_000, "dispositivo": "cuda"}, S)
    assert 0.65 < metricas["exactitud_verdadera"] <= techo + 0.006, metricas


def test_clave_desconocida_falla():
    with pytest.raises(KeyError, match="learning_rate_init"):
        paridad_torch.entrenar({"learning_rate": 0.1}, S)


def test_activacion_desconocida_falla():
    """La validación de `activacion` corre antes de resolver `dispositivo` (ver
    `entrenar`), así que rechazar un valor desconocido no toca la tarjeta de
    video: esta mitad corre siempre, con o sin tarjeta. El control positivo
    (una activación válida sí debe entrenar) está abajo, marcado `gpu`."""
    with pytest.raises(ValueError, match="activacion"):
        paridad_torch.entrenar({**RAPIDO, "activacion": "softplus"}, S)


@pytest.mark.gpu
def test_activacion_desconocida_falla_control_positivo():
    paridad_torch.entrenar({**RAPIDO, "activacion": "tanh"}, S)  # una valida no debe fallar


def test_dispositivo_desconocido_falla():
    """`_validar_dispositivo` es una comprobación de formato pura (una regex),
    y corre antes de consultar `torch.cuda.is_available()`: esta mitad
    negativa corre siempre, con o sin tarjeta de video -- así la integración
    continua, sin tarjeta (`-m "not gpu"`), también protege que
    `_PATRON_DISPOSITIVO` rechace cualquier valor que no sea `auto` ni
    `cuda(:N)`. El control positivo (un dispositivo válido sí debe entrenar)
    está abajo, marcado `gpu`."""
    with pytest.raises(ValueError, match="dispositivo"):
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "gpu"}, S)
    with pytest.raises(ValueError, match="dispositivo"):
        # "cuda" es un prefijo válido, pero "cudaXYZ" no: la regex de
        # `_validar_dispositivo` exige el formato completo, así que falla aquí con
        # un ValueError en español y no varias capas más abajo con un error de
        # torch en inglés.
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "cudaXYZ"}, S)
    with pytest.raises(ValueError, match="dispositivo"):
        # El taller entrena siempre en la tarjeta de video: "cpu" es tan
        # inválido como "gpu", no una alternativa que la regex acepte.
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "cpu"}, S)


@pytest.mark.gpu
def test_dispositivo_desconocido_falla_control_positivo():
    paridad_torch.entrenar({**RAPIDO, "dispositivo": "cuda"}, S)  # uno valido no debe fallar


def test_precision_desconocida_falla():
    """`_validar_precision` también corre antes de tocar la tarjeta de video:
    esta mitad negativa no necesita marca `gpu`. El control positivo (una
    precisión válida sí debe entrenar) está abajo, marcado `gpu`."""
    with pytest.raises(ValueError, match="precision"):
        paridad_torch.entrenar({**RAPIDO, "precision": "float16"}, S)


@pytest.mark.gpu
def test_precision_desconocida_falla_control_positivo():
    paridad_torch.entrenar({**RAPIDO, "precision": "float32"}, S)  # una valida no debe fallar


# --- Validación de entrenar_red: anchos, activación, precisión --------------------

_DATOS_ANCHO = (
    np.random.default_rng(0).standard_normal((20, 4)),
    np.random.default_rng(1).integers(0, 2, 20).astype(np.int64),
    np.random.default_rng(2).standard_normal((10, 4)),
    np.random.default_rng(3).integers(0, 2, 10).astype(np.int64),
)


def _entrenar_red_con_ancho(ancho):
    return paridad_torch.entrenar_red(
        (ancho,),
        activacion="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=8,
        epocas=1,
        datos=_DATOS_ANCHO,
        semillas=S,
        dispositivo="cuda",
    )


def test_ancho_invalido_en_entrenar_red_falla():
    """Sin esta validación, `ancho=0` entrenaría en silencio una red degenerada
    (n_parametros=1); ver el docstring de `entrenar_red`. `ancho=-3` fallaría con
    un `RuntimeError` de torch en inglés en vez de un `ValueError` en español.
    `True` es instancia de `int` (`isinstance(True, int)` da `True`), así que la
    validación lo excluye a mano, igual que en `_validar_hiperparametros_entrenamiento`.
    `entrenar_red` valida `ancho` antes de resolver `dispositivo` o tocar la
    tarjeta de video, así que esta mitad negativa corre siempre, con o sin
    tarjeta. El control positivo (un ancho válido sí debe entrenar) está
    abajo, marcado `gpu`."""
    with pytest.raises(ValueError, match="ancho"):
        _entrenar_red_con_ancho(0)
    with pytest.raises(ValueError, match="ancho"):
        _entrenar_red_con_ancho(-3)
    with pytest.raises(ValueError, match="ancho"):
        _entrenar_red_con_ancho(True)


@pytest.mark.gpu
def test_ancho_invalido_en_entrenar_red_falla_control_positivo():
    resultado = _entrenar_red_con_ancho(4)  # un ancho valido no debe fallar
    assert resultado.n_parametros > 0
    resultado_np = _entrenar_red_con_ancho(np.int64(4))  # numpy.int64 tambien es valido
    assert resultado_np.n_parametros == resultado.n_parametros


_BASE_ACTIVACION_PRECISION_ENTRENAR_RED = {
    "learning_rate_init": 1e-3,
    "alpha": 1e-4,
    "batch_size": 8,
    "epocas": 1,
    "datos": _DATOS_ANCHO,
    "semillas": S,
    "dispositivo": "cuda",
}


def test_activacion_y_precision_invalidas_en_entrenar_red_fallan():
    """`entrenar_red` valida `activacion`/`precision` contra
    `_ACTIVACIONES`/`_validar_precision` antes de construir la red y antes de
    resolver `dispositivo`: una activación o precisión inválidas deben fallar
    con `ValueError` en español, no con un `KeyError` de diccionario, y esta
    mitad negativa corre siempre, con o sin tarjeta de video. El control
    positivo (activación/precisión válidas sí deben entrenar) está abajo,
    marcado `gpu`."""
    with pytest.raises(ValueError, match="activacion"):
        paridad_torch.entrenar_red(
            (4,), activacion="softplus", **_BASE_ACTIVACION_PRECISION_ENTRENAR_RED
        )
    with pytest.raises(ValueError, match="precision"):
        paridad_torch.entrenar_red(
            (4,),
            activacion="relu",
            precision="float16",
            **_BASE_ACTIVACION_PRECISION_ENTRENAR_RED,
        )


@pytest.mark.gpu
def test_activacion_y_precision_invalidas_en_entrenar_red_fallan_control_positivo():
    paridad_torch.entrenar_red(
        (4,), activacion="relu", **_BASE_ACTIVACION_PRECISION_ENTRENAR_RED
    )  # validos no deben fallar


@pytest.mark.gpu
def test_dispositivo_cuda_con_indice_fuera_de_rango_falla_en_espanol():
    """dispositivo="cuda:N" con N >= device_count() debe fallar como dispositivo
    inválido: cumple la regex de formato y "CUDA disponible" es verdadero, pero
    el índice no existe."""
    indice_invalido = f"cuda:{torch.cuda.device_count()}"
    with pytest.raises(ValueError, match="dispositivo"):
        paridad_torch.entrenar({**RAPIDO, "dispositivo": indice_invalido}, S)


def test_epocas_menor_a_uno_falla():
    """`entrenar` valida `epocas` antes de resolver `dispositivo` o tocar la
    tarjeta de video, así que esta mitad negativa corre siempre, con o sin
    tarjeta. El control positivo (una `epocas` válida sí debe entrenar) está
    abajo, marcado `gpu`."""
    with pytest.raises(ValueError, match="epocas"):
        paridad_torch.entrenar({**RAPIDO, "epocas": 0}, S)


@pytest.mark.gpu
def test_epocas_menor_a_uno_falla_control_positivo():
    paridad_torch.entrenar({**RAPIDO, "epocas": 1}, S)  # una valida no debe fallar


def test_auto_sin_cuda_falla_en_espanol(monkeypatch):
    """`"auto"` significa la tarjeta de video: sin CUDA visible debe fallar con
    un `ValueError` que diga qué falta."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="tarjeta de video"):
        paridad_torch._resolver_dispositivo("auto")


def test_auto_con_cuda_resuelve_a_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert paridad_torch._resolver_dispositivo("auto") == "cuda"


@pytest.mark.gpu
def test_auto_por_defecto_resuelve_dentro_de_entrenar():
    """`_resolver_dispositivo` sola no basta: hay que probar que `entrenar` la usa."""
    sin_dispositivo = {k: v for k, v in RAPIDO.items() if k != "dispositivo"}
    assert _sin_reloj(paridad_torch.entrenar(sin_dispositivo, S)) == _sin_reloj(
        paridad_torch.entrenar(RAPIDO, S)
    )


def test_dispositivo_cuda_sin_gpu_falla_en_espanol_antes_de_entrenar(monkeypatch):
    """Antes de generar datos, no después: esta prueba parcha `generar` para que
    reviente si algo la llama.

    Las dos formas de pedir la tarjeta de video ("cuda" explícito y "auto")
    deben fallar por igual, con el mismo `ValueError`, sin CUDA visible.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _generar_no_debe_llamarse(*_args, **_kwargs):
        raise AssertionError(
            "generar() no debe llamarse: la validación de dispositivo debe fallar antes"
        )

    monkeypatch.setattr(paridad_torch, "generar", _generar_no_debe_llamarse)
    with pytest.raises(ValueError, match="tarjeta de video") as excinfo_cuda:
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "cuda"}, S)
    with pytest.raises(ValueError, match="tarjeta de video") as excinfo_auto:
        paridad_torch.entrenar({**RAPIDO, "dispositivo": "auto"}, S)
    # Las dos rutas ("cuda" explícito, que valida aquí mismo, y "auto", que
    # valida dentro de `_resolver_dispositivo`) usan el MISMO mensaje --
    # MENSAJE_SIN_TARJETA, la fuente única -- así que un tesista ve la misma
    # frase sin importar cuál de las dos formas de pedir la tarjeta usó.
    assert str(excinfo_cuda.value) == MENSAJE_SIN_TARJETA
    assert str(excinfo_auto.value) == MENSAJE_SIN_TARJETA


@pytest.mark.gpu
def test_prueba_finita_igual_contrato_que_mlp_paridad():
    # Heredar DEFAULTS de mlp_paridad trae n_prueba: el adaptador debe respetarla
    # igual (misma SEMILLA_PRUEBA), o el contrato se desvía entre los dos modelos.
    assert "exactitud_prueba" not in paridad_torch.entrenar({**RAPIDO, "dispositivo": "cuda"}, S)
    metricas = paridad_torch.entrenar(
        {**RAPIDO, "dispositivo": "cuda", "n_prueba": 20_000, "n_oraculo": 100_000}, S
    )
    assert abs(metricas["exactitud_prueba"] - metricas["exactitud_verdadera"]) < 0.015


@pytest.mark.gpu
def test_entrena_en_cuda_y_da_exactitud_razonable(monkeypatch, _cuda_precalentado):
    dispositivos_vistos = set()
    original = torch.nn.BCEWithLogitsLoss.forward

    def espia(self, entrada, objetivo):
        dispositivos_vistos.add(entrada.device.type)
        return original(self, entrada, objetivo)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", espia)
    metricas = paridad_torch.entrenar({"dispositivo": "cuda", "n_oraculo": 5_000}, S)
    assert metricas["exactitud_val"] > 0.6
    assert dispositivos_vistos == {"cuda"}


@pytest.mark.gpu
def test_auto_usa_cuda_dentro_de_entrenar_si_hay_gpu(monkeypatch, _cuda_precalentado):
    dispositivos_vistos = set()
    original = torch.nn.BCEWithLogitsLoss.forward

    def espia(self, entrada, objetivo):
        dispositivos_vistos.add(entrada.device.type)
        return original(self, entrada, objetivo)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", espia)
    sin_dispositivo = {k: v for k, v in RAPIDO.items() if k != "dispositivo"}
    paridad_torch.entrenar(sin_dispositivo, S)
    assert dispositivos_vistos == {"cuda"}


# --- Divergencia --------------------------------------------------------------------


@pytest.mark.gpu
def test_divergencia_real_con_learning_rate_absurdo():
    """`lr=1e20` desborda de verdad, pero SOLO en float32; en float64 no alcanza a desbordar.

    Con `precision="float64"` (el default de este módulo), `lr=1e20` no diverge en 5
    épocas, igual que `mlp_paridad.entrenar` con el mismo `learning_rate_init`: Adam
    normaliza el paso a ~`lr` sin importar la escala del gradiente, y 1e20 cabe en el rango
    de float64 (~1.8e308). En float32 (rango ~3.4e38) sí desborda en la primera época.
    `lr=10` es grande pero finito y no diverge en ninguna precisión.
    """
    with pytest.raises(Divergencia, match="no finita"):
        paridad_torch.entrenar({**RAPIDO, "learning_rate_init": 1e20, "precision": "float32"}, S)
    # grande y finito: no diverge, ni en float32 ni en float64 (el default).
    paridad_torch.entrenar({**RAPIDO, "learning_rate_init": 10.0, "precision": "float32"}, S)
    paridad_torch.entrenar({**RAPIDO, "learning_rate_init": 10.0}, S)


@pytest.mark.gpu
def test_divergencia_por_perdida_no_finita(monkeypatch):
    """La pérdida da NaN pero sigue viniendo de una operación diferenciable.

    `torch.tensor(nan)` a secas no tiene `grad_fn` (no viene de ningún cómputo
    sobre las entradas), así que `perdida.backward()` fallaría con un
    `RuntimeError` distinto al que se quiere probar aquí ("element 0 of
    tensors does not require grad"), un caso que no puede pasar con la
    pérdida real (siempre depende de los logits, que sí requieren grad).
    Multiplicar la pérdida real por `nan` conserva el grafo (sigue viniendo
    de `logits`) y da un valor no finito, que es el caso real que se quiere
    cubrir: la pérdida se vuelve NaN, pero `backward()` sigue siendo válido
    (produce gradientes NaN, inocuos porque nadie los mira hasta el fin de
    época).
    """
    original = torch.nn.BCEWithLogitsLoss.forward

    def perdida_nan(self, entrada, objetivo):
        return original(self, entrada, objetivo) * float("nan")

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", perdida_nan)
    with pytest.raises(Divergencia, match="no finita"):
        paridad_torch.entrenar(RAPIDO, S)


@pytest.mark.gpu
def test_otro_error_no_se_confunde_con_divergencia(monkeypatch):
    def explota(self, entrada, objetivo):
        raise RuntimeError("otra cosa")

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", explota)
    with pytest.raises(RuntimeError, match="otra cosa"):
        paridad_torch.entrenar(RAPIDO, S)


# --- Arquitectura: sin DataLoader, sin generador aleatorio de torch -----------------


def test_modulo_no_usa_dataloader_ni_rng_de_torch():
    """Datos residentes (sin `DataLoader`) y el generador aleatorio de torch nunca se toca.

    Verifica el CÓDIGO de las funciones que entrenan, no el módulo entero: el
    docstring del módulo SÍ menciona `DataLoader` en la sección que explica por
    qué este diseño no lo usa (una comparación de tiempos contra la alternativa
    con `DataLoader`/`torch.optim.Adam`) -- buscar en el texto completo
    encontraría esa palabra ahí, no en código real. Tampoco alcanza con el
    comportamiento: un entrenamiento que por casualidad diera el mismo
    resultado no probaría que el mecanismo es distinto.
    """
    codigo = "".join(
        inspect.getsource(f)
        for f in (
            paridad_torch.entrenar,
            paridad_torch.entrenar_red,
            paridad_torch._construir_red,
            paridad_torch._inicializar_pesos,
        )
    )
    for prohibido in ("DataLoader", "TensorDataset", "fork_rng", "torch.manual_seed"):
        assert prohibido not in codigo, prohibido
    assert not hasattr(paridad_torch, "DataLoader")
    assert not hasattr(paridad_torch, "TensorDataset")


@pytest.mark.gpu
def test_datos_se_suben_una_sola_vez_sin_importar_las_epocas(monkeypatch):
    """`entrenar_red` sube x_tr/y_tr/x_va/y_va UNA vez, no una vez por época ni por lote.

    Cuenta las llamadas a `subir` (el único punto por el que un array de
    numpy se convierte en tensor de dispositivo) DENTRO de `entrenar_red`
    exclusivamente (no de `entrenar`, que además sube `x`/`p` del oráculo vía
    `riesgo_verdadero_en_dispositivo`; mezclar las dos cosas haría la cuenta
    depender de `n_oraculo` y perdería el punto): deben ser exactamente 4
    (x_tr, y_tr, x_va, y_va), sin importar si hay 3 épocas o 9 ni cuántos
    lotes tenga cada una.
    """
    rng = np.random.default_rng(0)
    datos = (
        rng.standard_normal((RAPIDO["n_train"], 4)),
        rng.integers(0, 2, RAPIDO["n_train"]).astype(np.int64),
        rng.standard_normal((RAPIDO["n_val"], 4)),
        rng.integers(0, 2, RAPIDO["n_val"]).astype(np.int64),
    )
    llamadas = {"n": 0}
    original = paridad_torch.subir

    def espia(*args, **kwargs):
        llamadas["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(paridad_torch, "subir", espia)
    for epocas in (3, 9):
        llamadas["n"] = 0
        paridad_torch.entrenar_red(
            (RAPIDO["ancho"],),
            activacion="relu",
            learning_rate_init=1e-3,
            alpha=1e-4,
            batch_size=32,
            epocas=epocas,
            datos=datos,
            semillas=S,
            dispositivo="cuda",
        )
        assert llamadas["n"] == 4, epocas


def test_gen_batches_coincide_con_sklearn():
    from sklearn.utils import gen_batches

    for n, batch_size in [(130, 17), (2000, 64), (10, 3), (7, 3), (6, 3), (2, 3), (5, 100)]:
        esperado = [(s.start, s.stop) for s in gen_batches(n, batch_size)]
        real = list(paridad_torch._gen_batches(n, min(batch_size, n)))
        assert real == esperado, (n, batch_size)


def test_lr_schedule_epoca_coincide_con_sklearn_paso_a_paso():
    """`_lr_schedule_epoca` es una función pura (nunca toca la tarjeta de
    video): arma de una vez el vector de `lr_t` de Adam (la fórmula de
    sklearn, ver la sección "Por qué el paso de Adam se escribe a mano" del
    docstring del módulo) para los `n_pasos` minibatches de una época,
    arrancando en el paso global `t_inicial + 1`. Esta prueba la compara contra la misma
    fórmula calculada paso a paso con las mismas beta1/beta2 del módulo.
    """
    beta1, beta2 = paridad_torch._ADAM_BETA1, paridad_torch._ADAM_BETA2
    lr_init = 1e-3
    for t_inicial, n_pasos in [(0, 5), (7, 1), (100, 32)]:
        esperado = [
            lr_init * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
            for t in range(t_inicial + 1, t_inicial + 1 + n_pasos)
        ]
        real = paridad_torch._lr_schedule_epoca(t_inicial, n_pasos, lr_init)
        assert real.dtype == np.float64
        np.testing.assert_array_equal(real, np.array(esperado, dtype=np.float64))


@pytest.mark.gpu
def test_inicializacion_de_pesos_coincide_bit_a_bit_con_sklearn():
    """`_inicializar_pesos` reproduce `MLPClassifier._init_coef` bit a bit.

    La prueba espía `BaseMultilayerPerceptron._init_coef` de sklearn y captura el estado de
    `self._random_state` justo antes del sorteo y el `coef_init` que produce. El primer bloque
    compara sklearn contra la fórmula escrita en la prueba; el segundo construye la red con
    `paridad_torch._construir_red`, llama al `_inicializar_pesos` real desde ese mismo estado
    y compara sus pesos capa por capa con `np.array_equal`.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron

    original = BaseMultilayerPerceptron._init_coef
    capturas = []

    def espia(self, fan_in, fan_out, dtype):
        estado_antes = self._random_state.get_state()
        coef, inter = original(self, fan_in, fan_out, dtype)
        factor = 2.0 if self.activation == "logistic" else 6.0
        capturas.append(
            {
                "fan_in": fan_in,
                "fan_out": fan_out,
                "estado": estado_antes,
                "cota": math.sqrt(factor / (fan_in + fan_out)),
                "coef": coef.copy(),
                "inter": inter.copy(),
            }
        )
        return coef, inter

    x = np.random.RandomState(1).randn(10, 4)
    y = np.array([0, 1] * 5)
    for activacion, capas in [("relu", (3,)), ("logistic", (5,)), ("tanh", (6, 5))]:
        capturas.clear()
        BaseMultilayerPerceptron._init_coef = espia
        try:
            m = MLPClassifier(
                hidden_layer_sizes=capas,
                activation=activacion,
                solver="adam",
                random_state=42,
                max_iter=1,
                batch_size=10,
                shuffle=False,
            )
            m.partial_fit(x, y, classes=np.array([0, 1]))
        finally:
            BaseMultilayerPerceptron._init_coef = original

        # Reproduce cada sorteo con una RandomState propia puesta en el estado
        # capturado (así no importa qué más haya consumido self._random_state):
        # esto es exactamente lo que hace _inicializar_pesos, capa por capa.
        for captura in capturas:
            rng = np.random.RandomState()
            rng.set_state(captura["estado"])
            cota = captura["cota"]
            coef = rng.uniform(-cota, cota, (captura["fan_in"], captura["fan_out"]))
            inter = rng.uniform(-cota, cota, captura["fan_out"])
            np.testing.assert_array_equal(coef, captura["coef"])
            np.testing.assert_array_equal(inter, captura["inter"])

        # Llama a los nombres REALES del módulo (_construir_red +
        # _inicializar_pesos), no a una reimplementación dentro de la prueba:
        # así detecta un orden sesgo/peso invertido o un factor de Glorot fijo
        # directamente en los pesos, no solo en la exactitud tras entrenar.
        entradas = x.shape[1]
        red = paridad_torch._construir_red(
            capas, activacion, entradas, dispositivo="cuda", dtype=torch.float64
        )
        paridad_torch._inicializar_pesos(red, activacion, 42)
        capas_lineales = [m for m in red if isinstance(m, torch.nn.Linear)]
        assert len(capas_lineales) == len(capturas)
        for capa, captura in zip(capas_lineales, capturas, strict=True):
            peso = capa.weight.detach().cpu().numpy().T
            sesgo = capa.bias.detach().cpu().numpy()
            np.testing.assert_array_equal(peso, captura["coef"])
            np.testing.assert_array_equal(sesgo, captura["inter"])


@pytest.mark.gpu
def test_skip_init_no_toca_el_rng_global_de_torch():
    """`_construir_red` no debe consumir ni un solo sorteo del generador global de torch.

    Sin `skip_init`, `nn.Linear.__init__` llama a `reset_parameters`, que SÍ
    consume el generador aleatorio global de torch para su propia inicialización (que
    `_inicializar_pesos` sobrescribe por completo de todas formas, ver su
    docstring) -- eso obligaría a envolver la construcción en
    `torch.random.fork_rng` para no pisar el generador aleatorio global entre llamadas
    concurrentes, justo lo que este módulo evita. Esta prueba lo verifica en
    vez de asumirlo.
    """
    estado_antes = torch.get_rng_state().clone()
    paridad_torch._construir_red((4, 4), "relu", 3, dispositivo="cuda", dtype=torch.float64)
    assert torch.equal(estado_antes, torch.get_rng_state())


# --- Equivalencia con mlp_paridad: mismo problema, mismos hiperparámetros -----------

_CONFIGS_EQUIVALENCIA = [
    {"activacion": "logistic", "n_capas": 1, "ancho": 8},
    {"activacion": "tanh", "n_capas": 3, "ancho": 6},
    {"activacion": "identity", "n_capas": 1, "ancho": 5},
    {"activacion": "relu", "n_capas": 3, "ancho": 10, "alpha": 5.0},
    {"activacion": "relu", "n_capas": 1, "ancho": 8, "learning_rate_init": 1e-2},
]
_BASE_EQUIVALENCIA = {
    "n_train": 130,  # no es múltiplo de 17: ejercita el último lote parcial.
    "n_val": 60,
    "n_oraculo": 4_000,
    "epocas": 6,
    "batch_size": 17,
    "precision": "float64",
}
_SEMILLAS_EQUIVALENCIA = [Semillas(1, 2, 3), Semillas(11, 22, 33)]
_DISPOSITIVOS = ["cuda"]


@pytest.mark.gpu
@pytest.mark.parametrize("dispositivo", _DISPOSITIVOS)
@pytest.mark.parametrize("semilla", _SEMILLAS_EQUIVALENCIA, ids=lambda s: f"s{s.init}")
@pytest.mark.parametrize("indice_config", range(len(_CONFIGS_EQUIVALENCIA)))
class TestEquivalenciaConMlpParidad:
    """Criterio de aceptación (1): en la tarjeta de video, en float64, paridad_torch == mlp_paridad.

    5 configs (logistic/tanh/identity/relu, 1 y 3 capas, alpha grande, lr
    grande) x 2 semillas = 10 comparaciones. Cada una
    entrena DOS VECES (torch y sklearn) con la MISMA config y semilla y
    compara: (a) la exactitud_val de CADA ÉPOCA (vía `reportar`), no solo la
    final, y (b) `exactitud_verdadera` con tolerancia 1e-12 (el oráculo se
    calcula distinto —numpy puro en un caso, tensores en el otro— así que no
    se exige bit a bit ahí, pero 1e-12 es indistinguible de una predicción
    idéntica: con d+k_ruido=13 columnas float64, el error de redondeo de la
    posterior no llega ni cerca de eso).
    """

    def test_trayectoria_y_riesgo_verdadero_identicos(self, indice_config, semilla, dispositivo):
        cfg = {**_BASE_EQUIVALENCIA, **_CONFIGS_EQUIVALENCIA[indice_config]}
        # "precision" es una perilla de paridad_torch; mlp_paridad.DEFAULTS no la tiene.
        cfg_sklearn = {k: v for k, v in cfg.items() if k != "precision"}

        def _acumular(vistos):
            def reportar(epoca, exactitud_val):
                vistos.append(exactitud_val)
                return False

            return reportar

        trayectoria_sklearn = []
        metricas_sklearn = mlp_paridad.entrenar(
            cfg_sklearn, semilla, _acumular(trayectoria_sklearn)
        )

        trayectoria_torch = []
        metricas_torch = paridad_torch.entrenar(
            {**cfg, "dispositivo": dispositivo}, semilla, _acumular(trayectoria_torch)
        )

        assert trayectoria_torch == trayectoria_sklearn
        assert metricas_torch["exactitud_val"] == metricas_sklearn["exactitud_val"]
        assert metricas_torch["n_parametros"] == metricas_sklearn["n_parametros"]
        assert metricas_torch["epocas"] == metricas_sklearn["epocas"]
        assert (
            abs(metricas_torch["exactitud_verdadera"] - metricas_sklearn["exactitud_verdadera"])
            < 1e-12
        )


# --- Equivalencia con mlp_paridad a nivel de PESOS ---------------------------------

_CONFIGS_PESOS = [
    {"activacion": "logistic", "n_capas": 1, "ancho": 8},
    {"activacion": "tanh", "n_capas": 3, "ancho": 6},
    {"activacion": "relu", "n_capas": 3, "ancho": 10, "alpha": 5.0},
]


@pytest.mark.gpu
@pytest.mark.parametrize("dispositivo", _DISPOSITIVOS)
@pytest.mark.parametrize("indice_config", range(len(_CONFIGS_PESOS)))
def test_pesos_finales_coinciden_con_sklearn_hasta_el_redondeo(
    monkeypatch, indice_config, dispositivo
):
    """`TestEquivalenciaConMlpParidad` (arriba) compara EXACTITUDES -- conteos de
    aciertos sobre 60 puntos de validación y sobre el oráculo --, y esas métricas
    no cambian ante errores del orden de `1e-6` en los pesos: una fórmula de Adam
    ligeramente distinta a la de sklearn (por ejemplo, la corrección de sesgo de
    `v` dentro de la raíz en vez de fuera, como hace `torch.optim.Adam`, o un
    `_ADAM_EPSILON` de `1e-7` en vez de `1e-8`) puede seguir dando la misma
    exactitud en estas configuraciones chicas, aunque los pesos ya no coincidan.
    Esta prueba compara los pesos FINALES, no la exactitud: espía
    `MLPClassifier.partial_fit` para capturar el modelo real que usa
    `mlp_paridad.entrenar` internamente (sin duplicar su bucle de entrenamiento)
    y espía `paridad_torch.entrenar_red` para capturar `resultado.modelo` (mismo
    patrón que `test_entrenar_usa_entrenar_red_con_anchos_repetidos`, más abajo:
    el nombre `entrenar_red` se resuelve en el namespace del módulo en cada
    llamada, así que parchar el atributo del módulo intercepta la llamada que
    hace `entrenar`). `alpha=5.0` en la tercera config ejercita el gradiente de
    L2 a propósito: con alpha chico, regularizar también los sesgos (que
    sklearn nunca hace) o dividir entre el tamaño de lote nominal en vez del
    real casi no se nota; con alpha grande, sí.
    """
    from sklearn.neural_network import MLPClassifier

    cfg = {**_BASE_EQUIVALENCIA, **_CONFIGS_PESOS[indice_config]}
    cfg_sklearn = {k: v for k, v in cfg.items() if k != "precision"}

    modelos_sklearn = []
    original_partial_fit = MLPClassifier.partial_fit

    def espia_partial_fit(self, *args, **kwargs):
        resultado = original_partial_fit(self, *args, **kwargs)
        modelos_sklearn.append(self)
        return resultado

    monkeypatch.setattr(MLPClassifier, "partial_fit", espia_partial_fit)
    mlp_paridad.entrenar(cfg_sklearn, S)
    modelo_sklearn = modelos_sklearn[-1]

    redes_torch = []
    original_entrenar_red = paridad_torch.entrenar_red

    def espia_entrenar_red(*args, **kwargs):
        resultado = original_entrenar_red(*args, **kwargs)
        redes_torch.append(resultado.modelo)
        return resultado

    monkeypatch.setattr(paridad_torch, "entrenar_red", espia_entrenar_red)
    paridad_torch.entrenar({**cfg, "dispositivo": dispositivo}, S)
    capas_lineales = [m for m in redes_torch[-1] if isinstance(m, torch.nn.Linear)]

    assert len(capas_lineales) == len(modelo_sklearn.coefs_)
    for capa, w, b in zip(
        capas_lineales, modelo_sklearn.coefs_, modelo_sklearn.intercepts_, strict=True
    ):
        peso_torch = capa.weight.detach().cpu().numpy().T
        sesgo_torch = capa.bias.detach().cpu().numpy()
        assert np.abs(peso_torch - w).max() <= 1e-12
        assert np.abs(sesgo_torch - b).max() <= 1e-12


@pytest.mark.gpu
def test_float32_vs_float64_no_se_exige_equivalencia_pero_ambos_entrenan():
    """float32 es una perilla de rendimiento (ver la sección de precisión del
    docstring del módulo), no de fidelidad.

    No se compara contra sklearn en float32: la propia inicialización ya
    trunca al copiar (sklearn siempre sortea y guarda en float64). Solo se
    verifica que el modo funciona y produce una exactitud razonable.
    """
    metricas = paridad_torch.entrenar({**APRENDE, "precision": "float32"}, S)
    assert metricas["exactitud_val"] > 0.55, metricas


# --- Oráculo en el dispositivo --------------------------------------------------------


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_coincide_con_paridad():
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float64
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)
    red.eval()

    def predecir_numpy(x):
        with torch.no_grad():
            logits = red(torch.from_numpy(x.astype(np.float64)).to("cuda"))
        return (logits.squeeze(1) > 0).long().cpu().numpy()

    esperado = paridad.riesgo_verdadero(
        predecir_numpy,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=10_000,
        semilla=SEMILLA_ORACULO,
    )
    real = paridad_torch.riesgo_verdadero_en_dispositivo(
        red,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=10_000,
        semilla=SEMILLA_ORACULO,
        dispositivo="cuda",
    )
    assert abs(real - esperado) < 1e-12


@pytest.mark.gpu
def test_exactitud_en_dispositivo_coincide_con_calculo_manual():
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float64)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    red.eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 4))
    y = rng.integers(0, 2, 50).astype(np.int64)

    with torch.no_grad():
        logits = red(torch.from_numpy(x).to("cuda"))
    prediccion_manual = (logits.squeeze(1) > 0).long().cpu().numpy()
    esperado = float(np.mean(prediccion_manual == y))

    real = paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda")
    assert real == pytest.approx(esperado, abs=1e-12)


# --- entrenar_red: la API reusable para el CASH -------------------------------------


@pytest.mark.gpu
def test_entrenar_red_api_directa():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((60, 4))
    y = rng.integers(0, 2, 60).astype(np.int64)
    x_va = rng.standard_normal((20, 4))
    y_va = rng.integers(0, 2, 20).astype(np.int64)

    resultado = paridad_torch.entrenar_red(
        (6, 6),
        activacion="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=8,
        epocas=3,
        datos=(x, y, x_va, y_va),
        semillas=S,
        dispositivo="cuda",
    )
    assert isinstance(resultado, paridad_torch.ResultadoEntrenamiento)
    assert resultado.epocas == 3
    assert 0.0 <= resultado.exactitud_val <= 1.0
    assert resultado.n_parametros == 4 * 6 + 6 + 6 * 6 + 6 + 6 * 1 + 1
    assert resultado.segundos > 0.0
    assert isinstance(resultado.modelo, torch.nn.Sequential)


@pytest.mark.gpu
def test_entrenar_usa_entrenar_red_con_anchos_repetidos(monkeypatch):
    """`entrenar` (el contrato) debe llamar a `entrenar_red` con `(ancho,) * n_capas`."""
    vistos = {}
    original = paridad_torch.entrenar_red

    def espia(anchos, **kwargs):
        vistos["anchos"] = anchos
        return original(anchos, **kwargs)

    monkeypatch.setattr(paridad_torch, "entrenar_red", espia)
    paridad_torch.entrenar({**RAPIDO, "ancho": 12, "n_capas": 3}, S)
    assert vistos["anchos"] == (12, 12, 12)


# --- Determinismo (8): en proceso y entre procesos, en CUDA -------------------------

_CFG_DETERMINISMO = {
    "n_train": 300,
    "n_val": 100,
    "epocas": 8,
    "batch_size": 23,
    "n_capas": 3,
    "ancho": 32,
    "activacion": "relu",
    "n_oraculo": 5_000,
}


@pytest.mark.gpu
def test_determinismo_en_proceso_cuda(_cuda_precalentado):
    """Misma config y semillas -> métricas idénticas dos veces, en el MISMO proceso.

    Esta red (solo `nn.Linear` + activaciones, sin convoluciones ni
    operaciones con reducción atómica) da resultados bit-idénticos sin
    `CUBLAS_WORKSPACE_CONFIG` ni `torch.use_deterministic_algorithms(True)` (20 corridas
    repetidas con torch 2.14.0+cu130). Por eso `entrenar`
    NO activa ninguna de las dos: la regla del taller pide no cambiar estado
    global de torch en silencio, y activarlas sin necesidad sería justo eso
    (además de forzar un `CUBLAS_WORKSPACE_CONFIG` que tendría que fijarse
    ANTES de inicializar CUDA, algo que un módulo de biblioteca no puede
    garantizar por su cuenta). Si algún día una arquitectura o un tamaño de
    lote distintos dejan de ser deterministas, esta prueba lo va a detectar.
    """
    r1 = _sin_reloj(paridad_torch.entrenar({**_CFG_DETERMINISMO, "dispositivo": "cuda"}, S))
    r2 = _sin_reloj(paridad_torch.entrenar({**_CFG_DETERMINISMO, "dispositivo": "cuda"}, S))
    assert r1 == r2


@pytest.mark.gpu
def test_determinismo_entre_procesos_cuda():
    """Misma config y semillas -> métricas idénticas en DOS PROCESOS distintos."""
    codigo = (
        "import json\n"
        "from taller_hpo.problemas import paridad_torch\n"
        "from taller_hpo.semillas import Semillas\n"
        f"cfg = {_CFG_DETERMINISMO!r}\n"
        "cfg['dispositivo'] = 'cuda'\n"
        "r = paridad_torch.entrenar(cfg, Semillas(9, 8, 7))\n"
        "del r['segundos']\n"
        "print(json.dumps(r, sort_keys=True))\n"
    )
    entorno = dict(os.environ)
    salidas = []
    for _ in range(2):
        resultado = subprocess.run(
            [sys.executable, "-c", codigo],
            capture_output=True,
            text=True,
            check=True,
            env=entorno,
            timeout=60,
        )
        salidas.append(json.loads(resultado.stdout.strip().splitlines()[-1]))
    assert salidas[0] == salidas[1]


# --- Validación de hiperparámetros de entrenar_red ----------------------------------
# Mismo patrón que _entrenar_red_con_ancho/_DATOS_ANCHO de arriba: función auxiliar que arma
# entrenar_red con overrides puntuales sobre una base válida y conocida.


def _entrenar_red_valida(**overrides):
    base = {
        "activacion": "relu",
        "learning_rate_init": 1e-3,
        "alpha": 1e-4,
        "batch_size": 8,
        "epocas": 1,
        "datos": _DATOS_ANCHO,
        "semillas": S,
        "dispositivo": "cuda",
    }
    base.update(overrides)
    return paridad_torch.entrenar_red((4,), **base)


@pytest.mark.gpu
def test_batch_size_invalido_en_entrenar_red_falla():
    """Sin esta validación, `batch_size=0` no entrena en silencio, revienta con
    `ZeroDivisionError` dentro de `_gen_batches` (`n // batch_size`) -- el clamp
    real, `min(batch_size, n_train)`, solo se aplica a un valor YA válido, nunca
    a uno <= 0, así que deja
    pasar el 0 tal cual. `batch_size=-5` sí entrena en silencio, pero no como una
    versión recortada: `n // -5` (división entera) da un cociente negativo, así que
    `range(...)` en `_gen_batches` queda vacío y el único lote que se genera es
    `(0, n_train)` completo -- una red que ve todo el dataset como un solo minibatch,
    sin ningún error. `1.5` y `np.float64` no son enteros; `True` es `bool`, que
    `isinstance(..., numbers.Integral)` acepta pero esta validación excluye a mano."""
    for invalido in (0, -5, 1.5, True, np.float64(2.0)):
        with pytest.raises(ValueError, match="batch_size"):
            _entrenar_red_valida(batch_size=invalido)
    _entrenar_red_valida(batch_size=1)  # borde válido: mínimo permitido
    # Mayor que n_train SÍ se recorta -- a n_train, un único lote (ver
    # `batch_efectivo` en `entrenar_red`) -- pero sin aviso, a diferencia de
    # sklearn (que avisa con UserWarning en este mismo caso): se recorta igual
    # que sklearn pero sin aviso. En una búsqueda real no se activa
    # (`cash_paridad.DEFAULTS`: n_train=2000, mlp_batch_size=64); solo lo
    # activan pruebas con n_train chico, donde el aviso de sklearn sería
    # ruido bajo `-W error` (ver el comentario de `batch_efectivo` en
    # `entrenar_red`).
    _entrenar_red_valida(batch_size=1_000)


@pytest.mark.gpu
def test_learning_rate_init_invalido_en_entrenar_red_falla():
    """Sin esta validación, `learning_rate_init=0.0`/`-1e-3` entrenarían en
    silencio. `nan` también: ni `<= 0` ni `> 0` son ciertos para `nan`, así que
    sin un chequeo aparte se cuela (sklearn sí lo rechaza: su
    `Interval.__contains__` devuelve `False` con `nan`)."""
    for invalido in (0.0, -1e-3, True, "0.01", float("nan")):
        with pytest.raises(ValueError, match="learning_rate_init"):
            _entrenar_red_valida(learning_rate_init=invalido)
    _entrenar_red_valida(learning_rate_init=1e-8)  # borde válido: > 0


@pytest.mark.gpu
def test_alpha_invalido_en_entrenar_red_falla():
    """Sin esta validación, `alpha=-1.0` entrenaría en silencio (L2 negativo).
    `nan` igual que en `learning_rate_init`: `nan < 0` es `False`, así que hace
    falta `math.isnan`."""
    for invalido in (-1.0, -1e-6, True, "0.0", float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            _entrenar_red_valida(alpha=invalido)
    _entrenar_red_valida(alpha=0.0)  # borde válido: L2 se puede desactivar del todo


@pytest.mark.gpu
def test_epocas_invalida_en_entrenar_red_falla():
    """Sin esta validación, un llamador directo a `entrenar_red` -- como
    `cash_paridad.mlp` -- se saltaría la guarda de `epocas` que solo aplica
    `entrenar` antes de llegar aquí."""
    for invalido in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="epocas"):
            _entrenar_red_valida(epocas=invalido)
    _entrenar_red_valida(epocas=1)  # borde válido


# --- "auto" resuelto en entrenar_red -------------------------------------------------


@pytest.mark.gpu
def test_entrenar_red_resuelve_auto_con_cuda():
    """Sin resolver `dispositivo="auto"` aquí, `entrenar_red` fallaría con un
    error de torch en inglés varias capas más abajo: nunca llegaría a
    `_resolver_dispositivo`, que solo invoca `entrenar`."""
    resultado = _entrenar_red_valida(dispositivo="auto")
    assert isinstance(resultado, paridad_torch.ResultadoEntrenamiento)


def test_entrenar_red_resuelve_auto_y_sin_cuda_falla(monkeypatch):
    """Sin una tarjeta de video, "auto" debe fallar con un `ValueError` que
    diga qué falta."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="tarjeta de video"):
        _entrenar_red_valida(dispositivo="auto")


def test_entrenar_red_sin_cuda_falla_con_mensaje_claro_antes_de_tocar_datos(monkeypatch):
    """`dispositivo="cuda"` EXPLÍCITO (no "auto") y sin tarjeta de video visible:
    `entrenar_red` debe fallar con `MENSAJE_SIN_TARJETA` antes de subir ningún
    dato, nunca con el `RuntimeError` en inglés de torch ("No CUDA GPUs are
    available") que saldría varias líneas más abajo, dentro de `subir`.

    `_resolver_dispositivo` no revisa `torch.cuda.is_available()` para un
    "cuda" ya explícito (solo lo hace para "auto"): sin este chequeo aparte
    en `entrenar_red`, un dispositivo "cuda" sin tarjeta real llegaría intacto
    hasta la primera operación de tensor.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    llamado = {"subir": False}
    original_subir = paridad_torch.subir

    def _subir_no_debe_llamarse(*args, **kwargs):
        llamado["subir"] = True
        return original_subir(*args, **kwargs)

    monkeypatch.setattr(paridad_torch, "subir", _subir_no_debe_llamarse)
    with pytest.raises(ValueError) as excinfo:
        _entrenar_red_valida(dispositivo="cuda")
    assert str(excinfo.value) == MENSAJE_SIN_TARJETA
    assert not llamado["subir"], "subir() no debió llamarse: la falla debe ser antes"


def test_resolver_dispositivo_y_precision_modelo_fuera_de_cuda_falla():
    """`exactitud_en_dispositivo`/`riesgo_verdadero_en_dispositivo` con
    `dispositivo=None` (el default) DERIVAN el dispositivo del modelo -- y un
    modelo cuyos parámetros no están en la tarjeta de video falla con un
    `ValueError` claro. Esta prueba corre sin tarjeta de video: construye el
    modelo con los parámetros fuera de ella a propósito y nunca lo mueve a
    "cuda".
    """
    modelo = torch.nn.Sequential(torch.nn.Linear(4, 1)).double()
    assert next(modelo.parameters()).device.type == "cpu"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 4))
    y = rng.integers(0, 2, 10).astype(np.int64)

    with pytest.raises(ValueError, match="tarjeta de video"):
        paridad_torch.exactitud_en_dispositivo(modelo, x, y)
    with pytest.raises(ValueError, match="tarjeta de video"):
        paridad_torch.riesgo_verdadero_en_dispositivo(
            modelo, d=2, a=1.5, k_ruido=0, rotar=False, n=100, semilla=1
        )
    # Con "dispositivo" explícito (aunque no coincida con el dispositivo del
    # modelo), la ruta que SÍ valida/contrasta sigue funcionando: no es
    # esta guarda la que revienta primero.
    with pytest.raises(ValueError, match="dispositivo"):
        paridad_torch.exactitud_en_dispositivo(modelo, x, y, dispositivo="cuda")


# --- dtype/dispositivo derivados del modelo, y contrastados si vienen explícitos ----


@pytest.mark.gpu
def test_exactitud_en_dispositivo_deriva_precision_del_modelo():
    """Con un modelo float32 y `precision` omitida, un default fijo ("float64") para
    `_resolver_dispositivo_y_precision` haría que `modelo(x_t)` chocara de dtype
    (`RuntimeError` de torch en inglés): por eso se deriva de
    `next(modelo.parameters()).dtype` en vez de asumir un valor fijo."""
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float32)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    real = paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda")  # sin precision
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_exactitud_en_dispositivo_deriva_dispositivo_y_precision_del_modelo():
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float64)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    real = paridad_torch.exactitud_en_dispositivo(red, x, y)  # sin dispositivo NI precision
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_exactitud_en_dispositivo_con_modulo_sin_parametros_usa_float64():
    """`cash_paridad._envolver_como_modulo` da un `nn.Module` SIN parámetros propios
    (su `forward` llama una closure cerrada sobre pesos ya ajustados --
    `logistica`/`knn`/`krr`, nunca `mlp`): `next(modelo.parameters())` SIN default
    revienta con `StopIteration` en ese caso, lo que rompería las pruebas de
    `tests/test_cash_paridad.py` que entrenan `logistica`/`knn`/`krr` en cuda. Sin
    `precision`, un módulo así debe caer al default ("float64";
    `cash_paridad._predictor_desde_modulo` siempre les sube datos con `np.float64`
    a mano), no reventar.
    """

    class _SinParametros(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True) - 0.5  # logit arbitrario, determinista

    modulo = _SinParametros()
    assert list(modulo.parameters()) == []
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    real = paridad_torch.exactitud_en_dispositivo(modulo, x, y, dispositivo="cuda")
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_con_modulo_sin_parametros_usa_float64():
    """Mismo caso que la prueba de arriba, para `riesgo_verdadero_en_dispositivo`:
    `cash_paridad.entrenar` llama a las dos con el mismo `_Modulo` sin parámetros.
    """

    class _SinParametros(torch.nn.Module):
        def forward(self, x):
            return x.sum(dim=1, keepdim=True) - 0.5

    modulo = _SinParametros()
    real = paridad_torch.riesgo_verdadero_en_dispositivo(
        modulo,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=1_000,
        semilla=SEMILLA_ORACULO,
        dispositivo="cuda",
    )
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_exactitud_y_riesgo_en_dispositivo_sin_parametros_ni_dispositivo_caen_a_cuda():
    """Esta prueba cubre la rama hermana de las dos pruebas de arriba: ahí, el
    llamador pasa `dispositivo="cuda"` explícito; aquí ninguna de las dos
    llamadas lo pasa, así que `_resolver_dispositivo_y_precision` tiene que
    tomar la otra rama de un módulo SIN parámetros (`parametro is None` y
    `dispositivo is None`), la que cae al default `"cuda"`. Sin esta prueba,
    un cambio real de ese default a otro dispositivo pasaría la suite: las
    dos pruebas de arriba siempre pasan `dispositivo="cuda"` explícito y
    nunca ejercitan esa rama. El `forward` de
    este módulo anota, en una lista propia, el tipo de dispositivo que de
    verdad recibió en cada llamada, y esta prueba exige que la lista entera
    diga "cuda"."""

    class _SinParametrosAnotaDispositivo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dispositivos_vistos = []

        def forward(self, x):
            self.dispositivos_vistos.append(x.device.type)
            return x.sum(dim=1, keepdim=True) - 0.5

    modulo = _SinParametrosAnotaDispositivo()
    assert list(modulo.parameters()) == []
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)

    exactitud = paridad_torch.exactitud_en_dispositivo(modulo, x, y)  # sin dispositivo
    assert 0.0 <= exactitud <= 1.0

    riesgo = paridad_torch.riesgo_verdadero_en_dispositivo(
        modulo,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=1_000,
        semilla=SEMILLA_ORACULO,
    )  # sin dispositivo
    assert 0.0 <= riesgo <= 1.0

    assert modulo.dispositivos_vistos
    assert set(modulo.dispositivos_vistos) == {"cuda"}


@pytest.mark.gpu
def test_exactitud_en_dispositivo_precision_invalida_falla():
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float64)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    with pytest.raises(ValueError, match="precision"):
        paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda", precision="float16")
    paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda", precision="float64")


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_deriva_precision_del_modelo():
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float32
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)
    real = paridad_torch.riesgo_verdadero_en_dispositivo(
        red,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=1_000,
        semilla=SEMILLA_ORACULO,
        dispositivo="cuda",
    )
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_exactitud_en_dispositivo_dispositivo_no_coincide_con_modelo_falla():
    """Un `dispositivo` explícito que no es el del modelo debe fallar aquí mismo:
    sin este chequeo, fallaría varias capas más abajo con un error de torch en
    inglés ("Expected all tensors to be on the same device"). No hace falta una
    segunda tarjeta de video para probarlo: pedir un índice explícito distinto del real
    (`"cuda:1"` contra un modelo en `"cuda:0"`) ya ejercita la comparación, que
    ocurre antes de mover ningún tensor."""
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float64)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    with pytest.raises(ValueError, match="dispositivo"):
        paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda:1")
    paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda")  # coincide, no falla


@pytest.mark.gpu
def test_exactitud_en_dispositivo_precision_no_coincide_con_modelo_falla():
    """Un `precision` explícito que no es el del modelo debe fallar aquí mismo:
    sin este chequeo, fallaría con un choque de dtype de torch en inglés ("mat1
    and mat2 must have the same dtype")."""
    red = paridad_torch._construir_red((5,), "relu", 4, dispositivo="cuda", dtype=torch.float32)
    paridad_torch._inicializar_pesos(red, "relu", 3)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20, 4))
    y = rng.integers(0, 2, 20).astype(np.int64)
    with pytest.raises(ValueError, match="precision"):
        paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda", precision="float64")
    paridad_torch.exactitud_en_dispositivo(red, x, y, dispositivo="cuda", precision="float32")


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_auto_explicito_se_resuelve():
    """`dispositivo="auto"` pasado EXPLÍCITAMENTE (no `None`) debe resolverse
    aquí: sin hacerlo, se pasaría tal cual a `Tensor.to("auto")`, que no es un
    dispositivo real de torch."""
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float64
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)
    real = paridad_torch.riesgo_verdadero_en_dispositivo(
        red,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=1_000,
        semilla=SEMILLA_ORACULO,
        dispositivo="auto",
    )
    assert 0.0 <= real <= 1.0


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_auto_explicito_sin_cuda_falla(monkeypatch):
    """Igual que la prueba de arriba, pero sin CUDA visible: "auto" debe
    fallar con un mensaje claro, no resolver a un dispositivo distinto en
    silencio."""
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float64
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="tarjeta de video"):
        paridad_torch.riesgo_verdadero_en_dispositivo(
            red,
            d=DEFAULTS["d"],
            a=DEFAULTS["a"],
            k_ruido=DEFAULTS["k_ruido"],
            rotar=False,
            n=1_000,
            semilla=SEMILLA_ORACULO,
            dispositivo="auto",
        )


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_dispositivo_invalido_falla():
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float64
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)
    kwargs = dict(
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=1_000,
        semilla=SEMILLA_ORACULO,
    )
    for invalido in ("gpu", "cpu"):
        with pytest.raises(ValueError, match="dispositivo"):
            paridad_torch.riesgo_verdadero_en_dispositivo(red, dispositivo=invalido, **kwargs)
    paridad_torch.riesgo_verdadero_en_dispositivo(red, dispositivo="cuda", **kwargs)  # válido


# --- Oráculo con varios lotes ---------------------------------------------------------


@pytest.mark.gpu
def test_riesgo_verdadero_en_dispositivo_con_varios_lotes(monkeypatch):
    """`test_riesgo_verdadero_en_dispositivo_coincide_con_paridad` (arriba) usa
    `n=10_000` con el lote real de `_LOTE_ORACULO=65_536`: es un solo lote, nunca
    ejercita el bucle `for inicio in range(0, n, _LOTE_ORACULO)` con más de una
    vuelta -- la ruta real con `n_oraculo=200_000` (DEFAULTS) sí hace varias.
    Sin forzar varios lotes, una desalineación entre `p` y `x` a partir del
    segundo lote no la delataría ningún assert de esta suite. Se fuerza un lote
    chico (999, no divisor de 10_000: también ejercita el último lote parcial).
    Mismo patrón que esa prueba, copiado casi literal.
    """
    monkeypatch.setattr(paridad_torch, "_LOTE_ORACULO", 999)
    red = paridad_torch._construir_red(
        (6,), "relu", DEFAULTS["d"] + DEFAULTS["k_ruido"], dispositivo="cuda", dtype=torch.float64
    )
    paridad_torch._inicializar_pesos(red, "relu", 7)

    def predecir_numpy(x):
        with torch.no_grad():
            logits = red(torch.from_numpy(x.astype(np.float64)).to("cuda"))
        return (logits.squeeze(1) > 0).long().cpu().numpy()

    esperado = paridad.riesgo_verdadero(
        predecir_numpy,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=10_000,
        semilla=SEMILLA_ORACULO,
    )
    real = paridad_torch.riesgo_verdadero_en_dispositivo(
        red,
        d=DEFAULTS["d"],
        a=DEFAULTS["a"],
        k_ruido=DEFAULTS["k_ruido"],
        rotar=False,
        n=10_000,
        semilla=SEMILLA_ORACULO,
        dispositivo="cuda",
    )
    assert abs(real - esperado) < 1e-12


# --- CUDA graphs: equivalencia con/sin grafo, memoria, hilos, sincronización --------

_CFG_GRAFO = {
    "n_train": 90,
    "n_val": 40,
    "epocas": 5,
    "batch_size": 11,  # no múltiplo de n_train: ejercita el último lote parcial.
    "n_capas": 2,
    "ancho": 8,
    "activacion": "relu",
    "n_oraculo": 3_000,
}


@pytest.mark.gpu
def test_costo_captura_grafo_s_solo_se_reporta_al_capturar_en_cuda(_cuda_precalentado):
    """`ResultadoEntrenamiento.costo_captura_grafo_s` (ver su docstring) es `None`
    salvo que de verdad se capture un CUDA graph: con el interruptor apagado o
    con menos de dos épocas (nunca se llega a la época de captura) no hay nada
    que medir, y el campo debe quedarse en su default.
    """
    kw = dict(
        activacion="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=8,
        datos=_DATOS_ANCHO,
        semillas=S,
    )
    con_grafo = paridad_torch.entrenar_red((4,), epocas=3, dispositivo="cuda", **kw)
    assert con_grafo.costo_captura_grafo_s is not None
    assert con_grafo.costo_captura_grafo_s > 0

    sin_grafo = paridad_torch.entrenar_red(
        (4,), epocas=3, dispositivo="cuda", usar_cuda_graph=False, **kw
    )
    assert sin_grafo.costo_captura_grafo_s is None

    una_epoca = paridad_torch.entrenar_red((4,), epocas=1, dispositivo="cuda", **kw)
    assert una_epoca.costo_captura_grafo_s is None


@pytest.mark.gpu
def test_grafo_da_la_misma_trayectoria_que_eager_en_cuda(monkeypatch, _cuda_precalentado):
    """Con CUDA graph (`usar_cuda_graph=True`, default) y sin él, la misma config y semilla
    dan la misma trayectoria bit a bit: `exactitud_val` por época, pesos finales,
    `n_parametros` y `epocas`. Las dos rutas llaman a la misma `_ejecutar_epoca_cuda`.

    `entrenar()` no expone `usar_cuda_graph`, así que la prueba lo fuerza con un
    `monkeypatch` sobre `entrenar_red`. Los pesos se capturan espiando `entrenar_red`:
    las exactitudes son conteos de aciertos y no distinguen una diferencia minúscula de pesos.
    """

    def _acumular(vistos):
        def reportar(epoca, exactitud_val):
            vistos.append(exactitud_val)
            return False

        return reportar

    original_entrenar_red = paridad_torch.entrenar_red
    redes = {}

    def espia_con_grafo(*args, **kwargs):
        resultado = original_entrenar_red(*args, **kwargs)
        redes["con_grafo"] = resultado.modelo
        return resultado

    def espia_sin_grafo(*args, **kwargs):
        resultado = original_entrenar_red(*args, **{**kwargs, "usar_cuda_graph": False})
        redes["sin_grafo"] = resultado.modelo
        return resultado

    monkeypatch.setattr(paridad_torch, "entrenar_red", espia_con_grafo)
    trayectoria_con_grafo = []
    r_con_grafo = paridad_torch.entrenar(
        {**_CFG_GRAFO, "dispositivo": "cuda"}, S, _acumular(trayectoria_con_grafo)
    )

    monkeypatch.setattr(paridad_torch, "entrenar_red", espia_sin_grafo)
    trayectoria_sin_grafo = []
    r_sin_grafo = paridad_torch.entrenar(
        {**_CFG_GRAFO, "dispositivo": "cuda"}, S, _acumular(trayectoria_sin_grafo)
    )

    assert trayectoria_con_grafo == trayectoria_sin_grafo
    assert _sin_reloj(r_con_grafo) == _sin_reloj(r_sin_grafo)
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            redes["con_grafo"].parameters(), redes["sin_grafo"].parameters(), strict=True
        )
    )


@pytest.mark.gpu
def test_captura_de_grafo_es_segura_entre_hilos_en_cuda(_cuda_precalentado):
    """`taller_hpo.estudio.optimizar` fija `n_jobs=1`, pero Optuna en general
    implementa `n_jobs>1` con hilos, y CUDA solo permite UNA captura de stream de
    trabajo a la vez POR PROCESO: sin serializarlas, varios hilos que llaman a
    `entrenar(dispositivo="cuda")` a la vez y capturan su grafo al mismo tiempo
    fallan (`AcceleratorError`, "operation not permitted when stream is
    capturing"). Compara 4 llamadas concurrentes (una por hilo, con semillas
    distintas) contra las mismas 4 llamadas hechas en secuencia: deben coincidir
    exactamente, sin que ningún hilo falle. `ThreadPoolExecutor.result()` reenvía
    cualquier excepción del hilo al hilo principal (no se traga nada): si un hilo
    revienta, esta prueba falla con esa excepción real, no con un falso positivo.
    """
    import concurrent.futures

    def _entrenar(i):
        return _sin_reloj(
            paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda"}, Semillas(i, i, i))
        )

    secuencial = {i: _entrenar(i) for i in range(4)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {i: executor.submit(_entrenar, i) for i in range(4)}
        concurrente = {i: futuros[i].result() for i in range(4)}

    assert concurrente == secuencial


@pytest.mark.gpu
def test_captura_crea_el_stream_en_el_dispositivo_del_modelo(monkeypatch, _cuda_precalentado):
    """`torch.cuda.graph` sin `stream=` explícito usa `default_capture_stream`, un
    stream de CLASE creado UNA sola vez, perezosamente, en el dispositivo que esté
    activo la PRIMERA vez que el proceso captura algo (verificado en el código
    fuente instalado, `torch/cuda/graphs.py::graph.__init__`) -- no
    necesariamente el dispositivo de la red que se está entrenando. `entrenar_red`
    pasa en cambio un `torch.cuda.Stream(device=...)` propio, creado en el
    dispositivo REAL de los parámetros de la red (`parametros[0].device`, ver ese
    bucle), y lo pasa explícitamente a `torch.cuda.graph(..., stream=...)`.

    Con una sola tarjeta de video visible, las dos rutas terminan en el mismo índice de
    dispositivo, así que esta prueba no puede reproducir el grafo vacío que
    resultaría de capturar en el stream equivocado -- pero sí puede (y debe)
    verificar que el stream de captura se CREA explícitamente, en cada llamada,
    con el dispositivo real de la red, en vez de heredar el stream compartido de
    la clase (que no lleva ningún argumento de dispositivo).
    """
    dispositivos_pedidos = []
    original_stream = torch.cuda.Stream

    def stream_espia(*args, **kwargs):
        s = original_stream(*args, **kwargs)
        dispositivos_pedidos.append(kwargs.get("device"))
        return s

    monkeypatch.setattr(torch.cuda, "Stream", stream_espia)
    redes = []
    original_construir = paridad_torch._construir_red

    def espia_construir(*args, **kwargs):
        red = original_construir(*args, **kwargs)
        redes.append(red)
        return red

    monkeypatch.setattr(paridad_torch, "_construir_red", espia_construir)

    paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda:0"}, S)

    dispositivo_red = next(redes[-1].parameters()).device
    creados_para_la_red = [d for d in dispositivos_pedidos if d == dispositivo_red]
    assert creados_para_la_red, dispositivos_pedidos


@pytest.mark.gpu
def test_captura_usa_un_stream_propio_en_el_dispositivo_de_la_red(monkeypatch, _cuda_precalentado):
    """La prueba de arriba solo espía la CONSTRUCCIÓN de `torch.cuda.Stream`, no si
    ese stream de verdad llega a `torch.cuda.graph` -- las dos cosas son distintas:
    `entrenar_red` podría construir el stream correcto y aun así pasarlo por alto
    (p. ej. si se le olvidara el kwarg `stream=` en la llamada a `torch.cuda.graph`,
    que caería de vuelta al `default_capture_stream` COMPARTIDO de la clase, ver la
    sección "Captura en dispositivo explícito" del docstring del módulo). Esta
    prueba envuelve la CLASE `torch.cuda.graph` para capturar el `stream` que
    realmente recibe su `__init__`, y confirma que es el mismo objeto construido
    para la red (no `None`, no el `default_capture_stream` de clase)."""
    recibidos = []
    graph_original = torch.cuda.graph

    class GraphEspia(graph_original):
        def __init__(self, cuda_graph, *args, stream=None, **kwargs):
            recibidos.append(stream)
            super().__init__(cuda_graph, *args, stream=stream, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", GraphEspia)
    redes = []
    original_construir = paridad_torch._construir_red

    def espia_construir(*args, **kwargs):
        redes.append(original_construir(*args, **kwargs))
        return redes[-1]

    monkeypatch.setattr(paridad_torch, "_construir_red", espia_construir)

    paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda:0"}, S)

    assert len(recibidos) == 1
    assert recibidos[0] is not None
    assert recibidos[0] is not graph_original.default_capture_stream
    assert recibidos[0].device == next(redes[-1].parameters()).device


@pytest.mark.gpu
def test_grafo_vacio_falla_ruidosamente(monkeypatch, _cuda_precalentado):
    """Una captura que no graba ningún kernel (grafo vacío) no debe reportar en
    silencio, para siempre, la métrica de la época en que se capturó -- justo el
    modo de falla que la sección "Captura en dispositivo explícito" del docstring
    del módulo describe: en un estudio de HPO real corrompería los trials sin
    avisar. Simula esa captura vacía haciendo que `_ejecutar_epoca_cuda`, cuando
    corre DURANTE una captura de stream, devuelva la salida de la época anterior
    en vez de ejecutar nada nuevo -- así el `contador_static` grabado no avanza y
    `entrenar_red` debe detectarlo con el chequeo de la sección "Un solo
    `.item()`-equivalente por época" (ver ese bucle)."""
    original = paridad_torch._ejecutar_epoca_cuda
    previa = {}

    def ejecutar_sin_grabar_si_se_captura(*args):
        if torch.cuda.is_current_stream_capturing():
            return previa["salida"]  # ningún kernel grabado: grafo vacío
        previa["salida"] = original(*args)
        return previa["salida"]

    monkeypatch.setattr(paridad_torch, "_ejecutar_epoca_cuda", ejecutar_sin_grabar_si_se_captura)
    # torch avisa que el grafo capturado no tiene kernels (esperado: es justo lo que
    # este parche fuerza a propósito para simular la captura vacía); afirmarlo con
    # pytest.warns en vez de silenciarlo demuestra que el grafo quedó REALMENTE vacío
    # y no que el RuntimeError salió por otra causa.
    with pytest.warns(UserWarning, match="The CUDA Graph is empty"):
        with pytest.raises(RuntimeError, match="no avanzó en la época 1"):
            paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda"}, S)


@pytest.mark.gpu
def test_divergencia_en_epoca_0_eager_en_cuda():
    """La época 0 en cuda siempre corre eager (antes de cualquier captura), la
    misma ruta que ya cubre `test_divergencia_real_con_learning_rate_absurdo`.
    En float32 (rango ~3.4e38) un learning rate de `1e20` desborda de inmediato;
    en float64 no alcanza a desbordar (ver esa prueba para la explicación
    completa) -- por eso aquí se fuerza `precision="float32"`."""
    with pytest.raises(Divergencia, match="no finita"):
        paridad_torch.entrenar(
            {**RAPIDO, "dispositivo": "cuda", "learning_rate_init": 1e20, "precision": "float32"}, S
        )


@pytest.mark.gpu
def test_divergencia_se_detecta_dentro_de_un_replay_en_cuda(monkeypatch, _cuda_precalentado):
    """La detección de divergencia (requisito de este módulo desde su primera
    versión) también tiene que funcionar DENTRO de un `replay()`, la ruta por
    defecto con `dispositivo="auto"` en una máquina con tarjeta de video: el chequeo de
    `no_finito_epoca` de `_ejecutar_epoca_cuda` queda grabado en el grafo igual
    que el resto de la época (ver su docstring), así que un replay debe poder
    dispararlo sin que Python vuelva a ejecutarse dentro de la captura.

    Espía `_construir_red` para obtener una referencia a la red real y envenena
    sus pesos con NaN DESDE `reportar`, justo después de que termina la época 2
    (una época replay: la 0 es eager y la 1 es la que se captura) -- así la época
    3, otro replay, es la que debe detectar la no-finitud y lanzar `Divergencia`.
    """
    redes = []
    original_construir = paridad_torch._construir_red

    def espia_construir(*args, **kwargs):
        red = original_construir(*args, **kwargs)
        redes.append(red)
        return red

    monkeypatch.setattr(paridad_torch, "_construir_red", espia_construir)

    def reportar(epoca, exactitud_val):
        if epoca == 2:
            with torch.no_grad():
                for p in redes[-1].parameters():
                    p.fill_(float("nan"))
        return False

    with pytest.raises(Divergencia, match="época 3"):
        paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda"}, S, reportar)


@pytest.mark.gpu
def test_divergencia_por_pesos_no_finitos_al_final_de_un_replay_en_cuda(
    monkeypatch, _cuda_precalentado
):
    """El chequeo `~torch.isfinite(logits_va).all()` de `_ejecutar_epoca_cuda` (ver
    su docstring) es el ÚNICO que detecta pesos no finitos que aparecen DESPUÉS
    del último paso de Adam de la época: `suma_perdida` de esa misma época ya
    terminó de acumularse con pesos finitos (el envenenamiento llega tras el
    último minibatch), así que revisarla sola no alcanza. La prueba de arriba
    (`test_divergencia_se_detecta_dentro_de_un_replay_en_cuda`) envenena los
    pesos ENTRE épocas -- eso lo detecta la pérdida de la época SIGUIENTE, no la
    validación de la propia época -- así que no ejercita este chequeo.

    Envenena multiplicando los parámetros por `factor` (un tensor persistente
    capturado por referencia, como `orden_static`/`lr_static`) SOLO en el
    último minibatch de la época: como la condición se evalúa en Python
    DURANTE la captura, el `torch._foreach_mul_` correspondiente queda grabado
    en el grafo únicamente en esa posición, y se repite ahí en cada replay.
    Mientras `factor` vale 1.0 la multiplicación es inocua; `reportar` lo pone
    en NaN FUERA de cualquier captura, justo después de la penúltima época, así
    que la ÚLTIMA época (un replay) es la que debe detectar la no-finitud en la
    validación y lanzar `Divergencia`.
    """
    n_lotes = -(-_CFG_GRAFO["n_train"] // _CFG_GRAFO["batch_size"])
    factor = torch.ones((), dtype=torch.float64, device="cuda")
    original = paridad_torch._paso_adam_foreach
    llamadas = {"n": 0}

    def paso_con_veneno(parametros, gradientes, estado, neg_lr_t, **kwargs):
        original(parametros, gradientes, estado, neg_lr_t, **kwargs)
        llamadas["n"] += 1
        if llamadas["n"] % n_lotes == 0:  # último minibatch de la época: queda grabado
            with torch.no_grad():
                torch._foreach_mul_(list(parametros), factor)

    monkeypatch.setattr(paridad_torch, "_paso_adam_foreach", paso_con_veneno)
    ultima = _CFG_GRAFO["epocas"] - 1

    def reportar(epoca, exactitud_val):
        if epoca == ultima - 1:
            factor.fill_(float("nan"))  # fuera de la captura; la última época es replay
        return False

    with pytest.raises(Divergencia, match=f"época {ultima}"):
        paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda"}, S, reportar)
    # Solo 2 ejecuciones REALES de Python: época 0 (eager) + época 1 (capturada).
    # Las épocas 2, 3 y 4 son replay puro (ver el docstring del módulo): el veneno
    # de esas épocas viene del `factor` grabado, no de una llamada nueva a esta
    # función espía.
    assert llamadas["n"] == 2 * n_lotes


@pytest.mark.gpu
def test_perdida_no_finita_con_gradiente_finito_en_un_replay(monkeypatch, _cuda_precalentado):
    """La bandera de no-finitud de `_ejecutar_epoca_cuda` se arma sumando `perdida`
    a `suma_perdida` (ver su docstring): una pérdida no finita con GRADIENTE
    finito -- que no toca `logits_va`, así que `~torch.isfinite(logits_va).all()`
    tampoco la detecta -- solo la agarra esa suma. Envenena
    `BCEWithLogitsLoss.forward` sumándole un tensor persistente (`veneno`, NaN
    solo cuando `reportar` lo activa) que deja el GRADIENTE intacto: la pérdida
    reportada es no finita, pero backward() sigue siendo un número real. `reportar`
    activa el veneno DESPUÉS de que termina la época 2 (un replay: la 0 es eager y
    la 1 es la que se captura), así que la época 3, otro replay, es la que debe
    detectar la no-finitud."""
    veneno = torch.zeros((), dtype=torch.float64, device="cuda")
    original_forward = torch.nn.BCEWithLogitsLoss.forward

    def forward_con_veneno(self, entrada, objetivo):
        return original_forward(self, entrada, objetivo) + veneno

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", forward_con_veneno)

    def reportar(epoca, exactitud_val):
        if epoca == 2:
            veneno.fill_(float("nan"))
        return False

    with pytest.raises(Divergencia, match="época 3"):
        paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda"}, S, reportar)


@pytest.mark.gpu
def test_grafo_no_deja_fugas_de_memoria_en_cuda(_cuda_precalentado):
    """La memoria de la tarjeta de video vuelve a la línea base después de muchas llamadas. 20
    entrenamientos seguidos (cada uno construye su propio grafo/red/buffers, todo
    local a `entrenar_red`); memoria antes y después.

    `memory_allocated` (bytes con datos vivos) por sí solo no basta: el pool
    PRIVADO de memoria de un `CUDAGraph` todavía referenciado cuenta como memoria
    RESERVADA, no asignada -- una fuga que retenga objetos `CUDAGraph` (que
    `entrenar_red` no debe hacer: cada uno es local a la llamada y se libera al
    salir) pasaría inadvertida si solo se mira `memory_allocated`. Por eso se
    comparan las dos métricas.
    """
    import gc

    def _entrenar_una_vez():
        paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda", "n_oraculo": 500}, S)

    _entrenar_una_vez()  # calentamiento: primera asignación de contexto/cuBLAS
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_asignada = torch.cuda.memory_allocated()
    base_reservada = torch.cuda.memory_reserved()
    for _ in range(20):
        _entrenar_una_vez()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    final_asignada = torch.cuda.memory_allocated()
    final_reservada = torch.cuda.memory_reserved()
    assert final_asignada == base_asignada, (
        f"asignada: base={base_asignada} final={final_asignada} "
        f"diff={final_asignada - base_asignada} bytes"
    )
    assert final_reservada == base_reservada, (
        f"reservada: base={base_reservada} final={final_reservada} "
        f"diff={final_reservada - base_reservada} bytes"
    )


@pytest.mark.gpu
def test_modelo_devuelto_no_retiene_gradientes_del_pool_del_grafo(_cuda_precalentado):
    """`entrenar_red` pone `p.grad = None` en cada parámetro antes de devolver el
    modelo (ver ese bucle, al final de la función): sin eso, `resultado.modelo`
    retendría una referencia a un `.grad` que puede vivir en el pool PRIVADO de
    memoria del CUDA graph, retrasando su liberación (ver la sección de memoria
    del docstring del módulo) aunque nadie lea ese gradiente -- nada en el resto
    del taller (`entrenar`, `cash_paridad`, los pasos) usa `.grad` de un modelo ya
    entrenado."""
    resultado = paridad_torch.entrenar_red(
        (4,),
        activacion="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        batch_size=8,
        epocas=4,
        datos=_DATOS_ANCHO,
        semillas=S,
        dispositivo="cuda",
    )
    assert all(p.grad is None for p in resultado.modelo.parameters())


# torch emite este UserWarning una sola vez por proceso, la primera vez que se llama
# a `set_sync_debug_mode`; no tiene nada que ver con lo que prueba esta función, y sin
# el filtro la suite en `-W error` lo convertiría en una falla ajena a la prueba.
@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_ejecutar_epoca_cuda_sin_sincronizaciones_eager_y_replay(monkeypatch, _cuda_precalentado):
    """Esta prueba usa
    `torch.cuda.set_sync_debug_mode("error")` -- detecta CUALQUIER sincronización
    oculta, no solo esas tres -- alrededor de cada llamada a `_ejecutar_epoca_cuda`
    (cubre la época 0 eager Y la propia computación que se graba durante la
    captura) y de cada `CUDAGraph.replay()` (cubre el replay que calcula la
    época 1, justo tras capturar, y los replays normales): dentro de un
    `replay()` no hay NINGUNA sincronización posible (es un `cudaGraphLaunch`
    asíncrono puro, Python no vuelve a correr).

    Control positivo: `.item()` bajo "error" SÍ lanza `RuntimeError` -- si no
    lanzara, esta prueba no probaría nada, estaría confiando en un detector que
    quizá esté apagado.
    """
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError, match="synchroniz"):
            torch.zeros((), device="cuda").item()
    finally:
        torch.cuda.set_sync_debug_mode("default")

    original_ejecutar = paridad_torch._ejecutar_epoca_cuda
    original_replay = torch.cuda.CUDAGraph.replay
    llamadas = {"ejecutar": 0, "replay": 0}

    def ejecutar_espia(*args, **kwargs):
        llamadas["ejecutar"] += 1
        torch.cuda.set_sync_debug_mode("error")
        try:
            return original_ejecutar(*args, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def replay_espia(self):
        llamadas["replay"] += 1
        torch.cuda.set_sync_debug_mode("error")
        try:
            return original_replay(self)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    monkeypatch.setattr(paridad_torch, "_ejecutar_epoca_cuda", ejecutar_espia)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", replay_espia)

    paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda", "epocas": 4}, S)

    # época 0 (eager) + época 1 (grabada durante la captura) = 2 llamadas a
    # _ejecutar_epoca_cuda; época 1 (el replay que calcula la época 1, justo tras
    # capturar) + épocas 2 y 3 (replay normal) = 3 replays.
    assert llamadas["ejecutar"] == 2
    assert llamadas["replay"] == 3


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode is a prototype feature:UserWarning")
def test_una_sola_sincronizacion_por_epoca_en_cuda(_cuda_precalentado):
    """`orden_static`/`lr_static` se llenan con `.copy_(..., non_blocking=True)` desde
    buffers de host fijados, así que la única sincronización explícita por época es el
    `.tolist()` final.

    `torch.cuda.set_sync_debug_mode("warn")` convierte cada sincronización en un
    `UserWarning`. La prueba mide la diferencia entre `epocas=E` y `epocas=E+3`, no el total,
    porque el total incluye sincronizaciones de una sola vez ajenas al bucle. Quitar
    `non_blocking=True` agrega sincronizaciones y la prueba falla; el efecto de
    `pin_memory=True` no lo cuantifica.
    """

    def _sincronizaciones(epocas):
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as registro:
                warnings.simplefilter("always")
                paridad_torch.entrenar({**_CFG_GRAFO, "dispositivo": "cuda", "epocas": epocas}, S)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return len([w for w in registro if "synchronizing" in str(w.message)])

    base = _sincronizaciones(_CFG_GRAFO["epocas"])
    con_tres_epocas_mas = _sincronizaciones(_CFG_GRAFO["epocas"] + 3)
    assert con_tres_epocas_mas - base == 3
