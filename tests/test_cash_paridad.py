"""CASH sobre la paridad gaussiana: el algoritmo también es un hiperparámetro.

Casi todas las pruebas llevan la marca `gpu`. Las que no la llevan validan configuración
pura (algoritmo desconocido, dispositivo inválido, si `xgboost` trae CUDA) o comprueban un
hecho con scikit-learn puro: que `knn` no tiene empates de distancia en este dataset y que
`_gamma_krr` reproduce `SVC(gamma="scale")`.

`RAPIDO` recorta n_train/n_val/n_oraculo, `S` es una `Semillas` fija y `_sin_reloj` compara
todo menos "segundos". Las implementaciones de scikit-learn (`LogisticRegression`,
`KNeighborsClassifier`, `KernelRidge`) viven solo aquí, como referencia. `bosque` y `hgb` no
se comparan contra scikit-learn (sus árboles parten distinto): sus pruebas verifican el
contrato y el mapeo de parámetros contra el booster real de XGBoost.
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import xgboost
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from taller_hpo.problemas import paridad
from taller_hpo.problemas.cash_paridad import (
    _LOGISTICA_MAX_ITER,
    ALGORITMOS,
    DEFAULTS,
    _ajustar_baseline_gpu,
    _ajustar_logistica_gpu,
    _gamma_krr,
    _predecir_xgb_gpu,
    _resolver_dispositivo,
    entrenar,
)
from taller_hpo.problemas.mlp_paridad import SEMILLA_ORACULO, SEMILLA_PRUEBA, _controlador_hilos
from taller_hpo.problemas.paridad import error_bayes, generar
from taller_hpo.semillas import Semillas

RAPIDO = {"n_train": 200, "n_val": 100, "n_oraculo": 10_000}
S = Semillas(datos=1, orden=2, init=3)

# Overrides por algoritmo para que entrenar() sea rápido en la prueba: cada
# algoritmo trae SOLO lo mínimo para que su rama entrene rápido.
CONFIGS: dict[str, dict] = {
    "logistica": {"algoritmo": "logistica"},
    "bosque": {"algoritmo": "bosque", "bosque_n_arboles": 8},
    "hgb": {"algoritmo": "hgb", "hgb_max_iter": 8},
    "mlp": {"algoritmo": "mlp", "mlp_epocas": 4, "ancho_1": 8},
    "knn": {"algoritmo": "knn"},
    "krr": {"algoritmo": "krr"},
}

METRICAS_ESPERADAS = {"exactitud_val", "exactitud_verdadera", "n_parametros", "epocas", "segundos"}


def _sin_reloj(metricas):
    return {k: v for k, v in metricas.items() if k != "segundos"}


def _cfg(algoritmo, **extra):
    return {**RAPIDO, "dispositivo": "cuda", **CONFIGS[algoritmo], **extra}


_CUDA_PRECALENTADA = False


def _precalentar_cuda():
    """Precalienta CUDA (contexto, cuBLAS) una sola vez para todo el módulo -- la
    primera llamada en "cuda" del proceso paga la inicialización de CUDA, bien
    por encima de lo que tardaría cualquier prueba individual; pagarla aquí,
    antes de la primera prueba marcada `gpu`, evita que esa prueba en
    particular parezca (falsamente) la más lenta del archivo."""
    global _CUDA_PRECALENTADA
    if _CUDA_PRECALENTADA:
        return
    _CUDA_PRECALENTADA = True
    entrenar(_cfg("bosque", bosque_n_arboles=2, n_train=10, n_val=10, n_oraculo=10), S)
    # mlp_batch_size=4 (<= n_train=10): entrenar_red recorta el batch en silencio
    # (`batch_efectivo = min(batch_size, n_train)`, sin aviso, ver el comentario
    # sobre batch_efectivo en paridad_torch.entrenar_red), así que el
    # precalentamiento no necesita esperar ningún aviso.
    entrenar(
        _cfg("mlp", mlp_epocas=1, ancho_1=2, mlp_batch_size=4, n_train=10, n_val=10, n_oraculo=10),
        S,
    )


@pytest.fixture(autouse=True)
def _precalentar_gpu_si_corresponde(request):
    """Dispara `_precalentar_cuda` antes de la primera prueba marcada `gpu` de este
    módulo. Las pruebas sin esa marca (validación de configuración, fórmulas
    puras) no tocan la tarjeta de video y no deben pagar su costo ni depender
    de que haya una."""
    if request.node.get_closest_marker("gpu") is not None:
        _precalentar_cuda()


# --- Contrato por algoritmo -------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_contrato_por_algoritmo(algoritmo):
    """El contrato completo de métricas incluye `exactitud_prueba` siempre que se pida
    un conjunto de prueba: `entrenar` la calcula por un camino aparte según el
    algoritmo (ver `paridad_torch.exactitud_en_dispositivo` y
    `_predecir_xgb_gpu`)."""
    m = entrenar(_cfg(algoritmo, n_prueba=500), S)
    assert set(m) == METRICAS_ESPERADAS | {"exactitud_prueba"}
    assert 0.0 <= m["exactitud_val"] <= 1.0
    assert 0.0 <= m["exactitud_verdadera"] <= 1.0
    assert 0.0 <= m["exactitud_prueba"] <= 1.0
    assert m["n_parametros"] > 0.0
    assert m["epocas"] >= 1.0
    assert m["segundos"] >= 0.0


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_misma_semilla_mismo_resultado(algoritmo):
    cfg = _cfg(algoritmo)
    assert _sin_reloj(entrenar(cfg, S)) == _sin_reloj(entrenar(cfg, S))


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_datos_cambia_el_resultado_en_todos_los_algoritmos(algoritmo):
    cfg = _cfg(algoritmo)
    base = _sin_reloj(entrenar(cfg, S))
    otra = _sin_reloj(entrenar(cfg, Semillas(9, S.orden, S.init)))
    assert base != otra


# random_state solo mueve el resultado cuando el algoritmo lo usa de verdad (ver
# la sección de contrato del docstring del módulo): knn no tiene parámetros que
# ajustar; krr resuelve un sistema lineal cerrado por Cholesky, sin azar;
# logistica converge al único óptimo de un problema convexo con
# `torch.optim.LBFGS`, que aquí siempre arranca de pesos en cero (ver
# `_ajustar_logistica_gpu`); hgb, traducido a `XGBClassifier` con los defaults
# de este módulo, tampoco subsamplea filas ni columnas (`subsample`/
# `colsample_*` sin pasar, ver la sección "hgb" del docstring del módulo).
@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["bosque", "mlp"])
def test_init_cambia_el_resultado_donde_el_modelo_lo_usa(algoritmo):
    cfg = _cfg(algoritmo)
    base = _sin_reloj(entrenar(cfg, S))
    otra = _sin_reloj(entrenar(cfg, Semillas(S.datos, S.orden, 9)))
    assert base != otra


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["logistica", "krr", "knn", "hgb"])
def test_init_no_afecta_donde_el_modelo_no_lo_usa(algoritmo):
    cfg = _cfg(algoritmo)
    base = _sin_reloj(entrenar(cfg, S))
    otra = _sin_reloj(entrenar(cfg, Semillas(S.datos, S.orden, 9)))
    assert base == otra


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_orden_cambia_el_resultado_solo_en_mlp(algoritmo):
    cfg = _cfg(algoritmo)
    base = _sin_reloj(entrenar(cfg, S))
    otra = _sin_reloj(entrenar(cfg, Semillas(S.datos, 9, S.init)))
    if algoritmo == "mlp":
        assert base != otra
    else:
        assert base == otra


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_placebo_no_cambia_nada(algoritmo):
    cfg = _cfg(algoritmo)
    assert _sin_reloj(entrenar({**cfg, "placebo": 0.0}, S)) == _sin_reloj(
        entrenar({**cfg, "placebo": 123.0}, S)
    )


# Pares de valores EXTREMOS por perilla, para test_cada_perilla_cambia_el_resultado:
# sin esto, la implementación podría ignorar un hiperparámetro entero (p. ej. construir
# el modelo de XGBoost sin pasarle `logistica_C`/`hgb_learning_rate`) sin que
# ninguna prueba lo note -- las pruebas de contrato no revisan el VALOR de la
# métrica, solo que exista y esté en rango.
_PERILLAS_POR_ALGORITMO = [
    ("logistica", "logistica_C", 0.001, 1000.0),
    ("hgb", "hgb_learning_rate", 0.02, 0.5),
    ("hgb", "hgb_max_iter", 5, 30),
    ("bosque", "bosque_profundidad_max", 2, 8),
    ("bosque", "bosque_min_muestras_hoja", 1, 30),
    ("knn", "knn_n_vecinos", 1, 15),
    ("krr", "krr_alpha", 0.01, 10.0),
    ("mlp", "mlp_learning_rate_init", 1e-4, 1e-2),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("algoritmo", "perilla", "valor_bajo", "valor_alto"), _PERILLAS_POR_ALGORITMO
)
def test_cada_perilla_cambia_el_resultado(algoritmo, perilla, valor_bajo, valor_alto):
    """Mueve cada perilla a dos valores EXTREMOS y compara el resultado completo --
    la única forma de detectar que la implementación construye el modelo sin
    pasarle esa perilla."""
    base = _cfg(algoritmo, **{perilla: valor_bajo})
    otra = _cfg(algoritmo, **{perilla: valor_alto})
    assert _sin_reloj(entrenar(base, S)) != _sin_reloj(entrenar(otra, S))


@pytest.mark.gpu
def test_parametros_de_bosque_hgb_knn_no_afectan_a_mlp():
    base = _cfg("mlp")
    otra = {
        **base,
        "bosque_n_arboles": 999,
        "hgb_max_iter": 999,
        "krr_alpha": 50.0,
        "knn_n_vecinos": 1,
    }
    assert _sin_reloj(entrenar(base, S)) == _sin_reloj(entrenar(otra, S))


@pytest.mark.gpu
def test_parametros_de_mlp_no_afectan_a_bosque():
    base = _cfg("bosque")
    otra = {**base, "ancho_1": 999, "mlp_epocas": 999, "mlp_learning_rate_init": 10.0}
    assert _sin_reloj(entrenar(base, S)) == _sin_reloj(entrenar(otra, S))


@pytest.mark.gpu
def test_ancho_2_y_3_no_afectan_con_n_capas_1():
    base = _cfg("mlp", n_capas=1)
    otra = {**base, "ancho_2": 999, "ancho_3": 999}
    assert _sin_reloj(entrenar(base, S)) == _sin_reloj(entrenar(otra, S))


@pytest.mark.gpu
def test_ancho_3_no_afecta_con_n_capas_2():
    base = _cfg("mlp", n_capas=2, ancho_1=8, ancho_2=4)
    otra = {**base, "ancho_3": 999}
    assert _sin_reloj(entrenar(base, S)) == _sin_reloj(entrenar(otra, S))


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_nadie_supera_el_techo_de_bayes(algoritmo):
    # Mismo patrón que test_nadie_supera_el_techo_de_bayes en
    # tests/test_mlp_paridad.py: "0.006: error Monte Carlo del oráculo con
    # 50 000 muestras (< 3 EE)." -- aquí con n_oraculo=20 000 (menos muestras,
    # más margen) para que la prueba siga siendo rápida.
    techo = 1.0 - error_bayes(DEFAULTS["d"], DEFAULTS["a"])
    cfg = _cfg(algoritmo, n_oraculo=20_000)
    m = entrenar(cfg, S)
    assert m["exactitud_verdadera"] <= techo + 0.01


def test_algoritmo_desconocido_falla():
    """`_validar_combinaciones` corre antes de resolver o validar el dispositivo, así
    que esta prueba no necesita una tarjeta de video."""
    with pytest.raises(ValueError, match="algoritmo"):
        entrenar({**RAPIDO, "algoritmo": "red_neuronal_convolucional"}, S)


@pytest.mark.parametrize("n_capas", [0, 4])
def test_n_capas_fuera_de_rango_falla(n_capas):
    with pytest.raises(ValueError, match="n_capas"):
        entrenar(_cfg("mlp", n_capas=n_capas), S)


def test_clave_desconocida_falla():
    with pytest.raises(KeyError, match="bosque_arboles"):
        entrenar({"bosque_arboles": 10}, S)


@pytest.mark.gpu
def test_reportar_no_se_llama_fuera_de_mlp():
    """Los algoritmos de un solo `fit` (todos salvo mlp) nunca deben llamar a
    `reportar`."""
    llamadas = []

    def reportar(epoca, exactitud_val):
        llamadas.append((epoca, exactitud_val))
        return False

    for algoritmo in ("logistica", "bosque", "hgb", "knn", "krr"):
        entrenar(_cfg(algoritmo), S, reportar)
    assert llamadas == []


@pytest.mark.gpu
def test_reportar_recibe_cada_epoca_en_mlp_y_puede_parar():
    """`entrenar` (algoritmo="mlp") debe pasar el `reportar` real a
    `paridad_torch.entrenar_red`, no `None` -- si no, la poda de Optuna se
    apagaría en silencio."""
    llamadas = []

    def reportar(epoca, exactitud_val):
        llamadas.append((epoca, exactitud_val))
        return epoca == 1

    m = entrenar(_cfg("mlp", mlp_epocas=5), S, reportar)
    assert [e for e, _ in llamadas] == [0, 1]
    assert all(0.0 <= v <= 1.0 for _, v in llamadas)
    assert m["epocas"] == 2.0
    assert m["exactitud_val"] == llamadas[-1][1]


@pytest.mark.gpu
def test_divergencia_del_mlp_se_distingue_de_otros_errores():
    """`cash_paridad.Divergencia` debe ser LA MISMA clase que
    `mlp_paridad.Divergencia` (reexportada, ver el docstring del módulo), nunca
    una clase local que la sombree: `pasos/05_buscar_el_modelo.py` atrapa
    `cash_paridad.Divergencia` para podar un trial que diverge sin abortar la
    búsqueda completa, así que si `cash_paridad` definiera su propia clase con
    el mismo nombre, una divergencia real de `entrenar_red` escaparía de ese
    `except` y el paso 05 se caería en el primer trial que diverja.
    `mlp_learning_rate_init=1e200` diverge de verdad en la época 0: el paso de Adam escrito
    a mano en `paridad_torch.entrenar_red` mueve los pesos a una magnitud que ya no cabe
    en `float64` en la primera actualización."""
    from taller_hpo.problemas import cash_paridad as cash_paridad_mod
    from taller_hpo.problemas import mlp_paridad

    assert cash_paridad_mod.Divergencia is mlp_paridad.Divergencia
    with pytest.raises(cash_paridad_mod.Divergencia, match="época 0"):
        entrenar(_cfg("mlp", mlp_learning_rate_init=1e200), S)


@pytest.mark.gpu
def test_n_parametros_mlp_arquitectura_conocida():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    m = entrenar(_cfg("mlp", n_capas=1, ancho_1=8), S)
    assert m["n_parametros"] == entradas * 8 + 8 + 8 * 1 + 1


@pytest.mark.gpu
def test_n_parametros_logistica_es_dimensiones_mas_intercepto():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    m = entrenar(_cfg("logistica"), S)
    assert m["n_parametros"] == entradas + 1


@pytest.mark.gpu
def test_n_parametros_knn_es_el_dataset_completo():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    m = entrenar(_cfg("knn"), S)
    assert m["n_parametros"] == RAPIDO["n_train"] * entradas


@pytest.mark.gpu
def test_n_parametros_krr_es_un_coeficiente_dual_por_muestra():
    m = entrenar(_cfg("krr"), S)
    assert m["n_parametros"] == float(RAPIDO["n_train"])


@pytest.mark.gpu
def test_n_parametros_bosque_crece_con_mas_arboles():
    chico = entrenar(_cfg("bosque", bosque_n_arboles=5), S)
    grande = entrenar(_cfg("bosque", bosque_n_arboles=30), S)
    assert grande["n_parametros"] > chico["n_parametros"]


@pytest.mark.gpu
def test_n_parametros_hgb_crece_con_mas_iteraciones():
    chico = entrenar(_cfg("hgb", hgb_max_iter=5), S)
    grande = entrenar(_cfg("hgb", hgb_max_iter=30), S)
    assert grande["n_parametros"] > chico["n_parametros"]


@pytest.mark.gpu
def test_prueba_finita_solo_si_se_pide_y_cerca_de_la_verdad():
    assert "exactitud_prueba" not in entrenar(_cfg("logistica"), S)

    m = entrenar(_cfg("logistica", n_prueba=8_000, n_oraculo=40_000), S)
    # Igual que en mlp_paridad: con miles de muestras la prueba estima la
    # verdad con holgura suficiente para no fallar de forma intermitente.
    assert abs(m["exactitud_prueba"] - m["exactitud_verdadera"]) < 0.05


@pytest.mark.gpu
def test_prueba_finita_en_mlp():
    m = entrenar(_cfg("mlp", n_prueba=2_000, n_oraculo=10_000), S)
    assert 0.0 <= m["exactitud_prueba"] <= 1.0


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["logistica", "knn", "krr", "mlp"])
def test_oraculo_y_prueba_en_dispositivo_contra_predictor_independiente(algoritmo, monkeypatch):
    """Protege `exactitud_verdadera`/`exactitud_prueba` de logistica/knn/krr/mlp, que
    calculan el oráculo en el dispositivo, contra un predictor independiente: sin esta
    prueba, `entrenar()` podría usar `SEMILLA_PRUEBA` en vez de `SEMILLA_ORACULO` (o al
    revés), o recortar `n_oraculo`, y nada lo notaría.

    El predictor independiente es `KNeighborsClassifier`/`KernelRidge` para knn/krr, `(w, b)`
    de `_ajustar_logistica_gpu` evaluados en NumPy para logistica, y `resultado.modelo`
    espiado de `paridad_torch.entrenar_red` para mlp. Se compara con `abs=1e-12`: el oráculo
    de producción suma por lotes y el de la prueba en un solo paso, y el orden de la suma
    puede cambiar el último bit."""
    n_prueba = 500
    n_oraculo = 5_000
    cfg = _cfg(algoritmo, n_prueba=n_prueba, n_oraculo=n_oraculo)
    c = {**DEFAULTS, **cfg}
    problema = {k: c[k] for k in ("d", "a", "k_ruido", "rotar")}
    x, y = generar(c["n_train"] + c["n_val"], semilla=S.datos, **problema)
    x_tr, y_tr = x[: c["n_train"]], y[: c["n_train"]]

    if algoritmo == "logistica":
        w_t, b_t = _ajustar_logistica_gpu(x_tr, y_tr, c["logistica_C"], "cuda")
        w, b = w_t.cpu().numpy(), b_t.cpu().item()

        def predecir_independiente(x):
            return ((np.asarray(x) @ w + b) > 0).astype(np.int64)

    elif algoritmo == "knn":
        modelo_sklearn = KNeighborsClassifier(n_neighbors=c["knn_n_vecinos"], n_jobs=1).fit(
            x_tr, y_tr
        )

        def predecir_independiente(x):
            return modelo_sklearn.predict(np.asarray(x)).astype(np.int64)

    elif algoritmo == "krr":
        gamma = _gamma_krr(x_tr, "scale")
        y_pm1 = (2.0 * y_tr - 1.0).astype(np.float64)
        modelo_sklearn = KernelRidge(kernel="rbf", alpha=c["krr_alpha"], gamma=gamma).fit(
            x_tr, y_pm1
        )

        def predecir_independiente(x):
            return (modelo_sklearn.predict(np.asarray(x)) > 0.0).astype(np.int64)

    else:  # "mlp"
        from taller_hpo.problemas import paridad_torch

        capturado: dict = {}
        original = paridad_torch.entrenar_red

        def espia(anchos, **kwargs):
            resultado = original(anchos, **kwargs)
            capturado["modelo"] = resultado.modelo
            return resultado

        monkeypatch.setattr(paridad_torch, "entrenar_red", espia)

        def predecir_independiente(x):
            # paridad_torch.subir (no torch.as_tensor): x puede venir de
            # paridad._muestras_oraculo, marcado de solo lectura -- ver el
            # docstring de subir para el UserWarning que -W error volvería un
            # error duro.
            modulo = capturado["modelo"]
            modulo.eval()
            xt = paridad_torch.subir(np.asarray(x), np.float64, "cuda")
            with torch.no_grad():
                logits = modulo(xt).squeeze(1)
            return (logits > 0).cpu().numpy().astype(np.int64)

    m = entrenar(cfg, S)

    riesgo = paridad.riesgo_verdadero(
        predecir_independiente, n=n_oraculo, semilla=SEMILLA_ORACULO, **problema
    )
    assert m["exactitud_verdadera"] == pytest.approx(1.0 - riesgo, abs=1e-12)

    x_pr, y_pr = generar(n_prueba, semilla=SEMILLA_PRUEBA, **problema)
    exactitud_prueba_esperada = float(np.mean(predecir_independiente(x_pr) == y_pr))
    assert m["exactitud_prueba"] == pytest.approx(exactitud_prueba_esperada, abs=1e-12)


# --- dispositivo: validación, resolución de "auto", errores claros --------------------


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize(
    "dispositivo",
    ["cpu", "tpu", pytest.param("gpu", id="valor-gpu"), "cuda:", "cuda:-1", "cudaa"],
)
def test_dispositivo_invalido_falla(dispositivo, algoritmo, monkeypatch):
    """`"cpu"` es inválido como `"tpu"` o `"cudaa"`: `dispositivo` tiene la forma
    `auto|cuda(:N)` y cualquier otro valor falla antes de generar datos.

    La prueba recorre los 6 algoritmos porque `mlp` tiene una segunda validación en
    `paridad_torch.entrenar_red` que ocultaría un fallo de `_PATRON_DISPOSITIVO`; los otros
    cinco solo dependen de este módulo. `id="valor-gpu"` evita confundir el valor "gpu" con
    la marca `@pytest.mark.gpu`. El parche sobre `generar` exige que `_validar_dispositivo`
    rechace el valor antes de generar ningún dato."""
    import taller_hpo.problemas.cash_paridad as cash_paridad_mod

    def _generar_no_debe_llamarse(*_args, **_kwargs):
        raise AssertionError(
            "generar() no debe llamarse: la validación de dispositivo debe fallar antes"
        )

    monkeypatch.setattr(cash_paridad_mod, "generar", _generar_no_debe_llamarse)
    with pytest.raises(ValueError, match="debe ser 'auto' o 'cuda'"):
        entrenar(_cfg(algoritmo, dispositivo=dispositivo), S)


def test_dispositivo_auto_es_el_default():
    assert DEFAULTS["dispositivo"] == "auto"


def test_auto_siempre_resuelve_a_cuda():
    """ "auto" no depende de qué haya instalado ni de qué algoritmo sortee la
    búsqueda: siempre es "cuda" (ver la sección "dispositivo" del docstring del
    módulo). Función pura -- no necesita una tarjeta de video real."""
    assert _resolver_dispositivo("auto") == "cuda"


def test_resolver_dispositivo_deja_cuda_explicito_intacto():
    assert _resolver_dispositivo("cuda") == "cuda"
    assert _resolver_dispositivo("cuda:2") == "cuda:2"


def test_xgboost_con_cuda_lee_build_info(monkeypatch):
    """`_xgboost_con_cuda` debe leer `xgboost.build_info()["USE_CUDA"]` de verdad, no
    devolver un valor fijo: `_validar_disponibilidad_gpu` depende de esto para no
    aceptar `bosque`/`hgb` en una máquina con un xgboost compilado sin CUDA. No
    necesita una tarjeta de video real: solo lee un atributo del paquete."""
    from taller_hpo.problemas.cash_paridad import _xgboost_con_cuda

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})
    assert _xgboost_con_cuda() is False

    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})
    assert _xgboost_con_cuda() is True


def test_dispositivo_cuda_sin_torch_falla_claro():
    """Sin torch instalado, "cuda" (o "auto", que siempre resuelve a "cuda") debe
    detenerse con un `ValueError` claro en español que diga qué falta (ver la
    sección "dispositivo" del docstring del módulo). Esta prueba corre en un
    subproceso con un buscador en `sys.meta_path` que
    bloquea torch (mismo patrón que `_BloqueaTorch` en tests/test_juguete.py):
    bloquear con `sys.modules["torch"] = None` en vez de esto rompería scipy,
    que hace `import torch` opcional puertas adentro y necesita ver un
    ImportError real. No necesita una tarjeta de video real: torch nunca llega
    a importarse."""
    bloqueador = """
import sys

class _BloqueaTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"bloqueado para pruebas: {name}", name=name)
        return None

sys.meta_path.insert(0, _BloqueaTorch())
"""
    codigo = (
        bloqueador
        + """
from taller_hpo.problemas.cash_paridad import entrenar
from taller_hpo.semillas import Semillas

S = Semillas(1, 2, 3)
RAPIDO = {"n_train": 80, "n_val": 30, "n_oraculo": 200}
try:
    entrenar({**RAPIDO, "algoritmo": "mlp", "dispositivo": "auto"}, S)
    raise SystemExit("FALLO: dispositivo='auto' sin torch no lanzo ValueError")
except ValueError as error:
    assert "torch no" in str(error), str(error)
print("OK")
"""
    )
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", codigo],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


@pytest.mark.gpu
def test_cuda_explicito_con_indice_fuera_de_rango_falla():
    n = torch.cuda.device_count()
    with pytest.raises(ValueError, match="tarjeta"):
        entrenar(_cfg("mlp", dispositivo=f"cuda:{n + 5}"), S)


@pytest.mark.gpu
def test_cuda_pedido_sin_xgboost_falla_claro():
    """`bosque`/`hgb` sin xgboost -> ValueError claro; `mlp` no lo necesita.

    Simulado en subproceso con un buscador en `sys.meta_path` que bloquea
    xgboost (nunca `sys.modules["xgboost"] = None`: mismo patrón que
    `_BloqueaTorch` en tests/test_juguete.py, adaptado a xgboost).

    Prueba los dos algoritmos, `bosque` y `hgb`: las dos ramas comparten la misma
    validación (`_validar_disponibilidad_gpu`), pero probar solo una dejaría pasar
    un bug que afecte solo a `hgb` (p. ej. un `elif algoritmo == "bosque"` que se
    olvide del caso `hgb`) sin que la suite lo note.
    """
    bloqueador = """
import sys

class _BloqueaXgboost:
    def find_spec(self, name, path=None, target=None):
        if name == "xgboost" or name.startswith("xgboost."):
            raise ModuleNotFoundError(f"bloqueado para pruebas: {name}", name=name)
        return None

sys.meta_path.insert(0, _BloqueaXgboost())
"""
    codigo = (
        bloqueador
        + """
from taller_hpo.problemas.cash_paridad import entrenar
from taller_hpo.semillas import Semillas

S = Semillas(1, 2, 3)
for algoritmo in ("bosque", "hgb"):
    cfg = {"n_train": 50, "n_val": 20, "n_oraculo": 100, "dispositivo": "cuda",
           "algoritmo": algoritmo, "bosque_n_arboles": 2, "hgb_max_iter": 2}
    try:
        entrenar(cfg, S)
        raise SystemExit(f"FALLO: {algoritmo} no lanzó ValueError")
    except ValueError as error:
        assert "xgboost" in str(error), str(error)

cfg_mlp = {"n_train": 20, "n_val": 10, "n_oraculo": 20, "dispositivo": "cuda",
           "algoritmo": "mlp", "mlp_epocas": 1, "ancho_1": 2}
entrenar(cfg_mlp, S)  # mlp no necesita xgboost: no debe fallar.
print("OK")
"""
    )
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", codigo],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


@pytest.mark.gpu
def test_entrenar_logistica_knn_krr_mlp_no_carga_sklearn_ni_threadpoolctl():
    """`logistica`, `knn`, `krr` y `mlp` nunca importan `xgboost` dentro de
    `entrenar()` (solo `bosque`/`hgb` lo hacen, ver la sección XGBoost del
    docstring del módulo), así que en un proceso limpio `sklearn` y
    `threadpoolctl` deben quedar fuera de `sys.modules` al terminar. Cada
    algoritmo corre en su propio SUBPROCESO limpio (mismo patrón que
    `tests/test_contrato_mlp.py`), para que otro archivo de pruebas que ya
    haya importado `sklearn` en este mismo proceso no contamine el
    resultado. Con esto, un import perezoso de `sklearn` metido dentro de
    `entrenar_red` o de la rama de `knn` no pasa inadvertido."""
    for algoritmo in ("logistica", "knn", "krr", "mlp"):
        cfg = {**RAPIDO, "dispositivo": "cuda", **CONFIGS[algoritmo]}
        codigo = (
            "import sys\n"
            "from taller_hpo.problemas.cash_paridad import entrenar\n"
            "from taller_hpo.semillas import Semillas\n"
            f"entrenar({cfg!r}, Semillas(1, 2, 3))\n"
            "assert 'sklearn' not in sys.modules, "
            "sorted(m for m in sys.modules if 'sklearn' in m)\n"
            "assert 'threadpoolctl' not in sys.modules\n"
            "print('OK')\n"
        )
        resultado = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, timeout=60
        )
        assert resultado.returncode == 0, (algoritmo, resultado.stderr)
        assert "OK" in resultado.stdout, (algoritmo, resultado.stdout)


@pytest.mark.gpu
def test_entrenar_bosque_hgb_si_carga_sklearn_y_threadpoolctl():
    """Esta prueba fija, en vez de solo documentar, la excepción de la sección
    XGBoost del docstring del módulo: `bosque`/`hgb` importan `xgboost`, y ese
    import por sí solo ya carga `sklearn`/`threadpoolctl` (`XGBClassifier`
    hereda de `sklearn.base`), así que entrenar con cualquiera de los dos deja
    los dos módulos en `sys.modules`. Esta prueba corre en un SUBPROCESO
    limpio, igual que la de arriba, por el mismo motivo: sin este control, la
    excepción quedaría escrita en prosa sin que ninguna prueba la comprobara."""
    for algoritmo in ("bosque", "hgb"):
        cfg = {**RAPIDO, "dispositivo": "cuda", **CONFIGS[algoritmo]}
        codigo = (
            "import sys\n"
            "from taller_hpo.problemas.cash_paridad import entrenar\n"
            "from taller_hpo.semillas import Semillas\n"
            f"entrenar({cfg!r}, Semillas(1, 2, 3))\n"
            "assert 'sklearn' in sys.modules\n"
            "assert 'threadpoolctl' in sys.modules\n"
            "print('OK')\n"
        )
        resultado = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, timeout=60
        )
        assert resultado.returncode == 0, (algoritmo, resultado.stderr)
        assert "OK" in resultado.stdout, (algoritmo, resultado.stdout)


@pytest.mark.parametrize("dispositivo", ["auto", "cuda"])
@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_sin_tarjeta_de_video_falla_en_espanol_para_todo_algoritmo(
    algoritmo, dispositivo, monkeypatch
):
    """Protege el chequeo central de `_validar_disponibilidad_gpu`
    (`if not torch.cuda.is_available(): raise ValueError(...)`) para los seis
    algoritmos, no solo para los que ya cubren otras pruebas de este archivo:
    sin esta prueba, ese chequeo roto (por ejemplo `if False:` en vez de la
    condición real) deja a `bosque`/`hgb` seguir adelante con solo un aviso de
    xgboost, en vez de detenerse con el error claro que exige la
    sección "dispositivo" del docstring del módulo. Parcha
    `torch.cuda.is_available` para simular la ausencia de tarjeta de video sin
    tocar una real, y parcha `generar` para que reviente si algo la llama --
    mismo patrón que
    `test_dispositivo_cuda_sin_gpu_falla_en_espanol_antes_de_entrenar` en
    tests/test_paridad_torch.py. Sin la marca `gpu`: corre en la integración
    continua (`pytest -m "not gpu"`) y en cualquier máquina, con o sin tarjeta
    de video real, porque nunca llega a tocarla."""
    import taller_hpo.problemas.cash_paridad as cash_paridad_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _generar_no_debe_llamarse(*_args, **_kwargs):
        raise AssertionError(
            "generar() no debe llamarse: la validación de la tarjeta de video debe fallar antes"
        )

    monkeypatch.setattr(cash_paridad_mod, "generar", _generar_no_debe_llamarse)
    with pytest.raises(ValueError, match="tarjeta de video"):
        entrenar({**RAPIDO, **CONFIGS[algoritmo], "dispositivo": dispositivo}, S)


def test_subproceso_sin_gpu_visible_falla_en_espanol():
    """Mismo chequeo que la prueba anterior, pero con una ausencia de tarjeta de
    video REAL en vez de simulada: un subproceso propio con
    `CUDA_VISIBLE_DEVICES=""` (patrón de
    `test_subproceso_sin_gpu_visible_falla_al_importar` en
    tests/test_juguete.py) hace que `torch.cuda.is_available()` de verdad
    devuelva `False`, sin depender de un monkeypatch. Usa `bosque`: ejercita
    además la validación de xgboost, que nunca llega a alcanzarse porque la
    tarjeta de video falla primero."""
    codigo = (
        "from taller_hpo.problemas.cash_paridad import entrenar\n"
        "from taller_hpo.semillas import Semillas\n"
        "S = Semillas(1, 2, 3)\n"
        "cfg = {'n_train': 50, 'n_val': 20, 'n_oraculo': 100, 'dispositivo': 'cuda', "
        "'algoritmo': 'bosque', 'bosque_n_arboles': 2}\n"
        "try:\n"
        "    entrenar(cfg, S)\n"
        "    raise SystemExit('FALLO: entrenar no lanzo ValueError sin tarjeta de video')\n"
        "except ValueError as error:\n"
        "    assert 'tarjeta de video' in str(error), str(error)\n"
        "print('OK')\n"
    )
    entorno = dict(os.environ)
    entorno["CUDA_VISIBLE_DEVICES"] = ""
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", codigo],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "OK" in resultado.stdout


@pytest.mark.gpu
def test_determinismo_entre_procesos_cuda():
    """Misma config y semillas -> métricas idénticas en DOS PROCESOS distintos."""
    codigo = (
        "import json\n"
        "from taller_hpo.problemas.cash_paridad import entrenar\n"
        "from taller_hpo.semillas import Semillas\n"
        "cfg = {'n_train': 150, 'n_val': 60, 'n_oraculo': 2000, 'dispositivo': 'cuda', "
        "'algoritmo': 'bosque', 'bosque_n_arboles': 8}\n"
        "r = entrenar(cfg, Semillas(9, 8, 7))\n"
        "del r['segundos']\n"
        "print(json.dumps(r, sort_keys=True))\n"
    )
    salidas = []
    for _ in range(2):
        resultado = subprocess.run(
            [sys.executable, "-W", "error", "-c", codigo],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resultado.returncode == 0, resultado.stderr
        salidas.append(json.loads(resultado.stdout))
    assert salidas[0] == salidas[1]


# --- Equivalencia contra scikit-learn (referencia) --------------------------------------


_S_EQUIV = Semillas(1, 2, 3)
_CFG_EQUIV = {"n_train": 400, "n_val": 150, "n_oraculo": 20_000}


def _datos_equivalencia(**overrides):
    cfg = {**DEFAULTS, **_CFG_EQUIV, **overrides}
    problema = {k: cfg[k] for k in ("d", "a", "k_ruido", "rotar")}
    x, y = generar(cfg["n_train"] + cfg["n_val"], semilla=_S_EQUIV.datos, **problema)
    x_tr, y_tr = x[: cfg["n_train"]], y[: cfg["n_train"]]
    x_va, y_va = x[cfg["n_train"] :], y[cfg["n_train"] :]
    return cfg, problema, x_tr, y_tr, x_va, y_va


@pytest.mark.gpu
@pytest.mark.parametrize("capital_c", [1.0, 5.0, 0.2])
def test_logistica_coeficientes_y_predicciones_contra_sklearn(capital_c):
    """Llama a `_ajustar_logistica_gpu` -- la función real de producción, nunca una
    copia del bucle de LBFGS reescrita dentro de la prueba -- para que un bug real
    (p. ej. que ignorara `capital_c`) no pueda pasar sin que esta prueba lo note, y
    compara sus `(w, b)` contra un `LogisticRegression` forzado a converger cerca
    del óptimo exacto (`tol=1e-12`, no el default `tol=1e-4`): con las
    tolerancias de este módulo para `torch.optim.LBFGS`
    (`tolerance_grad=1e-10, tolerance_change=1e-14`, ver la sección "logistica" del
    docstring del módulo), `max|dw| <= 1.7e-7` medido en 6/6 configs de
    `n_train`x`C`, muy por debajo de `rtol=1e-4, atol=1e-9`. El intercepto (`b`) es
    mucho más chico en magnitud que `w` (~1e-4 con estos datos), así que se compara
    con una tolerancia absoluta propia en vez de `rtol` (que en un valor tan chico
    sería casi cero): medido, `|b_gpu - b_ref|` entre 1.0e-7 y 1.3e-7 en las 3
    configs de esta prueba, de ahí `< 1e-6` con margen. Las predicciones se
    comparan en dos direcciones: contra ESE sklearn de referencia (>= 99.9%,
    medido 100% en 6/6) y contra un `LogisticRegression` con su `tol=1e-4` por
    defecto (>= 99.5%, con margen sobre el 99.89% mínimo medido) -- la referencia
    que de verdad importa para verificar esta implementación.
    """
    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(
        algoritmo="logistica", logistica_C=capital_c
    )
    x_oraculo, _ = generar(cfg["n_oraculo"], semilla=999_999, **problema)
    with _controlador_hilos().limit(limits=1):
        modelo_defecto = LogisticRegression(
            C=capital_c, max_iter=_LOGISTICA_MAX_ITER, random_state=_S_EQUIV.init
        ).fit(x_tr, y_tr)
        # tol=1e-12 (no el default 1e-4): un ajuste de sklearn forzado a converger
        # cerca del óptimo exacto del MISMO objetivo convexo, para poder comparar
        # números contra un LBFGS que sí converge -- ver la sección "logistica" del
        # docstring del módulo.
        modelo_referencia = LogisticRegression(
            C=capital_c, tol=1e-12, max_iter=10_000, random_state=_S_EQUIV.init
        ).fit(x_tr, y_tr)
        pred_referencia = modelo_referencia.predict(x_oraculo)
        pred_defecto = modelo_defecto.predict(x_oraculo)
    w_ref = modelo_referencia.coef_.ravel()
    b_ref = modelo_referencia.intercept_[0]

    w_gpu_t, b_gpu_t = _ajustar_logistica_gpu(x_tr, y_tr, capital_c, "cuda")
    w_gpu = w_gpu_t.cpu().numpy()
    b_gpu = b_gpu_t.cpu().item()

    assert np.allclose(w_gpu, w_ref, rtol=1e-4, atol=1e-9)
    assert abs(b_gpu - b_ref) < 1e-6

    pred_gpu = ((x_oraculo @ w_gpu + b_gpu) > 0).astype(np.int64)

    assert np.mean(pred_gpu == pred_referencia) >= 0.999
    assert np.mean(pred_gpu == pred_defecto) >= 0.995


@pytest.mark.gpu
@pytest.mark.parametrize("capital_c", [1.0, 0.2])
def test_logistica_exactitud_val_sobre_validacion(capital_c):
    """Protege `exactitud_val` de `logistica` con un cálculo INDEPENDIENTE sobre las
    filas de VALIDACIÓN: sin esta prueba, un error que validara sobre el propio
    `x_tr`/`y_tr` (en vez de `x_va`/`y_va`) pasaría el resto de la suite sin que
    nada lo note -- `test_contrato_por_algoritmo` solo revisa que `exactitud_val`
    esté en `[0, 1]`, nunca su VALOR. Llama a `_ajustar_logistica_gpu` -- la
    función real de producción, nunca una copia del LBFGS reescrita aquí -- para
    recalcular `(w, b)` y evaluar sobre `x_va`/`y_va` por su cuenta."""
    cfg, _, x_tr, y_tr, x_va, y_va = _datos_equivalencia(
        algoritmo="logistica", logistica_C=capital_c
    )
    m = entrenar({**cfg, "dispositivo": "cuda"}, _S_EQUIV)
    w, b = _ajustar_logistica_gpu(x_tr, y_tr, capital_c, "cuda")
    xt = torch.as_tensor(x_va, dtype=torch.float64, device="cuda")
    pred = ((xt @ w + b) > 0).cpu().numpy().astype(np.int64)
    assert m["exactitud_val"] == float(np.mean(pred == y_va))


@pytest.mark.gpu
@pytest.mark.parametrize("k", [1, 2, 4, 5, 6, 8, 10])
def test_knn_predicciones_identicas_a_sklearn_incluso_con_empates_de_voto(k):
    """Un empate de VOTO (posible con `k` par: p. ej. 3 votos a 3 de 6 vecinos) es la
    única rama donde `knn` podría desempatar distinto de `KNeighborsClassifier`
    (ver la sección "knn" del docstring del módulo) -- por eso se prueban `k`
    pares además de impares, y se VERIFICA (no se asume) que sí hay empates de
    voto en los datos de esta config: probar solo valores sin empates dejaría
    esa rama sin ejercitar.
    """
    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo="knn", knn_n_vecinos=k)
    predecir_gpu, _, _, _ = _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")

    x_oraculo, _ = generar(cfg["n_oraculo"], semilla=999_999, **problema)
    # _controlador_hilos().limit(limits=1): KNeighborsClassifier.predict/kneighbors
    # sobre n_oraculo=20_000 filas corre BLAS por dentro (distancias), sin límite de
    # hilos propio más allá del n_jobs=1 de joblib -- por prevención en una máquina
    # compartida.
    with _controlador_hilos().limit(limits=1):
        modelo_sklearn = KNeighborsClassifier(n_neighbors=k, n_jobs=1).fit(x_tr, y_tr)
        pred_sklearn = modelo_sklearn.predict(x_oraculo)
        _, idx = modelo_sklearn.kneighbors(x_oraculo)
    pred_gpu = predecir_gpu(x_oraculo)
    assert np.array_equal(pred_gpu, pred_sklearn)

    votos = y_tr[idx].sum(axis=1)
    if k % 2 == 0:
        # Con k par SÍ hay empates de voto en este dataset -- se verifica en vez
        # de asumirlo, para que la rama de desempate se ejercite de verdad.
        assert (2 * votos == k).any()


def test_knn_sin_empates_de_distancia_en_la_frontera_k_esima():
    """`kneighbors(k+1)` usa los `n_train` puntos de referencia completos y no
    reserva una matriz de distancias densa, así que verifica la frontera k-ésima
    real sin el costo de memoria de calcularla a mano. Pura computación de
    scikit-learn: no necesita una tarjeta de video."""
    k = DEFAULTS["knn_n_vecinos"]
    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo="knn", knn_n_vecinos=k)
    x_oraculo, _ = generar(cfg["n_oraculo"], semilla=999_999, **problema)
    # Mismo límite de hilos que arriba (por prevención): kneighbors() sobre 20_000
    # filas corre BLAS por dentro.
    with _controlador_hilos().limit(limits=1):
        modelo_sklearn = KNeighborsClassifier(n_neighbors=k, n_jobs=1).fit(x_tr, y_tr)
        dist, _ = modelo_sklearn.kneighbors(x_oraculo, n_neighbors=k + 1)
    assert np.all(dist[:, k - 1] < dist[:, k])


@pytest.mark.gpu
@pytest.mark.parametrize("alpha", [1.0, 0.1, 5.0])
def test_krr_predicciones_identicas_a_sklearn_kernel_ridge(alpha):
    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo="krr", krr_alpha=alpha)
    predecir_gpu, _, _, _ = _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")

    gamma = _gamma_krr(x_tr, "scale")
    y_pm1 = (2.0 * y_tr - 1.0).astype(np.float64)
    x_oraculo, _ = generar(cfg["n_oraculo"], semilla=999_999, **problema)
    # _controlador_hilos().limit(limits=1): KernelRidge.fit/predict sobre 20 000 puntos
    # de consulta corre BLAS por dentro, sin límite de hilos propio -- mismo motivo de
    # precaución que test_knn_predicciones_identicas_a_sklearn_incluso_con_empates_de_voto,
    # por prevención en una máquina compartida.
    with _controlador_hilos().limit(limits=1):
        modelo_sklearn = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(x_tr, y_pm1)
        puntajes_sklearn = modelo_sklearn.predict(x_oraculo)
    pred_sklearn = (puntajes_sklearn > 0.0).astype(np.int64)
    pred_gpu = predecir_gpu(x_oraculo)
    assert np.array_equal(pred_gpu, pred_sklearn)


def test_gamma_krr_coincide_con_la_formula_scale_de_svc():
    """`_gamma_krr` debe reproducir la fórmula real de `SVC(gamma="scale")` (ver la
    sección "krr" del docstring del módulo) -- sin esta prueba, una fórmula de
    gamma equivocada pasaría el resto de la suite sin que nada fallara: los
    demás algoritmos no dependen de `_gamma_krr`. Función pura: no necesita una
    tarjeta de video."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 13))
    y = rng.integers(0, 2, 100)
    esperado = SVC(gamma="scale").fit(x, y)._gamma
    assert _gamma_krr(x, "scale") == pytest.approx(esperado)


def test_gamma_krr_con_x_constante_da_uno():
    x = np.zeros((10, 4))
    assert _gamma_krr(x, "scale") == 1.0


@pytest.mark.gpu
def test_krr_gamma_solo_acepta_scale():
    """`_gamma_krr` solo se llama después de resolver y validar el dispositivo (ver
    `entrenar`), así que este error también necesita una tarjeta de video para
    surgir a través de `entrenar()`."""
    with pytest.raises(ValueError, match="krr_gamma"):
        entrenar(_cfg("krr", krr_gamma="auto"), S)


@pytest.mark.gpu
def test_mlp_usa_entrenar_red_de_paridad_torch(monkeypatch):
    """`entrenar` (algoritmo="mlp") debe reusar `paridad_torch.entrenar_red`, no
    reimplementar su propio bucle -- mismo patrón que
    `test_entrenar_usa_entrenar_red_con_anchos_repetidos` en
    tests/test_paridad_torch.py."""
    from taller_hpo.problemas import paridad_torch

    vistos = {}
    original = paridad_torch.entrenar_red

    def espia(anchos, **kwargs):
        vistos["anchos"] = anchos
        return original(anchos, **kwargs)

    monkeypatch.setattr(paridad_torch, "entrenar_red", espia)
    entrenar(_cfg("mlp", n_capas=2, ancho_1=6, ancho_2=4), S)
    assert vistos["anchos"] == (6, 4)


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["knn", "krr"])
def test_knn_krr_prediccion_por_bloques_no_cambia_el_resultado(algoritmo, monkeypatch):
    """`_logit_knn`/`_logit_krr` trocean la consulta con `_filas_por_bloque` para no
    reservar una matriz de distancias `n_consulta x n_train` completa de una sola
    vez (ver su docstring: hasta 2.96 GiB en `krr` con los DEFAULTS de este módulo,
    y 8.83 GiB con `n_train=6000`, en una tarjeta de video de 24 GiB compartida
    con otros procesos). Fuerza un bloque de 7 filas (`monkeypatch`, en vez del
    real `2**24 // n_train`) para ejercitar el camino con VARIOS bloques --
    incluido el último, parcial -- y verifica que la predicción no cambia frente
    al camino sin trocear (un bloque de una sola pieza, `10**9` filas)."""
    import taller_hpo.problemas.cash_paridad as cash_paridad_mod

    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo=algoritmo)
    x_oraculo, _ = generar(2000, semilla=999_999, **problema)
    predecir, _, _, _ = _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")

    monkeypatch.setattr(cash_paridad_mod, "_filas_por_bloque", lambda n: 10**9)
    pred_entero = predecir(x_oraculo)

    monkeypatch.setattr(cash_paridad_mod, "_filas_por_bloque", lambda n: 7)
    pred_bloques = predecir(x_oraculo)

    assert np.array_equal(pred_entero, pred_bloques)


# --- bosque/hgb: determinismo, mapeo de parametros, sin aviso de xgboost ---------------


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["logistica", "knn", "krr"])
def test_logistica_knn_krr_calculan_de_verdad_en_cuda(algoritmo, monkeypatch):
    """El cómputo corre de verdad EN `cuda`, y no solo el dispositivo resuelto dice
    `cuda`. Esta prueba espía `paridad_torch.subir` (el único punto donde un array
    de NumPy sube a un dispositivo en este módulo) y verifica que el `dispositivo`
    recibido sea 'cuda' en cada llamada dentro de `_ajustar_baseline_gpu`."""
    from taller_hpo.problemas import paridad_torch

    vistos = set()
    original = paridad_torch.subir

    def espia(arr, dtype, dispositivo):
        vistos.add(dispositivo)
        return original(arr, dtype, dispositivo)

    monkeypatch.setattr(paridad_torch, "subir", espia)
    cfg, _, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo=algoritmo)
    _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")
    assert vistos == {"cuda"}


@pytest.mark.gpu
def test_logistica_calcula_la_perdida_de_verdad_en_cuda(monkeypatch):
    """Espiar `subir` (arriba) solo prueba que los datos SUBEN a cuda, no que el
    cómputo pesado corre ahí: una implementación que moviera `xt`/`w`/`b` al
    fuera de cuda para cada evaluación de la cerradura de LBFGS y trajera el
    resultado de vuelta pasaría esa prueba sin que nada lo notara. Esta prueba
    espía `BCEWithLogitsLoss.forward` (la pérdida que `torch.optim.LBFGS`
    evalúa en cada paso) y verifica el dispositivo real de la entrada."""
    original = torch.nn.BCEWithLogitsLoss.forward
    vistos = set()

    def espia(self, entrada, objetivo):
        vistos.add(entrada.device.type)
        return original(self, entrada, objetivo)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", espia)
    _, _, x_tr, y_tr, _, _ = _datos_equivalencia(algoritmo="logistica")
    w, b = _ajustar_logistica_gpu(x_tr, y_tr, 1.0, "cuda")
    assert vistos == {"cuda"}
    assert w.device.type == "cuda"
    assert b.device.type == "cuda"


@pytest.mark.gpu
def test_knn_calcula_distancias_y_topk_de_verdad_en_cuda(monkeypatch):
    """Mismo motivo que la prueba de `logistica` de arriba: `knn` calcula
    distancias (`torch.cdist`) y los k más cercanos (`torch.topk`), y las dos
    deben correr EN cuda. Esta prueba espía las dos funciones y verifica el
    dispositivo real de los tensores que reciben."""
    vistos_cdist = []
    vistos_topk = []
    original_cdist = torch.cdist
    original_topk = torch.topk

    def espia_cdist(x1, x2, *args, **kwargs):
        vistos_cdist.append((x1.device.type, x2.device.type))
        return original_cdist(x1, x2, *args, **kwargs)

    def espia_topk(x, *args, **kwargs):
        vistos_topk.append(x.device.type)
        return original_topk(x, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", espia_cdist)
    monkeypatch.setattr(torch, "topk", espia_topk)
    cfg, problema, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo="knn")
    predecir, _, _, _ = _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")
    x_oraculo, _ = generar(500, semilla=999_999, **problema)
    predecir(x_oraculo)
    assert vistos_cdist, "torch.cdist no se llamó"
    assert all(d == ("cuda", "cuda") for d in vistos_cdist)
    assert vistos_topk, "torch.topk no se llamó"
    assert all(d == "cuda" for d in vistos_topk)


@pytest.mark.gpu
def test_krr_factoriza_cholesky_de_verdad_en_cuda(monkeypatch):
    """`krr` arma su matriz de kernel con `torch.cdist`/`torch.exp` y resuelve el
    sistema con `torch.linalg.cholesky` + `torch.cholesky_solve`: las cuatro
    deben correr EN cuda. Espiar solo `cholesky`/`cholesky_solve` (como hace
    `test_knn_calcula_distancias_y_topk_de_verdad_en_cuda` con `cdist`/`topk`
    para `knn`) no basta aquí: una implementación que armara la matriz de
    kernel fuera de cuda y solo subiera el resultado a cuda antes de
    factorizarlo pasaría esas dos espías sin que nada lo note, porque el
    resultado final se ve idéntico sin importar dónde se calculó la matriz."""
    vistos_cdist = []
    vistos_exp = []
    vistos_cholesky = []
    vistos_solve = []
    original_cdist = torch.cdist
    original_exp = torch.exp
    original_cholesky = torch.linalg.cholesky
    original_solve = torch.cholesky_solve

    def espia_cdist(x1, x2, *args, **kwargs):
        vistos_cdist.append((x1.device.type, x2.device.type))
        return original_cdist(x1, x2, *args, **kwargs)

    def espia_exp(entrada, *args, **kwargs):
        vistos_exp.append(entrada.device.type)
        return original_exp(entrada, *args, **kwargs)

    def espia_cholesky(a, *args, **kwargs):
        vistos_cholesky.append(a.device.type)
        return original_cholesky(a, *args, **kwargs)

    def espia_solve(b, u, *args, **kwargs):
        vistos_solve.append((b.device.type, u.device.type))
        return original_solve(b, u, *args, **kwargs)

    monkeypatch.setattr(torch, "cdist", espia_cdist)
    monkeypatch.setattr(torch, "exp", espia_exp)
    monkeypatch.setattr(torch.linalg, "cholesky", espia_cholesky)
    monkeypatch.setattr(torch, "cholesky_solve", espia_solve)
    cfg, _, x_tr, y_tr, x_va, y_va = _datos_equivalencia(algoritmo="krr")
    _ajustar_baseline_gpu(cfg, _S_EQUIV, x_tr, y_tr, x_va, y_va, "cuda")
    assert vistos_cdist, "torch.cdist no se llamó"
    assert all(d == ("cuda", "cuda") for d in vistos_cdist)
    assert vistos_exp, "torch.exp no se llamó"
    assert all(d == "cuda" for d in vistos_exp)
    assert vistos_cholesky, "torch.linalg.cholesky no se llamó"
    assert all(d == "cuda" for d in vistos_cholesky)
    assert vistos_solve, "torch.cholesky_solve no se llamó"
    assert all(d == ("cuda", "cuda") for d in vistos_solve)


def _espiar_fit_xgb(monkeypatch):
    """Esta función espía `XGBClassifier.fit` y devuelve la lista de modelos ya
    entrenados (en orden), para que la prueba inspeccione su booster real
    después de `entrenar()`."""
    import xgboost as xgb

    modelos = []
    original_fit = xgb.XGBClassifier.fit

    def espia_fit(self, *args, **kwargs):
        resultado = original_fit(self, *args, **kwargs)
        modelos.append(self)
        return resultado

    monkeypatch.setattr(xgb.XGBClassifier, "fit", espia_fit)
    return modelos


@pytest.mark.gpu
def test_bosque_no_lanza_el_aviso_de_deprecacion_de_xgbrfclassifier():
    """XGBRFClassifier está deprecado en xgboost 3.4.1 (FutureWarning al construirlo,
    medido) -- este módulo NUNCA lo usa (ver la sección "bosque" del docstring),
    así que entrenar() con algoritmo="bosque" debe correr limpio bajo -W error."""
    entrenar(_cfg("bosque", bosque_n_arboles=4), S)  # no debe lanzar nada.


@pytest.mark.gpu
def test_predecir_xgb_gpu_usa_nthread_1(monkeypatch):
    """`_predecir_xgb_gpu` construye el `DMatrix` de predicción con `nthread=1`, la
    convención `n_jobs=1` de todo el módulo en una máquina compartida con otros
    procesos (ver su docstring) -- sin pasarlo, `xgboost.DMatrix` toma su propio
    default (`nthread=-1`, todos los núcleos). Entrena aparte, con el modelo ya
    ajustado ANTES de espiar, para que el espía solo vea el `DMatrix` de la
    predicción -- no los que XGBoost arma puertas adentro durante `.fit()`."""
    import xgboost as xgb

    cfg, _, x_tr, y_tr, x_va, _ = _datos_equivalencia(algoritmo="bosque", bosque_n_arboles=4)
    modelo = xgb.XGBClassifier(
        n_estimators=1,
        num_parallel_tree=4,
        learning_rate=1.0,
        tree_method="hist",
        device="cuda",
        random_state=0,
        n_jobs=1,
    )
    modelo.fit(x_tr, y_tr)

    original_init = xgb.DMatrix.__init__
    vistos = []

    def espia_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        vistos.append(self.nthread)

    monkeypatch.setattr(xgb.DMatrix, "__init__", espia_init)
    predecir = _predecir_xgb_gpu(modelo)
    predecir(x_va)
    assert vistos == [1], vistos


@pytest.mark.gpu
@pytest.mark.parametrize("dispositivo", ["auto", "cuda"])
@pytest.mark.parametrize("algoritmo", ["bosque", "hgb"])
def test_bosque_hgb_entrenan_de_verdad_en_el_device_cuda(algoritmo, dispositivo, monkeypatch):
    """Verifica, para bosque/hgb, el device REAL que reporta el booster entrenado
    (`Booster.save_config()`), no solo que `entrenar()` no haya lanzado ningún
    error.

    Esta prueba corre con `dispositivo="auto"`, además de "cuda" explícito:
    `pasos/05_buscar_el_modelo.py` nunca pasa `dispositivo` (su `FIJOS` está
    vacío), así que en producción
    bosque/hgb siempre resuelven desde "auto". Construir la config solo con
    `_cfg` (que fuerza "cuda" a mano, ver su definición) dejaría esa ruta sin
    ejercitar: una rama que convirtiera "auto" en otro valor que "cuda" dentro de bosque/hgb
    (o que pasara `c["dispositivo"]` sin resolver a XGBoost) pasaría
    inadvertida si ninguna prueba entrenara alguna vez desde "auto"."""
    modelos = _espiar_fit_xgb(monkeypatch)
    entrenar({**RAPIDO, **CONFIGS[algoritmo], "dispositivo": dispositivo}, S)
    cfg_booster = json.loads(modelos[-1].get_booster().save_config())
    assert cfg_booster["learner"]["generic_param"]["device"] == "cuda:0"


@pytest.mark.gpu
def test_bosque_es_una_sola_ronda_de_boosting_con_n_arboles_paralelos(monkeypatch):
    """`bosque` emula un bosque aleatorio con UNA sola ronda de boosting y
    `num_parallel_tree` árboles en paralelo, nunca boosting real (ver la sección
    "bosque" del docstring del módulo): `num_boosted_rounds()` debe ser SIEMPRE 1
    y el número de árboles en el volcado debe ser exactamente `bosque_n_arboles`
    (árboles en PARALELO dentro de esa ronda, no en serie)."""
    modelos = _espiar_fit_xgb(monkeypatch)
    entrenar(_cfg("bosque", bosque_n_arboles=7), S)
    modelo = modelos[-1]
    assert modelo.get_booster().num_boosted_rounds() == 1
    assert len(modelo.get_booster().get_dump()) == 7


@pytest.mark.gpu
def test_bosque_mapeo_de_parametros_documentado(monkeypatch):
    """Verifica el mapeo de parámetros documentado en la sección "bosque" del
    docstring del módulo -- subsample ~ 0.632 (1 - 1/e), colsample_bynode ~
    sqrt(n_features)/n_features, reg_lambda ~ 1e-5 -- vía `tree_train_param` del
    booster real (no solo que `entrenar()` no falle)."""
    modelos = _espiar_fit_xgb(monkeypatch)
    entrenar(_cfg("bosque"), S)
    tp = json.loads(modelos[-1].get_booster().save_config())["learner"]["gradient_booster"][
        "tree_train_param"
    ]
    n_features = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    assert float(tp["subsample"]) == pytest.approx(1.0 - 1.0 / math.e, abs=1e-3)
    assert float(tp["colsample_bynode"]) == pytest.approx(
        math.sqrt(n_features) / n_features, abs=1e-3
    )
    assert float(tp["lambda"]) == pytest.approx(1e-5, rel=1e-2)


@pytest.mark.gpu
@pytest.mark.parametrize("m", [1, 5])
def test_bosque_min_muestras_hoja_es_una_traduccion_exacta_a_min_child_weight(monkeypatch, m):
    """`base_score=0.5` fijo hace el hessian por fila EXACTAMENTE 0.25 (ver la sección
    "bosque" del docstring del módulo para la derivación), así que
    `min_child_weight = 0.25 * bosque_min_muestras_hoja` traduce la cobertura de
    cada hoja de vuelta a un número de muestras exacto: `cover / 0.25 >= m` para
    TODAS las hojas, y para `m=1` debe existir alguna hoja con `cover / 0.25 < 4`
    (esa segunda aserción distingue esta traducción exacta de un mapeo que dejara
    `base_score` libre, donde el hessian por fila -- y por lo tanto el número real
    de muestras exigido por hoja -- dependería de la media de las etiquetas del
    lote en vez de ser una constante conocida)."""
    modelos = _espiar_fit_xgb(monkeypatch)
    entrenar(_cfg("bosque", bosque_n_arboles=20, bosque_min_muestras_hoja=m), S)
    dump = modelos[-1].get_booster().get_dump(dump_format="json", with_stats=True)

    coberturas = []

    def _recolectar(nodo):
        hijos = nodo.get("children")
        if not hijos:
            coberturas.append(nodo["cover"])
        else:
            for hijo in hijos:
                _recolectar(hijo)

    for arbol in dump:
        _recolectar(json.loads(arbol))

    muestras_por_hoja = [c / 0.25 for c in coberturas]
    assert all(muestras >= m - 1e-6 for muestras in muestras_por_hoja), min(muestras_por_hoja)
    if m == 1:
        assert any(muestras < 4 for muestras in muestras_por_hoja), muestras_por_hoja


@pytest.mark.gpu
def test_hgb_mapeo_de_parametros_documentado(monkeypatch):
    """`hgb` debe crecer con `grow_policy="lossguide"` y `max_depth=0` (sin límite),
    no "depthwise" (ver la sección "hgb" del docstring del módulo), y debe respetar
    `hgb_learning_rate`/`hgb_max_iter` de verdad: verifica `grow_policy`/`max_depth`/
    `max_leaves` y también `eta` (el nombre interno de `learning_rate` en el
    booster, `tree_train_param["eta"]` -- confirmado leyendo `save_config()` de un
    booster real de este entorno) y `num_boosted_rounds() == hgb_max_iter`, para
    que una rama que ignorase esas dos perillas y usara un valor fijo no pase esta
    prueba."""
    modelos = _espiar_fit_xgb(monkeypatch)
    entrenar(_cfg("hgb", hgb_max_hojas=17, hgb_learning_rate=0.37, hgb_max_iter=11), S)
    modelo = modelos[-1]
    tp = json.loads(modelo.get_booster().save_config())["learner"]["gradient_booster"][
        "tree_train_param"
    ]
    assert tp["grow_policy"] == "lossguide"
    assert tp["max_depth"] == "0"
    assert int(tp["max_leaves"]) == 17
    assert float(tp["eta"]) == pytest.approx(0.37)
    assert modelo.get_booster().num_boosted_rounds() == 11


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["bosque", "hgb"])
def test_bosque_hgb_superan_un_piso_de_aprendizaje(algoritmo):
    """Si la regla de predicción de XGBoost estuviera INVERTIDA (p. ej.
    `(probabilidades > 0.5)` cambiado a `< 0.5`), `exactitud_verdadera` caería muy
    por debajo del azar (0.5) en vez de superarlo; con n_train=800 y los DEFAULTS
    de árboles/iteraciones (NO el override
    reducido de `CONFIGS`, que existe solo para que las demás pruebas sean rápidas)
    bosque/hgb superan un piso muy por encima del azar -- medido: bosque 0.6709,
    hgb 0.7467 con esta config. El piso de 0.6 deja margen bajo lo medido y muy
    por encima de una inversión (que daría ~1 - exactitud, bien abajo del azar)."""
    cfg = {
        "algoritmo": algoritmo,
        "dispositivo": "cuda",
        "n_train": 800,
        "n_val": 200,
        "n_oraculo": 20_000,
    }
    m = entrenar(cfg, S)
    assert m["exactitud_verdadera"] >= 0.6


@pytest.mark.gpu
@pytest.mark.parametrize("algoritmo", ["bosque", "hgb"])
def test_bosque_hgb_metricas_contra_predictor_independiente(algoritmo, monkeypatch):
    """Protege `exactitud_val`, `exactitud_verdadera`, `exactitud_prueba` y
    `n_parametros` de `entrenar` para bosque/hgb -- las cuatro métricas que
    `test_contrato_por_algoritmo` solo revisa por RANGO (`n_parametros > 0`, cada
    exactitud en `[0, 1]`), nunca por VALOR. Aquí se reconstruye un predictor
    INDEPENDIENTE a partir del booster real ya entrenado (espiado con
    `_espiar_fit_xgb`) -- no una llamada a `_predecir_xgb_gpu`, la función de
    producción -- y un conteo de nodos independiente a partir del volcado de texto de
    `get_dump()` (una línea por nodo, un recorrido distinto del JSON recursivo de
    `_contar_nodos_xgb`), y se comparan contra las métricas que devolvió `entrenar()`:
    si el oráculo usara la semilla equivocada, si `exactitud_val` se colara disfrazada
    de `exactitud_verdadera`, si la validación se calculara sobre el propio
    entrenamiento en vez del conjunto separado, o si `n_parametros` contara árboles en
    vez de nodos, esta prueba lo detecta porque calcula las cuatro cifras por su
    cuenta, desde cero."""
    import xgboost as xgb

    modelos = _espiar_fit_xgb(monkeypatch)
    n_prueba = 500
    n_oraculo = 5_000
    cfg = _cfg(algoritmo, n_prueba=n_prueba, n_oraculo=n_oraculo)
    m = entrenar(cfg, S)

    booster = modelos[-1].get_booster()

    def predecir_independiente(x: np.ndarray) -> np.ndarray:
        # nthread=1: misma convención n_jobs=1 del resto del módulo (ver
        # _predecir_xgb_gpu en cash_paridad.py) -- sin pasarlo, xgboost.DMatrix
        # toma su propio default (todos los núcleos) en una máquina compartida.
        dm = xgb.DMatrix(np.asarray(x, dtype=np.float64), nthread=1)
        return (booster.predict(dm) > 0.5).astype(np.int64)

    problema = {k: DEFAULTS[k] for k in ("d", "a", "k_ruido", "rotar")}
    riesgo = paridad.riesgo_verdadero(
        predecir_independiente, n=n_oraculo, semilla=SEMILLA_ORACULO, **problema
    )
    # Exacto (no approx): el booster es el mismo, así que el predictor
    # independiente y `_predecir_xgb_gpu` (usado dentro de `entrenar`) deben dar
    # el mismo bit -- medido, diferencia 0.0 en las dos métricas.
    assert m["exactitud_verdadera"] == 1.0 - riesgo

    x_pr, y_pr = generar(n_prueba, semilla=SEMILLA_PRUEBA, **problema)
    exactitud_prueba_esperada = float(np.mean(predecir_independiente(x_pr) == y_pr))
    assert m["exactitud_prueba"] == exactitud_prueba_esperada

    # exactitud_val: sobre el conjunto de VALIDACIÓN (las filas después de n_train),
    # nunca sobre el propio entrenamiento -- generar() con la misma semilla de datos
    # que usa entrenar() (S.datos) reproduce el split exacto de x_tr/x_va.
    x, y = generar(cfg["n_train"] + cfg["n_val"], semilla=S.datos, **problema)
    x_va, y_va = x[cfg["n_train"] :], y[cfg["n_train"] :]
    assert m["exactitud_val"] == float(np.mean(predecir_independiente(x_va) == y_va))

    # n_parametros: nodos totales, no árboles. get_dump() sin dump_format="json" da
    # texto plano con una línea por nodo -- un conteo independiente del recorrido
    # JSON recursivo de _contar_nodos_xgb.
    nodos_esperados = float(sum(len(arbol.strip().splitlines()) for arbol in booster.get_dump()))
    assert m["n_parametros"] == nodos_esperados


@pytest.mark.gpu
def test_cuda_con_xgboost_sin_build_cuda_falla_claro(monkeypatch):
    """Con xgboost instalado pero SIN CUDA compilada (`build_info()['USE_CUDA']`
    falso), `_validar_disponibilidad_gpu` debe fallar en español con un mensaje
    claro -- si solo verificara que xgboost se pueda IMPORTAR, el error real
    saldría varias capas más abajo, de xgboost, en inglés."""
    import taller_hpo.problemas.cash_paridad as cash_paridad_mod

    monkeypatch.setattr(cash_paridad_mod, "_xgboost_con_cuda", lambda: False)
    with pytest.raises(ValueError, match="compilado con soporte de CUDA"):
        entrenar(_cfg("bosque"), S)


@pytest.mark.parametrize("algoritmo", ["bosque", "hgb"])
def test_xgboost_sin_build_cuda_falla_claro_sin_necesitar_la_tarjeta(algoritmo, monkeypatch):
    """Esta prueba cubre, sin la marca `gpu`, la misma guarda que
    `test_cuda_con_xgboost_sin_build_cuda_falla_claro`: `bosque`/`hgb` se detienen
    en español cuando el `xgboost` instalado no trae CUDA compilado. Simula una
    tarjeta de video visible (parcha `torch.cuda.is_available` y
    `torch.cuda.device_count`) y un `xgboost` sin CUDA (parcha
    `_xgboost_con_cuda`), y parcha `generar` para que reviente si algo la llama:
    la validación tiene que fallar antes de tocar datos o tarjeta. Así corre en
    la integración continua (`pytest -m "not gpu"`) y en cualquier máquina."""
    import taller_hpo.problemas.cash_paridad as cash_paridad_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cash_paridad_mod, "_xgboost_con_cuda", lambda: False)

    def _generar_no_debe_llamarse(*_args, **_kwargs):
        raise AssertionError(
            "generar() no debe llamarse: la validación del build de xgboost debe fallar antes"
        )

    monkeypatch.setattr(cash_paridad_mod, "generar", _generar_no_debe_llamarse)
    with pytest.raises(ValueError, match="compilado con soporte de CUDA"):
        entrenar({**RAPIDO, **CONFIGS[algoritmo], "dispositivo": "cuda"}, S)
