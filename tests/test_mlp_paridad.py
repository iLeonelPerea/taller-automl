"""El modelo de referencia (`mlp_paridad`, scikit-learn) cumple el contrato
entrenar(config, semillas, reportar).

Configuración chica (`RAPIDO`, abajo) para que cada prueba entrene rápido:
pocas épocas y pocas filas de entrenamiento/validación/oráculo.
"""

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from taller_hpo.problemas.mlp_paridad import DEFAULTS, Divergencia, entrenar
from taller_hpo.problemas.paridad import error_bayes
from taller_hpo.semillas import Semillas, semillas_de

RAPIDO = {"epocas": 5, "n_train": 400, "n_val": 200, "n_oraculo": 20_000, "ancho": 16}
S = Semillas(datos=1, orden=2, init=3)


def _sin_reloj(metricas):
    return {k: v for k, v in metricas.items() if k != "segundos"}


def test_misma_semilla_mismo_resultado():
    assert _sin_reloj(entrenar(RAPIDO, S)) == _sin_reloj(entrenar(RAPIDO, S))


def test_cada_fuente_de_azar_cambia_el_resultado():
    base = _sin_reloj(entrenar(RAPIDO, S))
    for otra in (Semillas(9, 2, 3), Semillas(1, 9, 3), Semillas(1, 2, 9)):
        assert _sin_reloj(entrenar(RAPIDO, otra)) != base, otra


def test_orden_cambia_entre_epocas_y_es_igual_entre_arquitecturas(monkeypatch):
    original = MLPClassifier.partial_fit
    vistos = []

    def espia(self, x, y, *args, **kwargs):
        vistos.append(x[:10].copy())
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(MLPClassifier, "partial_fit", espia)
    entrenar({**RAPIDO, "ancho": 8}, S)
    angosta = list(vistos)
    vistos.clear()
    entrenar({**RAPIDO, "ancho": 32, "n_capas": 2}, S)

    assert len(angosta) == RAPIDO["epocas"]
    assert not np.array_equal(angosta[0], angosta[1])  # épocas con órdenes distintos
    for a, b in zip(angosta, vistos, strict=True):  # mismo orden para ambas arquitecturas
        np.testing.assert_array_equal(a, b)


def test_reportar_recibe_cada_epoca_y_puede_parar():
    llamadas = []

    def reportar(epoca, exactitud_val):
        llamadas.append((epoca, exactitud_val))
        return epoca == 2

    metricas = entrenar(RAPIDO, S, reportar)
    assert [e for e, _ in llamadas] == [0, 1, 2]
    assert all(0.0 <= v <= 1.0 for _, v in llamadas)
    assert metricas["epocas"] == 3
    assert metricas["exactitud_val"] == llamadas[-1][1]


def test_placebo_no_cambia_nada():
    assert _sin_reloj(entrenar({**RAPIDO, "placebo": 0.0}, S)) == _sin_reloj(
        entrenar({**RAPIDO, "placebo": 123.0}, S)
    )


def test_n_parametros_de_una_arquitectura_conocida():
    entradas = DEFAULTS["d"] + DEFAULTS["k_ruido"]
    metricas = entrenar({**RAPIDO, "ancho": 8, "n_capas": 1}, S)
    assert metricas["n_parametros"] == entradas * 8 + 8 + 8 * 1 + 1


def test_nadie_supera_el_techo_de_bayes():
    techo = 1.0 - error_bayes(DEFAULTS["d"], DEFAULTS["a"])
    for s in semillas_de(5, 3):
        metricas = entrenar({"epocas": 10, "n_oraculo": 50_000}, s)
        # 0.006: error Monte Carlo del oráculo con 50 000 muestras (< 3 EE).
        assert metricas["exactitud_verdadera"] <= techo + 0.006


def test_prueba_finita_solo_si_se_pide_y_cerca_de_la_verdad():
    assert "exactitud_prueba" not in entrenar(RAPIDO, S)

    metricas = entrenar({**RAPIDO, "n_prueba": 20_000, "n_oraculo": 100_000}, S)
    # Con 20 000 muestras, el error estándar de la exactitud es <= 0.0036:
    # la prueba estima la verdad dentro de 4 EE.
    assert abs(metricas["exactitud_prueba"] - metricas["exactitud_verdadera"]) < 0.015


def test_prueba_es_independiente_de_los_datos_de_entrenamiento():
    # La prueba sale de SEMILLA_PRUEBA: ninguna fila coincide con los datos de
    # entrenamiento y validación que genera la semilla `datos`.
    from taller_hpo.problemas import mlp_paridad
    from taller_hpo.problemas.paridad import generar

    problema = {k: DEFAULTS[k] for k in ("d", "a", "k_ruido", "rotar")}
    x_prueba, _ = generar(2000, semilla=mlp_paridad.SEMILLA_PRUEBA, **problema)
    x_datos, _ = generar(2000, semilla=S.datos, **problema)
    comunes = {tuple(f) for f in np.round(x_prueba, 12)} & {tuple(f) for f in np.round(x_datos, 12)}
    assert not comunes


def test_clave_desconocida_falla():
    with pytest.raises(KeyError, match="learning_rate_init"):
        entrenar({"learning_rate": 0.1}, S)


def test_todo_el_computo_corre_con_un_hilo_y_el_limite_se_restaura(monkeypatch):
    """Prueba de conducta, no de implementación: no le importa CÓMO entrenar()
    limita los hilos (threadpool_limits directo, un controlador cacheado, etc.),
    solo que el ajuste y las predicciones corran de verdad con 1 hilo BLAS (lo
    que usa MLPClassifier) y que, al salir de entrenar(), el límite previo del
    proceso quede restaurado.

    `hilos()` filtra por `user_api == "blas"` a propósito, no por descuido:
    `MLPClassifier.partial_fit`/`predict` no usan OpenMP (dependen solo del
    BLAS de numpy, OpenBLAS en este entorno) -- torch es una dependencia fija
    del taller (ver pyproject.toml) y trae su propio `libgomp`, que otro
    módulo de prueba puede cargar antes o después de que `_controlador_hilos()`
    se construya la primera vez; medir `max()` sobre TODAS las entradas de
    `threadpool_info()` (en vez de filtrar por `"blas"`) arriesgaría ver ese
    conteo heredado de OpenMP y no el 1 real del BLAS de numpy, aunque el
    límite sí se aplicó -- un falso negativo. Filtrar por `"blas"` sigue
    siendo fiel a lo que `MLPClassifier` de verdad usa, y la prueba sigue
    fallando si `limits=1` deja de aplicarse (quitar el límite, o mover el
    oráculo/la prueba fuera del `with`, la hace fallar igual).
    """
    import threadpoolctl

    def hilos():
        # Solo BLAS: ver el porqué en el docstring de esta prueba.
        blas = [
            i["num_threads"] for i in threadpoolctl.threadpool_info() if i["user_api"] == "blas"
        ]
        assert blas, "threadpoolctl no ve ninguna librería BLAS"
        return max(blas)

    vistos = []
    pf, pred = MLPClassifier.partial_fit, MLPClassifier.predict

    def espia_pf(self, *a, **k):
        vistos.append(("partial_fit", hilos()))
        return pf(self, *a, **k)

    def espia_pred(self, *a, **k):  # score() y el oráculo pasan por predict
        vistos.append(("predict", hilos()))
        return pred(self, *a, **k)

    monkeypatch.setattr(MLPClassifier, "partial_fit", espia_pf)
    monkeypatch.setattr(MLPClassifier, "predict", espia_pred)
    with threadpoolctl.threadpool_limits(limits=2):
        entrenar({**RAPIDO, "n_prueba": 100}, S)
        despues = hilos()
    assert {n for n, _ in vistos} == {"partial_fit", "predict"}
    assert all(h == 1 for _, h in vistos), vistos
    assert despues == 2


def test_divergencia_se_distingue_de_otros_errores(monkeypatch):
    def diverge(self, *args, **kwargs):
        raise ValueError("Solver produced non-finite parameter weights. The input data may ...")

    monkeypatch.setattr(MLPClassifier, "partial_fit", diverge)
    with pytest.raises(Divergencia, match="época 0"):
        entrenar(RAPIDO, S)

    def otro_error(self, *args, **kwargs):
        raise ValueError("otra cosa")

    monkeypatch.setattr(MLPClassifier, "partial_fit", otro_error)
    with pytest.raises(ValueError, match="otra cosa"):
        entrenar(RAPIDO, S)
