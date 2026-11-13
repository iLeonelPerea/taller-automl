"""El problema de juguete y su oráculo dicen la verdad.

Si el oráculo miente, todo lo que se mida después contra él (inflación del
ganador, regret) miente también. Por eso cada fórmula se contrasta con Monte
Carlo independiente, con semillas fijas y tolerancia de 3 errores estándar.
"""

import math

import numpy as np
import pytest

from taller_hpo.problemas import paridad
from taller_hpo.problemas.paridad import (
    _muestras_oraculo,
    error_bayes,
    generar,
    posterior,
    riesgo_verdadero,
)

N_MC = 400_000


def _clasificador_bayes(d, a, rotar=False):
    return lambda x: (posterior(x, d=d, a=a, rotar=rotar) > 0.5).astype(np.int64)


def _ee_proporcion(p, n):
    return math.sqrt(p * (1.0 - p) / n)


def test_error_bayes_valores_conocidos():
    # d = 1 es la BER de BPSK: Φ(-a).
    assert error_bayes(1, 1.5) == pytest.approx(0.5 * math.erfc(1.5 / math.sqrt(2)), abs=1e-15)
    # Referencias calculadas aparte con scipy.stats.norm.cdf (el módulo usa math.erfc).
    assert error_bayes(2, 1.5) == pytest.approx(0.12468799825496074, abs=1e-12)
    assert error_bayes(3, 1.5) == pytest.approx(0.17483508713335844, abs=1e-12)
    assert error_bayes(2, 50.0) == pytest.approx(0.0, abs=1e-12)  # señal enorme: sin error
    assert error_bayes(2, 1e-9) == pytest.approx(0.5, abs=1e-9)  # sin señal: moneda


@pytest.mark.parametrize("d", [1, 2, 3])
def test_error_bayes_coincide_con_monte_carlo(d):
    a = 1.5
    x, y = generar(N_MC, d=d, a=a, semilla=100 + d)
    # El clasificador de Bayes es la paridad de los signos, sin usar la fórmula.
    prediccion = (np.prod(np.sign(x), axis=1) > 0).astype(np.int64)
    error_empirico = float(np.mean(prediccion != y))
    teorico = error_bayes(d, a)
    assert abs(error_empirico - teorico) < 3 * _ee_proporcion(teorico, N_MC)


def test_rotacion_no_cambia_el_error_de_bayes_pero_si_la_frontera():
    d, a = 2, 1.5
    x, y = generar(N_MC, d=d, a=a, rotar=True, semilla=7)
    teorico = error_bayes(d, a)
    ee = _ee_proporcion(teorico, N_MC)

    con_rotacion = float(np.mean(_clasificador_bayes(d, a, rotar=True)(x) != y))
    assert abs(con_rotacion - teorico) < 3 * ee

    # Quien ignora la rotación y corta por ejes queda mucho peor que el techo.
    por_ejes = float(np.mean((np.prod(np.sign(x), axis=1) > 0).astype(np.int64) != y))
    assert por_ejes > teorico + 0.05


def test_misma_semilla_mismos_datos_y_el_ruido_no_mueve_lo_informativo():
    x1, y1 = generar(1000, d=2, semilla=3)
    x2, y2 = generar(1000, d=2, semilla=3)
    x3, _ = generar(1000, d=2, semilla=4)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert not np.array_equal(x1, x3)

    xr, yr = generar(1000, d=2, k_ruido=5, semilla=3)
    assert xr.shape == (1000, 7)
    np.testing.assert_array_equal(xr[:, :2], x1)
    np.testing.assert_array_equal(yr, y1)


def test_etiquetas_balanceadas():
    _, y = generar(N_MC, d=2, semilla=11)
    assert abs(float(np.mean(y)) - 0.5) < 3 * _ee_proporcion(0.5, N_MC)


def test_posterior_esta_calibrada():
    # En cada decil de P(y=1|x), la frecuencia real de y=1 coincide con la probabilidad media.
    d, a = 2, 1.5
    x, y = generar(N_MC, d=d, a=a, k_ruido=3, semilla=21)
    p = posterior(x, d=d, a=a)
    cortes = np.quantile(p, np.linspace(0, 1, 11))
    grupo = np.clip(np.searchsorted(cortes, p, side="right") - 1, 0, 9)
    for g in range(10):
        en_grupo = grupo == g
        n_g = int(en_grupo.sum())
        media_p = float(p[en_grupo].mean())
        frecuencia = float(y[en_grupo].mean())
        assert abs(frecuencia - media_p) < 3 * _ee_proporcion(media_p, n_g) + 1e-3, g


def test_riesgo_verdadero_de_bayes_y_de_una_moneda():
    d, a = 2, 1.5
    riesgo_bayes = riesgo_verdadero(_clasificador_bayes(d, a), d=d, a=a, semilla=31)
    assert riesgo_bayes == pytest.approx(error_bayes(d, a), abs=0.0035)

    riesgo_constante = riesgo_verdadero(
        lambda x: np.ones(len(x), dtype=np.int64), d=d, a=a, semilla=31
    )
    assert riesgo_constante == pytest.approx(0.5, abs=0.0035)


def test_riesgo_verdadero_con_ruido_y_rotacion():
    d, a = 2, 1.5
    riesgo = riesgo_verdadero(
        _clasificador_bayes(d, a, rotar=True), d=d, a=a, k_ruido=4, rotar=True, semilla=41
    )
    assert riesgo == pytest.approx(error_bayes(d, a), abs=0.0035)


def test_riesgo_verdadero_rechaza_predicciones_invalidas():
    with pytest.raises(ValueError, match="forma"):
        riesgo_verdadero(lambda x: np.zeros(3, dtype=np.int64), n=100, semilla=0)
    with pytest.raises(ValueError, match="0/1"):
        riesgo_verdadero(lambda x: np.full(len(x), 0.7), n=100, semilla=0)


@pytest.mark.parametrize(
    "kwargs",
    [{"d": 0}, {"a": 0.0}, {"a": -1.0}, {"k_ruido": -1}, {"d": 2.0}],
)
def test_parametros_invalidos(kwargs):
    with pytest.raises(ValueError):
        generar(10, semilla=0, **kwargs)


def test_n_invalido():
    with pytest.raises(ValueError):
        generar(0, semilla=0)


# --- F1: cache de las muestras del oráculo (_muestras_oraculo) --------------------


def test_muestras_oraculo_evita_regenerar_con_la_misma_clave(monkeypatch):
    """Con la misma (n, semilla, d, a, k_ruido, rotar), `generar` se llama una sola vez.

    `_muestras_oraculo.cache_clear()` al inicio evita que una prueba anterior
    (u otra prueba de este mismo módulo, si pytest-xdist las agrupa en el
    mismo worker) deje la clave ya cacheada y esta prueba pase por
    casualidad sin que la cache haga nada.
    """
    _muestras_oraculo.cache_clear()
    llamadas = []
    original = paridad.generar

    def espia(*args, **kwargs):
        llamadas.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(paridad, "generar", espia)
    riesgo_verdadero(lambda x: np.zeros(len(x), dtype=np.int64), d=2, a=1.5, n=50, semilla=17)
    riesgo_verdadero(lambda x: np.ones(len(x), dtype=np.int64), d=2, a=1.5, n=50, semilla=17)
    assert len(llamadas) == 1, "la segunda llamada con la misma clave no debió regenerar x"


def test_muestras_oraculo_resultado_identico_a_generar_sin_cache():
    """El valor que sale de la cache es idéntico al de generar x y p a mano, sin ella.

    Compara `_muestras_oraculo` contra una llamada directa a `generar` + `posterior`:
    mismos arreglos, no una aproximación.
    """
    x_cache, p_cache = _muestras_oraculo(321, 99, 2, 1.5, 3, True)
    x_directo, _ = generar(321, d=2, a=1.5, k_ruido=3, rotar=True, semilla=99)
    p_directo = posterior(x_directo, d=2, a=1.5, rotar=True)
    np.testing.assert_array_equal(x_cache, x_directo)
    np.testing.assert_array_equal(p_cache, p_directo)


def test_muestras_oraculo_es_de_solo_lectura():
    x, p = _muestras_oraculo(40, 5, 2, 1.5, 0, False)
    assert x.flags.writeable is False
    assert p.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        x[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        p[0] = 999.0


def test_llamador_no_puede_mutar_la_cache_para_llamadas_futuras():
    """Un llamador que intentara mutar in-place no lo logra: la cache queda intacta.

    `test_muestras_oraculo_es_de_solo_lectura` ya prueba que mutar lanza
    `ValueError`; esta prueba comprueba la consecuencia que de verdad importa:
    después de ese intento fallido (capturado), una llamada posterior con la
    MISMA clave sigue devolviendo los valores originales, no una versión a
    medio mutar. Sin `flags.writeable = False`, esta prueba fallaría: el
    primer `x[:] = 0.0` mutaría el array cacheado in-place, y la segunda
    llamada devolvería ceros en vez de las x originales.
    """
    x1, _ = _muestras_oraculo(40, 6, 2, 1.5, 0, False)
    original = x1.copy()
    with pytest.raises(ValueError, match="read-only"):
        x1[:] = 0.0
    x2, _ = _muestras_oraculo(40, 6, 2, 1.5, 0, False)
    np.testing.assert_array_equal(x2, original)
    assert x2 is x1, "misma clave -> el mismo objeto cacheado (lru_cache no copia)"


def test_muestras_oraculo_distintas_claves_no_comparten_cache():
    x_a, p_a = _muestras_oraculo(200, 1, 2, 1.5, 0, False)
    x_b, p_b = _muestras_oraculo(200, 2, 2, 1.5, 0, False)  # solo cambia la semilla
    assert not np.array_equal(x_a, x_b)
    x_c, p_c = _muestras_oraculo(200, 1, 3, 1.5, 0, False)  # solo cambia d
    assert x_c.shape[1] != x_a.shape[1]


def test_cache_no_confunde_int_y_float_en_las_claves():
    """`typed=True` evita que `n=100` (válido) caliente la cache para `n=100.0` (inválido).

    Con `typed=False` (el default de `lru_cache`), `100 == 100.0` y
    `hash(100) == hash(100.0)`, así que la clave de `n=100` y la de `n=100.0`
    colisionarían. `riesgo_verdadero` no valida nada por sí mismo: toda la
    validación de tipos vive dentro de `generar` (vía `_validar`), a la que
    `_muestras_oraculo` solo llama en un cache-miss. Sin `typed=True`, una vez
    que `n=100` (entero) calentara la clave, la MISMA clave (por `==`/`hash`)
    serviría para `n=100.0` sin volver a llamar a `generar` -- y por lo tanto
    sin volver a validar: el resultado saldría en silencio en vez de lanzar
    `ValueError`, un bug dependiente del orden de llamadas (pasaría o no según
    qué prueba hubiera corrido antes). Esta prueba cubre las dos direcciones:
    en frío (nadie calentó la clave entera) y después de calentarla a
    propósito; las dos deben lanzar. Usa las semillas 71/72, que ninguna otra
    prueba de este módulo usa, para no depender de qué haya cacheado una
    prueba anterior.
    """
    _muestras_oraculo.cache_clear()

    def cero(x):
        return np.zeros(len(x), dtype=np.int64)

    # n=100.0 en frío: nadie llamó todavía con n=100 (int).
    with pytest.raises(ValueError, match="entero"):
        riesgo_verdadero(cero, d=2, a=1.5, n=100.0, semilla=71)
    # Calienta la clave ENTERA real.
    riesgo_verdadero(cero, d=2, a=1.5, n=100, semilla=71)
    # Con la clave entera ya cacheada, n=100.0 sigue debiendo fallar: typed=True la
    # trata como una clave distinta, no como un cache-hit sobre la de n=100.
    with pytest.raises(ValueError, match="entero"):
        riesgo_verdadero(cero, d=2, a=1.5, n=100.0, semilla=71)

    # Lo mismo con d=2.0 en vez de d=2 (semilla distinta: clave distinta de arriba).
    with pytest.raises(ValueError, match="entero"):
        riesgo_verdadero(cero, d=2.0, a=1.5, n=90, semilla=72)
    riesgo_verdadero(cero, d=2, a=1.5, n=90, semilla=72)
    with pytest.raises(ValueError, match="entero"):
        riesgo_verdadero(cero, d=2.0, a=1.5, n=90, semilla=72)
