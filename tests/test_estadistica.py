"""Contratos de taller_hpo.estadistica.

Cada afirmación estadística se prueba en las dos direcciones cuando tiene
sentido: un caso que cae adentro de la banda esperada y una banda lo bastante
angosta como para que una implementación incorrecta (t por normal, Wald por
Wilson, orden de Holm sin acumular el máximo) la haga fallar. Las simulaciones
usan `numpy.random.default_rng` con semilla fija: mismos números en cada corrida.
"""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from taller_hpo.estadistica import (
    Pareado,
    Resumen,
    diferencia_pareada,
    holm,
    ic_proporcion,
    maximo_esperado_normal,
    resumen,
    semillas_necesarias,
)

# --- resumen ------------------------------------------------------------


def test_resumen_valores_conocidos():
    r = resumen([1.0, 2.0, 3.0, 4.0, 5.0], confianza=0.95)
    t_critico = stats.t.ppf(0.975, df=4)
    assert isinstance(r, Resumen)
    assert r.n == 5
    assert r.media == pytest.approx(3.0)
    assert r.de == pytest.approx(1.5811388300841898)
    assert r.ee == pytest.approx(1.5811388300841898 / math.sqrt(5))
    assert r.ic_bajo == pytest.approx(3.0 - t_critico * r.ee)
    assert r.ic_alto == pytest.approx(3.0 + t_critico * r.ee)
    assert r.confianza == 0.95


def test_resumen_confianza_invalida_lanza_valueerror():
    with pytest.raises(ValueError, match="confianza"):
        resumen([1.0, 2.0, 3.0], confianza=1.5)


def test_resumen_menos_de_dos_valores_lanza_valueerror():
    with pytest.raises(ValueError, match="al menos 2"):
        resumen([1.0])


def test_resumen_nan_lanza_valueerror():
    with pytest.raises(ValueError, match="no finitos"):
        resumen([1.0, float("nan"), 3.0])


def test_resumen_inf_lanza_valueerror():
    with pytest.raises(ValueError, match="no finitos"):
        resumen([1.0, float("inf"), 3.0])


def test_resumen_valores_finitos_no_lanza_error():
    resumen([1.0, 2.0, 3.0])  # no debe lanzar


def test_resumen_confianza_090_usa_su_propio_t_critico():
    # Ninguna otra prueba de resumen usa una confianza distinta de 0.95: un
    # error que ignore el parámetro `confianza` y siempre use alfa=0.05
    # (p. ej. fijar `alfa = 0.05` al inicio de la función) pasaría toda la
    # suite salvo esta.
    r90 = resumen([1.0, 2.0, 3.0, 4.0, 5.0], confianza=0.90)
    t_critico_90 = stats.t.ppf(0.95, df=4)
    assert r90.confianza == 0.90
    assert r90.ic_bajo == pytest.approx(3.0 - t_critico_90 * r90.ee)
    assert r90.ic_alto == pytest.approx(3.0 + t_critico_90 * r90.ee)

    r95 = resumen([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r90.ic_bajo != pytest.approx(r95.ic_bajo)
    assert r90.ic_alto != pytest.approx(r95.ic_alto)


def test_resumen_entrada_2d_lanza_valueerror():
    # Sin validar ndim, `v.size` cuenta todos los elementos (aplana en
    # silencio): `resumen([[1,2],[3,4]])` daria n=4 en vez de fallar. `holm`
    # ya valida esto mismo; resumen no lo hacia, una inconsistencia dentro
    # del propio modulo.
    with pytest.raises(ValueError, match="1-D"):
        resumen([[1.0, 2.0], [3.0, 4.0]])


# --- maximo_esperado_normal ----------------------------------------------


@pytest.mark.parametrize(
    ("t", "esperado"),
    [(10, 1.539), (30, 2.043), (100, 2.508), (1000, 3.241)],
)
def test_maximo_esperado_normal_valores_conocidos(t, esperado):
    assert maximo_esperado_normal(t) == pytest.approx(esperado, abs=1e-3)


def test_maximo_esperado_normal_con_t_uno_es_cero():
    assert maximo_esperado_normal(1) == 0.0


def test_maximo_esperado_normal_contra_monte_carlo():
    rng = np.random.default_rng(0)
    for t in (10, 100):
        muestras = rng.standard_normal((100_000, t))
        maximo_mc = float(muestras.max(axis=1).mean())
        # 0.03: margen por encima del error Monte Carlo esperado con 100 000
        # repeticiones (una implementación con la integral mal armada, p. ej.
        # sin el factor t o con Phi en vez de Phi^(t-1), falla esto por mucho más).
        assert maximo_esperado_normal(t) == pytest.approx(maximo_mc, abs=0.03)


def test_maximo_esperado_normal_t_invalido_lanza_valueerror():
    with pytest.raises(ValueError, match="t debe ser"):
        maximo_esperado_normal(0)


def test_maximo_esperado_normal_acepta_entero_numpy():
    # t puede llegar de un calculo con arreglos (np.int64), no solo de un
    # literal Python: `isinstance(t, int)` a secas lo rechazaria.
    assert maximo_esperado_normal(np.int64(10)) == pytest.approx(1.539, abs=1e-3)


def test_maximo_esperado_normal_rechaza_bool():
    # bool es subclase de int en Python: sin excluirlo explicitamente,
    # `isinstance(True, int)` pasa y True se trataria como t=1.
    with pytest.raises(ValueError, match="t debe ser"):
        maximo_esperado_normal(True)


# --- diferencia_pareada: cobertura del intervalo por simulación ------------


def _cobertura_ic_pareado(n: int, n_simulaciones: int, semilla: int) -> float:
    """Fracción de corridas donde el intervalo del 95% atrapa la media real."""
    rng = np.random.default_rng(semilla)
    mu, sigma = 2.0, 1.0
    aciertos = 0
    for _ in range(n_simulaciones):
        diferencias = rng.normal(mu, sigma, size=n)
        a = {i: float(v) for i, v in enumerate(diferencias)}
        b = dict.fromkeys(range(n), 0.0)
        resultado = diferencia_pareada(a, b, delta=0.5)
        if resultado.ic_bajo <= mu <= resultado.ic_alto:
            aciertos += 1
    return aciertos / n_simulaciones


def test_cobertura_ic_pareado_n5():
    # Con n=5 (df=4) la t y la normal difieren bastante: si el código usara
    # z en vez de t, la cobertura real caería bien por debajo de 0.93.
    cobertura = _cobertura_ic_pareado(n=5, n_simulaciones=1000, semilla=1)
    assert 0.93 <= cobertura <= 0.97


def test_cobertura_ic_pareado_n10():
    cobertura = _cobertura_ic_pareado(n=10, n_simulaciones=1000, semilla=2)
    assert 0.93 <= cobertura <= 0.97


# --- diferencia_pareada: veredictos y Wilcoxon -----------------------------


def test_veredicto_mejor_y_wilcoxon_none_con_menos_de_6_no_nulas():
    diferencias = [5.0, 4.5, 5.5, 4.8, 5.2]  # n=5, todas a favor de a, no constantes
    a = {i: 10.0 + d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(5), 10.0)
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert isinstance(resultado, Pareado)
    assert resultado.veredicto == "mejor"
    assert resultado.wilcoxon_p is None  # 5 no nulas: menos de 6
    assert resultado.ic_bajo > 0.0


def test_veredicto_peor():
    diferencias = [-5.0, -4.5, -5.5, -4.8, -5.2]
    a = {i: 10.0 + d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(5), 10.0)
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.veredicto == "peor"
    assert resultado.ic_alto < 0.0


def test_veredicto_equivalente():
    # Ruido chico y simétrico alrededor de 0 con una zona de indiferencia
    # (delta) mucho más ancha: el intervalo principal cruza 0, pero el TOST
    # cabe entero dentro de (-delta, delta). Esta prueba no ejercita la
    # frontera exacta (ic_alto_tost == delta): en flotante esa igualdad tiene
    # medida cero, y un error que cierre el intervalo TOST (`<=` en vez de
    # `<`) solo difiere del contrato justo en ese punto, que no ocurre en la
    # práctica con datos medidos.
    diferencias = [0.05, -0.03, 0.02, -0.04, 0.01, 0.03, -0.02, 0.04, -0.01, 0.00]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(10), 0.0)
    resultado = diferencia_pareada(a, b, delta=0.5)
    assert resultado.veredicto == "equivalente"
    assert resultado.ic_bajo <= 0.0 <= resultado.ic_alto  # el intervalo principal no decide


def test_veredicto_no_concluyente():
    # n chico (df=2) y mucha varianza: el intervalo principal cruza 0 y el TOST
    # se sale de (-delta, delta) por el otro lado. Ni "mejor"/"peor" ni
    # "equivalente".
    diferencias = [1.0, -0.5, 0.3]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(3), 0.0)
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.veredicto == "no concluyente"


def test_tost_usa_nivel_1_menos_2_alfa_no_1_menos_alfa():
    # media = 0, ee = 0.5, n = 10 (df = 9). t_{0.95, df=9} = 1.833, asi que el
    # TOST (nivel 1 - 2*alfa = 0.90) da +-0.9166: cabe dentro de (-1.0, 1.0)
    # pero no dentro de (-0.9, 0.9). Un error que use el intervalo de nivel
    # 1 - alfa (0.95, con t_{0.975,9} = 2.262 -> +-1.131) daría "no
    # concluyente" en el primer caso, donde el contrato pide "equivalente".
    base = np.array([1, -1, 2, -2, 3, -3, 0.5, -0.5, 1.5, -1.5])
    diferencias = base / base.std(ddof=1) * 0.5 * math.sqrt(10)
    a = {i: float(d) for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(10), 0.0)

    equivalente = diferencia_pareada(a, b, delta=1.0)
    assert equivalente.veredicto == "equivalente"

    no_concluyente = diferencia_pareada(a, b, delta=0.9)
    assert no_concluyente.veredicto == "no concluyente"


def test_tost_respeta_confianza_distinta_de_095():
    # Mismo patrón que test_tost_usa_nivel_1_menos_2_alfa_no_1_menos_alfa,
    # pero variando `confianza` en vez de `delta`: ninguna otra prueba del
    # TOST usa una confianza distinta de 0.95, así que un error que calcule
    # t_critico_tost con un nivel fijo (siempre alfa=0.05, ignorando el
    # parámetro `confianza`) pasaría el resto de la suite.
    # Con confianza=0.99 (alfa=0.01, TOST a nivel 1-2*alfa=0.98), ee=0.5 y
    # df=9: t_{0.99,9}=2.8214, margen TOST=+-1.41072. delta=1.0 no alcanza a
    # cubrirlo -> "no concluyente"; delta=1.5 sí -> "equivalente". Un error
    # que siguiera usando el nivel 0.95 (t_{0.95,9}=1.833, margen +-0.9166)
    # daría "equivalente" en el primer caso, donde el contrato pide "no
    # concluyente" al nivel de confianza pedido.
    base = np.array([1, -1, 2, -2, 3, -3, 0.5, -0.5, 1.5, -1.5])
    diferencias = base / base.std(ddof=1) * 0.5 * math.sqrt(10)
    a = {i: float(d) for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(10), 0.0)

    no_concluyente = diferencia_pareada(a, b, delta=1.0, confianza=0.99)
    assert no_concluyente.veredicto == "no concluyente"

    equivalente = diferencia_pareada(a, b, delta=1.5, confianza=0.99)
    assert equivalente.veredicto == "equivalente"


def test_p_valor_es_el_de_una_t_pareada_bilateral_de_scipy():
    # Ninguna otra prueba de este archivo asierta p_valor directamente: un
    # error que lo calculara unilateral (stats.t.sf en vez de 2*stats.t.sf)
    # pasaría toda la suite. El p-valor bilateral no cambia con la dirección
    # (el estadístico solo cambia de signo), así que la prueba lo verifica en
    # ambas.
    a = {0: 1.3, 1: 2.1, 2: 0.7, 3: 1.9, 4: 1.2}
    b = {0: 1.0, 1: 1.2, 2: 0.9, 3: 1.1, 4: 0.8}
    claves = sorted(a)
    esperado = stats.ttest_rel([a[k] for k in claves], [b[k] for k in claves]).pvalue

    resultado_max = diferencia_pareada(a, b, delta=0.1)
    resultado_min = diferencia_pareada(a, b, delta=0.1, direccion="minimizar")
    assert resultado_max.p_valor == pytest.approx(esperado)
    assert resultado_min.p_valor == pytest.approx(esperado)


def test_minimizar_invierte_el_signo():
    diferencias_a_menos_b = [-2.0, -1.5, -2.7, -1.9, -1.9]  # a siempre pierde
    a = {i: 10.0 + d for i, d in enumerate(diferencias_a_menos_b)}
    b = dict.fromkeys(range(5), 10.0)

    resultado_max = diferencia_pareada(a, b, delta=1.0, direccion="maximizar")
    resultado_min = diferencia_pareada(a, b, delta=1.0, direccion="minimizar")

    assert resultado_max.diferencia_media < 0.0
    assert resultado_min.diferencia_media == pytest.approx(-resultado_max.diferencia_media)
    assert resultado_max.veredicto == "peor"
    assert resultado_min.veredicto == "mejor"  # el mismo par, el signo invertido


def test_claves_distintas_lanza_valueerror():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {0: 1.0, 1: 2.0, 3: 3.0}
    with pytest.raises(ValueError, match="claves distintas"):
        diferencia_pareada(a, b, delta=1.0)


def test_mismas_claves_no_lanza_error():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {0: 0.5, 1: 1.5, 2: 2.5}
    diferencia_pareada(a, b, delta=1.0)  # no debe lanzar


def test_nan_en_a_lanza_valueerror():
    # Un NaN (p. ej. de un modelo que diverge) no debe colarse como
    # significancia falsa: sin esta validacion, ee sale NaN, `ee > 0.0` es
    # False, y la rama de "diferencias identicas" con media != 0 daba
    # p_valor = 0.0 (un veredicto "mejor"/"peor" sin respaldo real).
    a = {0: 1.0, 1: float("nan"), 2: 3.0}
    b = dict.fromkeys(range(3), 0.0)
    with pytest.raises(ValueError, match="no finitos"):
        diferencia_pareada(a, b, delta=1.0)


def test_inf_en_a_lanza_valueerror():
    a = {0: 1.0, 1: float("inf"), 2: 3.0}
    b = dict.fromkeys(range(3), 0.0)
    with pytest.raises(ValueError, match="no finitos"):
        diferencia_pareada(a, b, delta=1.0)


def test_valores_finitos_no_lanzan_error_por_no_finitos():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = dict.fromkeys(range(3), 0.0)
    diferencia_pareada(a, b, delta=1.0)  # no debe lanzar


def test_emparejamiento_es_por_clave_no_por_posicion():
    # b se construye con las mismas claves que a pero en orden de insercion
    # inverso. Si el codigo emparejara por posicion (list(a.values()) contra
    # list(b.values())) en vez de por clave, las diferencias no saldrian
    # todas iguales a 1.0: de_dif seria ~18.7 en vez de 0.0. Asertar solo la
    # media no alcanza, porque con valores permutados la media coincide.
    a = {k: float(k * k) for k in range(6)}
    b = {k: a[k] - 1.0 for k in reversed(range(6))}
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.de_dif == pytest.approx(0.0, abs=1e-9)
    assert resultado.diferencia_media == pytest.approx(1.0)


def test_direccion_invalida_lanza_valueerror():
    a = {0: 1.0, 1: 2.0}
    b = {0: 0.5, 1: 1.5}
    with pytest.raises(ValueError, match="direccion"):
        diferencia_pareada(a, b, delta=1.0, direccion="lateral")


def test_delta_no_positivo_lanza_valueerror():
    a = {0: 1.0, 1: 2.0}
    b = {0: 0.5, 1: 1.5}
    with pytest.raises(ValueError, match="delta"):
        diferencia_pareada(a, b, delta=0.0)


def test_diferencia_pareada_confianza_099_amplia_el_intervalo():
    # Ninguna otra prueba de diferencia_pareada usa una confianza distinta de
    # 0.95: un error que ignore el parámetro `confianza` y siempre use
    # alfa=0.05 pasaría toda la suite salvo esta.
    a = {0: 1.3, 1: 2.1, 2: 0.7, 3: 1.9, 4: 1.2}
    b = {0: 1.0, 1: 1.2, 2: 0.9, 3: 1.1, 4: 0.8}
    resultado = diferencia_pareada(a, b, delta=0.1, confianza=0.99)
    assert resultado.confianza == 0.99
    t_critico = stats.t.ppf(0.995, df=resultado.n - 1)
    assert (resultado.ic_alto - resultado.diferencia_media) == pytest.approx(
        t_critico * resultado.de_dif / math.sqrt(resultado.n)
    )
    # Ambos lados del intervalo, no solo ic_alto: un error que calculara
    # ic_bajo con un t crítico fijo (p. ej. siempre t.ppf(0.975, df)) pasaría
    # la aserción de arriba, y solo el lado bajo lo delata.
    assert (resultado.diferencia_media - resultado.ic_bajo) == pytest.approx(
        t_critico * resultado.de_dif / math.sqrt(resultado.n)
    )

    resultado_95 = diferencia_pareada(a, b, delta=0.1)
    assert (resultado.ic_alto - resultado.diferencia_media) != pytest.approx(
        resultado_95.ic_alto - resultado_95.diferencia_media
    )


def test_diferencia_pareada_confianza_0_5_lanza_valueerror():
    # Con confianza=0.5, alfa=0.5 y el nivel TOST (1 - 2*alfa) es 0: el
    # intervalo colapsa a un punto (t.ppf(1-alfa, df) = t.ppf(0.5, df) = 0,
    # asi que ic_bajo_tost == ic_alto_tost == media), un caso degenerado que
    # `_validar_confianza` (que solo pide (0, 1)) no rechaza. Con confianza
    # < 0.5 el nivel es negativo y el intervalo se invierte de verdad
    # (ic_bajo_tost > ic_alto_tost).
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {0: 0.5, 1: 1.5, 2: 2.5}
    with pytest.raises(ValueError, match="confianza"):
        diferencia_pareada(a, b, delta=1.0, confianza=0.5)


def test_diferencia_pareada_confianza_0_51_no_lanza_error():
    a = {0: 1.0, 1: 2.0, 2: 3.0}
    b = {0: 0.5, 1: 1.5, 2: 2.5}
    diferencia_pareada(a, b, delta=1.0, confianza=0.51)  # no debe lanzar


def test_mejor_tiene_precedencia_sobre_equivalente():
    # El contrato dice: "mejor" si el intervalo entero está sobre 0, evaluado
    # ANTES que el TOST. Estas diferencias son deliberadamente ambiguas entre
    # las dos ramas: el intervalo principal está sobre 0 (ic_bajo > 0 ->
    # "mejor"), pero el TOST (más angosto, nivel 1 - 2*alfa) también cabe
    # dentro de (-delta, delta), que sería "equivalente" si la función lo
    # evaluara primero. Un error que invierta el orden (evaluar el TOST antes
    # que ic_bajo > 0) declararía "equivalente" aquí.
    diferencias = [0.010, 0.011, 0.009, 0.0105, 0.0095, 0.010]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(len(diferencias)), 0.0)
    delta = 0.5
    resultado = diferencia_pareada(a, b, delta=delta)

    assert resultado.veredicto == "mejor"
    assert resultado.ic_bajo > 0.0

    # Demuestra la ambiguedad: el TOST tambien cabria en (-delta, delta).
    df = resultado.n - 1
    ee = resultado.de_dif / math.sqrt(resultado.n)
    t_critico_tost = stats.t.ppf(0.95, df=df)
    ic_bajo_tost = resultado.diferencia_media - t_critico_tost * ee
    ic_alto_tost = resultado.diferencia_media + t_critico_tost * ee
    assert -delta < ic_bajo_tost
    assert ic_alto_tost < delta


def test_peor_tiene_precedencia_sobre_equivalente():
    # Espejo del caso anterior: las mismas diferencias negadas dan "peor" y
    # no "equivalente", por la misma razon (ic_alto < 0.0 se evalua antes
    # que el TOST).
    diferencias = [-0.010, -0.011, -0.009, -0.0105, -0.0095, -0.010]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(len(diferencias)), 0.0)
    resultado = diferencia_pareada(a, b, delta=0.5)
    assert resultado.veredicto == "peor"
    assert resultado.ic_alto < 0.0


@pytest.mark.parametrize("delta", [float("nan"), float("inf")])
def test_delta_no_finito_lanza_valueerror(delta):
    # delta=nan pasaba `delta <= 0` (siempre False con nan) y quedaba "no
    # concluyente" en silencio; delta=inf pasaba la misma validacion y
    # declaraba "equivalente" con un margen infinito. Ambos deben fallar
    # explicitamente, igual que ya lo hace semillas_necesarias con delta=inf.
    a = {0: 1.0, 1: 2.0}
    b = {0: 0.5, 1: 1.5}
    with pytest.raises(ValueError, match="delta"):
        diferencia_pareada(a, b, delta=delta)


# --- diferencia_pareada: ee == 0 (diferencias todas identicas) -------------


def test_ee_cero_con_diferencias_todas_cero_da_equivalente():
    # Sin varianza entre semillas, el intervalo colapsa a un punto (0.0): ni
    # "mejor" (ic_bajo > 0.0) ni "peor" (ic_alto < 0.0). El TOST también
    # colapsa a ese punto, que cae dentro de (-delta, delta), así que el
    # veredicto es "equivalente" con p_valor = 1.0 (el caso límite de "sin
    # diferencia").
    a = dict.fromkeys(range(4), 1.0)
    b = dict.fromkeys(range(4), 1.0)
    resultado = diferencia_pareada(a, b, delta=0.1)
    assert resultado.veredicto == "equivalente"
    assert resultado.p_valor == pytest.approx(1.0)
    assert resultado.ic_bajo == pytest.approx(0.0)
    assert resultado.ic_alto == pytest.approx(0.0)


def test_ee_cero_con_diferencias_todas_iguales_a_delta_da_mejor():
    # Diferencia constante de 1.0 (mayor que delta=0.1): el intervalo colapsa
    # al punto 1.0, por encima de cero, así que el veredicto es "mejor" con
    # p_valor = 0.0 (el caso límite de "diferencia segura"). Un error que
    # deje p_valor = 1.0 siempre en la rama ee == 0, o que abra el TOST a
    # (-inf, inf) en esa rama, no cambia esta prueba porque el veredicto ya
    # sale de ic_bajo > 0.0 antes de llegar al TOST; lo que sí detecta es un
    # error que use `ic_bajo >= 0.0` con el caso de arriba, donde
    # ic_bajo == 0.0 exactamente.
    a = {i: 2.0 + i for i in range(4)}
    b = {i: 1.0 + i for i in range(4)}
    resultado = diferencia_pareada(a, b, delta=0.1)
    assert resultado.veredicto == "mejor"
    assert resultado.p_valor == pytest.approx(0.0)


# --- diferencia_pareada: regla de Wilcoxon (>= 6 diferencias no nulas) -----


def test_wilcoxon_p_es_float_con_6_diferencias_no_nulas():
    # 7 diferencias, una es 0.0 (no cuenta): 6 no nulas, exactamente el
    # piso. Un error que siempre devuelva None falla esto.
    diferencias = [0.5, 0.2, 0.3, -0.1, 0.4, 0.6, 0.0]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(len(diferencias)), 0.0)
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert isinstance(resultado.wilcoxon_p, float)
    assert resultado.wilcoxon_p == pytest.approx(0.0625)


def test_wilcoxon_p_es_none_con_5_diferencias_no_nulas_de_6_totales():
    # 6 diferencias totales pero solo 5 no nulas: por debajo del piso. Un
    # error que cuente diferencias totales en vez de no nulas daría aquí un
    # float (0.125) en vez de None.
    diferencias = [0.5, 0.2, 0.3, -0.1, 0.4, 0.0]
    a = {i: d for i, d in enumerate(diferencias)}
    b = dict.fromkeys(range(len(diferencias)), 0.0)
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.wilcoxon_p is None


# --- diferencia_pareada: rho -------------------------------------------


def test_rho_es_nan_si_alguna_varianza_es_cero():
    a = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}  # constante: varianza 0
    b = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert math.isnan(resultado.rho)


def test_rho_constante_no_emite_warning():
    # std(ddof=1) de 7 copias de 0.7 no da exactamente 0.0 en punto flotante
    # (da ~1.1e-16), asi que comparar contra 0.0 no detecta la entrada
    # constante y pearsonr emitiria ConstantInputWarning. warnings-como-error
    # convierte ese warning en una falla de la prueba si el criterio es malo.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = dict.fromkeys(range(7), 0.7)
        b = {i: float(i) for i in range(7)}
        resultado = diferencia_pareada(a, b, delta=1.0)
    assert math.isnan(resultado.rho)


def test_rho_alto_cuando_a_y_b_se_mueven_juntas():
    a = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}
    b = {0: 1.1, 1: 2.2, 2: 2.9, 3: 4.1, 4: 5.3}
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.rho > 0.9  # casi colineales


def test_rho_es_pearson_no_spearman():
    # xs y ys=xs**4 son monótonas: Spearman da 1.0 exacto. Pearson, no. Un
    # error que use spearmanr pasaría las pruebas de "rho alto" y de NaN,
    # pero fallaría esta.
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ys = xs**4
    a = {i: float(v) for i, v in enumerate(xs)}
    b = {i: float(v) for i, v in enumerate(ys)}
    resultado = diferencia_pareada(a, b, delta=1.0)
    assert resultado.rho == pytest.approx(np.corrcoef(xs, ys)[0, 1])
    assert resultado.rho < 0.99  # bien lejos del 1.0 que daria Spearman


# --- semillas_necesarias ---------------------------------------------------


def test_semillas_necesarias_respeta_el_piso_normal():
    de_dif, delta = 1.0, 0.4
    n = semillas_necesarias(de_dif, delta)
    piso = math.ceil(7.85 * (de_dif / delta) ** 2)
    assert n >= piso
    assert n >= 2


def test_semillas_necesarias_da_potencia_real_de_aprox_08():
    de_dif, delta = 1.0, 0.4
    n = semillas_necesarias(de_dif, delta)

    # Verificación independiente: no se llama a semillas_necesarias otra vez,
    # se mide la potencia con una t de una muestra de scipy directamente.
    rng = np.random.default_rng(3)
    n_simulaciones = 4000
    rechazos = sum(
        stats.ttest_1samp(rng.normal(delta, de_dif, size=n), popmean=0.0).pvalue < 0.05
        for _ in range(n_simulaciones)
    )
    potencia_real = rechazos / n_simulaciones
    assert 0.75 <= potencia_real <= 0.9


def test_semillas_necesarias_al_menos_dos():
    assert semillas_necesarias(de_dif=0.01, delta=10.0) == 2


def test_semillas_necesarias_de_dif_no_positivo_lanza_valueerror():
    with pytest.raises(ValueError, match="de_dif"):
        semillas_necesarias(de_dif=0.0, delta=1.0)


def test_semillas_necesarias_delta_no_finito_lanza_valueerror():
    # delta=inf pasaba la validacion original (`delta <= 0` es False con
    # inf) y llegaba al calculo de nc = delta*sqrt(n)/de_dif, tambien inf.
    with pytest.raises(ValueError, match="delta"):
        semillas_necesarias(de_dif=1.0, delta=float("inf"))


def test_semillas_necesarias_de_dif_extremo_devuelve_2_sin_colgarse():
    # Con de_dif=1e-12 y delta=1.0, la no centralidad en n=2
    # (delta*sqrt(2)/de_dif ~= 1.4e12) satura scipy.stats.nct: tanto la cola
    # superior como la inferior dan NaN ahi. Antes de este atajo, el codigo
    # trataba ese NaN como "potencia insuficiente" (NaN >= potencia es False)
    # y el loop ascendente jamas terminaba (colgado verificado > 15 s), o bien
    # (version con la validacion de potencia_de) lanzaba FloatingPointError.
    # La respuesta real no es un error: es trivial. Con una no centralidad asi
    # de grande la potencia verdadera en n=2 es 1 a precision de maquina para
    # cualquier `potencia < 1`, asi que el minimo exacto es 2, y el codigo lo
    # devuelve directo sin necesidad de llamar a nct.
    assert semillas_necesarias(de_dif=1e-12, delta=1.0) == 2


def test_semillas_necesarias_de_dif_extremo_moderado_tambien_da_2():
    # de_dif/delta = 1e-8: la no centralidad en n=2 (~1.4e8) sigue por encima
    # del umbral de saturacion (1e6), asi que toma el mismo atajo. Antes de
    # este atajo, este caso especifico ya respondia 2 pasando por scipy.stats.nct
    # sin error (comprobado con el codigo anterior): el atajo no le cambia el
    # valor, solo evita la llamada a nct.
    assert semillas_necesarias(de_dif=1e-8, delta=1.0) == 2


@pytest.mark.parametrize("alfa", [1e-8, 1e-4, 0.51, 0.999])
def test_semillas_alfa_fuera_del_dominio_lanza_valueerror(alfa):
    # Fuera de [1e-3, 0.5], scipy.stats.nct con df=1 no converge y devolvía una
    # potencia inflada en silencio (con alfa=1e-8 y de_dif=1.8e-8 daba 2 semillas
    # cuando la potencia real en n=2 es 0.783 < 0.8, medido por Monte Carlo).
    with pytest.raises(ValueError, match="alfa debe estar en"):
        semillas_necesarias(1.0, 1.0, alfa=alfa)


@pytest.mark.parametrize("alfa", [1e-3, 0.5])
def test_semillas_alfa_en_los_bordes_del_dominio_funciona(alfa):
    n = semillas_necesarias(1.0, 1.0, alfa=alfa)
    assert n >= 2


def test_semillas_no_convergencia_de_scipy_se_escala_a_error(monkeypatch):
    # Red de seguridad: si scipy avisa que la serie no convergió, el número que
    # devuelve no se usa. Sin el aviso, la misma llamada responde normal.
    original = stats.nct.sf

    def sf_que_no_converge(*args, **kwargs):
        warnings.warn("Series did not converge", RuntimeWarning, stacklevel=2)
        return original(*args, **kwargs)

    assert semillas_necesarias(1.0, 1.0) == 10
    monkeypatch.setattr(stats.nct, "sf", sf_que_no_converge)
    with pytest.raises(FloatingPointError, match="no pudo evaluar la potencia"):
        semillas_necesarias(1.0, 1.0)


def test_semillas_necesarias_razon_100_es_78491_no_el_piso_redondeado():
    # razon = de_dif/delta = 100: el piso del contrato, ceil(7.85*100**2) =
    # 78500, no es un piso real aquí. El n mínimo exacto (verificado por
    # fuerza bruta contra la potencia real de la t no central) es 78491, 9
    # por debajo. La causa es el propio 7.85: es (z_.975 + z_.8)**2 =
    # 7.8489... redondeado hacia arriba, y ese redondeo (0.0011 * razon**2 ~=
    # 11 aquí) supera el margen que le da la t sobre la normal a partir de
    # razon ~= 41. Con los defaults, el resultado puede quedar por debajo de
    # ese piso.
    assert semillas_necesarias(100.0, 1.0) == 78491


def test_semillas_alfa_grande_usa_ambas_colas_y_es_minimo():
    # Con alfa=0.5 la cola inferior bilateral de la t no central ya no es
    # despreciable: en n=2 aporta 0.0673 de potencia (0.5485 unilateral +
    # 0.0673 = 0.6158 bilateral), lo suficiente para cruzar 0.6 y que n=2 sea
    # el mínimo. Un error que sume solo la cola superior mide 0.5485 en n=2,
    # no le alcanza, y sube a n=3; un error que quite el refinamiento
    # descendente se queda en el techo normal (n=3) sin bajar. Los dos
    # errores dan 3, no 2.
    n = semillas_necesarias(1.55, 1.0, alfa=0.5, potencia=0.6)
    assert n == 2

    # Verificacion independiente por simulacion (vectorizada: 20000 filas de
    # tamano n=2 de una vez), sin llamar a semillas_necesarias de nuevo.
    rng = np.random.default_rng(11)
    x = rng.normal(1.0, 1.55, size=(20_000, n))
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / math.sqrt(n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    potencia_real = float((p < 0.5).mean())
    assert potencia_real == pytest.approx(0.6158, abs=0.01)
    assert potencia_real >= 0.6


@pytest.mark.parametrize(
    ("de_dif", "delta", "alfa", "potencia", "n_esperado"),
    [
        # Valores conocidos (G*Power) para una t pareada bilateral, dos
        # direcciones simultaneas: matan tanto la aproximacion normal pura
        # (M5: `max(2, piso_normal)`, que da 8/32/13 en vez de 10/34/15) como
        # cualquier n que no sea el minimo exacto.
        (1.0, 1.0, 0.05, 0.8, 10),
        (1.0, 0.5, 0.05, 0.8, 34),
        (1.0, 0.8, 0.05, 0.8, 15),
        # Casos con la cola inferior resuelta por simetría.
        (0.07478991596638657, 1.0, 0.05, 0.8, 2),
        (0.3, 1.0, 0.01, 0.9, 5),
    ],
)
def test_semillas_necesarias_valores_conocidos(de_dif, delta, alfa, potencia, n_esperado):
    assert semillas_necesarias(de_dif, delta, alfa=alfa, potencia=potencia) == n_esperado


# --- ic_proporcion ----------------------------------------------------------


def test_ic_proporcion_k_cero_ancho_positivo():
    bajo, alto = ic_proporcion(0, 20)
    assert alto > bajo
    assert alto - bajo > 0.0


def test_ic_proporcion_exacto_tambien_da_ancho_positivo_en_k_cero():
    bajo, alto = ic_proporcion(0, 20, metodo="exact")
    assert alto - bajo > 0.0


def test_ic_proporcion_default_es_wilson():
    # Un error que cambie el default a "exact" da 0.168433 en vez de
    # 0.161125: valores distintos, medidos en las dos direcciones.
    _, alto = ic_proporcion(0, 20)
    assert alto == pytest.approx(0.161125, abs=1e-5)


def test_ic_proporcion_exact_da_el_intervalo_de_clopper_pearson():
    _, alto = ic_proporcion(0, 20, metodo="exact")
    assert alto == pytest.approx(0.168433, abs=1e-5)


def test_ic_proporcion_mas_confianza_da_intervalo_mas_ancho():
    bajo_90, alto_90 = ic_proporcion(10, 40, confianza=0.90)
    bajo_99, alto_99 = ic_proporcion(10, 40, confianza=0.99)
    assert (alto_99 - bajo_99) > (alto_90 - bajo_90)


def test_ic_proporcion_confianza_invalida_lanza_valueerror():
    with pytest.raises(ValueError, match="confianza"):
        ic_proporcion(3, 10, confianza=1.0)


@pytest.mark.parametrize(
    ("k", "n"),
    [(5, 3), (-1, 10), (3, 0), (11, 10), (2.5, 10)],
)
def test_ic_proporcion_k_n_invalidos_lanza_valueerror(k, n):
    # scipy 1.18.1 ya no lanza con k/n invalidos: los enmascara a NaN por
    # dentro (`_binomtest.py`, `k = xp.where(valid, k, xp.nan)`), asi que sin
    # esta validacion propia estos casos devolverian (nan, nan) en silencio
    # en vez de fallar. Comprobado con el codigo sin la validacion: los 5
    # casos dan (nan, nan).
    with pytest.raises(ValueError, match="k"):
        ic_proporcion(k, n)


def test_ic_proporcion_rechaza_bool():
    # bool es subclase de int: sin excluirlo, ic_proporcion(True, 10) se
    # trataria como k=1.
    with pytest.raises(ValueError, match="k"):
        ic_proporcion(True, 10)


def test_ic_proporcion_k_n_en_los_bordes_no_lanza_error():
    ic_proporcion(0, 20)  # k=0
    ic_proporcion(20, 20)  # k=n


# --- holm --------------------------------------------------------------


def test_holm_contra_ejemplo_resuelto_a_mano():
    # m=4. Orden ascendente: 0.005(3), 0.01(0), 0.03(2), 0.04(1).
    # candidatos: 4*0.005=0.02; 3*0.01=0.03; 2*0.03=0.06; 1*0.04=0.04 (capado a 0.06
    # por el máximo acumulado). En el orden de entrada: [0.03, 0.06, 0.06, 0.02].
    p = [0.01, 0.04, 0.03, 0.005]
    ajustados = holm(p)
    assert ajustados == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_es_monotono_en_el_orden_de_los_p_originales():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.0, 0.2, size=12).tolist()
    ajustados = np.array(holm(p))
    orden = np.argsort(p)
    ajustados_en_orden = ajustados[orden]
    assert np.all(np.diff(ajustados_en_orden) >= -1e-12)


def test_holm_nunca_supera_uno():
    p = [0.9, 0.95, 0.99, 0.999]
    assert all(v <= 1.0 for v in holm(p))


def test_holm_un_solo_valor_no_cambia():
    assert holm([0.03]) == pytest.approx([0.03])


def test_holm_nan_lanza_valueerror():
    # Sin esta validacion, `max(x, nan)` devuelve x: el NaN hereda el maximo
    # acumulado del p anterior y sale "significativo" en silencio.
    with pytest.raises(ValueError, match="0, 1"):
        holm([0.01, float("nan")])


def test_holm_p_fuera_de_rango_lanza_valueerror():
    with pytest.raises(ValueError, match="0, 1"):
        holm([0.01, 1.5, -0.2])


def test_holm_p_validos_en_los_bordes_no_lanza_error():
    holm([0.0, 1.0])  # no debe lanzar


def test_holm_entrada_2d_lanza_valueerror():
    # Sin validar ndim, `np.argsort` y la indexacion por `orden` siguen de
    # largo sobre un arreglo 2-D y el error real aparece mucho mas abajo,
    # como un "the truth value of an array... is ambiguous" críptico en vez
    # de decir que el problema es la forma de la entrada.
    with pytest.raises(ValueError, match="1-D"):
        holm([[0.01, 0.02], [0.03, 0.04]])
