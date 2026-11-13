"""Paso 04: el frente de Pareto y su variante con restricción, de punta a punta.

`tests/test_pasos.py` ya corre `pasos/04_pareto.py` con PRESUPUESTO=humo. Este archivo
verifica además el contenido: que imprime el motor y las cinco respuestas numeradas, que
sus números tienen sentido y que el cálculo es el correcto. Una fixture de
`scope="module"` corre el paso una sola vez, porque paga dos búsquedas completas más la
re-evaluación del frente. Un "Warning" en stderr sería una regresión: el paso no usa
ninguna API experimental de Optuna.

Las pruebas de caja blanca envuelven `taller_hpo.problemas.juguete.entrenar` antes de
correr el paso con `runpy.run_path` en un subproceso nuevo, nunca en el proceso de pytest.
"""

from __future__ import annotations

import ast
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from scipy import stats as scipy_stats

from taller_hpo import estadistica
from taller_hpo.problemas.paridad import error_bayes

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "04_pareto.py"

# PRESUPUESTOS['humo'] en pasos/04_pareto.py: si cambia ahí, actualizar aquí
# (varias pruebas comparan contra estos números tal cual).
N_TOTAL_HUMO = 12
K_REEVALUAR_HUMO = 3
MAX_FRENTE_REEVALUAR_HUMO = 4
# Las mismas dos constantes que usa pasos/04_pareto.py para la semilla de
# re-evaluación del frente (BASE_REEVALUACION, BASE_DATOS): se recalculan aquí,
# de forma independiente, para verificar la secuencia real de semillas que ve
# cada punto (ver test_reevaluacion_del_frente_usa_las_mismas_semillas...).
_BASE_REEVALUACION = 20_260_935_000
_BASE_DATOS = 20_260_920

# DELTA_COMPARACION de pasos/04_pareto.py (0.02), en puntos porcentuales: el
# umbral de equivalencia que reconstruye _tost_desde_ic, para no repetir un
# "2.0" suelto en cada prueba que distingue 'equivalente' de 'no concluyente'.
DELTA_PCT = 2.0
# Tolerancia por el redondeo a 2 decimales de las cifras impresas.
TOL_TOST = 0.05


def _tost_desde_ic(ic_bajo: float, ic_alto: float, k: int) -> tuple[float, float]:
    """Reconstruye el intervalo TOST de `taller_hpo.estadistica.diferencia_pareada`
    a partir de un intervalo de confianza de 95% YA IMPRESO y el tamaño de
    muestra `k` real (post-Divergencia) que lo produjo: invierte ese
    intervalo para recuperar `media`/`ee` (`ee = semiancho / t.ppf(0.975, k -
    1)`) y arma el TOST con el mismo nivel `1 - 2*alfa` (`alfa=0.05`) que usa
    `diferencia_pareada` para decidir 'equivalente' (`media ± t.ppf(0.95, k -
    1) * ee`). Es lo que permite, en una prueba de caja negra sobre stdout,
    distinguir 'equivalente' (el TOST completo cabe en ±`DELTA_PCT`) de 'no
    concluyente' (no cabe) sin volver a llamar a `diferencia_pareada`: con el
    intervalo y `k` ya impresos, es la única reconstrucción posible sin
    fabricar datos sintéticos que reproduzcan la misma media y varianza."""
    media = (ic_bajo + ic_alto) / 2.0
    t_ic = float(scipy_stats.t.ppf(0.975, df=k - 1))
    ee = (ic_alto - ic_bajo) / (2.0 * t_ic)
    t_tost = float(scipy_stats.t.ppf(0.95, df=k - 1))
    return media - t_tost * ee, media + t_tost * ee


@pytest.fixture(scope="module")
def corrida_humo(tmp_path_factory) -> tuple[str, Path]:
    resultados = tmp_path_factory.mktemp("resultados")
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "Warning" not in resultado.stderr, resultado.stderr
    return resultado.stdout, resultados


@pytest.fixture(scope="module")
def salida_humo(corrida_humo) -> str:
    return corrida_humo[0]


@pytest.mark.gpu
def test_presupuesto_invalido_falla_con_mensaje(tmp_path):
    entorno = {**os.environ, "PRESUPUESTO": "enorme", "TALLER_RESULTADOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


@pytest.mark.gpu
def test_imprime_el_motor_antes_de_correr_nada(salida_humo):
    # Primera línea de salida: taller_hpo.problemas.juguete.DESCRIPCION es lo
    # único que le dice a un tesista, leyendo el log, en qué tarjeta de video
    # entrenó esta corrida -- por eso se imprime ANTES de correr cualquier
    # búsqueda, no mezclado entre las respuestas.
    primera_linea = salida_humo.splitlines()[0]
    assert primera_linea.startswith("Motor: "), salida_humo


@pytest.mark.gpu
def test_deja_su_propio_png_con_contenido(corrida_humo):
    _, resultados = corrida_humo
    figura = resultados / "04_pareto.png"
    assert figura.exists()
    # No solo "existe": un PNG vacío o truncado también "existiría".
    assert figura.stat().st_size > 1000


@pytest.mark.gpu
def test_imprime_las_cinco_respuestas_numeradas(salida_humo):
    for inicio in ("1. El frente tiene", "4. Con n_parametros <="):
        assert inicio in salida_humo, salida_humo
    # La respuesta 2 (comparación pareada de los extremos del frente), la 3
    # (inflación del frente) y la 5 (comparación contra la restricción) tienen
    # redacciones distintas según si hubo suficientes puntos para compararlos
    # -- las tres normalmente aparecen en un presupuesto humo con 12 trials,
    # pero el texto exacto depende de cuántos sobrevivan a la Divergencia.
    assert re.search(
        r"^2\. (Re-evaluado con las mismas|El frente tiene un solo punto|Menos de dos puntos)",
        salida_humo,
        re.MULTILINE,
    ), salida_humo
    assert re.search(
        r"^3\. (Frente re-evaluado CONTRA EL CONTROL|Sin control disponible|"
        r"Ningún punto representativo)",
        salida_humo,
        re.MULTILINE,
    ), salida_humo
    assert re.search(
        r"^5\. (El mejor punto del frente|Sin ningún trial factible|La comparación pareada)",
        salida_humo,
        re.MULTILINE,
    ), salida_humo


@pytest.mark.gpu
def test_imprime_la_linea_base_y_sus_numeros_son_consistentes(salida_humo):
    """`pasos/04_pareto.py` siembra DEFAULTS como línea base de la búsqueda de
    Pareto (`estudio.sembrar_linea_base(study_pareto, ...,
    DISTRIBUCIONES_ESPACIO)`) y la imprime aparte de la respuesta 1 -- si el
    frente la deja dominada (lo normal: DEFAULTS es `n_capas=1, ancho=64`, 961
    parámetros, mucho más caro que el punto más caro que suele ganar el frente
    en humo/taller) esa es la lección del paso, no un error.
    """
    m = re.search(
        r"línea base \(DEFAULTS, n_capas=(?P<n_capas>\d+), ancho=(?P<ancho>\d+)\): "
        r"(?P<params>\d+) parámetros, exactitud_val (?P<val>[\d.]+) -- "
        r"(?P<veredicto>SÍ quedó en el frente|quedó DOMINADA \(otro trial la iguala o supera en "
        r"las dos métricas y la supera en al menos una\))\.",
        salida_humo,
    )
    assert m, salida_humo
    assert int(m.group("n_capas")) == 1
    assert int(m.group("ancho")) == 64
    # 15*ancho + 1 con n_capas=1 (ver _n_parametros_mlp en pasos/04_pareto.py).
    assert int(m.group("params")) == 15 * 64 + 1
    assert 0.0 <= float(m.group("val")) <= 1.0


# --- el frente impreso en una línea parseable: sin dominados, sin usar TODOS los trials ---

_RE_FRENTE_LINEA = re.compile(r"^frente: (?P<lista>\[.*\])$", re.MULTILINE)


@pytest.mark.gpu
def test_frente_es_parseable_y_no_tiene_puntos_dominados(salida_humo):
    """Además de la respuesta 1 en prosa, el paso imprime `frente: [(costo,
    exactitud), ...]` en una línea aparte, ordenada barato -> caro. Verificar
    que la exactitud es ESTRICTAMENTE creciente en esa lista (nunca empata ni
    baja) atrapa dos errores concretos: usar TODOS los trials en vez de
    `study.best_trials` (con 12 trials evaluados casi seguro produce una
    secuencia no monótona al ordenar por costo) y invertir la dirección de
    `n_parametros` a maximizar (rompe qué cuenta como "no dominado" y con eso
    la relación costo/exactitud del frente)."""
    m = _RE_FRENTE_LINEA.search(salida_humo)
    assert m, salida_humo
    frente = ast.literal_eval(m.group("lista"))
    assert len(frente) >= 1, salida_humo
    costos = [c for c, _ in frente]
    exactitudes = [e for _, e in frente]
    assert costos == sorted(costos), "el frente debe imprimirse barato -> caro"
    for i in range(1, len(frente)):
        assert exactitudes[i] > exactitudes[i - 1], (
            "cada punto más caro del frente debe tener exactitud estrictamente mayor que el "
            "anterior; si no, está dominado por él y no debería estar en study.best_trials",
            frente,
        )


@pytest.mark.gpu
def test_veredicto_de_linea_base_es_consistente_con_el_frente(salida_humo):
    """El veredicto ('SÍ quedó en el frente' / 'quedó DOMINADA') tiene que
    coincidir con la lista `frente: [...]` de verdad: 'SÍ' solo si el par
    (n_parametros, exactitud_val) de la línea base aparece en esa lista, y
    'DOMINADA' solo si algún punto del frente la iguala o supera en las dos
    métricas y la supera en al menos una -- invertir esa condición (usar `not
    in` donde iba `in`) haría que el texto dijera lo contrario de lo que la
    propia lista impresa muestra."""
    m_frente = _RE_FRENTE_LINEA.search(salida_humo)
    assert m_frente, salida_humo
    frente = ast.literal_eval(m_frente.group("lista"))
    m_base = re.search(
        r"línea base \(DEFAULTS, n_capas=\d+, ancho=\d+\): (?P<params>\d+) parámetros, "
        r"exactitud_val (?P<val>[\d.]+) -- (?P<veredicto>SÍ quedó en el frente|quedó DOMINADA)",
        salida_humo,
    )
    assert m_base, salida_humo
    base_params = int(m_base.group("params"))
    base_val = round(float(m_base.group("val")), 3)
    esta_en_frente = any(p == base_params and round(e, 3) == base_val for p, e in frente)
    if m_base.group("veredicto").startswith("SÍ"):
        assert esta_en_frente, (frente, base_params, base_val)
    else:
        assert not esta_en_frente, (frente, base_params, base_val)
        dominante = any(
            p <= base_params and e >= base_val and (p < base_params or e > base_val)
            for p, e in frente
        )
        assert dominante, (
            "si la línea base no está en el frente, algún punto debe dominarla de verdad",
            frente,
            base_params,
            base_val,
        )


# --- los números de cada respuesta tienen sentido, no solo están presentes ---

_RE_RESPUESTA_1 = re.compile(
    r"1\. El frente tiene (?P<n_frente>\d+) configuraciones no dominadas de las "
    r"(?P<n_total>\d+) evaluadas: exactitud_val entre (?P<exact_min>[\d.]+) y "
    r"(?P<exact_max>[\d.]+), n_parametros entre (?P<param_min>[\d.]+) y (?P<param_max>[\d.]+)\."
)
# Respuesta 2: delta_busqueda (una diferencia CRUDA, sin re-evaluar) se juzga
# contra un intervalo de PREDICCIÓN de una diferencia nueva -- no contra el
# intervalo de la media re-evaluada (ver `_frase_delta_busqueda` en
# pasos/04_pareto.py) -- así que el texto trae AMBOS: el intervalo de la
# media re-evaluada (para el veredicto
# "SÍ compra"/"peor"/"equivalente"/"no concluyente" de la comparación pareada) y
# el intervalo de predicción por separado (para sobreestimaba/subestimaba/
# coincide/no concluyente de delta_busqueda).
_RE_RESPUESTA_2 = re.compile(
    r"2\. Re-evaluado con las mismas (?P<k>\d+) semillas nuevas para los dos: el extremo más "
    r"caro del frente \((?P<caro_params>[\d.]+) parámetros, reevaluado (?P<caro_reeval>[\d.]+)\) "
    r"contra el más barato \((?P<barato_params>[\d.]+) parámetros, reevaluado "
    r"(?P<barato_reeval>[\d.]+)\) (?P<frase>.+?) \(intervalo de confianza (?P<conf>\d+)% "
    r"\[(?P<ic_bajo>[+-][\d.]+), (?P<ic_alto>[+-][\d.]+)\]\)\. El valor de la búsqueda por sí "
    r"solo, sin re-evaluar, sugería (?P<delta_busqueda>[+-][\d.]+) puntos por "
    r"(?P<delta_param>[+-][\d.]+) parámetros más -- contra el intervalo de PREDICCIÓN de una "
    r"diferencia nueva con esas mismas (?P<k_ext>\d+) re-evaluaciones pareadas "
    r"\(\[(?P<pred_bajo>[+-][\d.]+), (?P<pred_alto>[+-][\d.]+)\]\): (?P<frase_busqueda>.+?)\."
)
_RE_RESPUESTA_3_RESUMEN = re.compile(
    r"3\. Frente re-evaluado CONTRA EL CONTROL \(línea base, que nadie seleccionó\), con las "
    r"mismas (?P<k_reeval>\d+) semillas nuevas para cada punto y el control: de los "
    r"(?P<n_reeval>\d+) puntos, (?P<n_arriba>\d+) infla\(n\) más que el control, "
    r"(?P<n_abajo>\d+) menos, (?P<n_no>\d+) no se distinguen de él; (?P<frase_agregado>.+?)\.$",
    re.MULTILINE,
)
# frase_agregado (ver pasos/04_pareto.py) tiene DOS partes, separadas por "; ":
# (1) la inflación típica del frente CONTRA SU PROPIA BÚSQUEDA (nulo exacto,
# resumen() de las M inflaciones individuales contra 0 -- nunca contra una
# resta con el control tratado como constante) y (2), si hay control, la MISMA
# comparación pero contra el control, sumando su propia varianza de predicción
# al error estándar del agregado.
_RE_FRASE_ABSOLUTA = re.compile(
    r"^la inflación típica del frente contra su propia búsqueda \(nulo exacto, sin mirar el "
    r"control\) es (?P<media>[+-][\d.]+) puntos \(intervalo de confianza (?P<conf>\d+)% "
    r"\[(?P<ic_bajo>[+-][\d.]+), "
    r"(?P<ic_alto>[+-][\d.]+)\]\) -- "
    r"(?P<frase>se distingue de 0 por encima|se distingue de 0 por debajo|no se distingue de 0)"
)
_RE_FRASE_VS_CONTROL = re.compile(
    r"contra el control, sumando su propia varianza de predicción, "
    r"(?:el frente infla (?:MÁS|MENOS) (?P<diff_a>[+-][\d.]+) puntos \(intervalo (?P<conf_a>\d+)% "
    r"\[(?P<bajo_a>[+-][\d.]+), (?P<alto_a>[+-][\d.]+)\]\)"
    r"|no concluyente \(diferencia (?P<diff_b>[+-][\d.]+) puntos, intervalo (?P<conf_b>\d+)% "
    r"\[(?P<bajo_b>[+-][\d.]+), (?P<alto_b>[+-][\d.]+)\]\))$"
)
_RE_RESPUESTA_3_NO_VISTA = re.compile(
    r"La parte que ninguna re-evaluación con los MISMOS datos puede ver -- la reevaluada contra "
    r"la verdad del oráculo -- promedia (?P<no_vista>[+-][\d.]+) puntos "
    r"\(intervalo de confianza (?P<conf>\d+)% "
    r"\[(?P<ic_bajo>[+-][\d.]+), (?P<ic_alto>[+-][\d.]+)\], rango (?P<nv_bajo>[+-][\d.]+) a "
    r"(?P<nv_alto>[+-][\d.]+)\): (?P<frase>.+?)\.$",
    re.MULTILINE,
)
_RE_RESPUESTA_3_PUNTO = re.compile(
    r"^   - (?P<n_param>[\d.]+) parámetros: búsqueda (?P<val_busqueda>[\d.]+), reevaluada "
    r"(?P<val_reeval>[\d.]+) \(intervalo de confianza \[(?P<ic_bajo>[\d.]+), "
    r"(?P<ic_alto>[\d.]+)\]\), verdadera "
    r"(?P<verdad>[\d.]+): inflación \(vs 0, intervalo de PREDICCIÓN de una observación nueva\) "
    r"(?P<inflacion>[+-][\d.]+) puntos "
    r"\(intervalo \[(?P<infl_bajo>[+-][\d.]+), (?P<infl_alto>[+-][\d.]+)\]\)"
    r"(?:, vs control (?P<frase_control>.+?) \(diferencia de inflaciones "
    r"(?P<rel_media>[+-][\d.]+) puntos, intervalo \[(?P<rel_bajo>[+-][\d.]+), "
    r"(?P<rel_alto>[+-][\d.]+)\]\))?\.$",
    re.MULTILINE,
)
_RE_CONTROL_LINEA = re.compile(
    r"^   control \(línea base\): búsqueda (?P<busqueda>[\d.]+), reevaluada (?P<reeval>[\d.]+) "
    r"\(intervalo de confianza \[(?P<ic_bajo>[\d.]+), (?P<ic_alto>[\d.]+)\]\): inflación propia "
    r"\(vs 0, intervalo de "
    r"PREDICCIÓN de una observación nueva\) "
    r"(?P<inflacion>[+-][\d.]+) puntos \(intervalo \[(?P<infl_bajo>[+-][\d.]+), "
    r"(?P<infl_alto>[+-][\d.]+)\]\) -- (?P<frase>.+?)\.$",
    re.MULTILINE,
)
_RE_RESPUESTA_4 = re.compile(
    r"4\. Con n_parametros <= (?P<presupuesto>[\d.]+): (?P<factibles>\d+) de "
    r"(?P<completos>\d+) trials completos fueron factibles \((?P<infactibles>\d+) "
    r"infactibles\); el mejor trial de ESTA búsqueda con restricción, sin mirar la restricción "
    r"\(exactitud_val (?P<sin_val>[\d.]+), (?P<sin_params>[\d.]+) parámetros\): "
    r"(?P<frase_ratio>.+?)\. El ganador FACTIBLE tiene (?P<factible_params>[\d.]+) parámetros\. "
    r"(?P<frase_costo>.+?); (?P<frase_propia>.+)\.$",
    re.MULTILINE,
)
_RE_COSTO_REAL_CALCULADO = re.compile(
    r"^El costo real, con el MISMO mecanismo de selección en los dos lados \(el mejor punto del "
    r"frente en total contra el mejor dentro del presupuesto, los dos re-evaluados con las mismas "
    r"(?P<k>\d+) semillas nuevas\): sin filtrar (?P<media_total>[\d.]+), dentro del presupuesto "
    r"(?P<media_presupuesto>[\d.]+) -- (?P<frase>.+?) \(intervalo de confianza (?P<conf>\d+)% "
    r"\[(?P<ic_bajo>[+-][\d.]+), (?P<ic_alto>[+-][\d.]+)\]\)$"
)
# frase_propia (respuesta 4) tiene tres formas posibles, según si hubo control
# disponible para juzgar al ganador FACTIBLE de la búsqueda con restricción:
# sin control (ninguna línea base), con control pero SIN veredicto de
# selección posible (por construcción no debería ocurrir si hay control: se
# deja como red de seguridad) sale como "desviación realizada"; con control Y
# suficientes semillas sale la comparación de tres ramas (ver
# `_ic_relativo_control`).
_RE_PROPIA_CALCULADA = re.compile(
    r"^aparte -- un fenómeno DISTINTO del costo de la restricción --, búsqueda "
    r"(?P<busqueda>[\d.]+), re-evaluada (?P<reeval>[\d.]+) \((?P<desviacion>[+-][\d.]+) puntos\): "
    r"contra el control \(línea base\), (?P<frase_control>.+?) \(diferencia de inflaciones "
    r"(?P<rel_media>[+-][\d.]+) puntos, intervalo de confianza \[(?P<rel_bajo>[+-][\d.]+), "
    r"(?P<rel_alto>[+-][\d.]+)\]\)$"
)
_RE_PROPIA_SIN_CONTROL = re.compile(
    r"^aparte -- desviación realizada, por sí sola NO es evidencia de selección \(hace falta un "
    r"control\): búsqueda (?P<busqueda>[\d.]+), re-evaluada (?P<reeval>[\d.]+) "
    r"\((?P<desviacion>[+-][\d.]+) puntos\)$"
)
_RE_RESPUESTA_5 = re.compile(
    r"5\. El mejor punto del frente con n_parametros <= (?P<presupuesto>[\d.]+).*? tiene "
    r"(?P<p_params>[\d.]+) parámetros y exactitud_val (?P<p_val>[\d.]+) en la búsqueda\. "
    r"Re-evaluado con las mismas (?P<k>\d+) semillas nuevas que la config de la restricción: "
    r"restricción (?P<media_restriccion>[\d.]+), frente (?P<media_frente>[\d.]+) "
    r"\(intervalo de confianza "
    r"(?P<conf>\d+)% \[(?P<ic_bajo>[+-][\d.]+), (?P<ic_alto>[+-][\d.]+)\] sobre la diferencia\): "
    r"en ESTA réplica de cada búsqueda, la variante con restricción (?P<frase>.+?) frente a "
    r"buscar en dos objetivos y filtrar después\."
)
# La nota descriptiva que _nota_semillas_necesarias (pasos/04_pareto.py)
# imprime justo después de cada comparación pareada de este paso (respuestas
# 2, 4 y 5, cuando aplica). "no aplica" (de_dif medido 0.0) no matchea este
# patrón a propósito: esa rama no tiene N que recalcular.
_RE_NOTA_SEMILLAS = re.compile(
    r"^   \(semillas para DETECTAR una diferencia de ±(?P<delta>\d+) con potencia 80%: "
    r"(?P<n>\d+); esta corrida usó (?P<k_usado>\d+)\)$",
    re.MULTILINE,
)


@pytest.mark.gpu
def test_humo_los_numeros_tienen_sentido(salida_humo):
    m1 = _RE_RESPUESTA_1.search(salida_humo)
    assert m1, salida_humo
    n_frente = int(m1.group("n_frente"))
    n_total = int(m1.group("n_total"))
    exact_min, exact_max = float(m1.group("exact_min")), float(m1.group("exact_max"))
    param_min, param_max = float(m1.group("param_min")), float(m1.group("param_max"))
    assert n_total <= N_TOTAL_HUMO, "PRESUPUESTOS['humo']['n'] es 12: no evalúa más trials que eso"
    assert 1 <= n_frente <= n_total, "el frente no puede tener más puntos que trials evaluados"
    assert 0.0 <= exact_min <= exact_max <= 1.0, "exactitud_val vive en [0, 1]"
    assert 0.0 < param_min <= param_max, "n_parametros de una red real siempre es positivo"

    m3 = _RE_RESPUESTA_3_RESUMEN.search(salida_humo)
    assert m3, salida_humo
    n_reeval = int(m3.group("n_reeval"))
    k_reeval = int(m3.group("k_reeval"))
    n_arriba, n_abajo, n_no = (
        int(m3.group("n_arriba")),
        int(m3.group("n_abajo")),
        int(m3.group("n_no")),
    )
    assert 1 <= n_reeval <= n_frente, "no se puede re-evaluar más puntos de los que tiene el frente"
    assert k_reeval == K_REEVALUAR_HUMO, (
        "PRESUPUESTOS['humo']['k_reevaluar'] es 3: si cambia, actualizar aquí"
    )
    assert n_arriba + n_abajo + n_no == n_reeval, (
        "las tres categorías del veredicto CONTRA EL CONTROL (infla más / infla menos / no se "
        "distingue) deben cubrir TODOS los puntos re-evaluados, sin traslape"
    )

    puntos = _RE_RESPUESTA_3_PUNTO.findall(salida_humo)
    assert len(puntos) == n_reeval, (
        "debe haber una línea de detalle por cada punto re-evaluado",
        salida_humo,
    )
    control_m = _RE_CONTROL_LINEA.search(salida_humo)
    assert control_m, (
        "debe imprimirse la línea del control cuando hay control disponible",
        salida_humo,
    )

    # Orden de los grupos de _RE_RESPUESTA_3_PUNTO: n_param, val_busqueda,
    # val_reeval, ic_bajo, ic_alto, verdad, inflacion, infl_bajo, infl_alto,
    # frase_control, rel_media, rel_bajo, rel_alto.
    arriba_detalle = sum(1 for p in puntos if p[9] == "infla más que el control")
    abajo_detalle = sum(1 for p in puntos if p[9] == "infla menos que el control")
    no_detalle = sum(1 for p in puntos if p[9] == "no se distingue del control")
    assert (arriba_detalle, abajo_detalle, no_detalle) == (n_arriba, n_abajo, n_no), (
        "el conteo agregado de la respuesta 3 debe coincidir con las líneas de detalle"
    )
    reevaluadas_por_params: dict[str, float] = {}
    busqueda_por_params: dict[str, float] = {}
    diffs_no_vistas_detalle: list[float] = []
    inflaciones_pct: list[float] = []
    # de_punto de CADA punto (reconstruido del semiancho YA impreso de su
    # "reevaluada (intervalo [...])"), guardado por n_parametros: la respuesta 2 lo
    # reusa más abajo para reconstruir el intervalo de predicción de
    # delta_busqueda con los DOS extremos reales de esta corrida, en vez de
    # aceptar la prosa de _frase_delta_busqueda sin verificar contra qué
    # número se comparó.
    de_por_params: dict[str, float] = {}
    # Cota de caja NEGRA para el margen de "diferencia de inflaciones" (rel_bajo/
    # rel_alto) de cada punto: reconstruye de_punto y de_control de los
    # semianchos YA impresos de sus respectivas "reevaluada (intervalo [...])"
    # (de = semiancho*sqrt(n)/t, la inversa de como `taller_hpo.estadistica.resumen`
    # arma ese intervalo) y exige que el semiancho impreso de la diferencia de
    # inflaciones sea >= al margen NO PAREADO teórico (t*sqrt(de_punto**2 +
    # de_control**2)) menos una tolerancia de 0.2 puntos por el redondeo a 2
    # decimales -- mata angostar el margen en el SITIO DE LLAMADA de
    # `_ic_relativo_control` (p. ej. `par_control.de_dif / 2.0`), que una
    # prueba sintética de esa función sola no puede detectar porque el defecto
    # está fuera de ella.
    from scipy import stats as scipy_stats_para_cota

    m_descartadas_frente = re.search(
        r"\((?P<n>\d+) semilla\(s\) descartada\(s\) en esta re-evaluación por Divergencia",
        salida_humo,
    )
    n_sobrevivientes = k_reeval - (
        int(m_descartadas_frente.group("n")) if m_descartadas_frente else 0
    )
    t_critico_cota = float(scipy_stats_para_cota.t.ppf(0.975, df=n_sobrevivientes - 1))
    control_reeval_val_temprano = float(control_m.group("reeval"))
    ancho_media_control_temprano = float(control_m.group("ic_alto")) - control_reeval_val_temprano
    de_control_temprano = ancho_media_control_temprano * (n_sobrevivientes**0.5) / t_critico_cota
    for p in puntos:
        n_param_txt = p[0]
        val_busqueda, val_reeval = float(p[1]), float(p[2])
        ic_bajo, ic_alto = float(p[3]), float(p[4])
        verdad = float(p[5])
        inflacion_pct = float(p[6])
        infl_bajo, infl_alto = float(p[7]), float(p[8])
        frase_control = p[9]
        rel_media, rel_bajo, rel_alto = float(p[10]), float(p[11]), float(p[12])
        reevaluadas_por_params[n_param_txt] = val_reeval
        busqueda_por_params[n_param_txt] = val_busqueda
        diffs_no_vistas_detalle.append(val_reeval - verdad)
        inflaciones_pct.append(inflacion_pct)
        assert 0.0 <= val_busqueda <= 1.0
        assert 0.0 <= ic_bajo <= ic_alto <= 1.0, "el intervalo de la reevaluación vive en [0, 1]"
        assert 0.0 <= verdad <= 1.0
        assert infl_bajo <= infl_alto
        # La inflación (vs 0) impresa DEBE ser 100*(búsqueda - reevaluada) --
        # un signo invertido (p. ej. por una direccion="minimizar" indebida) lo
        # rompe. Tolerancia de 0.15 puntos por el redondeo a 3 decimales.
        assert abs(inflacion_pct - 100 * (val_busqueda - val_reeval)) <= 0.15, (
            "la inflación (vs 0) impresa no coincide con 100*(búsqueda - reevaluada)",
            p,
        )
        assert rel_bajo <= rel_alto
        # Cota de semiancho (ver el comentario antes del loop): angostar el
        # margen de este punto contra el control por debajo de lo que sus
        # propias dispersiones reevaluadas justifican debe fallar aquí.
        ancho_media_punto = ic_alto - val_reeval
        de_punto = ancho_media_punto * (n_sobrevivientes**0.5) / t_critico_cota
        de_por_params[n_param_txt] = de_punto
        margen_teorico_pct = 100 * t_critico_cota * (de_punto**2 + de_control_temprano**2) ** 0.5
        semiancho_impreso = rel_alto - rel_media
        assert semiancho_impreso >= margen_teorico_pct - 0.2, (
            "el semiancho de 'diferencia de inflaciones' debe ser al menos el margen NO "
            "pareado teórico (t*sqrt(de_punto**2 + de_control**2)) -- un margen más angosto "
            "indica que se angostó en el sitio de llamada, no solo en la fórmula de "
            "_ic_relativo_control",
            semiancho_impreso,
            margen_teorico_pct,
            p,
        )
        # El margen de la inflación (vs 0) de ESTE punto tiene que ser el
        # mismo tipo de intervalo de PREDICCIÓN que ya se exige más abajo para
        # el control (sqrt(k+1) veces el margen de su propia media
        # reevaluada, `_margen_prediccion` en pasos/04_pareto.py) -- nunca el
        # error estándar de esa media (angosto por construcción). Mismo
        # cociente sqrt(n_sobrevivientes + 1) que la verificación del control,
        # reconstruido de las líneas YA impresas.
        ancho_infl_punto = (infl_alto - inflacion_pct) / 100.0
        assert ancho_media_punto > 0.0, p
        assert ancho_infl_punto / ancho_media_punto == pytest.approx(
            (n_sobrevivientes + 1) ** 0.5, rel=0.05
        ), (
            "el margen de la inflación (vs 0) de este punto debe ser un intervalo de "
            "PREDICCIÓN (sqrt(k+1) veces el margen de su propia media reevaluada), no el "
            "error estándar de esa media",
            ancho_infl_punto,
            ancho_media_punto,
            n_sobrevivientes,
            p,
        )
        # El veredicto CONTRA EL CONTROL tiene que ser consistente con el signo
        # del intervalo relativo de esa misma línea: intercambiar los dos argumentos
        # de `diferencia_pareada` al construir esa comparación invertiría este
        # intervalo sin cambiar su magnitud, y esto lo detecta.
        if rel_bajo > 0.0:
            assert frase_control == "infla más que el control", p
        elif rel_alto < 0.0:
            assert frase_control == "infla menos que el control", p
        else:
            assert frase_control == "no se distingue del control", p

    # El control (línea base): su propia inflación (vs 0) sigue la misma regla
    # de tres ramas, y es el ancla contra la que se mide cada punto de arriba.
    control_busqueda = float(control_m.group("busqueda"))
    control_reeval_val = float(control_m.group("reeval"))
    control_infl_bajo = float(control_m.group("infl_bajo"))
    control_infl_alto = float(control_m.group("infl_alto"))
    control_inflacion_pct = float(control_m.group("inflacion"))
    control_frase = control_m.group("frase")
    assert 0.0 <= control_busqueda <= 1.0
    assert control_infl_bajo <= control_infl_alto
    assert abs(control_inflacion_pct - 100 * (control_busqueda - control_reeval_val)) <= 0.15
    # El margen de la inflación propia del control tiene que ser un intervalo
    # de PREDICCIÓN de una observación nueva (`_margen_prediccion`: t*de*
    # sqrt(1 + 1/k)) -- NO el error estándar de su propia media reevaluada
    # (t*de/sqrt(k), angosto por construcción). El cociente de los dos anchos
    # es sqrt(1+1/k) * sqrt(k) = sqrt(k+1): reconstruir ambos anchos de las
    # líneas YA impresas y compararlos mata una regresión de vuelta al
    # intervalo de la media.
    control_reeval_ic_alto = float(control_m.group("ic_alto"))
    ancho_media_control = control_reeval_ic_alto - control_reeval_val
    ancho_infl_control = (control_infl_alto - control_inflacion_pct) / 100.0
    assert ancho_media_control > 0.0, control_m.group(0)
    assert ancho_infl_control / ancho_media_control == pytest.approx(
        (k_reeval + 1) ** 0.5, rel=0.05
    ), (
        "el margen de la inflación propia del control debe ser un intervalo de PREDICCIÓN "
        "(sqrt(k+1) veces el margen de su propia media reevaluada), no el error estándar de "
        "esa media",
        ancho_infl_control,
        ancho_media_control,
        k_reeval,
    )
    if control_infl_bajo > 0.0:
        assert control_frase == "se distingue de 0 por encima", control_m.group(0)
    elif control_infl_alto < 0.0:
        assert control_frase == "se distingue de 0 por debajo", control_m.group(0)
    else:
        assert control_frase == "no se distingue de 0", control_m.group(0)

    # rel_media de CADA punto tiene que ser, hasta el redondeo, inflación del
    # punto MENOS inflación del control -- ata las dos mediciones (ya
    # verificadas cada una por separado arriba) entre sí.
    for p in puntos:
        inflacion_pct = float(p[6])
        rel_media = float(p[10])
        assert abs(rel_media - (inflacion_pct - control_inflacion_pct)) <= 0.6, (
            "la diferencia de inflaciones impresa debe ser inflación del punto MENOS "
            "inflación del control (hasta el redondeo de ambas a 2 decimales)",
            p,
            control_inflacion_pct,
        )

    # El agregado (frase_agregado) sale de resumen() sobre las inflaciones de
    # los M puntos CONTRA 0 (nunca de restar una extracción fija del control a
    # cada punto): su media impresa debe ser la media de las inflaciones YA
    # impresas línea por línea, y su intervalo debe corresponder al veredicto en
    # prosa -- las dos verificaciones atan el texto ("infla MÁS"/"MENOS"/"no
    # concluyente") al intervalo que lo acompaña, en vez de aceptar cualquier frase
    # fija sin mirarlo.
    frase_agregado = m3.group("frase_agregado")
    m_abs = _RE_FRASE_ABSOLUTA.search(frase_agregado)
    assert m_abs, frase_agregado
    media_abs = float(m_abs.group("media"))
    ic_bajo_abs, ic_alto_abs = float(m_abs.group("ic_bajo")), float(m_abs.group("ic_alto"))
    assert ic_bajo_abs <= ic_alto_abs
    assert media_abs == pytest.approx(statistics.mean(inflaciones_pct), abs=0.1), (
        "la media agregada impresa (contra 0) debe ser la media de las inflaciones de los "
        "puntos, ya impresas en las líneas de detalle",
        media_abs,
        inflaciones_pct,
    )
    frase_abs = m_abs.group("frase")
    if ic_bajo_abs > 0.0:
        assert frase_abs == "se distingue de 0 por encima", frase_agregado
    elif ic_alto_abs < 0.0:
        assert frase_abs == "se distingue de 0 por debajo", frase_agregado
    else:
        assert frase_abs == "no se distingue de 0", frase_agregado

    m_vs_control = _RE_FRASE_VS_CONTROL.search(frase_agregado)
    assert m_vs_control, frase_agregado
    if m_vs_control.group("diff_a") is not None:
        diff_control = float(m_vs_control.group("diff_a"))
        ic_bajo_c = float(m_vs_control.group("bajo_a"))
        ic_alto_c = float(m_vs_control.group("alto_a"))
        assert ic_bajo_c > 0.0 or ic_alto_c < 0.0, (
            "'el frente infla MÁS/MENOS' solo debe imprimirse si el intervalo no cruza 0",
            frase_agregado,
        )
    else:
        diff_control = float(m_vs_control.group("diff_b"))
        ic_bajo_c = float(m_vs_control.group("bajo_b"))
        ic_alto_c = float(m_vs_control.group("alto_b"))
        assert ic_bajo_c <= 0.0 <= ic_alto_c, (
            "'no concluyente' solo debe imprimirse si el intervalo cruza 0",
            frase_agregado,
        )
    assert ic_bajo_c <= ic_alto_c
    # Reconstrucción independiente del semiancho de "contra el control,
    # sumando su propia varianza de predicción" (`_ic_agregado_vs_control` en
    # pasos/04_pareto.py): ee del semiancho YA impreso del agregado contra 0
    # (m_abs, arriba) y de_c del semiancho YA impreso de la línea del control
    # (de_control_temprano, calculado antes del loop de puntos). Pasar
    # de_control=0.0 en el sitio de llamada (o dividir ee) angosta este
    # semiancho sin que ninguna otra prueba de este archivo lo note, porque
    # media_abs/ic_bajo_abs/ic_alto_abs (el agregado CONTRA 0) no dependen de
    # de_control en absoluto.
    t_critico_agregado = float(scipy_stats_para_cota.t.ppf(0.975, df=n_reeval - 1))
    ee_agregado_reconstruido = (ic_alto_abs - media_abs) / t_critico_agregado
    # de_control_temprano vive en escala CRUDA [0, 1] (viene de las líneas
    # "reevaluada (intervalo [...])", nunca multiplicadas por 100): hay que pasarlo a
    # PUNTOS PORCENTUALES (*100) antes de combinarlo con ee_agregado_reconstruido,
    # que sí sale de cifras YA impresas en puntos (ic_alto_abs/media_abs).
    de_control_pct = de_control_temprano * 100.0
    semiancho_vs_control_esperado = (
        t_critico_agregado
        * (ee_agregado_reconstruido**2 + de_control_pct**2 * (1.0 + 1.0 / n_sobrevivientes)) ** 0.5
    )
    semiancho_vs_control_impreso = ic_alto_c - diff_control
    assert semiancho_vs_control_impreso == pytest.approx(semiancho_vs_control_esperado, rel=0.05), (
        "el semiancho de 'contra el control, sumando su propia varianza de predicción' debe "
        "ser t*sqrt(ee_agregado**2 + de_control**2*(1 + 1/k)) -- pasar de_control=0.0 (o "
        "angostar ee) en el sitio de llamada lo rompe sin que el agregado contra 0 cambie",
        semiancho_vs_control_impreso,
        semiancho_vs_control_esperado,
    )
    # diff_control = media_abs (agregado contra 0) MENOS la inflación propia
    # del control -- la misma resta que hace el paso, verificada aquí con los
    # dos números YA extraídos de líneas independientes.
    assert diff_control == pytest.approx(media_abs - control_inflacion_pct, abs=0.15), (
        "la diferencia contra el control debe ser la media agregada (contra 0) menos la "
        "inflación propia del control",
        diff_control,
        media_abs,
        control_inflacion_pct,
    )

    m3_no_vista = _RE_RESPUESTA_3_NO_VISTA.search(salida_humo)
    assert m3_no_vista, salida_humo
    no_vista = float(m3_no_vista.group("no_vista"))
    nv_bajo, nv_alto = float(m3_no_vista.group("nv_bajo")), float(m3_no_vista.group("nv_alto"))
    ic_nv_bajo = float(m3_no_vista.group("ic_bajo"))
    ic_nv_alto = float(m3_no_vista.group("ic_alto"))
    assert nv_bajo <= no_vista <= nv_alto, "la media de la parte no vista cae en su propio rango"
    assert ic_nv_bajo <= no_vista <= ic_nv_alto, (
        "el intervalo de la parte no vista contiene su media"
    )
    # "promedia" tiene que ser la media de (reevaluada - verdadera) de las
    # líneas de detalle YA impresas -- un signo invertido (reevaluada menos
    # verdadera vs. verdadera menos reevaluada) lo cambia de signo sin que
    # ninguna otra prueba lo note.
    assert no_vista == pytest.approx(100 * statistics.mean(diffs_no_vistas_detalle), abs=0.15), (
        "el 'promedia' de la parte no vista debe ser 100*media(reevaluada - verdadera) de las "
        "líneas de detalle ya impresas",
        no_vista,
        diffs_no_vistas_detalle,
    )
    frase_no_vista = m3_no_vista.group("frase")
    # La frase de la parte no vista tiene que salir del intervalo, nunca de una media
    # cruda sin ramas: "no_distingue" cuando el intervalo cruza 0, y en ese caso la
    # cola "hace falta un conjunto de validación nuevo" NO debe imprimirse
    # (contradiría al propio veredicto).
    if ic_nv_bajo > 0.0:
        assert frase_no_vista.startswith("la validación de este dataset SOBREESTIMA la verdad"), (
            frase_no_vista,
            ic_nv_bajo,
            ic_nv_alto,
        )
        assert "eso no es algo" in frase_no_vista
    elif ic_nv_alto < 0.0:
        assert frase_no_vista.startswith("la validación de este dataset SUBESTIMA la verdad"), (
            frase_no_vista,
            ic_nv_bajo,
            ic_nv_alto,
        )
        assert "eso no es algo" in frase_no_vista
    else:
        assert (
            frase_no_vista == "no se distingue de 0: esta re-evaluación no detecta un sesgo "
            "propio de este dataset"
        ), frase_no_vista
        assert "eso no es algo" not in frase_no_vista

    m_frente = _RE_FRENTE_LINEA.search(salida_humo)
    assert m_frente, salida_humo
    frente_lista = ast.literal_eval(m_frente.group("lista"))

    tiene_respuesta_2_completa = re.search(
        r"^2\. Re-evaluado con las mismas", salida_humo, re.MULTILINE
    )
    m2 = _RE_RESPUESTA_2.search(salida_humo)
    if tiene_respuesta_2_completa:
        assert m2, salida_humo
    if m2 is not None:
        # Solo se imprime esta forma de la respuesta 2 si el frente tiene >= 2
        # puntos Y >= 2 sobrevivieron a la re-evaluación -- si no, ver la otra
        # rama en test_imprime_las_cinco_respuestas_numeradas.
        caro_params, barato_params = m2.group("caro_params"), m2.group("barato_params")
        assert caro_params in reevaluadas_por_params, (caro_params, reevaluadas_por_params)
        assert barato_params in reevaluadas_por_params, (barato_params, reevaluadas_por_params)
        # Los índices representativos de _indices_representativos SIEMPRE
        # incluyen los dos extremos (ver su docstring en pasos/04_pareto.py):
        # el barato/caro de esta respuesta deben ser el punto MÁS BARATO y el
        # MÁS CARO del frente COMPLETO (frente_lista[0]/frente_lista[-1]), no
        # un índice representativo intermedio cualquiera -- mata, p. ej.,
        # tomar barato = puntos_reevaluados[1] en vez de puntos_reevaluados[0].
        assert barato_params == str(frente_lista[0][0]), (
            "el extremo barato de la respuesta 2 debe ser el punto MÁS BARATO del frente "
            "completo (el primero de 'frente: [...]')",
            barato_params,
            frente_lista,
        )
        assert caro_params == str(frente_lista[-1][0]), (
            "el extremo caro de la respuesta 2 debe ser el punto MÁS CARO del frente "
            "completo (el último de 'frente: [...]')",
            caro_params,
            frente_lista,
        )
        delta_pct = 100 * (
            reevaluadas_por_params[caro_params] - reevaluadas_por_params[barato_params]
        )
        # Las "reevaluado" impresas EN la propia respuesta 2 deben coincidir
        # con las "reevaluada" que ya imprimió la respuesta 3 para esos mismos
        # dos puntos: son el mismo número, calculado una sola vez.
        assert float(m2.group("caro_reeval")) == pytest.approx(
            reevaluadas_por_params[caro_params], abs=0.001
        )
        assert float(m2.group("barato_reeval")) == pytest.approx(
            reevaluadas_por_params[barato_params], abs=0.001
        )
        frase = m2.group("frase")
        ic_bajo_2, ic_alto_2 = float(m2.group("ic_bajo")), float(m2.group("ic_alto"))
        if frase.startswith("SÍ compra"):
            assert ic_bajo_2 > 0.0, (frase, ic_bajo_2)
            monto = float(re.search(r"SÍ compra ([\d.]+) puntos", frase).group(1))
            assert monto == pytest.approx(delta_pct, abs=0.1), (monto, delta_pct)
        elif frase.startswith("resulta PEOR"):
            assert ic_alto_2 < 0.0, (frase, ic_alto_2)
            monto = float(re.search(r"pierde ([\d.]+) puntos", frase).group(1))
            assert -monto == pytest.approx(delta_pct, abs=0.1), (monto, delta_pct)
        elif frase.startswith("no compra nada distinguible"):
            assert ic_bajo_2 <= 0.0 <= ic_alto_2, (frase, ic_bajo_2, ic_alto_2)
            monto = float(re.search(r"diferencia ([+-][\d.]+) puntos", frase).group(1))
            assert monto == pytest.approx(delta_pct, abs=0.1), (monto, delta_pct)
            # 'equivalente' (a diferencia de 'no concluyente') exige que el TOST
            # COMPLETO quepa dentro de ±DELTA_PCT -- un intervalo de confianza de
            # 95% que cruza 0 no alcanza por sí solo (ver
            # taller_hpo.estadistica.diferencia_pareada): reconstruir el TOST
            # desde el intervalo YA impreso atrapa el error que
            # vacía 'no concluyente' -> 'equivalente' (o viceversa) en el sitio
            # de llamada (`elif comparacion.veredicto == "equivalente":` con la
            # rama de 'no concluyente' fusionada ahí).
            tost_bajo_2, tost_alto_2 = _tost_desde_ic(ic_bajo_2, ic_alto_2, int(m2.group("k_ext")))
            assert -DELTA_PCT + TOL_TOST < tost_bajo_2 and tost_alto_2 < DELTA_PCT - TOL_TOST, (
                "'no compra nada distinguible' (equivalente) exige que el TOST reconstruido "
                "quepa dentro de ±DELTA_PCT -- si no cabe, el veredicto real era 'no "
                "concluyente'",
                tost_bajo_2,
                tost_alto_2,
                frase,
            )
        else:
            assert frase.startswith("da un resultado no concluyente"), frase
            # 'no concluyente' (a diferencia de 'equivalente') no exige que el
            # intervalo quepa dentro de ±delta_pct, pero SÍ exige que cruce 0 -- si
            # no, el veredicto real habría sido 'mejor'/'peor', no este.
            assert ic_bajo_2 <= 0.0 <= ic_alto_2, (frase, ic_bajo_2, ic_alto_2)
            monto = float(re.search(r"diferencia ([+-][\d.]+) puntos", frase).group(1))
            assert monto == pytest.approx(delta_pct, abs=0.1), (monto, delta_pct)
            # Y el TOST reconstruido NO debe caber entero en ±DELTA_PCT -- si
            # cupiera, el veredicto real habría sido 'equivalente', no este (el
            # error inverso al de arriba).
            tost_bajo_2, tost_alto_2 = _tost_desde_ic(ic_bajo_2, ic_alto_2, int(m2.group("k_ext")))
            assert tost_bajo_2 <= -DELTA_PCT + TOL_TOST or tost_alto_2 >= DELTA_PCT - TOL_TOST, (
                "'da un resultado no concluyente' exige que el TOST reconstruido NO quepa "
                "dentro de ±DELTA_PCT -- si cupiera, el veredicto real era 'equivalente'",
                tost_bajo_2,
                tost_alto_2,
                frase,
            )

        # delta_busqueda (crudo, sin re-evaluar) se juzga contra el intervalo de
        # PREDICCIÓN de una diferencia NO pareada (`_margen_prediccion_no_pareada`
        # en pasos/04_pareto.py), no contra el intervalo de la media re-evaluada.
        # Reconstruir las cuatro ramas aquí ata la interpretación en prosa al
        # intervalo impreso, en vez de una frase fija. La rama 'coincide'
        # compara la DESVIACIÓN de delta_busqueda respecto al intervalo contra
        # ±delta_pct -- nunca los extremos del intervalo contra 0 (ver
        # test_frase_delta_busqueda_coincide_... más abajo, con los dos casos
        # sintéticos que distinguen una comparación de la otra).
        delta_busqueda_2 = float(m2.group("delta_busqueda"))
        # delta_busqueda se ata a los valores de búsqueda ya impresos en el
        # detalle de la respuesta 3: tiene que ser la diferencia CRUDA de esos
        # mismos dos extremos -- un signo invertido (p. ej. restar al revés)
        # lo rompe.
        assert caro_params in busqueda_por_params and barato_params in busqueda_por_params
        assert delta_busqueda_2 == pytest.approx(
            100 * (busqueda_por_params[caro_params] - busqueda_por_params[barato_params]),
            abs=0.1,
        ), (delta_busqueda_2, busqueda_por_params, caro_params, barato_params)
        pred_bajo_2, pred_alto_2 = float(m2.group("pred_bajo")), float(m2.group("pred_alto"))
        frase_busqueda = m2.group("frase_busqueda")
        delta_pct_umbral = 2.0
        assert pred_bajo_2 <= pred_alto_2

        # Reconstrucción independiente del intervalo de PREDICCIÓN no pareado
        # de delta_busqueda (`_margen_prediccion_no_pareada` en
        # pasos/04_pareto.py), a partir de TRES semianchos ya impresos en
        # otro lado: de_dif del intervalo PAREADO de comparacion_extremos
        # (ic_bajo_2/ic_alto_2, ya verificado arriba contra delta_pct) y
        # de_caro/de_barato de las líneas de detalle de la respuesta 3
        # (de_por_params, llenado en el loop de arriba). Volver al margen
        # PAREADO (`_margen_prediccion(de_dif, k)`) o angostar cualquiera de
        # los tres términos en el sitio de llamada cambia pred_bajo_2/
        # pred_alto_2 YA IMPRESOS sin que la FORMA del texto lo delate --
        # esto sí lo detecta, porque compara contra el número reconstruido.
        assert caro_params in de_por_params and barato_params in de_por_params
        de_caro, de_barato = de_por_params[caro_params], de_por_params[barato_params]
        k_ext_2 = int(m2.group("k_ext"))
        t_critico_k_ext = float(scipy_stats_para_cota.t.ppf(0.975, df=k_ext_2 - 1))
        semiancho_media_2 = (ic_alto_2 - delta_pct) / 100.0
        assert semiancho_media_2 > 0.0, (ic_bajo_2, ic_alto_2, delta_pct)
        de_dif_2 = semiancho_media_2 * (k_ext_2**0.5) / t_critico_k_ext
        margen_pred_esperado_pct = (
            100 * t_critico_k_ext * (de_dif_2**2 / k_ext_2 + de_caro**2 + de_barato**2) ** 0.5
        )
        semiancho_pred_impreso = (pred_alto_2 - pred_bajo_2) / 2.0
        assert semiancho_pred_impreso == pytest.approx(margen_pred_esperado_pct, abs=0.3), (
            "el semiancho del intervalo de PREDICCIÓN de delta_busqueda debe ser "
            "t*sqrt(de_dif**2/k + de_caro**2 + de_barato**2) -- NO pareado -- volver al margen "
            "pareado, o angostar cualquiera de los tres términos en el sitio de llamada, lo "
            "rompe sin que la forma del texto cambie",
            semiancho_pred_impreso,
            margen_pred_esperado_pct,
        )
        # Piso asintótico (n -> infinito, `_margen_asintotico_no_pareada`):
        # decide, de forma INDEPENDIENTE del resto de la clasificación, si la
        # rama 'no concluyente' de más abajo debe decir 'una sola corrida' (el
        # piso YA es >= delta_pct_umbral: ninguna cantidad de semillas de
        # re-evaluación resolvería nada) o 'con k semillas' (el piso es
        # menor: más semillas SÍ podrían resolverlo). Pasar 0.0 en el sitio de
        # llamada fuerza siempre la segunda rama sin que ningún otro número
        # impreso lo delate -- por eso se reconstruye aquí, en vez de aceptar
        # cualquiera de las dos redacciones.
        z_critico_asintotico = float(scipy_stats_para_cota.norm.ppf(0.975))
        margen_asintotico_esperado_pct = (
            100 * z_critico_asintotico * (de_caro**2 + de_barato**2) ** 0.5
        )
        if delta_busqueda_2 > pred_alto_2:
            assert frase_busqueda == (
                "la búsqueda la sobreestimaba más de lo que explica una sola corrida"
            ), (frase_busqueda, delta_busqueda_2, pred_alto_2)
        elif delta_busqueda_2 < pred_bajo_2:
            assert frase_busqueda == (
                "la búsqueda la subestimaba más de lo que explica una sola corrida"
            ), (frase_busqueda, delta_busqueda_2, pred_bajo_2)
        elif (
            delta_busqueda_2 - delta_pct_umbral < pred_bajo_2
            and pred_alto_2 < delta_busqueda_2 + delta_pct_umbral
        ):
            assert frase_busqueda.startswith("coincide dentro de"), frase_busqueda
        elif margen_asintotico_esperado_pct >= delta_pct_umbral:
            assert frase_busqueda.startswith("no concluyente: una sola corrida"), (
                "el piso asintótico reconstruido (de_caro/de_barato, independiente de "
                "delta_busqueda) ya es mayor o igual al umbral: la rama debe decir que ninguna "
                "cantidad de semillas resolvería esto, no ofrecer 'más semillas' como salida",
                frase_busqueda,
                margen_asintotico_esperado_pct,
            )
        else:
            assert frase_busqueda.startswith("no concluyente: con"), (
                "el piso asintótico reconstruido queda bajo el umbral: más semillas de "
                "re-evaluación SÍ podrían resolver esta comparación, y la rama debe decirlo",
                frase_busqueda,
                margen_asintotico_esperado_pct,
            )

    m4 = _RE_RESPUESTA_4.search(salida_humo)
    assert m4, salida_humo
    presupuesto_parametros = m4.group("presupuesto")
    factibles, completos = int(m4.group("factibles")), int(m4.group("completos"))
    infactibles = int(m4.group("infactibles"))
    sin_val, sin_params = float(m4.group("sin_val")), float(m4.group("sin_params"))
    factible_params = float(m4.group("factible_params"))
    assert presupuesto_parametros == "100"
    assert completos <= n_total
    assert factibles + infactibles == completos
    assert 0.0 <= sin_val <= 1.0
    assert sin_params > 0.0
    # El ganador FACTIBLE tiene que respetar de verdad la restricción que dice
    # obedecer -- invertir el signo de `set_constraint` (factible con
    # n_parametros >= presupuesto en vez de <=) lo rompe sin que el resto del
    # texto cambie de forma.
    assert factible_params <= float(presupuesto_parametros), (
        "el ganador FACTIBLE de la búsqueda con restricción debe respetar "
        "n_parametros <= PRESUPUESTO_PARAMETROS",
        factible_params,
        presupuesto_parametros,
    )

    frase_ratio = m4.group("frase_ratio")
    if frase_ratio.startswith("cae dentro"):
        assert sin_params <= float(presupuesto_parametros), (frase_ratio, sin_params)
    else:
        m_ratio = re.search(r"([\d.]+)x el presupuesto -- queda fuera de ella", frase_ratio)
        assert m_ratio, frase_ratio
        ratio = float(m_ratio.group(1))
        # Con un presupuesto que de verdad restringe: tiene que dejar trials
        # fuera, y el mejor trial SIN mirar la restricción tiene que costar
        # más que el presupuesto -- si no, la restricción no restringía nada.
        assert factibles < completos, "un presupuesto que de verdad restringe deja algo fuera"
        assert infactibles > 0
        assert sin_params > float(presupuesto_parametros), (
            "el mejor trial SIN restricción debe superar el presupuesto: si no, la "
            "restricción no estaba restringiendo nada"
        )
        assert ratio == pytest.approx(sin_params / float(presupuesto_parametros), abs=0.05)

    frase_costo = m4.group("frase_costo")
    frente_mas_caro = frente_lista[-1][0]
    if frase_costo.startswith("el mejor punto del frente ya cabe en el presupuesto"):
        # mejor_frente_total y punto_comparable son el mismo trial (caso
        # degenerado: ver pasos/04_pareto.py, rama `ETIQUETA_TOTAL ==
        # ETIQUETA_PRESUPUESTO`) -- eso SOLO puede pasar si el punto MÁS CARO
        # del frente completo (frente_lista[-1], el mismo que entra a
        # mejor_frente_total) ya cabe dentro del presupuesto: si costara más,
        # punto_comparable habría tenido que ser otro punto (uno más barato) y
        # las dos etiquetas no podrían coincidir. Una condición degenerada
        # siempre verdadera (`if True` en vez de la comparación real) tomaría
        # esta rama incluso cuando el punto más caro excede el presupuesto, y
        # esto lo detecta.
        assert frente_mas_caro <= float(presupuesto_parametros), (
            "la rama degenerada del costo real solo debe imprimirse si el punto más caro del "
            "frente cabe dentro del presupuesto",
            frente_mas_caro,
            presupuesto_parametros,
            frente_lista,
        )
    elif not frase_costo.startswith("El paso no pudo calcular el costo real"):
        if frente_mas_caro > float(presupuesto_parametros):
            assert _RE_COSTO_REAL_CALCULADO.search(frase_costo), (
                "el punto más caro del frente excede el presupuesto: esta rama debe traer el "
                "costo real calculado, no la degenerada ni 'no pudo calcular'",
                frase_costo,
                frente_mas_caro,
                presupuesto_parametros,
            )
        m_costo = _RE_COSTO_REAL_CALCULADO.search(frase_costo)
        assert m_costo, frase_costo
        media_total = float(m_costo.group("media_total"))
        media_presupuesto = float(m_costo.group("media_presupuesto"))
        ic_bajo_costo = float(m_costo.group("ic_bajo"))
        ic_alto_costo = float(m_costo.group("ic_alto"))
        assert ic_bajo_costo <= ic_alto_costo
        # "sin filtrar" SIEMPRE es el punto más caro del frente completo
        # (frente_ordenado[-1], el último de la lista "frente: [...]" -- ver
        # test_frente_es_parseable_..., la exactitud es estrictamente
        # creciente): su reevaluada ya se imprimió en el detalle de la
        # respuesta 3 cuando ese punto es representativo. Atarlas aquí mata un
        # "costo real" vaciado que no toque de verdad esa config.
        params_mas_caro = str(frente_lista[-1][0])
        if params_mas_caro in reevaluadas_por_params:
            assert media_total == pytest.approx(
                reevaluadas_por_params[params_mas_caro], abs=0.001
            ), (
                "'sin filtrar' del costo real debe ser la reevaluada del punto MÁS CARO del "
                "frente completo",
                media_total,
                reevaluadas_por_params,
            )
        frase_costo_valor = m_costo.group("frase")
        # diferencia_media_pct > 0 significa que el punto SIN filtrar le gana
        # al punto DENTRO del presupuesto, o sea la restricción SÍ cuesta. Un
        # error que intercambie los dos lados de esa diferencia_pareada
        # invierte el signo aquí sin cambiar el |valor| impreso.
        diferencia_media_pct = 100 * (media_total - media_presupuesto)
        # El CENTRO del intervalo (nunca solo si contiene 0) tiene que coincidir con
        # esta resta de medias YA impresas, en TODAS las ramas -- no solo en
        # "SÍ cuesta"/"GANA": alimentar diferencia_pareada con el mismo punto
        # dos veces (p. ej. diferencia_pareada(total, total)) deja media_total
        # y media_presupuesto correctas (calculadas aparte con resumen()) pero
        # vacía el intervalo a [+0.00, +0.00] -- las ramas "equivalente"/"no
        # concluyente" de abajo solo piden que el intervalo contenga 0, y eso también
        # es cierto para un intervalo vaciado que contradice sus propias medias.
        assert (ic_bajo_costo + ic_alto_costo) / 2.0 == pytest.approx(
            diferencia_media_pct, abs=0.15
        ), (
            "el centro del intervalo del costo real debe ser 100*(media sin filtrar - media dentro "
            "del presupuesto), en cualquier rama del veredicto",
            ic_bajo_costo,
            ic_alto_costo,
            diferencia_media_pct,
        )
        # media_presupuesto tiene que ser la MISMA reevaluada que ya imprimió
        # la respuesta 3 para el punto que la respuesta 5 dice usar (cuando
        # ese punto es representativo): ata el costo real a la config real del
        # punto dentro del presupuesto, igual que ya se hace arriba para el
        # punto sin filtrar.
        m5_temprano = _RE_RESPUESTA_5.search(salida_humo)
        if m5_temprano is not None:
            params_dentro_presupuesto = str(int(round(float(m5_temprano.group("p_params")))))
            if params_dentro_presupuesto in reevaluadas_por_params:
                assert media_presupuesto == pytest.approx(
                    reevaluadas_por_params[params_dentro_presupuesto], abs=0.001
                ), (
                    "'dentro del presupuesto' del costo real debe ser la reevaluada del punto "
                    "que la respuesta 5 usa como 'el mejor punto del frente con n_parametros "
                    "<= presupuesto'",
                    media_presupuesto,
                    reevaluadas_por_params,
                )
        if frase_costo_valor.startswith("la restricción SÍ cuesta"):
            assert ic_bajo_costo > 0.0, frase_costo
            assert diferencia_media_pct > 0.0, (frase_costo, diferencia_media_pct)
            monto = float(re.search(r"SÍ cuesta ([\d.]+) puntos", frase_costo_valor).group(1))
            assert monto == pytest.approx(diferencia_media_pct, abs=0.1), (
                monto,
                diferencia_media_pct,
            )
        elif frase_costo_valor.startswith("la restricción GANA"):
            assert ic_alto_costo < 0.0, frase_costo
            assert diferencia_media_pct < 0.0, (frase_costo, diferencia_media_pct)
            monto = float(re.search(r"GANA ([\d.]+) puntos", frase_costo_valor).group(1))
            assert -monto == pytest.approx(diferencia_media_pct, abs=0.1), (
                monto,
                diferencia_media_pct,
            )
        elif frase_costo_valor.startswith("la restricción no cuesta nada distinguible"):
            assert ic_bajo_costo <= 0.0 <= ic_alto_costo, frase_costo
            # 'equivalente' exige el TOST completo dentro de ±DELTA_PCT --
            # reconstruido del intervalo de confianza de 95% YA impreso y el `k` real
            # (costo_restriccion_real.n,
            # `m_costo.group("k")`) que lo produjo. Mata fusionar 'equivalente' con
            # 'no concluyente' en el sitio de llamada.
            k_costo_tost = int(m_costo.group("k"))
            tost_bajo_costo, tost_alto_costo = _tost_desde_ic(
                ic_bajo_costo, ic_alto_costo, k_costo_tost
            )
            assert (
                -DELTA_PCT + TOL_TOST < tost_bajo_costo and tost_alto_costo < DELTA_PCT - TOL_TOST
            ), (
                "'la restricción no cuesta nada distinguible' (equivalente) exige que el TOST "
                "reconstruido quepa dentro de ±DELTA_PCT -- si no cabe, el veredicto real era "
                "'no concluyente'",
                tost_bajo_costo,
                tost_alto_costo,
                frase_costo,
            )
        else:
            assert frase_costo_valor.startswith("no concluyente con"), frase_costo_valor
            # 'no concluyente' tiene que salir de un intervalo que de verdad cruza 0
            # -- si no, el veredicto real habría sido 'mejor'/'peor'/
            # 'equivalente', no este.
            assert ic_bajo_costo <= 0.0 <= ic_alto_costo, (
                frase_costo,
                ic_bajo_costo,
                ic_alto_costo,
            )
            # Y el TOST reconstruido NO debe caber entero en ±DELTA_PCT -- si
            # cupiera, el veredicto real habría sido 'equivalente' (el error
            # inverso al de arriba).
            k_costo_tost = int(m_costo.group("k"))
            tost_bajo_costo, tost_alto_costo = _tost_desde_ic(
                ic_bajo_costo, ic_alto_costo, k_costo_tost
            )
            assert (
                tost_bajo_costo <= -DELTA_PCT + TOL_TOST or tost_alto_costo >= DELTA_PCT - TOL_TOST
            ), (
                "'no concluyente con' exige que el TOST reconstruido NO quepa dentro de "
                "±DELTA_PCT -- si cupiera, el veredicto real era 'equivalente'",
                tost_bajo_costo,
                tost_alto_costo,
                frase_costo,
            )

    frase_propia = m4.group("frase_propia")
    if frase_propia.startswith("aparte, la desviación de esta búsqueda"):
        pass  # Divergencia descartó demasiadas semillas: nada más que verificar.
    elif frase_propia.startswith("aparte -- desviación realizada"):
        m_sin_control = _RE_PROPIA_SIN_CONTROL.search(frase_propia)
        assert m_sin_control, frase_propia
        busqueda_propia = float(m_sin_control.group("busqueda"))
        reeval_propia = float(m_sin_control.group("reeval"))
        desviacion_propia = float(m_sin_control.group("desviacion"))
        assert 0.0 <= busqueda_propia <= 1.0
        assert 0.0 <= reeval_propia <= 1.0
        assert desviacion_propia == pytest.approx(100 * (busqueda_propia - reeval_propia), abs=0.15)
    else:
        m_propia = _RE_PROPIA_CALCULADA.search(frase_propia)
        assert m_propia, frase_propia
        busqueda_propia = float(m_propia.group("busqueda"))
        reeval_propia = float(m_propia.group("reeval"))
        desviacion_propia = float(m_propia.group("desviacion"))
        rel_bajo_propia, rel_alto_propia = (
            float(m_propia.group("rel_bajo")),
            float(m_propia.group("rel_alto")),
        )
        frase_control_propia = m_propia.group("frase_control")
        assert 0.0 <= busqueda_propia <= 1.0
        assert 0.0 <= reeval_propia <= 1.0
        # Fenómeno DISTINTO del costo real: la desviación realizada de la
        # búsqueda con restricción, búsqueda menos re-evaluada.
        assert desviacion_propia == pytest.approx(100 * (busqueda_propia - reeval_propia), abs=0.15)
        assert rel_bajo_propia <= rel_alto_propia
        # El veredicto CONTRA EL CONTROL tiene que ser consistente con el
        # signo del intervalo relativo -- mismo chequeo que ya hace la respuesta 3
        # para cada punto del frente, aquí para el ganador con restricción.
        if rel_bajo_propia > 0.0:
            assert frase_control_propia == "infla más que el control", frase_propia
        elif rel_alto_propia < 0.0:
            assert frase_control_propia == "infla menos que el control", frase_propia
        else:
            assert frase_control_propia == "no se distingue del control", frase_propia

    m5 = _RE_RESPUESTA_5.search(salida_humo)
    assert m5, salida_humo
    p_params, p_val = float(m5.group("p_params")), float(m5.group("p_val"))
    assert param_min <= p_params <= param_max, "el punto de comparación viene del frente"
    assert p_params <= float(presupuesto_parametros), (
        "el punto del frente elegido para comparar debe respetar el mismo presupuesto"
    )
    assert 0.0 <= p_val <= 1.0
    media_restriccion_5 = float(m5.group("media_restriccion"))
    media_frente_5 = float(m5.group("media_frente"))
    assert 0.0 <= media_restriccion_5 <= 1.0
    assert 0.0 <= media_frente_5 <= 1.0
    # "frente" de la respuesta 5 tiene que ser la MISMA reevaluada que ya
    # imprimió la respuesta 3 para ese punto (`punto_comparable`), no una
    # segunda re-evaluación con otra base de semillas: las dos vienen ahora de
    # la MISMA llamada a `_reevaluar_o_divergente` (ver pasos/04_pareto.py).
    # Mata reintroducir una llamada aparte para la comparación final, que le
    # daría a `punto_comparable` dos medias re-evaluadas distintas sin que
    # ninguna prueba de caja negra sobre stdout lo note (las dos tendrían la
    # forma numérica esperada, solo el valor difiere).
    p_params_txt = str(int(round(p_params)))
    if p_params_txt in reevaluadas_por_params:
        assert media_frente_5 == pytest.approx(reevaluadas_por_params[p_params_txt], abs=0.001), (
            "'frente' de la respuesta 5 debe ser la misma reevaluada que ya imprimió la "
            "respuesta 3 para el punto dentro del presupuesto -- una segunda re-evaluación con "
            "otra base de semillas le daría un valor distinto",
            media_frente_5,
            reevaluadas_por_params,
        )
    ic_bajo, ic_alto = float(m5.group("ic_bajo")), float(m5.group("ic_alto"))
    assert ic_bajo <= ic_alto
    frase = m5.group("frase")
    # Las dos medias re-evaluadas se calculan DIRECTO de los datos (resumen
    # sobre cada config por separado), sin pasar por diferencia_pareada: si se
    # intercambiaran restricción y frente al armar esa comparación pareada, el
    # intervalo invertiría signo pero estas dos medias no -- por eso su diferencia
    # tiene que apuntar en el mismo sentido, y con la misma magnitud, que el
    # veredicto en prosa.
    diferencia_medias_5 = 100 * (media_restriccion_5 - media_frente_5)
    # El veredicto en prosa tiene que ser consistente con el intervalo que lo acompaña:
    # "gana" solo si el intervalo entero queda sobre 0, "pierde" solo si queda bajo 0.
    if frase.startswith("gana"):
        assert ic_bajo > 0.0, (frase, ic_bajo, ic_alto)
        assert diferencia_medias_5 > 0.0, (frase, media_restriccion_5, media_frente_5)
        monto = float(re.search(r"gana por ([\d.]+) puntos", frase).group(1))
        assert monto == pytest.approx(diferencia_medias_5, abs=0.1), (monto, diferencia_medias_5)
    elif frase.startswith("pierde"):
        assert ic_alto < 0.0, (frase, ic_bajo, ic_alto)
        assert diferencia_medias_5 < 0.0, (frase, media_restriccion_5, media_frente_5)
        monto = float(re.search(r"pierde por ([\d.]+) puntos", frase).group(1))
        assert -monto == pytest.approx(diferencia_medias_5, abs=0.1), (monto, diferencia_medias_5)
    else:
        assert frase.startswith("es equivalente") or frase.startswith(
            "da un resultado no concluyente"
        ), frase
        # Ni "equivalente" ni "no concluyente" son válidos si el intervalo no cruza
        # 0 -- en ese caso el veredicto real habría sido "gana"/"pierde".
        assert ic_bajo <= 0.0 <= ic_alto, (frase, ic_bajo, ic_alto)
        # "gana"/"pierde" ya comparan su monto contra diferencia_medias_5
        # arriba; estas dos ramas embeben el mismo número con signo bajo la
        # etiqueta "diferencia" -- verificarlo aquí es lo que detecta un
        # intercambio de argumentos en diferencia_pareada incluso cuando el
        # intervalo de esta réplica cae de cualquier lado del cero (un veredicto
        # "no concluyente" o "equivalente" no cambia de palabra al invertir
        # el signo de la diferencia, así que sin este chequeo el intercambio
        # pasaría desapercibido).
        monto_con_signo = float(re.search(r"diferencia ([+-][\d.]+) puntos", frase).group(1))
        assert monto_con_signo == pytest.approx(diferencia_medias_5, abs=0.1), (
            monto_con_signo,
            diferencia_medias_5,
            frase,
        )
        # "es equivalente" exige el TOST completo dentro de ±DELTA_PCT; "da un
        # resultado no concluyente" exige que NO quepa -- reconstruido del
        # intervalo de confianza de 95% YA impreso y el `k` real (comparacion.n,
        # `m5.group("k")`, que
        # el propio texto de la respuesta 5 ya imprime como "las mismas {k}
        # semillas nuevas"). Mata fusionar las dos ramas en el sitio de
        # llamada (`elif comparacion.veredicto == "equivalente":` con "no
        # concluyente" adentro).
        k5_tost = int(m5.group("k"))
        tost_bajo_5, tost_alto_5 = _tost_desde_ic(ic_bajo, ic_alto, k5_tost)
        if frase.startswith("es equivalente"):
            assert -DELTA_PCT + TOL_TOST < tost_bajo_5 and tost_alto_5 < DELTA_PCT - TOL_TOST, (
                "'es equivalente' exige que el TOST reconstruido quepa dentro de ±DELTA_PCT "
                "-- si no cabe, el veredicto real era 'no concluyente'",
                tost_bajo_5,
                tost_alto_5,
                frase,
            )
        else:
            assert tost_bajo_5 <= -DELTA_PCT + TOL_TOST or tost_alto_5 >= DELTA_PCT - TOL_TOST, (
                "'da un resultado no concluyente' exige que el TOST reconstruido NO quepa "
                "dentro de ±DELTA_PCT -- si cupiera, el veredicto real era 'es equivalente'",
                tost_bajo_5,
                tost_alto_5,
                frase,
            )

    # "Superan el techo de Bayes": sus conteos tienen que coincidir con las
    # líneas de detalle (ya extraídas en `puntos`, arriba) cuya búsqueda o
    # verdadera de verdad lo superan; si NINGUNA línea lo supera, la frase
    # entera no debe imprimirse. Mata tanto una cuenta fija desconectada de
    # las líneas ya impresas como una condición `if` invertida (imprimir la
    # línea cuando nadie la supera, o callarla cuando alguien sí).
    techo_bayes = 1.0 - error_bayes(juguete.DEFAULTS["d"], juguete.DEFAULTS["a"])
    val_sobre_techo_esperado = sum(1 for p in puntos if float(p[1]) > techo_bayes)
    verdad_sobre_techo_esperado = sum(1 for p in puntos if float(p[5]) > techo_bayes)
    m_techo = re.search(
        r"^   Superan el techo de Bayes \([\d.]+\): (?P<partes>.+)\.$", salida_humo, re.MULTILINE
    )
    if val_sobre_techo_esperado or verdad_sobre_techo_esperado:
        assert m_techo, (
            "al menos un punto de detalle supera el techo de Bayes en esta corrida: la línea "
            "'Superan el techo de Bayes' debe imprimirse",
            val_sobre_techo_esperado,
            verdad_sobre_techo_esperado,
            salida_humo,
        )
        partes_techo = m_techo.group("partes")
        if val_sobre_techo_esperado:
            m_val_techo = re.search(r"(?P<n>\d+) exactitud_val de la búsqueda", partes_techo)
            assert m_val_techo and int(m_val_techo.group("n")) == val_sobre_techo_esperado, (
                "el conteo de 'exactitud_val de la búsqueda' debe coincidir con cuántas líneas "
                "de detalle tienen su búsqueda sobre el techo",
                partes_techo,
                val_sobre_techo_esperado,
            )
        else:
            assert "exactitud_val de la búsqueda" not in partes_techo, partes_techo
        if verdad_sobre_techo_esperado:
            m_verdad_techo = re.search(
                r"(?P<n>\d+) exactitud_verdadera media \((?P<k>\d+) semillas", partes_techo
            )
            assert m_verdad_techo, partes_techo
            assert int(m_verdad_techo.group("n")) == verdad_sobre_techo_esperado, (
                "el conteo de 'exactitud_verdadera media' debe coincidir con cuántas líneas de "
                "detalle tienen su verdadera sobre el techo",
                partes_techo,
                verdad_sobre_techo_esperado,
            )
            # El n de semillas de esta nota debe ser el n REAL que sobrevivió
            # (`punto['resumen_verdad'].n`), no K_REEVALUAR a secas: con
            # Divergencia descartando semillas, los dos difieren.
            assert int(m_verdad_techo.group("k")) == n_sobrevivientes, (
                m_verdad_techo.group(0),
                n_sobrevivientes,
            )
        else:
            assert "exactitud_verdadera media" not in partes_techo, partes_techo
    else:
        assert m_techo is None, (
            "ningún punto de detalle supera el techo de Bayes en esta corrida: la línea "
            "'Superan el techo de Bayes' no debe imprimirse",
            salida_humo,
        )


@pytest.mark.gpu
def test_humo_es_determinista(tmp_path_factory, corrida_humo):
    """Semillas fijas en todo el paso (dataset, búsqueda, re-evaluación,
    restricción y comparación pareada): dos corridas con el mismo PRESUPUESTO
    deben imprimir exactamente lo mismo.

    Reusa `corrida_humo` (que ya corrió el módulo UNA vez) como una de las dos
    corridas a comparar, en vez de lanzar dos corridas nuevas: esta prueba
    necesitaba dos ejecuciones independientes para comparar, pero no necesita
    que NINGUNA de las dos sea distinta del resto de la suite -- reusar la del
    fixture ahorra una corrida completa de tarjeta de video compartida. Se compara todo
    MENOS la última línea ("figura: ..."): esa sí cambia entre corridas porque
    cada una usa su propio TALLER_RESULTADOS, y esa ruta no es parte de
    ninguna respuesta numerada.
    """
    salida_1, _ = corrida_humo
    resultados_2 = tmp_path_factory.mktemp("resultados_det")
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados_2)}
    resultado_2 = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado_2.returncode == 0, resultado_2.stderr
    assert salida_1.rsplit("figura:", 1)[0] == resultado_2.stdout.rsplit("figura:", 1)[0]


@pytest.mark.gpu
def test_humo_evalua_a_lo_sumo_los_trials_del_presupuesto(salida_humo):
    """`PRESUPUESTOS['humo']['n']` (12) y `['k_reevaluar']` (3) fijan un TECHO
    para cuántos trials evalúa cada búsqueda y cuántas semillas nuevas ve cada
    punto re-evaluado -- no un piso exacto (un trial puede fallar por
    `Divergencia` y quedar fuera de COMPLETE). Se verifica en los NÚMEROS que
    imprimen las respuestas 1 y 4, no en el código fuente: así una corrida con
    algún trial divergente (poco probable con `epocas=3`, pero posible) sigue
    siendo una prueba de comportamiento válida, no un detector de cambios de
    texto que hay que actualizar a mano cada vez que se toca un comentario.
    """
    m1 = _RE_RESPUESTA_1.search(salida_humo)
    assert m1, salida_humo
    assert int(m1.group("n_total")) <= N_TOTAL_HUMO

    m4 = _RE_RESPUESTA_4.search(salida_humo)
    assert m4, salida_humo
    assert int(m4.group("completos")) <= N_TOTAL_HUMO

    m3 = _RE_RESPUESTA_3_RESUMEN.search(salida_humo)
    assert m3, salida_humo
    assert int(m3.group("k_reeval")) == K_REEVALUAR_HUMO


def _de_dif_desde_ic_pct(ic_bajo_pct: float, ic_alto_pct: float, k: int) -> float:
    """Reconstruye `de_dif` (escala CRUDA [0, 1], la que espera
    `taller_hpo.estadistica.semillas_necesarias`) a partir de un intervalo de
    confianza de 95% YA IMPRESO en PUNTOS PORCENTUALES y el tamaño de muestra
    `k` real (post-
    Divergencia) que lo produjo: invierte `ee = semiancho / t.ppf(0.975, k-1)`
    y `de_dif = ee * sqrt(k)` -- la misma álgebra que ya usa este archivo para
    reconstruir `de_dif_2` dentro de `test_humo_los_numeros_tienen_sentido`,
    factorizada aquí para reusarla también en las respuestas 4 y 5."""
    t_critico = float(scipy_stats.t.ppf(0.975, df=k - 1))
    ee_pct = (ic_alto_pct - ic_bajo_pct) / (2.0 * t_critico)
    return (ee_pct * (k**0.5)) / 100.0


def _verificar_nota_semillas(salida: str, fin: int, de_dif: float) -> bool:
    """Busca la nota '(semillas para DETECTAR ...)' justo después de la
    posición `fin` (el final de la respuesta que la trae) y, si es la rama
    numérica (de_dif medido > 0), recalcula su N contra
    `estadistica.semillas_necesarias(de_dif, 0.02)` con tolerancia ±1 -- el
    mismo patrón y la misma tolerancia que
    `test_respuesta_1_frase_potencia_recalcula_semillas_necesarias` en
    tests/test_paso_05.py (`de_dif` reconstruido desde el intervalo impreso
    puede diferir en su último dígito del `de_dif` exacto que usó el paso).
    Devuelve
    si de verdad verificó algo (False en la rama 'no aplica', que no tiene N
    que recalcular)."""
    ventana = salida[fin : fin + 400]
    m_nota = _RE_NOTA_SEMILLAS.search(ventana)
    if m_nota is None:
        assert "no aplica" in ventana, (
            "se esperaba una nota de semillas_necesarias justo después de esta respuesta",
            ventana,
        )
        return False
    if de_dif <= 0.0:
        return False
    n_esperado = estadistica.semillas_necesarias(de_dif, 0.02)
    n_impreso = int(m_nota.group("n"))
    assert abs(n_impreso - n_esperado) <= 1, (m_nota.group(0), n_esperado, de_dif)
    return True


@pytest.mark.gpu
def test_notas_semillas_necesarias_recalculan_desde_el_ic_impreso(salida_humo):
    """Las notas que acompañan las respuestas 2, 4 y 5 ('semillas para
    DETECTAR una diferencia de ±2 con potencia 80%: N; esta corrida usó K')
    no llevan una cifra fija de ninguna corrida: `N` tiene que ser
    `estadistica.semillas_necesarias(de_dif, 0.02)` con el `de_dif` MEDIDO de
    esa comparación en ESTA corrida. Aquí se reconstruye `de_dif` de forma
    independiente, a partir del intervalo de confianza de 95% ya impreso de
    cada comparación (como
    en tests/test_paso_05.py), y se recalcula `N` contra él -- mata pasar un
    `de_dif` distinto (p. ej. `comparacion.de_dif / 3.0`) en el sitio de
    llamada, algo que ninguna prueba sintética de `_nota_semillas_necesarias`
    sola puede detectar porque el defecto está fuera de ella."""
    verificadas = 0

    m2 = _RE_RESPUESTA_2.search(salida_humo)
    if m2 is not None:
        de_dif_2 = _de_dif_desde_ic_pct(
            float(m2.group("ic_bajo")), float(m2.group("ic_alto")), int(m2.group("k_ext"))
        )
        if _verificar_nota_semillas(salida_humo, m2.end(), de_dif_2):
            verificadas += 1

    m4 = _RE_RESPUESTA_4.search(salida_humo)
    assert m4, salida_humo
    m_costo = _RE_COSTO_REAL_CALCULADO.search(m4.group("frase_costo"))
    if m_costo is not None:
        de_dif_4 = _de_dif_desde_ic_pct(
            float(m_costo.group("ic_bajo")),
            float(m_costo.group("ic_alto")),
            int(m_costo.group("k")),
        )
        if _verificar_nota_semillas(salida_humo, m4.end(), de_dif_4):
            verificadas += 1

    m5 = _RE_RESPUESTA_5.search(salida_humo)
    if m5 is not None:
        de_dif_5 = _de_dif_desde_ic_pct(
            float(m5.group("ic_bajo")), float(m5.group("ic_alto")), int(m5.group("k"))
        )
        if _verificar_nota_semillas(salida_humo, m5.end(), de_dif_5):
            verificadas += 1

    assert verificadas > 0, (
        "esta corrida de humo no tuvo ninguna comparación con de_dif > 0 para poder "
        "verificar su nota de semillas_necesarias",
        salida_humo,
    )


# --- funciones puras de clasificación: probadas con ast, sin correr el paso ---


def _cargar_funciones_puras(
    nombres: tuple[str, ...], *, extra_espacio_nombres: dict | None = None
) -> dict:
    """Extrae, compila y devuelve las funciones puras nombradas de
    `pasos/04_pareto.py` SIN ejecutar el resto del script (que entrena en la
    tarjeta de video): parsea el archivo con `ast`, se queda solo con los
    `FunctionDef` de nivel superior cuyo nombre está en `nombres`, los compila
    en un módulo nuevo y devuelve `{nombre: función}`. Así se prueban TODAS
    las ramas de la lógica de clasificación con valores sintéticos, en las
    dos direcciones, sin pagar el costo -- ni el de la tarjeta de video -- de
    una corrida real.

    `extra_espacio_nombres` inyecta nombres adicionales al namespace aislado
    ANTES de ejecutar el módulo -- por ejemplo, una `Divergencia` de prueba
    para `_reevaluar_o_divergente`, que la referencia por nombre de módulo
    (`except Divergencia:`) sin importarla ella misma (la importa
    `pasos/04_pareto.py` de `taller_hpo.problemas.juguete`, fuera del AST
    extraído aquí)."""
    import numpy as np
    from scipy import stats as scipy_stats

    arbol = ast.parse(PASO.read_text(encoding="utf-8"))
    nodos = [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name in nombres]
    faltantes = set(nombres) - {n.name for n in nodos}
    assert not faltantes, f"no se encontraron estas funciones en {PASO}: {faltantes}"
    modulo = ast.Module(body=nodos, type_ignores=[])
    ast.fix_missing_locations(modulo)
    # _margen_prediccion usa `scipy_stats` y `np` del módulo del paso -- sin
    # sembrarlos aquí, ejecutar solo su AST (sin el resto de los imports del
    # archivo) daría NameError.
    espacio_nombres: dict = {"np": np, "scipy_stats": scipy_stats, **(extra_espacio_nombres or {})}
    exec(compile(modulo, filename=str(PASO), mode="exec"), espacio_nombres)
    return {nombre: espacio_nombres[nombre] for nombre in nombres}


def test_veredicto_relativo_las_tres_ramas_y_sus_bordes():
    f = _cargar_funciones_puras(("_veredicto_relativo",))["_veredicto_relativo"]
    assert f(0.5, 1.0) == "arriba"
    assert f(-1.0, -0.5) == "abajo"
    assert f(-1.0, 1.0) == "no_distingue"
    assert f(-1.0, 0.0) == "no_distingue", "ic_alto == 0.0 no es < 0.0: no debe ser 'abajo'"
    assert f(0.0, 1.0) == "no_distingue", "ic_bajo == 0.0 no es > 0.0: no debe ser 'arriba'"


def test_frase_control_punto_y_frase_ic_cero_cubren_las_tres_ramas():
    funcs = _cargar_funciones_puras(("_frase_control_punto", "_frase_ic_cero"))
    assert funcs["_frase_control_punto"]("arriba") == "infla más que el control"
    assert funcs["_frase_control_punto"]("abajo") == "infla menos que el control"
    assert funcs["_frase_control_punto"]("no_distingue") == "no se distingue del control"
    assert funcs["_frase_ic_cero"]("arriba") == "se distingue de 0 por encima"
    assert funcs["_frase_ic_cero"]("abajo") == "se distingue de 0 por debajo"
    assert funcs["_frase_ic_cero"]("no_distingue") == "no se distingue de 0"


def test_frase_extremos_las_cuatro_ramas():
    f = _cargar_funciones_puras(("_frase_extremos",))["_frase_extremos"]
    assert f("mejor", 5.0, 2.0) == "SÍ compra 5.00 puntos de exactitud_val"
    frase_peor = f("peor", -5.0, 2.0)
    assert "PEOR" in frase_peor and "pierde 5.00 puntos" in frase_peor
    assert "no compra nada distinguible" in f("equivalente", 0.5, 2.0)
    assert "no concluyente" in f("no concluyente", 0.5, 2.0)


def test_frase_delta_busqueda_las_cuatro_ramas_en_las_dos_direcciones():
    f = _cargar_funciones_puras(("_frase_delta_busqueda",))["_frase_delta_busqueda"]
    # sobreestimaba: delta por ENCIMA del intervalo de predicción.
    assert "sobreestimaba" in f(10.0, -2.0, 5.0, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)
    # subestimaba: delta por DEBAJO -- la comparación es bilateral: la
    # búsqueda puede sobre O sub-estimar la diferencia frente a re-evaluar.
    assert "subestimaba" in f(-10.0, -2.0, 5.0, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)
    # coincide: el intervalo entero cae dentro de +-delta_pct DE LA DESVIACIÓN
    # respecto a delta_busqueda, no de los extremos contra 0.
    assert "coincide dentro de" in f(0.5, -1.0, 1.0, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)
    # no concluyente: delta dentro del intervalo, pero el intervalo no cabe
    # entero en +-delta_pct alrededor de delta_busqueda.
    assert "no concluyente" in f(0.0, -10.0, 10.0, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)


def test_frase_delta_busqueda_coincide_compara_la_desviacion_no_los_extremos_contra_cero():
    """Caso A y B: demuestran que 'coincide' debe comparar delta_busqueda_pct
    contra el intervalo de predicción -- nunca los extremos del intervalo
    contra 0, que es una pregunta distinta (si el intervalo es chico en
    términos absolutos, no si delta_busqueda_pct cae cerca de él)."""
    f = _cargar_funciones_puras(("_frase_delta_busqueda",))["_frase_delta_busqueda"]
    # Caso A: búsqueda +20.0 con predicción [+18.5, +21.5]. La desviación
    # [-1.5, +1.5] cabe dentro de ±2 -> 'coincide'.
    frase_a = f(20.0, 18.5, 21.5, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)
    assert frase_a.startswith("coincide dentro de"), frase_a
    # Caso B: búsqueda +1.8 con predicción [-1.0, +1.9]. La desviación
    # [-0.1, +2.8] NO cabe dentro de ±2 (2.8 > 2.0) -> 'no concluyente', pese a
    # que los extremos crudos del intervalo sí caían dentro de ±2.
    frase_b = f(1.8, -1.0, 1.9, delta_pct=2.0, k=5, margen_asintotico_pct=0.5)
    assert frase_b.startswith("no concluyente"), frase_b


def test_frase_delta_busqueda_no_concluyente_distingue_piso_irresoluble_de_mas_semillas():
    """Cuando el piso asintótico (el margen con infinitas semillas de
    re-evaluación) ya es mayor que ±delta_pct, ninguna cantidad de semillas
    resuelve nada: la frase debe decirlo así, en vez de invitar a agregar más
    semillas -- que sí ayudaría en el caso contrario. Las dos ramas comparten
    el mismo delta_busqueda/intervalo (ninguno de los dos resuelve la
    pregunta): lo único que cambia es el piso asintótico."""
    f = _cargar_funciones_puras(("_frase_delta_busqueda",))["_frase_delta_busqueda"]
    irresoluble = f(0.0, -10.0, 10.0, delta_pct=2.0, k=5, margen_asintotico_pct=4.4)
    assert "ninguna cantidad de semillas" in irresoluble, irresoluble
    assert "más semillas de re-evaluación sí podrían resolverlo" not in irresoluble
    resoluble = f(0.0, -10.0, 10.0, delta_pct=2.0, k=5, margen_asintotico_pct=1.0)
    assert "con 5 semillas" in resoluble, resoluble
    assert "más semillas de re-evaluación sí podrían resolverlo" in resoluble


def test_margen_prediccion_positivo_simetrico_y_cero_con_de_dif_cero():
    f = _cargar_funciones_puras(("_margen_prediccion",))["_margen_prediccion"]
    assert f(0.0, 5) == 0.0, "sin dispersión (de_dif == 0.0) no hay nada que predecir"
    margen = f(0.02, 5)
    assert margen > 0.0
    # Crece con de_dif y decrece con n (más observaciones angostan el
    # intervalo, aunque nunca a cero por el "+1" de la varianza de predicción).
    assert f(0.04, 5) > margen
    assert f(0.02, 20) < margen


def test_margen_prediccion_no_pareada_combina_las_tres_varianzas():
    f = _cargar_funciones_puras(("_margen_prediccion_no_pareada",))["_margen_prediccion_no_pareada"]
    assert f(0.0, 0.0, 0.0, 5) == 0.0
    margen = f(0.02, 0.03, 0.04, 5)
    assert margen > 0.0
    # Cualquiera de las tres fuentes de varianza ensancha el margen: la de la
    # diferencia PAREADA (incertidumbre del centro) y las dos poblacionales NO
    # pareadas (dispersión de una diferencia nueva).
    assert f(0.05, 0.03, 0.04, 5) > margen
    assert f(0.02, 0.05, 0.04, 5) > margen
    assert f(0.02, 0.03, 0.06, 5) > margen
    # Más n angosta la parte de_dif**2/n (menos incertidumbre en el centro),
    # pero nunca por debajo del piso asintótico (de_a**2 + de_b**2, fijo).
    assert f(0.02, 0.03, 0.04, 50) < margen


def test_margen_asintotico_no_pareada_es_el_piso_de_margen_prediccion_no_pareada():
    funcs = _cargar_funciones_puras(
        ("_margen_prediccion_no_pareada", "_margen_asintotico_no_pareada")
    )
    assert funcs["_margen_asintotico_no_pareada"](0.0, 0.0) == 0.0
    piso = funcs["_margen_asintotico_no_pareada"](0.03, 0.04)
    assert piso > 0.0
    # El margen con n finito siempre es MAYOR que el piso asintótico (el t
    # crítico con pocos grados de libertad supera a z, y de_dif**2/n solo
    # puede sumar, nunca restar) -- incluso con n grande.
    assert funcs["_margen_prediccion_no_pareada"](0.02, 0.03, 0.04, 5) > piso
    assert funcs["_margen_prediccion_no_pareada"](0.02, 0.03, 0.04, 10_000) > piso


def test_ic_relativo_control_resta_la_media_pareada_y_usa_margen_no_pareado():
    """`_ic_relativo_control` compara dos trials DISTINTOS de la búsqueda (el
    punto y el control, cada uno con su propia semilla): el margen tiene que
    salir de la varianza NO pareada (de_dif**2/n + de_punto**2 + de_control**2,
    la misma fórmula que `_margen_prediccion_no_pareada`), nunca de
    `_margen_prediccion` a secas sobre `de_dif` (válido solo si punto y
    control re-evaluados con la misma semilla tuvieran correlación 0, algo
    que este paso no mide -- ver el docstring de la función). Calculado aquí
    a mano con scipy, sin volver a invocar ninguna función de
    `pasos/04_pareto.py`, en las dos direcciones: con `de_punto`/`de_control`
    en 0 el margen se reduce (pero no colapsa, por el término `de_dif**2/n`)
    y con cualquiera de los tres > 0 el margen crece."""
    from scipy import stats as scipy_stats_local

    # _ic_relativo_control llama a _margen_prediccion_no_pareada por su
    # nombre: las dos tienen que compilarse JUNTAS en el mismo módulo
    # aislado, o la llamada interna da NameError.
    funcs = _cargar_funciones_puras(("_ic_relativo_control", "_margen_prediccion_no_pareada"))
    f = funcs["_ic_relativo_control"]
    d_crudo, diferencia_media, de_dif, de_punto, de_control, n = 0.10, 0.04, 0.02, 0.03, 0.05, 9
    media, ic_bajo, ic_alto = f(d_crudo, diferencia_media, de_dif, de_punto, de_control, n)
    assert media == pytest.approx(d_crudo - diferencia_media)
    assert ic_bajo <= media <= ic_alto
    t_critico = float(scipy_stats_local.t.ppf(0.975, df=n - 1))
    margen_esperado = t_critico * (de_dif**2 / n + de_punto**2 + de_control**2) ** 0.5
    assert (ic_alto - media) == pytest.approx(margen_esperado)
    assert (media - ic_bajo) == pytest.approx(margen_esperado)
    # de_punto = de_control = 0 -> el margen NO colapsa a 0 (a diferencia de
    # _margen_prediccion con de_dif == 0): sigue quedando el término
    # de_dif**2/n, la incertidumbre de estimar el centro con las n
    # re-evaluaciones pareadas. Mata volver a `_margen_prediccion` (que sí
    # colapsaría a 0 aquí porque solo mira de_dif, y de_dif > 0 en este caso
    # no alcanza para distinguir las dos fórmulas por sí solo).
    media_sin_poblacion, ic_bajo_sp, ic_alto_sp = f(d_crudo, diferencia_media, de_dif, 0.0, 0.0, n)
    margen_sin_poblacion = t_critico * (de_dif**2 / n) ** 0.5
    assert margen_sin_poblacion > 0.0
    assert (ic_alto_sp - media_sin_poblacion) == pytest.approx(margen_sin_poblacion)
    assert margen_sin_poblacion < margen_esperado, (
        "quitar la dispersión poblacional de punto y control debe angostar el margen frente "
        "al caso con de_punto/de_control > 0"
    )
    # Con las tres varianzas en 0, el intervalo sí colapsa a la media.
    media_0, ic_bajo_0, ic_alto_0 = f(d_crudo, diferencia_media, 0.0, 0.0, 0.0, n)
    assert ic_bajo_0 == ic_alto_0 == pytest.approx(media_0)


def test_ic_agregado_vs_control_suma_la_varianza_de_prediccion_del_control():
    """Con `de_control > 0` el intervalo debe ser MÁS ANCHO que tratando la
    inflación del control como una constante conocida (`de_control == 0`) --
    el control es UNA extracción, tan incierta como cualquier punto
    individual -- calculado con la fórmula a mano, en las dos direcciones."""
    f = _cargar_funciones_puras(("_ic_agregado_vs_control",))["_ic_agregado_vs_control"]
    diff, ic_bajo, ic_alto = f(0.05, 0.01, 10, 0.02, 0.03, 9)
    assert diff == pytest.approx(0.03)
    assert ic_bajo <= diff <= ic_alto
    diff_sin_control, ic_bajo_sin, ic_alto_sin = f(0.05, 0.01, 10, 0.02, 0.0, 9)
    assert diff_sin_control == pytest.approx(diff)
    assert (ic_alto - diff) > (ic_alto_sin - diff_sin_control), (
        "la varianza de predicción del control (de_control > 0) debe ensanchar el intervalo "
        "frente a "
        "tratar su inflación como una constante conocida (de_control == 0)",
        ic_alto,
        ic_alto_sin,
    )
    # Calculado a mano con scipy: t_critico(df=9) * sqrt(ee**2 + de**2*(1+1/n)).
    from scipy import stats as scipy_stats_local

    t_critico = float(scipy_stats_local.t.ppf(0.975, df=9))
    se_esperado = (0.01**2 + 0.03**2 * (1.0 + 1.0 / 9.0)) ** 0.5
    assert (ic_alto - diff) == pytest.approx(t_critico * se_esperado)


def test_frase_costo_restriccion_las_cuatro_ramas():
    f = _cargar_funciones_puras(("_frase_costo_restriccion",))["_frase_costo_restriccion"]
    assert f("mejor", 5.0, 2.0, 5) == "la restricción SÍ cuesta 5.00 puntos"
    assert f("peor", 5.0, 2.0, 5) == "la restricción GANA 5.00 puntos"
    assert f("equivalente", 5.0, 2.0, 5) == "la restricción no cuesta nada distinguible (±2)"
    assert f("no concluyente", 5.0, 2.0, 5) == "no concluyente con 5 semillas"


def test_frase_no_vista_las_tres_ramas():
    f = _cargar_funciones_puras(("_frase_no_vista",))["_frase_no_vista"]
    assert f("arriba") == "la validación de este dataset SOBREESTIMA la verdad"
    assert f("abajo") == "la validación de este dataset SUBESTIMA la verdad"
    assert "no se distingue de 0" in f("no_distingue")


def test_reevaluar_o_divergente_descarta_la_semilla_completa_para_todas_las_configs():
    """`_reevaluar_o_divergente` (a diferencia de `taller_hpo.honestidad.reevaluar`)
    tiene que descartar una semilla COMPLETA -- para TODAS las configs, no
    solo la que divergió -- en cuanto CUALQUIER config diverge con ella (ver
    su docstring en pasos/04_pareto.py): así las listas resultantes quedan
    con el mismo largo y la MISMA semilla en cada posición para cada config,
    que es lo que `taller_hpo.estadistica.diferencia_pareada` necesita para
    emparejar. Se prueba con una `Divergencia` de mentira inyectada en el
    namespace aislado (`_cargar_funciones_puras` no ejecuta el resto del
    archivo, que es donde se importa la real de
    `taller_hpo.problemas.juguete`) y un `entrenar` falso que la lanza para
    la config 'b' solo en la semilla 1 -- nunca para 'a', y nunca en las
    semillas 0 o 2.

    Mata conservar el resultado parcial de 'a' en la semilla 1 (rompería el
    pareo semilla a semilla que promete el docstring: 'a' tendría 3
    resultados y 'b' solo 2, sin forma de saber cuál de los 3 de 'a' no
    tiene pareja)."""

    class DivergenciaFalsa(Exception):
        pass

    f = _cargar_funciones_puras(
        ("_reevaluar_o_divergente",), extra_espacio_nombres={"Divergencia": DivergenciaFalsa}
    )["_reevaluar_o_divergente"]

    def entrenar_falso(config, semilla, reportar):
        if config["marca"] == "b" and semilla == 1:
            raise DivergenciaFalsa("mock: b diverge en la semilla 1")
        return {"valor": semilla, "marca": config["marca"]}

    configs = {"a": {"marca": "a"}, "b": {"marca": "b"}}
    resultado, descartadas = f(configs, entrenar_falso, [0, 1, 2])

    assert descartadas == 1, (descartadas, resultado)
    assert len(resultado["a"]) == len(resultado["b"]) == 2, (
        "las dos configs deben quedar con el MISMO largo tras descartar la semilla 1 "
        "COMPLETA -- no solo de la config que divergió",
        resultado,
    )
    # Pareo semilla a semilla: la semilla 1 (la que divergió para 'b') debe
    # faltar de LAS DOS listas, y las semillas 0 y 2 deben seguir en la MISMA
    # posición relativa para 'a' y 'b'.
    assert [r["valor"] for r in resultado["a"]] == [0, 2], resultado["a"]
    assert [r["valor"] for r in resultado["b"]] == [0, 2], resultado["b"]


def test_n_repr_frente_incluye_el_mas_uno_y_recorta_despues():
    """`_n_repr_frente(n_frente, maximo)` decide cuántos puntos del frente
    re-evaluar: TODOS (`n_frente`) cuando el frente ya tiene a lo sumo
    `maximo + 1` puntos -- el "+1" es lo que de verdad cuesta no recortar
    nada, un solo entrenamiento extra por semilla -- y solo `maximo` en caso
    contrario. Probada en las dos direcciones con `MAX_FRENTE_REEVALUAR_HUMO`
    (4, el valor real de `PRESUPUESTOS['humo']['max_frente_reevaluar']` en
    pasos/04_pareto.py) y con otro `maximo` distinto (6), para no depender de
    una sola constante: revertir el "+1" a un simple `n_frente <= maximo`
    (`MAX_FRENTE_REEVALUAR_HUMO` frentes de tamaño exacto `maximo + 1` dejarían
    de re-evaluarse completos) lo detecta el caso `(5, 4) -> 5`."""
    f = _cargar_funciones_puras(("_n_repr_frente",))["_n_repr_frente"]
    assert f(MAX_FRENTE_REEVALUAR_HUMO, MAX_FRENTE_REEVALUAR_HUMO) == MAX_FRENTE_REEVALUAR_HUMO
    assert (
        f(MAX_FRENTE_REEVALUAR_HUMO + 1, MAX_FRENTE_REEVALUAR_HUMO) == MAX_FRENTE_REEVALUAR_HUMO + 1
    )
    assert f(MAX_FRENTE_REEVALUAR_HUMO + 2, MAX_FRENTE_REEVALUAR_HUMO) == MAX_FRENTE_REEVALUAR_HUMO
    assert f(7, 6) == 7
    assert f(8, 6) == 6


# --- DISTRIBUCIONES_ESPACIO no puede desincronizarse a mano de espacio() ---


@pytest.mark.gpu
def test_distribuciones_espacio_coincide_con_las_distributions_reales(tmp_path):
    """`DISTRIBUCIONES_ESPACIO` es una copia a mano del espacio real que
    arma `espacio()`, usada solo para validar la línea base ANTES de
    encolarla (`taller_hpo.estudio.sembrar_linea_base`). Si `espacio()`
    cambia sin actualizar esa copia, `sembrar_linea_base` seguiría validando
    contra un espacio viejo sin que nada lo marque: se verifica contra las
    `distributions` reales de un trial cualquiera que Optuna sí registró."""
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    codigo = (
        "import runpy\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "trial = next(\n"
        "    t for t in ns['trials_pareto_completos']\n"
        "    if t.user_attrs.get('rol') != 'linea_base'\n"
        ")\n"
        "assert trial.distributions == ns['DISTRIBUCIONES_ESPACIO'], (\n"
        "    dict(trial.distributions), ns['DISTRIBUCIONES_ESPACIO']\n"
        ")\n"
        "print('DISTRIBUCIONES_OK')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "DISTRIBUCIONES_OK" in resultado.stdout, resultado.stdout


@pytest.mark.gpu
def test_mejor_restriccion_es_el_argmax_de_los_factibles_con_constraints_validas(tmp_path):
    """Caja blanca dirigida al ganador de la respuesta 4:
    `mejor_restriccion = study_restriccion.best_trial if factibles else None`.
    Una prueba de caja negra sobre stdout no puede distinguir "el mejor
    trial FACTIBLE" de "cualquier trial factible" (p. ej. el segundo mejor,
    o el peor): se lee el namespace real del script (`runpy.run_path` lo
    devuelve) y se verifica, con los propios `factibles`/`trials_restriccion`
    que el paso calculó, que `mejor_restriccion.value` es el máximo de
    `t.value` entre los factibles -- ni el segundo mejor
    (`sorted(factibles, key=lambda t: t.value)[-2]`) ni el peor -- y que
    TODAS sus restricciones son <= 0 (la definición misma de "factible").

    De paso, ata `n_infactibles` (calculado dentro del `else` de la respuesta
    4) a `len(trials_restriccion) - len(factibles)`: con las semillas fijas
    de humo la búsqueda queda en 6 factibles y 6 infactibles, así que esta
    comparación en particular no distingue la fórmula correcta de
    `n_infactibles = len(factibles)` (las dos dan 6 aquí por coincidencia) --
    ver `test_n_infactibles_no_es_len_factibles_con_split_forzado_asimetrico`,
    más abajo, que rompe esa coincidencia a propósito.
    """
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    codigo = (
        "import runpy\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "factibles = ns['factibles']\n"
        "trials_restriccion = ns['trials_restriccion']\n"
        "mejor_restriccion = ns['mejor_restriccion']\n"
        "assert factibles, 'esta prueba necesita al menos un trial factible en esta corrida'\n"
        "assert mejor_restriccion is not None\n"
        "assert all(v <= 0.0 for v in mejor_restriccion.constraints.values()), (\n"
        "    'el ganador de la respuesta 4 debe ser FACTIBLE: todas sus restricciones <= 0',\n"
        "    dict(mejor_restriccion.constraints),\n"
        ")\n"
        "esperado = max(t.value for t in factibles)\n"
        "assert mejor_restriccion.value == esperado, (\n"
        "    'mejor_restriccion debe ser el trial de mayor value ENTRE LOS FACTIBLES -- no el "
        "segundo mejor ni el peor',\n"
        "    mejor_restriccion.value,\n"
        "    esperado,\n"
        ")\n"
        "n_infactibles_esperado = len(trials_restriccion) - len(factibles)\n"
        "print(f'N_INFACTIBLES_ESPERADO={n_infactibles_esperado}')\n"
        "if 'busqueda_factible' in ns:\n"
        "    assert ns['busqueda_factible'] == mejor_restriccion.value, (\n"
        "        'la respuesta 4 debe juzgar al ganador FACTIBLE, no a otro trial',\n"
        "        ns['busqueda_factible'],\n"
        "        mejor_restriccion.value,\n"
        "    )\n"
        "print(f'VALOR_FACTIBLE={mejor_restriccion.value!r}')\n"
        "print('ARGMAX_OK')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "ARGMAX_OK" in resultado.stdout, resultado.stdout

    m4 = _RE_RESPUESTA_4.search(resultado.stdout)
    assert m4, resultado.stdout
    m_esperado = re.search(r"N_INFACTIBLES_ESPERADO=(\d+)", resultado.stdout)
    assert m_esperado, resultado.stdout
    assert int(m4.group("infactibles")) == int(m_esperado.group(1)), (
        "el 'infactibles' impreso por la respuesta 4 debe ser "
        "len(trials_restriccion) - len(factibles), no len(factibles) a secas",
        m4.group("infactibles"),
        m_esperado.group(1),
    )

    # El "búsqueda X" de la parte "aparte" de la respuesta 4 tiene que ser el
    # value del ganador FACTIBLE. Si saliera de otro trial (p. ej. el mejor sin
    # mirar la restricción, que es infactible), la comparación contra el
    # control juzgaría a un trial que la búsqueda con restricción nunca eligió.
    m_valor = re.search(r"VALOR_FACTIBLE=(\S+)", resultado.stdout)
    assert m_valor, resultado.stdout
    valor_factible = float(m_valor.group(1))
    frase_propia = m4.group("frase_propia")
    m_propia = _RE_PROPIA_CALCULADA.search(frase_propia) or _RE_PROPIA_SIN_CONTROL.search(
        frase_propia
    )
    assert m_propia, frase_propia
    assert float(m_propia.group("busqueda")) == pytest.approx(
        round(valor_factible, 3), abs=0.0005
    ), (frase_propia, valor_factible)


@pytest.mark.gpu
def test_n_infactibles_no_es_len_factibles_con_split_forzado_asimetrico(tmp_path):
    """Caja blanca CON SESGO REAL que rompe a propósito la coincidencia de
    `test_mejor_restriccion_es_el_argmax_...` (arriba): con las semillas
    fijas de humo, `len(factibles)` y `len(trials_restriccion) -
    len(factibles)` dan el MISMO número (6 y 6), así que ninguna prueba que
    compare el 'infactibles' impreso contra esa resta puede distinguirla de
    `n_infactibles = len(factibles)` (una fórmula distinta que da el mismo
    número aquí, solo por coincidencia). Aquí se envuelve
    `taller_hpo.problemas.juguete.entrenar` para forzar, SOLO en las llamadas
    de la búsqueda con restricción (se distingue de la de Pareto por
    `metricas`, la misma técnica de
    `test_respuesta4_propia_detecta_sesgo_inyectado_solo_en_esa_busqueda`), un
    `n_parametros` de 10 (factible) para los dos primeros trials
    (`trial.number < 2`) y de 100_000 (infactible, muy por encima del
    presupuesto de 100) para el resto: con `N_TOTAL_HUMO=12` trials, eso deja
    2 factibles y 10 infactibles -- números BIEN distintos entre sí -- así
    que `len(factibles)` (2) y `len(trials_restriccion) - len(factibles)`
    (10) ya no pueden coincidir por casualidad.
    """
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    envoltura = (
        "import inspect\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    trial = loc.get('trial')\n"
        "    metricas = loc.get('metricas')\n"
        "    es_restriccion = metricas is not None and list(metricas) == ['exactitud_val']\n"
        "    if trial is not None and es_restriccion:\n"
        "        resultado['n_parametros'] = 10.0 if trial.number < 2 else 100_000.0\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "print(f'N_FACTIBLES={len(ns[\"factibles\"])}')\n"
        "print(f'N_COMPLETOS={len(ns[\"trials_restriccion\"])}')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    n_factibles = int(re.search(r"N_FACTIBLES=(\d+)", salida).group(1))
    n_completos = int(re.search(r"N_COMPLETOS=(\d+)", salida).group(1))
    n_infactibles_esperado = n_completos - n_factibles
    assert n_factibles != n_infactibles_esperado, (
        "esta prueba necesita un split asimétrico para tener poder de detección -- si el sesgo "
        "no lo logró, ajustar los n_parametros forzados",
        n_factibles,
        n_infactibles_esperado,
    )

    m4 = _RE_RESPUESTA_4.search(salida)
    assert m4, salida
    assert int(m4.group("factibles")) == n_factibles, (m4.group(0), n_factibles)
    assert int(m4.group("infactibles")) == n_infactibles_esperado, (
        "con un split forzado de 2 factibles y 10 infactibles, el 'infactibles' impreso debe "
        "ser la RESTA (len(completos) - len(factibles) = 10), no len(factibles) (2) -- si "
        "coincide con len(factibles), la fórmula se vació a esa variable en el sitio de "
        "impresión",
        m4.group("infactibles"),
        n_infactibles_esperado,
        n_factibles,
    )


@pytest.mark.gpu
def test_indices_reevaluar_cubre_el_frente_completo_cuando_cabe_en_el_mas_uno(tmp_path):
    """Caja blanca dirigida al borde del "+1" de `_n_repr_frente`
    (pasos/04_pareto.py): `indices_reevaluar` en la corrida real (no solo la
    función pura probada con valores sintéticos) tiene que cubrir el frente
    COMPLETO cuando `len(frente_ordenado) <= MAX_FRENTE_REEVALUAR + 1` -- en
    humo, `MAX_FRENTE_REEVALUAR_HUMO` es 4, así que un frente de hasta 5
    puntos se re-evalúa entero. Si el frente de esta corrida excede ese
    tamaño, se verifica en cambio que `indices_reevaluar` tenga exactamente
    `MAX_FRENTE_REEVALUAR_HUMO` elementos y que incluya los dos extremos --
    la otra mitad de la misma regla."""
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    codigo = (
        "import runpy\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "n_frente = len(ns['frente_ordenado'])\n"
        "indices = ns['indices_reevaluar']\n"
        "maximo = ns['MAX_FRENTE_REEVALUAR']\n"
        "print(f'N_FRENTE={n_frente}')\n"
        "print(f'INDICES={list(indices)}')\n"
        "print(f'MAXIMO={maximo}')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    n_frente = int(re.search(r"N_FRENTE=(\d+)", resultado.stdout).group(1))
    indices = ast.literal_eval(re.search(r"INDICES=(\[.*\])", resultado.stdout).group(1))
    maximo = int(re.search(r"MAXIMO=(\d+)", resultado.stdout).group(1))
    if n_frente <= maximo + 1:
        assert indices == list(range(n_frente)), (
            "con un frente de a lo sumo MAX_FRENTE_REEVALUAR + 1 puntos, indices_reevaluar debe "
            "cubrirlo COMPLETO -- no solo MAX_FRENTE_REEVALUAR representantes",
            indices,
            n_frente,
            maximo,
        )
    else:
        assert len(indices) == maximo, (indices, maximo)
        assert 0 in indices and (n_frente - 1) in indices, (
            "los índices representativos siempre deben incluir los dos extremos del frente",
            indices,
            n_frente,
        )


# --- verificación de caja blanca: qué semillas y qué métrica ve cada llamada real ---


@pytest.mark.gpu
def test_reevaluacion_usa_semillas_nuevas_y_no_optimiza_el_oraculo(tmp_path):
    """Envuelve `taller_hpo.problemas.juguete.entrenar` en un subproceso nuevo para
    verificar cuatro cosas que stdout no muestra:

    1. Re-evaluar un punto del frente usa semillas nuevas: las llamadas posteriores a la
       primera comparten `datos` con ella pero tienen `orden`/`init` distintos.
    2. Ninguna "reevaluada" sale del oráculo: la envoltura fuerza `exactitud_verdadera` a
       0.0, así que con el código correcto ninguna "reevaluada" impresa vale 0.000.
    3. El ganador con restricción, el punto del frente al mismo presupuesto y el control
       entran a la misma llamada a `_reevaluar_o_divergente` con las mismas semillas,
       como exige `diferencia_pareada`. Las etiquetas reales se leen de `marco.f_globals`.
    4. La config que entrena bajo el rol 'restriccion' es el ganador factible
       (`mejor_restriccion`), atado por `n_parametros` al `factible_params` de la
       respuesta 4.
    """
    registro = tmp_path / "llamadas.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_REGISTRO_LLAMADAS": str(registro),
    }
    envoltura = (
        "import inspect\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_llamadas = []\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    rol = etiqueta\n"
        "    if marco is not None and etiqueta is not None:\n"
        "        g = marco.f_globals\n"
        "        if etiqueta == g.get('RESTRICCION_ETIQUETA'):\n"
        "            rol = 'restriccion'\n"
        "        elif etiqueta == g.get('ETIQUETA_PRESUPUESTO'):\n"
        "            rol = 'frente'\n"
        "        elif etiqueta == g.get('CONTROL_ETIQUETA'):\n"
        "            rol = 'control'\n"
        "    _llamadas.append(\n"
        "        {\n"
        "            'config': dict(config),\n"
        "            'datos': semillas.datos,\n"
        "            'orden': semillas.orden,\n"
        "            'init': semillas.init,\n"
        "            'n_parametros': resultado['n_parametros'],\n"
        "            'etiqueta': rol,\n"
        "        }\n"
        "    )\n"
        "    resultado['exactitud_verdadera'] = 0.0\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "with open(os.environ['TALLER_REGISTRO_LLAMADAS'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(_llamadas, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    m = _RE_FRENTE_LINEA.search(resultado.stdout)
    assert m, resultado.stdout
    frente = ast.literal_eval(m.group("lista"))
    assert all(exactitud != 0.0 for _, exactitud in frente), (
        "el frente no debe depender de exactitud_verdadera (forzada aquí a 0.0): si una "
        "exactitud del frente sale 0.0, el código está usando el oráculo para construirlo"
    )

    # Ninguna "reevaluada" de la respuesta 3 puede salir 0.000 con el oráculo
    # forzado a 0.0 -- si el código usara exactitud_verdadera como la métrica
    # "reevaluada" (en vez de exactitud_val), saldría exactamente 0.
    puntos = _RE_RESPUESTA_3_PUNTO.findall(resultado.stdout)
    assert puntos, resultado.stdout
    assert all(float(p[2]) != 0.0 for p in puntos), (
        "una 'reevaluada' en 0.000 con el oráculo forzado a 0.0 indica que el código usó "
        "exactitud_verdadera en vez de exactitud_val",
        puntos,
    )

    llamadas = json.loads(registro.read_text(encoding="utf-8"))
    assert len(llamadas) > 0

    por_config: dict[str, list[dict]] = {}
    for llamada in llamadas:
        clave = json.dumps(llamada["config"], sort_keys=True)
        por_config.setdefault(clave, []).append(llamada)
    reevaluadas = [v for v in por_config.values() if len(v) > 1]
    assert reevaluadas, "ninguna config se llamó más de una vez: revisar la selección del frente"
    for llamadas_de_una_config in reevaluadas:
        primera, *resto = llamadas_de_una_config
        for llamada in resto:
            assert llamada["datos"] == primera["datos"], (
                "la re-evaluación debe compartir los MISMOS datos que vio la búsqueda"
            )
            assert (llamada["orden"], llamada["init"]) != (primera["orden"], primera["init"]), (
                "re-evaluar con la MISMA orden/init que la búsqueda no mide nada nuevo "
                "(la inflación saldría siempre +0.00)"
            )

    # El ganador FACTIBLE que la respuesta 4 ya imprimió: la config que
    # entrena bajo 'restriccion' en la re-evaluación conjunta (abajo) tiene
    # que ser ESE mismo trial, no otro (p. ej. mejor_sin_restriccion, que
    # puede exceder PRESUPUESTO_PARAMETROS y por lo tanto no ser factible).
    m4 = _RE_RESPUESTA_4.search(resultado.stdout)
    assert m4, resultado.stdout
    factible_params = float(m4.group("factible_params"))

    # Las llamadas de la re-evaluación conjunta (respuestas 4 y 5): dos
    # etiquetas siempre ('restriccion'/'frente'), y una tercera ('control')
    # si había línea base disponible -- ver el docstring de esta prueba.
    llamadas_finales = [
        ll for ll in llamadas if ll.get("etiqueta") in ("restriccion", "frente", "control")
    ]
    etiquetas_presentes: list[str] = []
    for ll in llamadas_finales:
        if ll["etiqueta"] not in etiquetas_presentes:
            etiquetas_presentes.append(ll["etiqueta"])
    assert {"restriccion", "frente"} <= set(etiquetas_presentes), llamadas_finales
    tam_grupo = len(etiquetas_presentes)
    assert len(llamadas_finales) == tam_grupo * K_REEVALUAR_HUMO, (
        "la re-evaluación conjunta debe hacer exactamente una llamada por etiqueta y por "
        "semilla, sin llamadas de más ni de menos",
        llamadas_finales,
    )
    for i in range(0, len(llamadas_finales), tam_grupo):
        grupo = llamadas_finales[i : i + tam_grupo]
        assert {ll["etiqueta"] for ll in grupo} == set(etiquetas_presentes), grupo
        semillas_grupo = {(ll["orden"], ll["init"]) for ll in grupo}
        assert len(semillas_grupo) == 1, (
            "todas las configs de la re-evaluación conjunta deben ver la MISMA semilla en "
            "cada posición -- si no, la comparación pareada no está emparejando nada",
            grupo,
        )
        datos_grupo = {ll["datos"] for ll in grupo}
        assert len(datos_grupo) == 1, grupo

    restriccion_calls = [ll for ll in llamadas_finales if ll["etiqueta"] == "restriccion"]
    frente_calls = [ll for ll in llamadas_finales if ll["etiqueta"] == "frente"]
    assert len(restriccion_calls) == len(frente_calls) == K_REEVALUAR_HUMO
    for restriccion, frente_cmp in zip(restriccion_calls, frente_calls, strict=True):
        assert restriccion["config"] != frente_cmp["config"], (
            "las dos configs de la re-evaluación conjunta deben ser distintas entre sí",
            restriccion,
            frente_cmp,
        )
        assert restriccion["n_parametros"] <= 100.0, (
            "la config 'restriccion' de la re-evaluación conjunta debe respetar "
            "PRESUPUESTO_PARAMETROS <= 100 -- si excede eso, no es el ganador FACTIBLE",
            restriccion,
        )
        assert restriccion["n_parametros"] == pytest.approx(factible_params, abs=0.5), (
            "la config 'restriccion' de la re-evaluación conjunta debe ser, en parámetros, "
            "el mismo ganador FACTIBLE que ya imprimió la respuesta 4 -- no otro trial de la "
            "búsqueda con restricción (p. ej. el mejor SIN mirar la restricción, que puede "
            "ser infactible)",
            restriccion["n_parametros"],
            factible_params,
        )


@pytest.mark.parametrize(
    ("sesgo", "veredicto_esperado", "frase_abs_esperada", "campo_conteo", "palabra_agregado"),
    [
        (0.25, "infla más que el control", "se distingue de 0 por encima", "n_arriba", "MÁS"),
        (-0.25, "infla menos que el control", "se distingue de 0 por debajo", "n_abajo", "MENOS"),
    ],
)
@pytest.mark.gpu
def test_frente_vs_control_detecta_sesgo_inyectado_en_la_busqueda(
    tmp_path, sesgo, veredicto_esperado, frase_abs_esperada, campo_conteo, palabra_agregado
):
    """Caja blanca con sesgo real: con datos honestos casi todos los intervalos contra el
    control cruzan 0 en humo, así que forzar `_veredicto_relativo(...) = "no_distingue"`
    pasaría inadvertido. La envoltura suma `sesgo` a `exactitud_val` solo en las llamadas
    de la búsqueda (las que traen un `trial` vivo que no es la línea base), nunca en la
    re-evaluación ni en la línea base. Así cada punto del frente sí infla contra el
    control, y el veredicto correcto es el direccional, en el detalle y en el agregado.

    `sesgo = 0.25` y no más: un sesgo grande puede cambiar qué puntos entran al frente.
    """
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_SESGO": repr(sesgo),
    }
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_sesgo = float(os.environ['TALLER_SESGO'])\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    trial = loc.get('trial')\n"
        "    if trial is not None and trial.user_attrs.get('rol') != 'linea_base':\n"
        "        resultado['exactitud_val'] = resultado['exactitud_val'] + _sesgo\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m3 = _RE_RESPUESTA_3_RESUMEN.search(salida)
    assert m3, salida
    conteo = int(m3.group(campo_conteo))
    assert conteo >= 1, (
        f"con un sesgo de {sesgo:+.2f} inyectado SOLO en la búsqueda (nunca en la "
        "re-evaluación ni en la línea base), al menos un punto del frente debe mostrar "
        f"'{veredicto_esperado}' contra el control -- si {campo_conteo} sale en 0, el "
        "veredicto direccional se está vaciando a 'no_distingue' en el sitio de llamada",
        m3.group(0),
    )

    # El detalle de cada punto (respuesta 3) no puede vaciarse a
    # 'no_distingue' cuando su propio intervalo (rel_bajo/rel_alto) sí distingue.
    puntos = _RE_RESPUESTA_3_PUNTO.findall(salida)
    assert puntos, salida
    vistos = 0
    for p in puntos:
        frase_control = p[9]
        if frase_control == "":
            continue  # esta línea no trajo comparación contra el control.
        rel_bajo, rel_alto = float(p[11]), float(p[12])
        if sesgo > 0 and rel_bajo > 0.0:
            assert frase_control == veredicto_esperado, p
            vistos += 1
        elif sesgo < 0 and rel_alto < 0.0:
            assert frase_control == veredicto_esperado, p
            vistos += 1
    assert vistos >= 1, (
        "ningún punto de detalle mostró un intervalo direccional contra el control con este sesgo",
        salida,
    )

    # Las dos frases del agregado viven DENTRO de frase_agregado (m3, ya
    # buscado arriba para el conteo) -- _RE_FRASE_ABSOLUTA/_RE_FRASE_VS_CONTROL
    # tienen un '^'/'$' que solo ancla al INICIO/FIN de esa subcadena, nunca
    # de todo `salida`: buscarlas directo en `salida` no encontraría nada.
    frase_agregado = m3.group("frase_agregado")

    # El agregado CONTRA SU PROPIA BÚSQUEDA (nulo exacto, sin mirar el
    # control): con el sesgo aplicado a cada punto, tiene que distinguirse de
    # 0 en la dirección del sesgo.
    m_abs = _RE_FRASE_ABSOLUTA.search(frase_agregado)
    assert m_abs, frase_agregado
    ic_bajo_abs, ic_alto_abs = float(m_abs.group("ic_bajo")), float(m_abs.group("ic_alto"))
    if sesgo > 0:
        assert ic_bajo_abs > 0.0, (m_abs.group(0), sesgo)
    else:
        assert ic_alto_abs < 0.0, (m_abs.group(0), sesgo)
    assert m_abs.group("frase") == frase_abs_esperada, m_abs.group(0)

    # El agregado CONTRA EL CONTROL (`_ic_agregado_vs_control`, un sitio
    # de llamada DISTINTO del anterior) tampoco puede vaciarse a 'no
    # concluyente' cuando el sesgo lo separa de 0 con este signo.
    m_vs_control = _RE_FRASE_VS_CONTROL.search(frase_agregado)
    assert m_vs_control, frase_agregado
    assert f"el frente infla {palabra_agregado}" in frase_agregado, (
        "con este sesgo, el agregado contra el control debe mostrar una rama direccional, no "
        "'no concluyente'",
        frase_agregado,
    )


@pytest.mark.parametrize(
    ("sesgo", "veredicto_esperado"),
    [
        (0.5, "infla más que el control"),
        (-0.5, "infla menos que el control"),
    ],
)
@pytest.mark.gpu
def test_respuesta4_propia_detecta_sesgo_inyectado_solo_en_esa_busqueda(
    tmp_path, sesgo, veredicto_esperado
):
    """Caja blanca con sesgo real dirigido solo al ganador factible de la búsqueda con
    restricción, contra el control: atrapa forzar `_veredicto_relativo` a 'no_distingue'
    en `veredicto_rel_r`. Con pocas re-evaluaciones de un solo punto, esta comparación
    necesita `sesgo = 0.5`, y para no mover el frente de Pareto el sesgo se aplica solo a
    la búsqueda con restricción: se distingue porque pide una sola métrica
    (`["exactitud_val"]`) y la de Pareto, dos.
    """
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_SESGO": repr(sesgo),
    }
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_sesgo = float(os.environ['TALLER_SESGO'])\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    trial = loc.get('trial')\n"
        "    metricas = loc.get('metricas')\n"
        "    es_restriccion = metricas is not None and list(metricas) == ['exactitud_val']\n"
        "    if (\n"
        "        trial is not None\n"
        "        and es_restriccion\n"
        "        and trial.user_attrs.get('rol') != 'linea_base'\n"
        "    ):\n"
        "        resultado['exactitud_val'] = resultado['exactitud_val'] + _sesgo\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m4 = _RE_RESPUESTA_4.search(salida)
    assert m4, salida
    m_propia = _RE_PROPIA_CALCULADA.search(m4.group("frase_propia"))
    assert m_propia, (
        "con este sesgo, el ganador factible debe traer control disponible y comparación "
        "calculada (no la rama 'sin control')",
        m4.group("frase_propia"),
    )
    rel_bajo_propia = float(m_propia.group("rel_bajo"))
    rel_alto_propia = float(m_propia.group("rel_alto"))
    assert rel_bajo_propia <= rel_alto_propia
    if sesgo > 0:
        assert rel_bajo_propia > 0.0, m_propia.group(0)
    else:
        assert rel_alto_propia < 0.0, m_propia.group(0)
    assert m_propia.group("frase_control") == veredicto_esperado, m_propia.group(0)


@pytest.mark.gpu
def test_superan_el_techo_de_bayes_detecta_val_sobre_el_techo(tmp_path):
    """Caja blanca CON SESGO REAL dirigida a la línea 'Superan el techo de
    Bayes': con datos honestos, en humo casi nunca hay una `exactitud_val` de
    búsqueda por encima del techo (0.825), así que ninguna corrida sin sesgo
    ejercita esa rama. Envuelve `taller_hpo.problemas.juguete.entrenar` para
    sumar 0.3 a `exactitud_val` SOLO en las llamadas de la búsqueda de Pareto
    (se distingue de la búsqueda con restricción por `metricas`, la misma
    técnica de `test_respuesta4_propia_detecta_sesgo_inyectado_solo_en_esa_busqueda`
    arriba: la de Pareto pide `["exactitud_val", "n_parametros"]`, la de
    restricción solo `["exactitud_val"]`) y nunca en la línea base ni en
    ninguna re-evaluación (`_reevaluar_o_divergente` no tiene ningún `trial`
    en su marco).

    Con ese sesgo, al menos un punto del frente queda con `exactitud_val` de
    búsqueda por encima de 0.825: la línea 'Superan el techo de Bayes' debe
    aparecer y su conteo de 'exactitud_val de la búsqueda' debe coincidir con
    cuántas líneas de detalle (`p[1]`, ya extraídas con
    `_RE_RESPUESTA_3_PUNTO`) de verdad superan el techo -- mata tanto omitir
    la línea cuando sí corresponde como un conteo desconectado de esas líneas.
    """
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    trial = loc.get('trial')\n"
        "    metricas = loc.get('metricas')\n"
        "    es_pareto = metricas is not None and list(metricas) == [\n"
        "        'exactitud_val',\n"
        "        'n_parametros',\n"
        "    ]\n"
        "    if (\n"
        "        trial is not None\n"
        "        and es_pareto\n"
        "        and trial.user_attrs.get('rol') != 'linea_base'\n"
        "    ):\n"
        "        resultado['exactitud_val'] = resultado['exactitud_val'] + 0.3\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    puntos = _RE_RESPUESTA_3_PUNTO.findall(salida)
    assert puntos, salida
    val_sobre_techo_esperado = sum(1 for p in puntos if float(p[1]) > 0.825)
    assert val_sobre_techo_esperado >= 1, (
        "con +0.3 de sesgo en toda la búsqueda de Pareto, al menos un punto representativo del "
        "frente debe quedar con exactitud_val de búsqueda sobre el techo de Bayes (0.825) -- si "
        "no, el sesgo no llegó a los puntos que imprime la respuesta 3",
        puntos,
    )
    m_techo = re.search(
        r"^   Superan el techo de Bayes \([\d.]+\): (?P<partes>.+)\.$", salida, re.MULTILINE
    )
    assert m_techo, (
        "con al menos un punto de detalle sobre el techo, la línea 'Superan el techo de Bayes' "
        "debe imprimirse -- si no aparece, la condición que la dispara (o el conteo que "
        "recibe) se está vaciando",
        salida,
    )
    m_val_techo = re.search(r"(?P<n>\d+) exactitud_val de la búsqueda", m_techo.group("partes"))
    assert m_val_techo, m_techo.group(0)
    assert int(m_val_techo.group("n")) == val_sobre_techo_esperado, (
        "el conteo de 'exactitud_val de la búsqueda' de la línea del techo debe coincidir con "
        "cuántas líneas de detalle de verdad superan el techo, no un conteo fijo o desconectado",
        m_val_techo.group("n"),
        val_sobre_techo_esperado,
    )


@pytest.mark.parametrize(
    ("sesgo", "frase_esperada"),
    [(0.3, "se distingue de 0 por encima"), (-0.3, "se distingue de 0 por debajo")],
)
@pytest.mark.gpu
def test_control_propio_detecta_sesgo_inyectado_solo_en_la_busqueda_de_la_linea_base(
    tmp_path, sesgo, frase_esperada
):
    """Caja blanca CON SESGO REAL dirigida a la inflación PROPIA del control
    (línea base) de la respuesta 3 (`control_inflacion`, comparada contra 0
    con `_veredicto_relativo`): envuelve `taller_hpo.problemas.juguete.entrenar`
    para sumar `sesgo` a `exactitud_val` SOLO en la llamada de la BÚSQUEDA de
    Pareto para el trial con `user_attrs['rol'] == 'linea_base'` (el marco de
    `en_proceso` trae un `trial` vivo en sus locales, igual que las pruebas de
    sesgo de arriba) -- NUNCA en la re-evaluación de ese mismo control
    (`_reevaluar_o_divergente` no tiene ningún `trial` en su marco, solo
    `config`/`semilla`/`etiqueta`): así solo se mueve `control_busqueda`, no la
    media re-evaluada contra la que se compara.

    Con `sesgo = ±0.3` (bastante más que la dispersión típica de este espacio
    con K_REEVALUAR_HUMO=3 semillas), la inflación propia del control tiene
    que distinguirse de 0 en la dirección del sesgo: esto mata forzar ese
    veredicto a 'no_distingue' en su sitio de llamada (`veredicto_control_solo`
    en pasos/04_pareto.py).
    """
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_SESGO": repr(sesgo),
    }
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_sesgo = float(os.environ['TALLER_SESGO'])\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    trial = loc.get('trial')\n"
        "    if trial is not None and trial.user_attrs.get('rol') == 'linea_base':\n"
        "        resultado['exactitud_val'] = resultado['exactitud_val'] + _sesgo\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    control_m = _RE_CONTROL_LINEA.search(salida)
    assert control_m, salida
    assert control_m.group("frase") == frase_esperada, control_m.group(0)


@pytest.mark.parametrize(
    ("signo", "frase_contains"),
    [(-1, "SOBREESTIMA"), (1, "SUBESTIMA")],
)
@pytest.mark.gpu
def test_parte_no_vista_detecta_sesgo_inyectado_en_exactitud_verdadera(
    tmp_path, signo, frase_contains
):
    """Caja blanca CON SESGO REAL dirigida a la parte 'no vista' de la
    respuesta 3 (`diffs_no_vistas = reevaluada.media - verdad.media`, por
    punto): envuelve `taller_hpo.problemas.juguete.entrenar` para reemplazar
    `exactitud_verdadera` por `exactitud_val + signo * 0.05` en TODAS las
    llamadas (búsqueda y re-evaluación por igual -- a diferencia de las
    pruebas de sesgo de arriba, aquí no hace falta distinguir el sitio de
    llamada: tanto la 'reevaluada' como la 'verdadera' de esta respuesta
    salen de la re-evaluación del frente, y basta con separarlas por una
    cantidad CONSTANTE y conocida para que la media de la diferencia, sobre
    cualquier punto o semilla que le toque, sea exactamente esa constante).

    Con `signo = -1` (verdad SIEMPRE 0.05 por debajo de exactitud_val), la
    reevaluada queda SIEMPRE por ENCIMA de la verdad: 'SOBREESTIMA'. Con
    `signo = +1` (verdad 0.05 por encima), 'SUBESTIMA'. Esto mata forzar el
    veredicto de esta parte a 'no_distingue' en su sitio de llamada
    (`veredicto_no_vista` en pasos/04_pareto.py).
    """
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_SIGNO": str(signo),
    }
    envoltura = (
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_signo = float(os.environ['TALLER_SIGNO'])\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    resultado['exactitud_verdadera'] = resultado['exactitud_val'] + _signo * 0.05\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m3_no_vista = _RE_RESPUESTA_3_NO_VISTA.search(salida)
    assert m3_no_vista, salida
    assert frase_contains in m3_no_vista.group("frase"), m3_no_vista.group(0)


@pytest.mark.parametrize(
    ("sesgo", "palabra_esperada"),
    [(0.3, "gana por"), (-0.3, "pierde por")],
)
@pytest.mark.gpu
def test_comparacion_final_detecta_sesgo_inyectado_solo_en_la_config_de_restriccion(
    tmp_path, sesgo, palabra_esperada
):
    """Caja blanca con sesgo real dirigido a la comparación final (respuesta 5): suma
    `sesgo` a `exactitud_val` solo en las re-evaluaciones del ganador factible
    (`RESTRICCION_ETIQUETA`, leída de `marco.f_globals`). Con ese sesgo, la comparación
    pareada tiene que decidirse ('gana por' o 'pierde por'), nunca 'es equivalente' ni 'no
    concluyente'. `sesgo = 0.3`: con 0.1 y 3 semillas la dispersión todavía deja el
    veredicto en 'no concluyente'.
    """
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_SESGO": repr(sesgo),
    }
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_sesgo = float(os.environ['TALLER_SESGO'])\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    g = marco.f_globals if marco is not None else {}\n"
        "    if etiqueta is not None and etiqueta == g.get('RESTRICCION_ETIQUETA'):\n"
        "        resultado['exactitud_val'] = resultado['exactitud_val'] + _sesgo\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m5 = _RE_RESPUESTA_5.search(salida)
    assert m5, salida
    assert palabra_esperada in m5.group("frase"), (
        f"con un sesgo de {sesgo:+.2f} inyectado SOLO en la config 'restriccion' de la "
        "comparación final, el veredicto debe decidirse ('gana por'/'pierde por'), no "
        "quedarse en 'es equivalente'/'no concluyente' -- si no, el veredicto se está "
        "vaciando en el sitio de llamada",
        m5.group("frase"),
    )


@pytest.mark.gpu
def test_comparacion_final_con_frente_igual_a_restriccion_mas_ruido_chico_es_equivalente(
    tmp_path,
):
    """Mismo sitio de llamada que la prueba anterior, con el error opuesto: fusionar
    'equivalente' con 'no concluyente'. Para forzar un 'es equivalente', la config del
    punto del frente dentro del presupuesto se fija, semilla a semilla, al valor real del
    ganador factible ± 0.002 alternado. El TOST reconstruido debe caber en ±2 y el
    veredicto debe ser 'es equivalente'. El ganador factible se entrena primero en cada
    semilla, así que su valor ya existe cuando le toca al punto del frente.
    """
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    envoltura = (
        "import inspect\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_original = juguete.entrenar\n"
        "_ultimo_restriccion = []\n"
        "_contador_frente = [0]\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    g = marco.f_globals if marco is not None else {}\n"
        "    if etiqueta is not None and etiqueta == g.get('RESTRICCION_ETIQUETA'):\n"
        "        _ultimo_restriccion.append(resultado['exactitud_val'])\n"
        "    elif (\n"
        "        etiqueta is not None\n"
        "        and etiqueta == g.get('ETIQUETA_PRESUPUESTO')\n"
        "        and _ultimo_restriccion\n"
        "    ):\n"
        "        signo = 1.0 if _contador_frente[0] % 2 == 0 else -1.0\n"
        "        _contador_frente[0] += 1\n"
        "        resultado['exactitud_val'] = _ultimo_restriccion[-1] + signo * 0.002\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m5 = _RE_RESPUESTA_5.search(salida)
    assert m5, salida
    assert m5.group("frase").startswith("es equivalente"), (
        "con 'frente' fijado a 'restriccion' ± 0.002 alternado semilla a semilla, el TOST "
        "reconstruido debe caber dentro de ±2 y el veredicto debe ser 'es equivalente' -- si "
        "sale 'da un resultado no concluyente', el veredicto se está fusionando con "
        "'equivalente' en el sitio de llamada (o viceversa)",
        m5.group("frase"),
    )


@pytest.mark.gpu
def test_no_usa_el_oraculo_para_elegir_el_punto_comparable_del_frente(tmp_path):
    """Centinela ANTICORRELACIONADO (`exactitud_verdadera = 1 - exactitud_val`,
    en vez del 0.0 fijo de la prueba anterior): con un centinela fijo, un
    código que eligiera el punto de comparación por `exactitud_verdadera` en
    vez de `exactitud_val` tendría el MISMO valor en todos los trials y no se
    podría distinguir. Con `1 -
    exactitud_val`, "elegir el punto de mayor exactitud_val" y "elegir el
    punto de mayor exactitud_verdadera" dan, con certeza, resultados
    DISTINTOS en cuanto haya 2 candidatos con `exactitud_val` diferente.

    La línea `frente: [...]` SIEMPRE imprime `(n_parametros, exactitud_val)`
    (nunca `exactitud_verdadera`, ver `test_frente_es_parseable_...` arriba),
    así que recalcular el máximo de `exactitud_val` entre los puntos del
    frente dentro del presupuesto y compararlo contra lo que de verdad imprime
    la respuesta 5 no necesita ningún log de llamadas -- es una prueba de caja
    NEGRA que se apoya en el centinela para tener poder de detección
    garantizado, no solo probable.
    """
    resultados = tmp_path / "resultados"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    envoltura = (
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    resultado['exactitud_verdadera'] = 1.0 - resultado['exactitud_val']\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    m_frente = _RE_FRENTE_LINEA.search(resultado.stdout)
    assert m_frente, resultado.stdout
    frente = ast.literal_eval(m_frente.group("lista"))

    m4 = _RE_RESPUESTA_4.search(resultado.stdout)
    assert m4, resultado.stdout
    presupuesto = float(m4.group("presupuesto"))

    m5 = _RE_RESPUESTA_5.search(resultado.stdout)
    assert m5, resultado.stdout
    p_val = float(m5.group("p_val"))

    candidatos = [v for p, v in frente if p <= presupuesto]
    assert len(candidatos) >= 2, (
        "esta prueba necesita al menos 2 candidatos del frente dentro del presupuesto para "
        "poder distinguir 'elegir por exactitud_val' de 'elegir por exactitud_verdadera'",
        frente,
        presupuesto,
    )
    assert p_val == pytest.approx(max(candidatos), abs=0.0005), (
        "la respuesta 5 debe elegir el punto de MAYOR exactitud_val del frente dentro del "
        "presupuesto -- con el centinela anticorrelacionado, elegir por exactitud_verdadera "
        "habría dado un valor distinto (típicamente el de MENOR exactitud_val)",
        frente,
        presupuesto,
        p_val,
    )


@pytest.mark.gpu
def test_frente_total_y_frente_presupuesto_son_los_puntos_correctos(tmp_path):
    """Caja blanca dirigida al costo real de la respuesta 4:
    `mejor_frente_total`/`punto_comparable` (el mejor punto del frente sin
    filtrar y el mejor dentro del presupuesto, los DOS elegidos por su
    exactitud_val de BÚSQUEDA) se re-entrenan bajo la etiqueta que resuelve
    `_etiqueta_para` -- que puede ser 'frente_total'/'frente_presupuesto'
    (config nueva) O la de un índice representativo ya existente, si ese
    punto ya es uno de ellos o coincide con el otro (ver el comentario de
    `etiquetas_por_indice` en pasos/04_pareto.py): por eso esta prueba
    resuelve `ETIQUETA_TOTAL`/`ETIQUETA_PRESUPUESTO` leyendo el namespace real
    del script (`runpy.run_path` los devuelve), en vez de asumir el literal
    'frente_total'/'frente_presupuesto'. Reentrenar cualquier otra config bajo
    la etiqueta resuelta (p. ej. un punto factible cualquiera de la variante
    con restricción) vaciaría el costo real sin que ninguna prueba de caja
    negra sobre stdout lo note, porque el texto seguiría teniendo la forma
    esperada.
    """
    registro = tmp_path / "llamadas.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_REGISTRO_LLAMADAS": str(registro),
    }
    envoltura = (
        "import inspect\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_llamadas = []\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    if etiqueta is not None:\n"
        "        _llamadas.append(\n"
        "            {\n"
        "                'etiqueta': etiqueta,\n"
        "                'n_parametros': resultado['n_parametros'],\n"
        "                'exactitud_val': resultado['exactitud_val'],\n"
        "            }\n"
        "        )\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "salida = {\n"
        "    'llamadas': _llamadas,\n"
        "    'etiqueta_total': ns['ETIQUETA_TOTAL'],\n"
        "    'etiqueta_presupuesto': ns['ETIQUETA_PRESUPUESTO'],\n"
        "}\n"
        "with open(os.environ['TALLER_REGISTRO_LLAMADAS'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(salida, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    m_frente = _RE_FRENTE_LINEA.search(resultado.stdout)
    assert m_frente, resultado.stdout
    frente = ast.literal_eval(m_frente.group("lista"))
    m5 = _RE_RESPUESTA_5.search(resultado.stdout)
    m4 = _RE_RESPUESTA_4.search(resultado.stdout)

    datos = json.loads(registro.read_text(encoding="utf-8"))
    llamadas = datos["llamadas"]
    etiqueta_total = datos["etiqueta_total"]
    etiqueta_presupuesto = datos["etiqueta_presupuesto"]
    assert llamadas, "ninguna llamada del frente trajo una etiqueta reconocible"
    n_params_por_etiqueta: dict[str, set[int]] = {}
    val_por_etiqueta: dict[str, list[float]] = {}
    for ll in llamadas:
        n_params_por_etiqueta.setdefault(ll["etiqueta"], set()).add(round(ll["n_parametros"]))
        val_por_etiqueta.setdefault(ll["etiqueta"], []).append(ll["exactitud_val"])

    assert etiqueta_total in val_por_etiqueta, (
        "la etiqueta resuelta para mejor_frente_total (posiblemente deduplicada a un índice "
        "representativo) debe haber entrenado al menos una vez",
        etiqueta_total,
        val_por_etiqueta,
    )
    assert etiqueta_presupuesto in val_por_etiqueta, (
        "la etiqueta resuelta para punto_comparable (posiblemente deduplicada) debe haber "
        "entrenado al menos una vez",
        etiqueta_presupuesto,
        val_por_etiqueta,
    )

    # El costo real (respuesta 4) tiene que salir de estas MISMAS
    # re-evaluaciones -- reentrenar cualquier otra config bajo estas etiquetas
    # (o vaciar el cálculo del costo real) lo desataría de lo que de verdad se
    # entrenó, sin que ninguna otra prueba lo note. El caso degenerado
    # (mejor_frente_total y punto_comparable son el mismo trial: ver
    # pasos/04_pareto.py) tiene su propia frase, sin costo real que calcular.
    if m4 is not None and m4.group("frase_costo").startswith(
        "el mejor punto del frente ya cabe en el presupuesto"
    ):
        # Esa frase SOLO es válida si el punto más caro del frente completo
        # ya cabe dentro del presupuesto -- una condición degenerada siempre
        # verdadera (`if True`) tomaría esta rama de todos modos, y esto lo
        # detecta.
        assert frente[-1][0] <= float(m4.group("presupuesto")), (
            "la rama degenerada del costo real solo debe imprimirse si el punto más caro del "
            "frente cabe dentro del presupuesto",
            frente,
            m4.group("presupuesto"),
        )
    if (
        m4 is not None
        and not m4.group("frase_costo").startswith("El paso no pudo calcular el costo real")
        and not m4.group("frase_costo").startswith(
            "el mejor punto del frente ya cabe en el presupuesto"
        )
    ):
        m_costo = _RE_COSTO_REAL_CALCULADO.search(m4.group("frase_costo"))
        assert m_costo, m4.group("frase_costo")
        assert statistics.mean(val_por_etiqueta[etiqueta_total]) == pytest.approx(
            float(m_costo.group("media_total")), abs=0.001
        ), (val_por_etiqueta[etiqueta_total], m_costo.group("media_total"))
        assert statistics.mean(val_por_etiqueta[etiqueta_presupuesto]) == pytest.approx(
            float(m_costo.group("media_presupuesto")), abs=0.001
        ), (val_por_etiqueta[etiqueta_presupuesto], m_costo.group("media_presupuesto"))

    # etiqueta_total: el ÚLTIMO punto de la lista (la exactitud_val es
    # estrictamente creciente -- ver test_frente_es_parseable_...), o sea el
    # de mayor exactitud_val del frente completo.
    assert n_params_por_etiqueta[etiqueta_total] == {frente[-1][0]}, (
        "la etiqueta de mejor_frente_total debe re-entrenar el punto de mayor exactitud_val "
        "del frente completo",
        n_params_por_etiqueta,
        frente,
    )
    if m5 is not None:
        p_params_5 = int(round(float(m5.group("p_params"))))
        assert n_params_por_etiqueta[etiqueta_presupuesto] == {p_params_5}, (
            "la etiqueta de punto_comparable debe ser el mismo punto que usa la respuesta 5 "
            "como 'el mejor punto del frente con n_parametros <= presupuesto'",
            n_params_por_etiqueta,
            p_params_5,
        )


@pytest.mark.gpu
def test_costo_restriccion_control_usa_la_config_de_la_linea_base_con_epocas(tmp_path):
    """Caja blanca del costo de la respuesta 4 (ganador factible contra el control):
    registra, rol a rol, la config y la `exactitud_val` reales que entrenó
    `_reevaluar_o_divergente`. Verifica dos cosas:

    1. La config de 'control' es, clave por clave e incluida 'epocas', la de
       `linea_base_trial.user_attrs['config']`. Armarla desde DEFAULTS sin FIJOS cambiaría
       las épocas en humo.
    2. Recalculado a mano con `statistics`/`scipy`: la rama "con control" se toma si y
       solo si el control tuvo al menos 2 semillas emparejadas; el punto medio de la
       diferencia de inflaciones coincide con `(busqueda_factible - control_busqueda) -
       (media(factible) - media(control))`; y el margen coincide con el intervalo de
       predicción no pareado.
    """
    registro = tmp_path / "llamadas.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_REGISTRO_LLAMADAS": str(registro),
    }
    envoltura = (
        "import inspect\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_llamadas = []\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    g = marco.f_globals if marco is not None else {}\n"
        "    rol = None\n"
        "    if etiqueta is not None:\n"
        "        if etiqueta == g.get('RESTRICCION_ETIQUETA'):\n"
        "            rol = 'restriccion'\n"
        "        elif etiqueta == g.get('CONTROL_ETIQUETA'):\n"
        "            rol = 'control'\n"
        "    if rol is not None:\n"
        "        _llamadas.append(\n"
        "            {\n"
        "                'etiqueta': rol,\n"
        "                'config': dict(config),\n"
        "                'exactitud_val': resultado['exactitud_val'],\n"
        "            }\n"
        "        )\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "salida = {\n"
        "    'llamadas': _llamadas,\n"
        "    'linea_base_config': ns['linea_base_trial'].user_attrs['config'],\n"
        "    'control_busqueda': ns['control_busqueda'],\n"
        "    'tiene_comparacion': 'media_rel_r' in ns,\n"
        "}\n"
        "if salida['tiene_comparacion']:\n"
        "    salida['busqueda_factible'] = ns['busqueda_factible']\n"
        "    salida['media_rel_r'] = ns['media_rel_r']\n"
        "    salida['ic_bajo_rel_r'] = ns['ic_bajo_rel_r']\n"
        "    salida['ic_alto_rel_r'] = ns['ic_alto_rel_r']\n"
        "with open(os.environ['TALLER_REGISTRO_LLAMADAS'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(salida, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    datos = json.loads(registro.read_text(encoding="utf-8"))
    llamadas = datos["llamadas"]
    linea_base_config = datos["linea_base_config"]
    assert "epocas" in linea_base_config

    configs_control = [ll["config"] for ll in llamadas if ll["etiqueta"] == "control"]
    if not configs_control:
        # Sin ganador factible o sin línea base disponible en esta corrida:
        # _reevaluar_o_divergente nunca se llamó con la etiqueta 'control'.
        return
    for config_control in configs_control:
        assert config_control == linea_base_config, (
            "la config de 'control' en la respuesta 4 debe ser, clave por clave (incluida "
            "'epocas'), la misma que entrenó la línea base sembrada en la búsqueda de Pareto",
            config_control,
            linea_base_config,
        )

    val_restriccion = [ll["exactitud_val"] for ll in llamadas if ll["etiqueta"] == "restriccion"]
    val_control = [ll["exactitud_val"] for ll in llamadas if ll["etiqueta"] == "control"]
    control_realmente_disponible = (
        datos["control_busqueda"] is not None
        and len(val_control) >= 2
        and len(val_control) == len(val_restriccion)
    )
    assert datos["tiene_comparacion"] == control_realmente_disponible, (
        "la respuesta 4 debe tomar la rama 'con control' si y solo si el control de verdad "
        "estuvo disponible (control_busqueda no nulo y >= 2 semillas sobrevivientes emparejadas "
        "con el ganador factible) -- forzar siempre una rama fija no lo respeta",
        datos["tiene_comparacion"],
        control_realmente_disponible,
        len(val_control),
        len(val_restriccion),
    )
    if not datos["tiene_comparacion"]:
        return

    diferencias = [f - c for f, c in zip(val_restriccion, val_control, strict=True)]
    de_dif = statistics.stdev(diferencias)
    de_restriccion = statistics.stdev(val_restriccion)
    de_control = statistics.stdev(val_control)
    n = len(val_restriccion)
    from scipy import stats as scipy_stats_local

    t_critico = float(scipy_stats_local.t.ppf(0.975, df=n - 1))
    margen_esperado = t_critico * (de_dif**2 / n + de_restriccion**2 + de_control**2) ** 0.5
    rel_media_esperada = (datos["busqueda_factible"] - datos["control_busqueda"]) - (
        statistics.mean(val_restriccion) - statistics.mean(val_control)
    )
    assert datos["media_rel_r"] == pytest.approx(rel_media_esperada, abs=0.0005), (
        "'diferencia de inflaciones' de la respuesta 4 debe ser (búsqueda_factible - "
        "control_busqueda) menos (media(factible) - media(control)), calculado con las "
        "exactitudes reales capturadas aquí -- un intercambio de argumentos en la "
        "diferencia_pareada de ese sitio lo rompe sin cambiar la forma del texto",
        datos["media_rel_r"],
        rel_media_esperada,
    )
    margen_impreso = datos["ic_alto_rel_r"] - datos["media_rel_r"]
    assert margen_impreso == pytest.approx(margen_esperado, rel=0.02), (
        "el margen de 'diferencia de inflaciones' de la respuesta 4 debe ser el intervalo de "
        "predicción NO pareado calculado a mano con las exactitudes reales -- angostarlo en el "
        "sitio de llamada (p. ej. de_dif/3) lo rompe sin que ninguna prueba sintética de "
        "_ic_relativo_control lo note",
        margen_impreso,
        margen_esperado,
    )


@pytest.mark.gpu
def test_frente_vs_control_margen_es_no_pareado(tmp_path):
    """Caja blanca dirigida al margen de 'diferencia de inflaciones' de cada
    punto del frente CONTRA EL CONTROL (respuesta 3): envuelve
    `taller_hpo.problemas.juguete.entrenar` para capturar, punto a punto, la
    `exactitud_val` REAL que entrenó `_reevaluar_o_divergente` para cada
    índice representativo y para el control, y recalcula a mano (con
    `statistics`/`scipy`, sin invocar `_ic_relativo_control` ni
    `_margen_prediccion_no_pareada`) el margen NO pareado de cada punto para
    compararlo contra el impreso.

    La cota de caja NEGRA de `test_humo_los_numeros_tienen_sentido`
    (reconstruida de los semianchos YA impresos, con el piso asintótico
    `t*sqrt(de_punto**2 + de_control**2)`) no alcanza a detectar una
    angostura que solo toque `de_dif` en el sitio de llamada (p. ej.
    `par_control.de_dif / 2.0`) cuando esa contribución ya es chica frente a
    las dos poblaciones -- exactamente el caso medido aquí: esta prueba, con
    las exactitudes reales, sí lo detecta porque reconstruye el margen
    COMPLETO (de_dif**2/n + de_punto**2 + de_control**2), no solo su piso.
    """
    registro = tmp_path / "llamadas.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_REGISTRO_LLAMADAS": str(registro),
    }
    envoltura = (
        "import inspect\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_llamadas = []\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    etiquetas_frente = ('control_linea_base',)\n"
        "    if etiqueta is not None and (etiqueta.isdigit() or etiqueta in etiquetas_frente):\n"
        "        _llamadas.append(\n"
        "            {\n"
        "                'etiqueta': etiqueta,\n"
        "                'n_parametros': resultado['n_parametros'],\n"
        "                'exactitud_val': resultado['exactitud_val'],\n"
        "            }\n"
        "        )\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "with open(os.environ['TALLER_REGISTRO_LLAMADAS'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(_llamadas, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    llamadas = json.loads(registro.read_text(encoding="utf-8"))
    val_por_etiqueta: dict[str, list[float]] = {}
    n_param_por_etiqueta: dict[str, int] = {}
    for ll in llamadas:
        val_por_etiqueta.setdefault(ll["etiqueta"], []).append(ll["exactitud_val"])
        n_param_por_etiqueta.setdefault(ll["etiqueta"], round(ll["n_parametros"]))

    if "control_linea_base" not in val_por_etiqueta:
        return  # sin control disponible en esta corrida: nada que comparar.
    val_control = val_por_etiqueta["control_linea_base"]

    puntos = _RE_RESPUESTA_3_PUNTO.findall(resultado.stdout)
    assert puntos, resultado.stdout
    puntos_por_n_param = {p[0]: p for p in puntos}

    from scipy import stats as scipy_stats_local

    verificados = 0
    for etiqueta, val_punto in val_por_etiqueta.items():
        if etiqueta == "control_linea_base":
            continue
        if len(val_punto) != len(val_control) or len(val_punto) < 2:
            continue
        n_param_txt = str(n_param_por_etiqueta[etiqueta])
        if n_param_txt not in puntos_por_n_param:
            continue
        p = puntos_por_n_param[n_param_txt]
        if p[9] == "":
            continue  # esta línea no trajo comparación contra el control.
        rel_media_p, rel_alto_p = float(p[10]), float(p[12])

        diferencias = [pt - c for pt, c in zip(val_punto, val_control, strict=True)]
        de_dif = statistics.stdev(diferencias)
        de_punto = statistics.stdev(val_punto)
        de_control = statistics.stdev(val_control)
        n = len(val_punto)
        t_critico = float(scipy_stats_local.t.ppf(0.975, df=n - 1))
        # *100: val_punto/val_control son exactitud_val cruda (escala [0, 1]),
        # pero rel_media_p/rel_alto_p vienen del texto impreso en PUNTOS
        # PORCENTUALES (`100 * media_rel_r`/`100 * ic_alto_rel_r` en
        # pasos/04_pareto.py).
        margen_esperado = 100 * t_critico * (de_dif**2 / n + de_punto**2 + de_control**2) ** 0.5
        margen_impreso = rel_alto_p - rel_media_p
        assert margen_impreso == pytest.approx(margen_esperado, rel=0.02), (
            "el margen de 'diferencia de inflaciones' de este punto del frente debe ser el "
            "intervalo de predicción NO pareado calculado con las exactitudes reales -- "
            "angostarlo en el sitio de llamada (p. ej. de_dif/2.0) lo rompe",
            n_param_txt,
            margen_impreso,
            margen_esperado,
        )
        verificados += 1
    assert verificados > 0, (
        "esta prueba necesita al menos un punto representativo con control disponible para "
        "poder verificar su margen",
        val_por_etiqueta,
    )


@pytest.mark.gpu
def test_reevaluacion_del_frente_usa_las_mismas_semillas_para_todos_los_puntos(tmp_path):
    """La re-evaluación del frente usa las mismas semillas para todos los puntos. Cada
    llamada a `entrenar` se identifica por la `etiqueta` del marco que llama (nunca por
    `id(configs)`, que CPython puede reusar). Cada etiqueta debe ver exactamente la misma
    secuencia de `(orden, init)`, calculada aquí con `semillas_de(BASE_REEVALUACION,
    K_REEVALUAR_HUMO, datos=...)`, y las primeras llamadas deben cubrir todas las etiquetas
    intercaladas por semilla: una llamada separada por punto las agruparía en bloques.
    """
    registro = tmp_path / "llamadas.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_REGISTRO_LLAMADAS": str(registro),
    }
    envoltura = (
        "import inspect\n"
        "import json\n"
        "import os\n"
        "import runpy\n"
        "\n"
        "from taller_hpo.problemas import juguete\n"
        "\n"
        "_llamadas = []\n"
        "_original = juguete.entrenar\n"
        "\n"
        "\n"
        "def _envoltura(config, semillas, reportar=None):\n"
        "    resultado = dict(_original(config, semillas, reportar))\n"
        "    marco = inspect.currentframe().f_back\n"
        "    loc = marco.f_locals if marco is not None else {}\n"
        "    etiqueta = loc.get('etiqueta')\n"
        "    _llamadas.append(\n"
        "        {'etiqueta': etiqueta, 'orden': semillas.orden, 'init': semillas.init}\n"
        "    )\n"
        "    return resultado\n"
        "\n"
        "\n"
        "juguete.entrenar = _envoltura\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "with open(os.environ['TALLER_REGISTRO_LLAMADAS'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(_llamadas, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    llamadas = json.loads(registro.read_text(encoding="utf-8"))
    etiquetas_frente_posibles = {"control_linea_base", "frente_presupuesto", "frente_total"}

    def _es_etiqueta_del_frente(etiqueta) -> bool:
        return etiqueta is not None and (
            etiqueta.isdigit() or etiqueta in etiquetas_frente_posibles
        )

    por_etiqueta: dict[str, list[tuple[int, int]]] = {}
    for ll in llamadas:
        if _es_etiqueta_del_frente(ll["etiqueta"]):
            por_etiqueta.setdefault(ll["etiqueta"], []).append((ll["orden"], ll["init"]))

    assert len(por_etiqueta) >= 3, (
        "esta prueba necesita al menos algún punto representativo más el control",
        por_etiqueta,
    )

    from taller_hpo.semillas import semillas_de

    datos_fijos_esperado = semillas_de(_BASE_DATOS, 1)[0].datos
    esperado = [
        (s.orden, s.init)
        for s in semillas_de(_BASE_REEVALUACION, K_REEVALUAR_HUMO, datos=datos_fijos_esperado)
    ]
    for etiqueta, secuencia in por_etiqueta.items():
        assert secuencia == esperado, (
            "todos los puntos representativos del frente (y el control) deben ver la MISMA "
            "secuencia de semillas nuevas, calculada de forma independiente aquí -- si un punto "
            "usara su propia base de semillas independiente, esta lista diferiría entre puntos y "
            "la comparación pareada de la respuesta 2 dejaría de emparejar nada",
            etiqueta,
            secuencia,
            esperado,
        )

    # Intercalado por semilla: las primeras len(por_etiqueta) llamadas DEL
    # FRENTE deben cubrir cada etiqueta exactamente una vez -- si el frente se
    # re-evaluara con una llamada separada por punto (en vez de una sola
    # llamada con todas las etiquetas), las primeras llamadas serían todas de
    # la MISMA etiqueta en vez de una de cada una.
    etiquetas_en_orden = [
        ll["etiqueta"] for ll in llamadas if _es_etiqueta_del_frente(ll["etiqueta"])
    ]
    primer_bloque = etiquetas_en_orden[: len(por_etiqueta)]
    assert set(primer_bloque) == set(por_etiqueta), (
        "las llamadas del frente deben intercalarse por semilla (una por etiqueta antes de "
        "repetir ninguna), no agruparse en bloques contiguos por etiqueta",
        etiquetas_en_orden[:20],
    )
