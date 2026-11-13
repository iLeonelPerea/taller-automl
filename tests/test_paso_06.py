"""Paso 06: honestidad. El paso corre de punta a punta y sus respuestas dicen lo que miden.

`tests/test_pasos.py` ya cubre el código de salida y el PNG. Aquí se verifica el contenido:
las seis respuestas numeradas aparecen, sus números tienen sentido (por ejemplo, la
"verdad" de las preguntas 2 y 6 es el mismo número, porque viene del mismo trial), reanudar
no cambia las respuestas, y varios números se recalculan de forma independiente. La fixture
`corrida` corre el paso una sola vez con PRESUPUESTO=humo y reparte su salida.
"""

from __future__ import annotations

import ast
import math
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
from scipy import stats as scipy_stats

from taller_hpo import estadistica, estudio, honestidad
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.semillas import semillas_de

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "06_honestidad.py"

FLOAT = r"[+-]?[0-9]+\.[0-9]+"

# Deben coincidir con pasos/06_honestidad.py: mismas semillas y mismo
# PRESUPUESTOS["humo"], para recalcular en proceso lo que el paso ya calculó
# en su propio subproceso y comparar contra lo impreso.
SEMILLA_PLAN = 20_260_601
SEMILLA_TPE = 20_260_602
SEMILLA_REPETICIONES = 20_260_603
N_BUSQUEDA_HUMO = 8
N_EVAL_HUMO = 3
R_HUMO = 2

_TEXTO_BRECHA = (
    r"no concluyente: con estas semillas la brecha no se distingue de cero|"
    r"distinta de cero: esa parte \(la suerte del conjunto de validación\) no la ve "
    r"re-evaluar con los mismos datos"
)
_RE_R1 = re.compile(
    rf"1\. Diferencia búsqueda - re-evaluación: la búsqueda vio ({FLOAT}), la re-evaluación "
    rf"\((\d+) semillas\) da ({FLOAT}) \[({FLOAT}), ({FLOAT})\]: "
    rf"({FLOAT}) puntos \(intervalo de confianza \[({FLOAT}), ({FLOAT})\]\) -- "
    rf"([\w áéíóú]+)\. Control \(línea base, "
    rf"que nadie eligió\): se desvía de su propia re-evaluación ({FLOAT}) puntos "
    rf"\(intervalo \[({FLOAT}), ({FLOAT})\]\) -- ([\w áéíóú]+); la inflación del ganador "
    r"(SÍ se distingue|NO se distingue) de ese control: diferencia de inflaciones "
    rf"({FLOAT}) puntos contra ±({FLOAT}) puntos que da UNA semilla cualquiera sin selección "
    rf"\(intervalo de predicción t, (\d+) grados? de libertad\)\. Revelación del oráculo "
    rf"\(sin usarla "
    rf"para elegir\): en verdad, el ganador re-evaluado promedia ({FLOAT}) "
    rf"\(intervalo \[({FLOAT}), ({FLOAT})\]\); la re-evaluación \(mismos datos\) queda ({FLOAT}) "
    rf"puntos respecto a la verdad \(intervalo \[({FLOAT}), ({FLOAT})\]\) -- ({_TEXTO_BRECHA})\."
)
_RE_R2 = re.compile(
    rf"2\. Contra el oráculo: la validación del ganador durante la búsqueda fue ({FLOAT}), "
    rf"su exactitud verdadera era ({FLOAT}) -- la validación ya mentía ({FLOAT}) "
    r"puntos antes de re-evaluar nada\."
)
_TEXTO_BRECHA_R3 = (
    r"no concluyente: con estas semillas la brecha no se distingue de cero|"
    r"distinta de cero: esa parte \(la suerte del conjunto de validación\) no la ve "
    r"re-evaluar con los mismos datos; para verla del todo hace falta un conjunto nuevo "
    r"\(pregunta 6\)"
)
_RE_R3 = re.compile(
    rf"3\. Re-evaluado, el ganador \(({FLOAT})\) contra la línea base \(({FLOAT})\): "
    rf"diferencia ({FLOAT}) puntos \(intervalo de confianza \[({FLOAT}), ({FLOAT})\]\), "
    r"veredicto '([\w áéíóú]+)' "
    rf"con delta=0\.01 y (\d+) semillas\. En verdad \(revelación del oráculo\), la misma "
    rf"comparación pareada da ({FLOAT}) puntos \(brecha val-verdad: ({FLOAT}) puntos, intervalo "
    rf"\[({FLOAT}), ({FLOAT})\]\) -- ({_TEXTO_BRECHA_R3})\."
)
_RE_R4 = re.compile(
    rf"4\. Con una desviación estándar de las diferencias de ({FLOAT}) \(con solo (\d+) "
    rf"semillas de evaluación, intervalo de confianza "
    rf"(\d+)% de esa desviación "
    rf"\[({FLOAT}), ({FLOAT})\]\), detectar una diferencia del tamaño de delta=0\.01 \(la mínima "
    rf"que importa\) con potencia 80% y alfa 0\.05 exige (\d+) semillas con el punto central "
    rf"\(entre (\d+) y (\d+) según esa misma incertidumbre; hubo (\d+)\); la "
    rf"diferencia observada \(({FLOAT}), (\d+) veces delta\) por sí sola necesitaría (\d+) "
    r"semillas para detectarse con potencia 80%, pero planear con la diferencia observada es "
    r"circular -- ya pasó por la selección y es ruidosa; planea con delta, y con (\d+) "
    r"semillas: (sí, las semillas alcanzaron la potencia planeada|"
    r"no, las semillas no alcanzaron la potencia planeada|"
    r"no concluyente \(el n necesario va de \d+ a \d+\)) \(veredicto '([\w áéíóú]+)'\)\."
)
_RE_R5 = re.compile(
    r"5\. En (\d+) búsquedas completas repetidas \(otras semillas\), la parte de la inflación "
    rf"debida a elegir el máximo promedia ({FLOAT}) puntos \(intervalo de confianza "
    rf"\[({FLOAT}), ({FLOAT})\]\) -- "
    r"([\w= ]+)\. La inflación cruda \(sin aislar el desvío común del conjunto de "
    rf"validación\) promedia ({FLOAT}) puntos \(intervalo \[({FLOAT}), ({FLOAT})\]\)\. Referencia "
    r"teórica \(cota superior si los (\d+) desvíos val-verdad son iid con la misma sigma; "
    r"con la misma verdad sería igualdad; sigma medido en los trials de la búsqueda del "
    rf"ganador\): sigma·E\[max de (\d+) normales\] = ({FLOAT}) \* ({FLOAT}) = ({FLOAT}) "
    rf"\(({FLOAT}) puntos; intervalo (\d+)% de la cota, propagando el intervalo de sigma: "
    # El grupo final se ancla al FINAL DE LÍNEA (re.MULTILINE), no al primer '.'
    # encontrado: la respuesta se imprime como una sola línea física (el
    # print de pasos/06_honestidad.py concatena f-strings sin '\n' de por
    # medio), así que exigir '$' fuerza al no codicioso a expandirse hasta el
    # punto final real aunque el texto de la comparación con la teoría
    # incluya números con decimales (p. ej. "2.97") antes de ese punto.
    rf"\[({FLOAT}), ({FLOAT})\]\)\. ¿Coincide con la teoría\? (.+?)\.$",
    re.MULTILINE,
)
_RE_R6 = re.compile(
    rf"6\. Prueba sellada \(n=5000, abierta una sola vez\): exactitud_prueba=({FLOAT}) "
    rf"\(intervalo de confianza 95% \[({FLOAT}), ({FLOAT})\]\) vs\. verdad=({FLOAT}) "
    rf"vs\. validación=({FLOAT}) -- "
    r"el intervalo (contiene|NO contiene) la verdad y (separa|NO separa) la prueba de la "
    r"validación\."
)

_VEREDICTOS_VALIDOS = {"mejor", "peor", "equivalente", "no concluyente"}


def _correr(presupuesto: str, resultados: Path, estudios: Path, timeout: float = 110):
    entorno = {
        **os.environ,
        "PRESUPUESTO": presupuesto,
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_ESTUDIOS": str(estudios),
    }
    return subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=timeout
    )


# Espacio de búsqueda, copiado de pasos/06_honestidad.py::_espacio (mismo
# archivo que se está probando, así que es el patrón canónico: los mismos
# rangos y el mismo orden de suggest_* que usa el paso, necesarios para
# reconstruir en proceso las R búsquedas repetidas de la pregunta 5 con las
# MISMAS semillas y comparar el resultado contra lo impreso.
def _espacio(trial: optuna.Trial) -> dict:
    return {
        "n_capas": trial.suggest_int("n_capas", 1, 2),
        "ancho": trial.suggest_int("ancho", 16, 128),
        "activacion": trial.suggest_categorical("activacion", ["relu", "tanh", "logistic"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 128),
    }


@pytest.fixture(scope="module")
def corrida(tmp_path_factory) -> dict:
    resultados = tmp_path_factory.mktemp("resultados_humo")
    estudios = tmp_path_factory.mktemp("estudios_humo")
    resultado = _correr("humo", resultados, estudios)
    assert resultado.returncode == 0, resultado.stderr
    return {"stdout": resultado.stdout, "resultados": resultados, "estudios": estudios}


def _nodo_funcion_pura(nombre: str) -> ast.FunctionDef:
    """`ast.FunctionDef` de una función pura de `pasos/06_honestidad.py`, leída con
    `ast` -- mismo patrón que `tests/test_pasos.py::_nodo_veredictos_franjas`
    (leer con `ast` en vez de importar: el paso corre código de nivel de módulo
    que entrena el juguete de verdad al importarlo).
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"), filename=str(PASO))
    return next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.FunctionDef) and nodo.name == nombre
    )


def _funcion_pura(nombre: str):
    """Ejecuta SOLO la función `nombre` de pasos/06_honestidad.py en un espacio de
    nombres propio (sin correr el resto del paso, que entrena el juguete de
    verdad), con `math` y `scipy_stats` disponibles -- mismo patrón que
    tests/test_pasos.py::veredictos_franjas: un error en su lógica rompe
    esta prueba y la del paso completo por igual, porque las dos corren la
    MISMA fuente.
    """
    modulo = ast.Module(body=[_nodo_funcion_pura(nombre)], type_ignores=[])
    ast.fix_missing_locations(modulo)
    espacio_nombres = {"math": math, "scipy_stats": scipy_stats}
    exec(compile(modulo, str(PASO), "exec"), espacio_nombres)
    return espacio_nombres[nombre]


@pytest.fixture(scope="module")
def veredicto_ic():
    return _funcion_pura("_veredicto_ic")


@pytest.fixture(scope="module")
def distingue_de_una_semilla():
    return _funcion_pura("_distingue_de_una_semilla")


@pytest.fixture(scope="module")
def alcanzo_potencia():
    return _funcion_pura("_alcanzo_potencia")


@pytest.fixture(scope="module")
def texto_brecha():
    return _funcion_pura("_texto_brecha")


@pytest.fixture(scope="module")
def comparacion_teoria_fn():
    return _funcion_pura("_comparacion_teoria")


def _pareado_sintetico(diferencias: list, *, delta: float = 0.01):
    """`Pareado` sobre `diferencias` contra un control fijo en 0.0 -- para ejercitar
    `_distingue_de_una_semilla` con datos sintéticos sin entrenar el juguete.
    """
    return estadistica.diferencia_pareada(
        dict(enumerate(diferencias)),
        {i: 0.0 for i in range(len(diferencias))},
        delta=delta,
        direccion="maximizar",
    )


def _estudio_ganador_persistido(estudios: Path) -> optuna.Study:
    destino = estudios / "06_honestidad_ganador_humo.log"
    return optuna.load_study(
        study_name="06_honestidad_ganador", storage=estudio.abrir_storage(destino)
    )


@pytest.mark.gpu
def test_paso_06_presupuesto_invalido_falla_con_mensaje(tmp_path):
    resultado = _correr("gigante", tmp_path / "resultados", tmp_path / "estudios", timeout=30)
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


@pytest.mark.gpu
def test_paso_06_humo_corre_y_contesta_las_seis_preguntas(corrida):
    stdout = corrida["stdout"]

    for inicio in (
        "1. Diferencia búsqueda - re-evaluación",
        "2. Contra el oráculo",
        "3. Re-evaluado, el ganador",
        "4. Con una desviación estándar de las diferencias de",
        "5. En 2 búsquedas completas repetidas",
        "6. Prueba sellada",
    ):
        assert inicio in stdout, stdout

    pngs = list(corrida["resultados"].glob("06_honestidad.png"))
    assert pngs, "el paso no guardó su figura"
    assert pngs[0].stat().st_size > 0


@pytest.mark.gpu
def test_paso_06_humo_los_numeros_tienen_sentido(corrida):
    stdout = corrida["stdout"]

    assert "Presupuesto: humo (búsqueda=8, re-evaluación=3, R=2)" in stdout, stdout

    # --- pregunta 1 ---
    m1 = _RE_R1.search(stdout)
    assert m1, stdout
    (
        val_busqueda1,
        media_reeval1,
        ic_bajo_r1,
        ic_alto_r1,
        inflacion1,
        ic_bajo_i1,
        ic_alto_i1,
    ) = (float(m1.group(i)) for i in (1, 3, 4, 5, 6, 7, 8))
    n_eval1 = int(m1.group(2))
    veredicto_ganador1 = m1.group(9)
    inflacion_base1, ic_bajo_ib1, ic_alto_ib1 = (float(m1.group(i)) for i in (10, 11, 12))
    veredicto_base1 = m1.group(13)
    distingue1 = m1.group(14)
    residuo_busqueda1, margen_una_semilla1 = (float(m1.group(i)) for i in (15, 16))
    gl_distingue1 = int(m1.group(17))
    media_verdad1, ic_bajo_v1, ic_alto_v1 = (float(m1.group(i)) for i in (18, 19, 20))
    brecha_reeval_verdad1 = float(m1.group(21))
    ic_bajo_brecha1, ic_alto_brecha1 = (float(m1.group(i)) for i in (22, 23))
    texto_brecha1 = m1.group(24)
    assert n_eval1 == N_EVAL_HUMO, "PRESUPUESTOS['humo']['n_eval']: si cambia, actualizar esto"
    assert 0.0 <= val_busqueda1 <= 1.0, "exactitud_val vive en [0, 1]"
    assert 0.0 <= media_reeval1 <= 1.0
    assert 0.0 <= media_verdad1 <= 1.0
    assert ic_bajo_r1 <= media_reeval1 <= ic_alto_r1, (
        "la media tiene que caer en su propio intervalo"
    )
    assert ic_bajo_v1 <= media_verdad1 <= ic_alto_v1
    # inflacion1 se verifica a precisión completa (no con val_busqueda1/
    # media_reeval1, ya redondeados a 3 decimales cada uno) en
    # test_paso_06_reevaluacion_pareada_recalculada_coincide.
    assert ic_bajo_i1 <= inflacion1 <= ic_alto_i1, (
        "la inflación tiene que caer en su propio intervalo"
    )
    if ic_bajo_i1 > 0.0:
        assert veredicto_ganador1 == "positiva", veredicto_ganador1
    elif ic_alto_i1 < 0.0:
        assert veredicto_ganador1 == "negativa", veredicto_ganador1
    else:
        assert veredicto_ganador1 == "no concluyente", veredicto_ganador1
    assert ic_bajo_brecha1 <= brecha_reeval_verdad1 <= ic_alto_brecha1, (
        "la brecha val-verdad tiene que caer en su propio intervalo"
    )
    if ic_bajo_brecha1 <= 0.0 <= ic_alto_brecha1:
        assert texto_brecha1.startswith("no concluyente"), texto_brecha1
    else:
        assert texto_brecha1.startswith("distinta de cero"), texto_brecha1
    assert margen_una_semilla1 >= 0.0, "el margen de predicción de una semilla no es negativo"
    assert gl_distingue1 == n_eval1 - 1, "los grados de libertad del intervalo son n_eval - 1"
    assert (distingue1 == "SÍ se distingue") == (abs(residuo_busqueda1) > margen_una_semilla1), (
        "'distingue' tiene que salir de |residuo| > margen de predicción de una semilla nueva"
    )
    # --- control de la línea base (nadie la eligió) y el veredicto que lo compara ---
    assert ic_bajo_ib1 <= inflacion_base1 <= ic_alto_ib1
    if ic_bajo_ib1 > 0.0:
        assert veredicto_base1 == "positiva", veredicto_base1
    elif ic_alto_ib1 < 0.0:
        assert veredicto_base1 == "negativa", veredicto_base1
    else:
        assert veredicto_base1 == "no concluyente", veredicto_base1
    assert distingue1 in ("SÍ se distingue", "NO se distingue")
    # La condición EXACTA que decide `distingue1` (el intervalo de predicción
    # de una semilla nueva, no el intervalo de la media de `pareado`) se verifica en
    # test_paso_06_reevaluacion_pareada_recalculada_coincide, con `pareado`
    # recalculado a precisión completa.

    # --- pregunta 2: mismo trial ganador que la pregunta 1 ---
    m2 = _RE_R2.search(stdout)
    assert m2, stdout
    val_busqueda2, verdad2, mentira2 = (float(g) for g in m2.groups())
    assert val_busqueda2 == pytest.approx(val_busqueda1, abs=1e-9), (
        "la 1 y la 2 imprimen el mismo valor de búsqueda del mismo ganador"
    )
    assert 0.0 <= verdad2 <= 1.0
    # mentira2 se verifica a precisión completa (no con estos val_busqueda2/
    # verdad2, ya redondeados a 3 decimales cada uno) en
    # test_paso_06_respuesta_2_contra_el_estudio_persistido.

    # --- pregunta 3 ---
    m3 = _RE_R3.search(stdout)
    assert m3, stdout
    media_ganador3, media_base3, diferencia3, ic_bajo3, ic_alto3 = (
        float(m3.group(i)) for i in range(1, 6)
    )
    veredicto3 = m3.group(6)
    n3 = int(m3.group(7))
    diferencia_verdad3 = float(m3.group(8))
    brecha_val_verdad3 = float(m3.group(9))
    ic_bajo_brecha3, ic_alto_brecha3 = (float(m3.group(i)) for i in (10, 11))
    texto_brecha3 = m3.group(12)
    assert n3 == n_eval1, "la re-evaluación de la pregunta 3 usa las mismas plan.evaluacion"
    assert veredicto3 in _VEREDICTOS_VALIDOS, veredicto3
    assert ic_bajo3 <= diferencia3 <= ic_alto3
    assert math.isfinite(diferencia_verdad3)
    # diferencia3 y brecha_val_verdad3 se verifican a precisión completa (no con
    # media_ganador3/media_base3, ya redondeados a 3 decimales cada uno) en
    # test_paso_06_reevaluacion_pareada_recalculada_coincide.
    assert ic_bajo_brecha3 <= brecha_val_verdad3 <= ic_alto_brecha3, (
        "la brecha val-verdad pareada tiene que caer en su propio intervalo"
    )
    if ic_bajo_brecha3 <= 0.0 <= ic_alto_brecha3:
        assert texto_brecha3.startswith("no concluyente"), texto_brecha3
    else:
        assert texto_brecha3.startswith("distinta de cero"), texto_brecha3

    # --- pregunta 4 ---
    m4 = _RE_R4.search(stdout)
    assert m4, stdout
    de_dif4 = float(m4.group(1))
    n_eval4 = int(m4.group(2))
    ic_pct4 = int(m4.group(3))
    de_dif_bajo4 = float(m4.group(4))
    de_dif_alto4 = float(m4.group(5))
    n_delta4 = int(m4.group(6))
    n_delta_bajo4 = int(m4.group(7))
    n_delta_alto4 = int(m4.group(8))
    n_hubo4 = int(m4.group(9))
    diferencia_obs4 = float(m4.group(10))
    veces_delta4 = int(m4.group(11))
    n_obs4 = int(m4.group(12))
    n_semillas4 = int(m4.group(13))
    alcanzo4 = m4.group(14)
    veredicto4_texto = m4.group(15)
    assert de_dif4 >= 0.0
    assert ic_pct4 == 95
    assert de_dif_bajo4 <= de_dif4 <= de_dif_alto4, "de_dif tiene que caer en su propio intervalo"
    assert n_delta4 >= 2, "estadistica.semillas_necesarias nunca baja de 2"
    assert n_delta_bajo4 <= n_delta4 <= n_delta_alto4, (
        "el n del punto central tiene que caer en el rango que da la incertidumbre de de_dif"
    )
    assert n_obs4 >= 2
    assert n_eval4 == n_hubo4 == n3 == n_semillas4, (
        "las cuatro se refieren a las mismas plan.evaluacion"
    )
    assert diferencia_obs4 == pytest.approx(abs(diferencia3) / 100.0, abs=1.5e-3), (
        "la diferencia observada de la pregunta 4 es |diferencia_media| de la pregunta 3, "
        "en unidades de exactitud (no en puntos)"
    )
    assert veces_delta4 == round(diferencia_obs4 / 0.01)
    if n_semillas4 >= n_delta_alto4:
        assert alcanzo4 == "sí, las semillas alcanzaron la potencia planeada", alcanzo4
    elif n_semillas4 < n_delta_bajo4:
        assert alcanzo4 == "no, las semillas no alcanzaron la potencia planeada", alcanzo4
    else:
        esperado4 = f"no concluyente (el n necesario va de {n_delta_bajo4} a {n_delta_alto4})"
        assert alcanzo4 == esperado4, alcanzo4
    assert veredicto4_texto == veredicto3, (
        "la respuesta 4 reporta el MISMO pareado que la 3: el veredicto no puede diferir"
    )

    # --- pregunta 5 ---
    m5 = _RE_R5.search(stdout)
    assert m5, stdout
    r5 = int(m5.group(1))
    media_infl5, ic_bajo_infl5, ic_alto_infl5 = (float(m5.group(i)) for i in (2, 3, 4))
    veredicto_texto5 = m5.group(5)
    media_cruda5, ic_bajo_cruda5, ic_alto_cruda5 = (float(m5.group(i)) for i in (6, 7, 8))
    n_busqueda5a, n_busqueda5b = int(m5.group(9)), int(m5.group(10))
    sigma5, max_esperado5, referencia5, referencia_puntos5 = (
        float(m5.group(i)) for i in (11, 12, 13, 14)
    )
    pct_cota5 = int(m5.group(15))
    referencia_baja_puntos5, referencia_alta_puntos5 = (float(m5.group(i)) for i in (16, 17))
    comparacion_texto5 = m5.group(18)
    assert r5 == R_HUMO, "PRESUPUESTOS['humo']['r']: si cambia, actualizar esto"
    assert n_busqueda5a == n_busqueda5b == N_BUSQUEDA_HUMO
    assert ic_bajo_infl5 <= media_infl5 <= ic_alto_infl5
    assert ic_bajo_cruda5 <= media_cruda5 <= ic_alto_cruda5
    assert sigma5 > 0.0, "sigma es una desviación estándar de datos no todos idénticos"
    # abs, no rel: sigma5/max_esperado5/referencia5 son tres redondeos
    # independientes de la misma cuenta -- multiplicar los dos primeros ya
    # redondeados acumula más error que el 1% relativo de un número tan
    # chico (~0.03).
    assert referencia5 == pytest.approx(sigma5 * max_esperado5, abs=1e-3)
    assert referencia_puntos5 == pytest.approx(100 * referencia5, abs=1e-2)
    assert pct_cota5 == 95
    assert referencia_baja_puntos5 <= referencia_puntos5 <= referencia_alta_puntos5, (
        "el intervalo propagado de la cota tiene que contener el punto de la cota"
    )
    if ic_bajo_infl5 > 0.0:
        assert veredicto_texto5 == "positiva", veredicto_texto5
    elif ic_alto_infl5 < 0.0:
        assert veredicto_texto5 == "negativa", veredicto_texto5
    else:
        assert veredicto_texto5 == f"no concluyente con R={r5}", veredicto_texto5
    # La consistencia EXACTA de esta frase con la condición que arma el paso se
    # verifica en test_paso_06_inflaciones_por_repeticion_recalculadas_coinciden,
    # con los valores recalculados a precisión completa: aquí los números ya
    # pasaron por el redondeo a 2 decimales de la impresión, así que cerca del
    # límite (referencia_baja/alta_puntos5 ~= ic_bajo/ic_alto) una comparación
    # estricta con esos mismos números redondeados podría no coincidir con la
    # rama que el paso, con su propia precisión completa, sí tomó.
    if ic_alto_infl5 < 0.0:
        assert comparacion_texto5.startswith("no: la inflación es negativa"), comparacion_texto5
    elif ic_bajo_infl5 <= 0.0 <= ic_alto_infl5:
        # Con el intervalo conteniendo el 0, la rama exacta (¿"sí" con coletilla, o
        # "no concluyente"?) depende de si ic_alto también queda por debajo de
        # la cota -- una comparación que aquí se hace con números YA
        # redondeados a 2 decimales, así que cerca del límite podría no
        # coincidir con la rama que el paso, con su propia precisión completa,
        # sí tomó (la consistencia EXACTA se verifica en
        # test_paso_06_inflaciones_por_repeticion_recalculadas_coinciden, con
        # los valores recalculados a precisión completa).
        assert comparacion_texto5.startswith("sí:") or comparacion_texto5 == (
            f"no concluyente con R={r5}: el intervalo medido contiene el 0 y se solapa con "
            "el de la cota"
        ), comparacion_texto5
    else:
        assert comparacion_texto5.startswith(
            ("sí:", "no: la media medida", f"no concluyente con R={r5}: ")
        ), comparacion_texto5

    # --- pregunta 6 ---
    m6 = _RE_R6.search(stdout)
    assert m6, stdout
    exactitud_prueba6, ic_bajo_p6, ic_alto_p6, verdad6, validacion6 = (
        float(m6.group(i)) for i in range(1, 6)
    )
    contiene6, separa6 = m6.group(6), m6.group(7)
    assert 0.0 <= exactitud_prueba6 <= 1.0
    assert ic_bajo_p6 <= exactitud_prueba6 <= ic_alto_p6
    assert validacion6 == pytest.approx(val_busqueda1, abs=1e-9)
    assert (contiene6 == "contiene") == (ic_bajo_p6 <= verdad6 <= ic_alto_p6)
    assert (separa6 == "separa") == (not (ic_bajo_p6 <= validacion6 <= ic_alto_p6))


@pytest.mark.gpu
def test_paso_06_respuesta_2_contra_el_estudio_persistido(corrida):
    """La verdad y el valor de búsqueda de las respuestas 2 y 6 tienen que salir del
    trial `top_1` REAL del estudio persistido -- nunca de otro trial (p. ej. el
    `top_2`, que también tendría un valor y una verdad plausibles). Contrastar
    directo contra `honestidad.mejores_de(study, ...)["top_1"]` es lo único que
    distingue "el paso lee el trial correcto" de "el paso lee un trial cualquiera
    y el texto sigue pareciendo razonable": ninguna otra prueba de este archivo
    reconstruye `top_1` desde el estudio y lo compara con las respuestas 2/6.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    top_1 = honestidad.mejores_de(study, 3, incluir_linea_base=True)["top_1"]
    trial_top_1 = next(t for t in study.trials if t.number == top_1["trial"])

    m2 = _RE_R2.search(corrida["stdout"])
    m6 = _RE_R6.search(corrida["stdout"])
    assert m2 and m6, corrida["stdout"]
    val_busqueda2, verdad2, mentira2 = (float(g) for g in m2.groups())
    verdad6, validacion6 = float(m6.group(4)), float(m6.group(5))

    assert val_busqueda2 == pytest.approx(trial_top_1.value, abs=5e-4)
    assert verdad2 == pytest.approx(trial_top_1.user_attrs["exactitud_verdadera"], abs=5e-4)
    assert mentira2 == pytest.approx(
        100 * (trial_top_1.value - trial_top_1.user_attrs["exactitud_verdadera"]), abs=0.05
    )
    assert validacion6 == pytest.approx(trial_top_1.value, abs=5e-4)
    assert verdad6 == pytest.approx(trial_top_1.user_attrs["exactitud_verdadera"], abs=5e-4)


@pytest.mark.gpu
def test_paso_06_protocolo_persistido_incluye_motor_y_n_busqueda(corrida):
    """El protocolo persistido debe traer `juguete.PROTOCOLO_MOTOR` y
    `n_busqueda`, para que reabrir con más trials que los que vio la pregunta 5
    sea detectable a simple vista.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    protocolo = study.user_attrs["protocolo"]
    assert protocolo["motor"] == juguete.MOTOR
    assert protocolo["dispositivo"] == juguete.DISPOSITIVO.split(":", 1)[0]
    assert protocolo["n_busqueda"] == N_BUSQUEDA_HUMO


@pytest.mark.gpu
def test_paso_06_protocolo_persistido_incluye_la_politica_de_semillas(corrida):
    """docs/arquitectura.md exige que el protocolo describa la política de
    semillas (además de sampler/pruner): sin esto, reabrir esta carpeta de
    `estudios/` con otra `SEMILLA_PLAN` cambiaría el conjunto de validación de
    TODOS los trials ya corridos sin que `crear_estudio` lo detecte -- la
    guarda de `crear_estudio` compara el protocolo entero, y sin este campo la
    segunda corrida terminaría con exit 0 y una respuesta 1 que fabrica una
    inflación con otro conjunto de validación.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    protocolo = study.user_attrs["protocolo"]
    plan_esperado = honestidad.plan_semillas(SEMILLA_PLAN, N_EVAL_HUMO)

    assert protocolo["politica_semillas"]["tipo"] == "CRN"
    assert protocolo["politica_semillas"]["plan_base"] == SEMILLA_PLAN
    assert protocolo["politica_semillas"]["busqueda"] == {
        "datos": plan_esperado.busqueda.datos,
        "orden": plan_esperado.busqueda.orden,
        "init": plan_esperado.busqueda.init,
    }
    assert protocolo["sampler_kwargs"]["seed_base"] == SEMILLA_TPE
    assert protocolo["sampler_kwargs"]["n_startup_trials"] == 5
    assert protocolo["sampler_kwargs"]["multivariate"] is True
    assert protocolo["pruner"] == "NopPruner"


@pytest.mark.gpu
def test_paso_06_prueba_sellada_recalculada_coincide(corrida):
    """Reentrena en proceso la config del top_1 persistido con `plan.busqueda`
    (la MISMA semilla que produjo ese modelo durante la búsqueda) y compara
    contra `exactitud_prueba` impresa. Un error que abriera la prueba
    sellada con una semilla de `plan.evaluacion` en vez de `plan.busqueda`
    entrenaría un modelo distinto y daría un número distinto. También recalcula
    el intervalo de Wilson de forma independiente (`estadistica.ic_proporcion`)
    y lo compara contra el impreso: un error que calculara ese intervalo a
    otro nivel de confianza (p. ej. 60%) pero siguiera imprimiendo "intervalo
    de confianza 95%" no cambia el punto (`exactitud_prueba`), así que sin
    este chequeo no se detecta.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    top_1 = honestidad.mejores_de(study, 3, incluir_linea_base=True)["top_1"]
    plan = honestidad.plan_semillas(SEMILLA_PLAN, N_EVAL_HUMO)
    recalculada = juguete.entrenar({**top_1["params"], "n_prueba": 5000}, plan.busqueda, None)[
        "exactitud_prueba"
    ]
    ic_bajo_recalc, ic_alto_recalc = estadistica.ic_proporcion(round(recalculada * 5000), 5000)

    m6 = _RE_R6.search(corrida["stdout"])
    assert m6, corrida["stdout"]
    assert recalculada == pytest.approx(float(m6.group(1)), abs=6e-4)
    assert ic_bajo_recalc == pytest.approx(float(m6.group(2)), abs=6e-4)
    assert ic_alto_recalc == pytest.approx(float(m6.group(3)), abs=6e-4)


@pytest.mark.gpu
def test_paso_06_reevaluacion_pareada_recalculada_coincide(corrida, distingue_de_una_semilla):
    """Recalcula la re-evaluación pareada del ganador contra la línea base con
    `plan_semillas(SEMILLA_PLAN, N_EVAL_HUMO).evaluacion` -- las MISMAS
    semillas que usa el paso para las respuestas 1 y 3 -- y compara contra lo
    impreso: la inflación del ganador Y de su control (línea base), el
    intervalo completo (no solo el punto) de la diferencia pareada y su
    `de_dif`, y el resumen de la revelación del oráculo. Comparar
    `pareado.ic_bajo/ic_alto` (no solo `diferencia_media`) es lo que detecta
    un emparejamiento roto (p. ej. invertir la lista de la línea base antes
    de emparejar): la media pareada puede coincidir por casualidad mientras
    el intervalo y `de_dif` no.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    seleccion = honestidad.mejores_de(study, 3, incluir_linea_base=True)
    plan = honestidad.plan_semillas(SEMILLA_PLAN, N_EVAL_HUMO)
    reevaluados = honestidad.reevaluar(
        {"top_1": seleccion["top_1"]["params"], "linea_base": seleccion["linea_base"]["params"]},
        juguete.entrenar,
        plan.evaluacion,
    )
    val_ganador = [m["exactitud_val"] for m in reevaluados["top_1"]]
    val_base = [m["exactitud_val"] for m in reevaluados["linea_base"]]
    verdad_ganador = [m["exactitud_verdadera"] for m in reevaluados["top_1"]]
    verdad_base = [m["exactitud_verdadera"] for m in reevaluados["linea_base"]]

    inflacion = honestidad.inflacion_ganador(
        seleccion["top_1"]["valor"], val_ganador, direccion="maximizar"
    )
    inflacion_base = honestidad.inflacion_ganador(
        seleccion["linea_base"]["valor"], val_base, direccion="maximizar"
    )
    pareado = estadistica.diferencia_pareada(
        dict(enumerate(val_ganador)), dict(enumerate(val_base)), delta=0.01, direccion="maximizar"
    )
    resumen_verdad_ganador = estadistica.resumen(verdad_ganador)
    pareado_verdad = estadistica.diferencia_pareada(
        dict(enumerate(verdad_ganador)),
        dict(enumerate(verdad_base)),
        delta=0.01,
        direccion="maximizar",
    )

    m1 = _RE_R1.search(corrida["stdout"])
    m3 = _RE_R3.search(corrida["stdout"])
    m4 = _RE_R4.search(corrida["stdout"])
    assert m1 and m3 and m4, corrida["stdout"]

    # --- pregunta 1: inflación del ganador y de su control (línea base) ---
    assert 100 * inflacion.inflacion == pytest.approx(float(m1.group(6)), abs=0.05)
    assert 100 * inflacion.ic_bajo == pytest.approx(float(m1.group(7)), abs=0.05)
    assert 100 * inflacion.ic_alto == pytest.approx(float(m1.group(8)), abs=0.05)
    if inflacion.ic_bajo > 0.0:
        veredicto_ganador_esperado = "positiva"
    elif inflacion.ic_alto < 0.0:
        veredicto_ganador_esperado = "negativa"
    else:
        veredicto_ganador_esperado = "no concluyente"
    assert m1.group(9) == veredicto_ganador_esperado
    assert 100 * inflacion_base.inflacion == pytest.approx(float(m1.group(10)), abs=0.05)
    assert 100 * inflacion_base.ic_bajo == pytest.approx(float(m1.group(11)), abs=0.05)
    assert 100 * inflacion_base.ic_alto == pytest.approx(float(m1.group(12)), abs=0.05)
    assert resumen_verdad_ganador.media == pytest.approx(float(m1.group(18)), abs=5e-4)
    assert resumen_verdad_ganador.ic_bajo == pytest.approx(float(m1.group(19)), abs=5e-4)
    assert resumen_verdad_ganador.ic_alto == pytest.approx(float(m1.group(20)), abs=5e-4)
    assert 100 * (inflacion.media_reevaluada - resumen_verdad_ganador.media) == pytest.approx(
        float(m1.group(21)), abs=0.05
    )

    # --- "se distingue": intervalo de PREDICCIÓN de una semilla nueva (no el
    # intervalo de la MEDIA de `pareado`) sobre `d_valor_busqueda`, vía la
    # MISMA función pura que usa el paso ---
    d_valor_busqueda = seleccion["top_1"]["valor"] - seleccion["linea_base"]["valor"]
    residuo_esperado = d_valor_busqueda - pareado.diferencia_media
    t_pred_esperado = float(scipy_stats.t.ppf(0.975, pareado.n - 1))
    margen_esperado = t_pred_esperado * pareado.de_dif * math.sqrt(1.0 + 1.0 / pareado.n)
    distingue_esperado = distingue_de_una_semilla(d_valor_busqueda, pareado)
    assert (m1.group(14) == "SÍ se distingue") == distingue_esperado
    assert distingue_esperado == (abs(residuo_esperado) > margen_esperado)
    assert 100 * residuo_esperado == pytest.approx(float(m1.group(15)), abs=0.05)
    assert 100 * margen_esperado == pytest.approx(float(m1.group(16)), abs=0.05)
    assert pareado.n - 1 == int(m1.group(17))

    # --- pregunta 1: brecha val-verdad de la re-evaluación, punto a punto ---
    brecha_reeval_verdad_valores = [v - t for v, t in zip(val_ganador, verdad_ganador, strict=True)]
    resumen_brecha_reeval_verdad = estadistica.resumen(brecha_reeval_verdad_valores)
    assert 100 * resumen_brecha_reeval_verdad.media == pytest.approx(float(m1.group(21)), abs=0.05)
    assert 100 * resumen_brecha_reeval_verdad.ic_bajo == pytest.approx(
        float(m1.group(22)), abs=0.05
    )
    assert 100 * resumen_brecha_reeval_verdad.ic_alto == pytest.approx(
        float(m1.group(23)), abs=0.05
    )

    # --- pregunta 3: diferencia pareada de validación (intervalo completo) y su "en verdad" ---
    assert 100 * pareado.diferencia_media == pytest.approx(float(m3.group(3)), abs=0.05)
    assert 100 * pareado.ic_bajo == pytest.approx(float(m3.group(4)), abs=0.05)
    assert 100 * pareado.ic_alto == pytest.approx(float(m3.group(5)), abs=0.05)
    assert pareado.veredicto == m3.group(6)
    assert 100 * pareado_verdad.diferencia_media == pytest.approx(float(m3.group(8)), abs=0.05)

    # --- pregunta 3: brecha val-verdad pareada, punto a punto ---
    brecha_val_verdad_valores = [
        (gv - bv) - (gt - bt)
        for gv, bv, gt, bt in zip(val_ganador, val_base, verdad_ganador, verdad_base, strict=True)
    ]
    resumen_brecha_val_verdad = estadistica.resumen(brecha_val_verdad_valores)
    assert 100 * resumen_brecha_val_verdad.media == pytest.approx(float(m3.group(9)), abs=0.05)
    assert 100 * resumen_brecha_val_verdad.ic_bajo == pytest.approx(float(m3.group(10)), abs=0.05)
    assert 100 * resumen_brecha_val_verdad.ic_alto == pytest.approx(float(m3.group(11)), abs=0.05)

    # --- pregunta 4: de_dif es la clave de semillas_necesarias ---
    assert pareado.de_dif == pytest.approx(float(m4.group(1)), abs=5e-4)


@pytest.mark.gpu
def test_paso_06_semillas_necesarias_recalculadas_coinciden(corrida):
    """`n` (respuesta 4) sale de `estadistica.semillas_necesarias`, no de un
    `delta` distinto al documentado: recalcularlo a partir del `de_dif`
    impreso (con 4 decimales, suficientes para que el recálculo no cruce un
    escalón de la función) atrapa un error que usara, p.ej., delta=0.05.
    También recalcula el intervalo chi-cuadrado de `de_dif` (gl = n_eval - 1)
    y el rango de `semillas_necesarias` en sus dos extremos: un error que
    imprimiera el rango sin de verdad usar el intervalo de `de_dif` (p. ej.
    un porcentaje fijo alrededor del punto) no coincidiría con este recálculo.
    """
    m4 = _RE_R4.search(corrida["stdout"])
    assert m4, corrida["stdout"]
    de_dif = float(m4.group(1))
    n_eval4 = int(m4.group(2))
    de_dif_bajo_impreso = float(m4.group(4))
    de_dif_alto_impreso = float(m4.group(5))
    n_delta_impreso = int(m4.group(6))
    n_delta_bajo_impreso = int(m4.group(7))
    n_delta_alto_impreso = int(m4.group(8))
    diferencia_obs = float(m4.group(10))
    n_obs_impreso = int(m4.group(12))
    assert n_delta_impreso == estadistica.semillas_necesarias(de_dif, delta=0.01)
    assert n_obs_impreso == estadistica.semillas_necesarias(de_dif, delta=diferencia_obs)

    gl = n_eval4 - 1
    de_dif_bajo_recalc = de_dif * math.sqrt(gl / scipy_stats.chi2.ppf(0.975, gl))
    de_dif_alto_recalc = de_dif * math.sqrt(gl / scipy_stats.chi2.ppf(0.025, gl))
    assert de_dif_bajo_recalc == pytest.approx(de_dif_bajo_impreso, abs=5e-4)
    assert de_dif_alto_recalc == pytest.approx(de_dif_alto_impreso, abs=5e-4)
    assert n_delta_bajo_impreso == estadistica.semillas_necesarias(de_dif_bajo_recalc, delta=0.01)
    assert n_delta_alto_impreso == estadistica.semillas_necesarias(de_dif_alto_recalc, delta=0.01)


@pytest.mark.gpu
def test_paso_06_referencia_de_la_pregunta_5_usa_n_busqueda_no_r(corrida):
    """`maximo_esperado_normal` de la respuesta 5 tiene que evaluarse en
    `N_BUSQUEDA` (8 en humo), no en `R` (2 en humo): con humo los dos son
    fáciles de confundir por ser ambos enteros chicos, así que se recalcula
    contra la función real en vez de solo comprobar consistencia interna.
    """
    m5 = _RE_R5.search(corrida["stdout"])
    assert m5, corrida["stdout"]
    n_busqueda_impreso = int(m5.group(9))
    max_esperado_impreso = float(m5.group(12))
    assert n_busqueda_impreso == N_BUSQUEDA_HUMO
    assert max_esperado_impreso == pytest.approx(
        estadistica.maximo_esperado_normal(N_BUSQUEDA_HUMO), abs=1e-3
    )


@pytest.mark.gpu
def test_paso_06_sigma_viene_de_los_trials_de_la_busqueda_del_ganador(corrida):
    """`sigma` sale de `exactitud_val - exactitud_verdadera` de los trials
    COMPLETE de la búsqueda DEL GANADOR (preguntas 1-4), nunca de la
    re-evaluación: se recalcula directo sobre el estudio persistido.
    """
    study = _estudio_ganador_persistido(corrida["estudios"])
    completos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    ruido = [t.value - t.user_attrs["exactitud_verdadera"] for t in completos]
    sigma_esperada = float(np.std(ruido, ddof=1))

    m5 = _RE_R5.search(corrida["stdout"])
    assert m5, corrida["stdout"]
    assert float(m5.group(11)) == pytest.approx(sigma_esperada, abs=1e-4)


@pytest.mark.gpu
def test_paso_06_inflaciones_por_repeticion_recalculadas_coinciden(corrida, comparacion_teoria_fn):
    """Reconstruye las `R_HUMO` búsquedas repetidas de la pregunta 5 con las
    MISMAS semillas (`SEMILLA_REPETICIONES`, `SEMILLA_TPE`) y el mismo
    `N_BUSQUEDA_HUMO`, calcula la inflación de cada una con el ganador elegido
    por VALIDACIÓN (`honestidad.mejores_de`, que ordena por `trial.value`) y
    compara el resumen resultante contra lo impreso. Un error que eligiera
    el ganador de cada repetición por el ORÁCULO en vez de por validación
    entrenaría los mismos modelos pero promediaría la inflación de un ganador
    distinto, y el resumen no coincidiría.
    """
    semillas_repeticiones = semillas_de(SEMILLA_REPETICIONES, R_HUMO)
    inflaciones: list[float] = []
    for r, semilla_r in enumerate(semillas_repeticiones):
        objetivo_r = en_proceso(
            juguete.entrenar,
            _espacio,
            metricas=["exactitud_val"],
            semillas=semilla_r,
            atributos=["exactitud_verdadera"],
        )
        estudio_r = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=SEMILLA_TPE + 1000 * (r + 1), n_startup_trials=5, multivariate=True
            ),
            pruner=optuna.pruners.NopPruner(),
        )
        estudio.optimizar(
            estudio_r,
            objetivo_r,
            n_total=N_BUSQUEDA_HUMO,
            atrapar=(juguete.Divergencia, MetricaNoFinita),
        )
        completos_r = estudio_r.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        desvios_r = {t.number: t.value - t.user_attrs["exactitud_verdadera"] for t in completos_r}
        ganador_r = honestidad.mejores_de(estudio_r, 1, incluir_linea_base=False)["top_1"]
        # Media de TODOS los desvíos, incluido el ganador (ver el porqué en el
        # docstring de pasos/06_honestidad.py, celda de la pregunta 5).
        inflaciones.append(desvios_r[ganador_r["trial"]] - st.mean(desvios_r.values()))

    resumen_recalculado = estadistica.resumen(inflaciones)

    m5 = _RE_R5.search(corrida["stdout"])
    assert m5, corrida["stdout"]
    media_impresa = float(m5.group(2))
    ic_bajo_impreso = float(m5.group(3))
    ic_alto_impreso = float(m5.group(4))
    comparacion_texto5 = m5.group(18)
    assert 100 * resumen_recalculado.media == pytest.approx(media_impresa, abs=0.05)
    assert 100 * resumen_recalculado.ic_bajo == pytest.approx(ic_bajo_impreso, abs=0.05)
    assert 100 * resumen_recalculado.ic_alto == pytest.approx(ic_alto_impreso, abs=0.05)

    # La frase "¿coincide con la teoría?" se recalcula llamando a la MISMA
    # `_comparacion_teoria` que usa el paso (no una copia manual de sus ramas),
    # con la MISMA precisión completa (nunca con los números ya redondeados a
    # 2 decimales de la impresión, donde un caso al filo de la cota podría
    # tomar la rama contraria por el propio redondeo) y propagando el
    # intervalo chi-cuadrado de sigma exactamente como lo hace el paso.
    study = _estudio_ganador_persistido(corrida["estudios"])
    completos_ganador = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    ruido_ganador = [t.value - t.user_attrs["exactitud_verdadera"] for t in completos_ganador]
    sigma_exacta = float(np.std(ruido_ganador, ddof=1))
    maximo_esperado_exacto = estadistica.maximo_esperado_normal(N_BUSQUEDA_HUMO)
    gl_sigma = len(ruido_ganador) - 1
    alfa_dedif = 0.05
    sigma_bajo_exacta = sigma_exacta * math.sqrt(
        gl_sigma / scipy_stats.chi2.ppf(1 - alfa_dedif / 2, gl_sigma)
    )
    sigma_alto_exacta = sigma_exacta * math.sqrt(
        gl_sigma / scipy_stats.chi2.ppf(alfa_dedif / 2, gl_sigma)
    )
    referencia_baja_puntos_exacta = 100 * sigma_bajo_exacta * maximo_esperado_exacto
    referencia_alta_puntos_exacta = 100 * sigma_alto_exacta * maximo_esperado_exacto
    # resumen_recalculado.ic_bajo/ic_alto están en fracción (0-1): a puntos
    # (*100) antes de comparar, igual que en el paso.
    ic_bajo_infl_exacto = 100 * resumen_recalculado.ic_bajo
    ic_alto_infl_exacto = 100 * resumen_recalculado.ic_alto
    esperado5 = comparacion_teoria_fn(
        ic_bajo_infl_exacto,
        ic_alto_infl_exacto,
        referencia_baja_puntos_exacta,
        referencia_alta_puntos_exacta,
        R_HUMO,
    )
    assert comparacion_texto5 == esperado5, (comparacion_texto5, esperado5)


def test_paso_06_separa_control_usa_la_funcion_pura_en_el_codigo_fuente():
    """Prueba estática: `separa_control` (que decide si la respuesta 1 dice 'SÍ
    se distingue' o 'NO se distingue') tiene que salir de una llamada a
    `_distingue_de_una_semilla` (el intervalo de PREDICCIÓN de una semilla
    nueva) -- no de una heurística en línea que compare `d_valor_busqueda`
    contra el intervalo de la MEDIA de `pareado` (una regla que
    test_distingue_de_una_semilla_calibracion_sin_seleccion demuestra que
    sobre-declara selección sin ninguna selección real). Un cambio real que
    volviera a esa heurística seguiría imprimiendo texto con forma válida (los
    regex de arriba no distinguen un método del otro), así que esta prueba
    verifica el árbol sintáctico del código fuente, no el stdout.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"))
    (asignacion,) = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and nodo.targets[0].id == "separa_control"
    ]
    assert (
        isinstance(asignacion.value, ast.Call)
        and isinstance(asignacion.value.func, ast.Name)
        and asignacion.value.func.id == "_distingue_de_una_semilla"
    ), (
        "separa_control debe salir de una llamada a _distingue_de_una_semilla, no de otra "
        f"cuenta en línea; nodo real: {ast.dump(asignacion.value)}"
    )


def test_paso_06_usa_delta_0_01_y_la_prueba_sellada_en_el_codigo_fuente():
    """Prueba estática (no depende de correr el paso): un error que
    cambiara `delta=0.01` por otro valor en alguna de las dos llamadas a
    `diferencia_pareada`, o que abriera la prueba entrenando directo en vez de
    pasar por `PruebaSellada.sellar`/`abrir`, seguiría imprimiendo texto con
    forma válida (los regex de arriba no distinguen delta=0.01 de delta=0.05,
    ni "sellada" de "entrenada directo") -- por eso esta prueba verifica el
    árbol sintáctico del código fuente, no el stdout.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"))

    llamadas_diferencia_pareada = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr == "diferencia_pareada"
    ]
    assert len(llamadas_diferencia_pareada) == 2, (
        "el paso llama a diferencia_pareada exactamente dos veces (validación y verdad)"
    )
    for llamada in llamadas_diferencia_pareada:
        deltas = [kw.value for kw in llamada.keywords if kw.arg == "delta"]
        assert len(deltas) == 1, "delta debe pasarse como keyword explícito"
        (delta_nodo,) = deltas
        assert isinstance(delta_nodo, ast.Constant) and delta_nodo.value == 0.01, (
            "las dos comparaciones pareadas de este paso usan delta=0.01"
        )

    llamadas_sellar = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr == "sellar"
    ]
    llamadas_abrir = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr == "abrir"
    ]
    assert len(llamadas_sellar) == 1, "la prueba se sella exactamente una vez"
    assert len(llamadas_abrir) == 1, "la prueba se abre exactamente una vez"


def _llamada_de_asignacion(nombre: str) -> ast.Call:
    """El `ast.Call` que asigna la variable `nombre` en pasos/06_honestidad.py --
    mismo patrón que test_paso_06_separa_control_usa_la_funcion_pura_en_el_
    codigo_fuente: se busca la asignación de nivel de módulo (o de una rama
    condicional del módulo, que `ast.walk` recorre igual), sin ejecutar nada.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"))
    (asignacion,) = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and nodo.targets[0].id == nombre
    ]
    assert isinstance(asignacion.value, ast.Call), (
        f"{nombre} debe asignarse directo desde una llamada, nodo real: "
        f"{ast.dump(asignacion.value)}"
    )
    return asignacion.value


def _mismo_argumento(nodo: ast.AST, fuente: str) -> bool:
    """¿`nodo` (un argumento real, tomado del código fuente del paso) es
    sintácticamente idéntico a la expresión `fuente` (p. ej. `"pareado.n"`)?
    Comparar por `ast.dump` en vez de por texto evita falsos negativos por
    espacios entre líneas.
    """
    esperado = ast.parse(fuente, mode="eval").body
    return ast.dump(nodo) == ast.dump(esperado)


def test_paso_06_las_decisiones_se_cablean_con_los_argumentos_documentados():
    """Las funciones puras de este archivo ya tienen pruebas por rama (arriba),
    pero eso no verifica CON QUÉ argumentos las llama el paso: un error que
    cableara, p. ej., `_alcanzo_potencia(pareado.n, n_delta, n_delta)` (el punto
    central en vez del rango `[n_delta_bajo, n_delta_alto]`) seguiría imprimiendo
    texto con forma válida, y `PRESUPUESTO=humo` rara vez entra en la rama que
    ese error afecta -- por eso esta prueba verifica el árbol sintáctico del
    código fuente (los argumentos reales de cada llamada), no el stdout de una
    corrida.
    """
    llamada_distingue = _llamada_de_asignacion("separa_control")
    assert isinstance(llamada_distingue.func, ast.Name)
    assert llamada_distingue.func.id == "_distingue_de_una_semilla"
    assert len(llamada_distingue.args) == 2 and not llamada_distingue.keywords
    assert _mismo_argumento(llamada_distingue.args[0], "d_valor_busqueda")
    assert _mismo_argumento(llamada_distingue.args[1], "pareado")

    llamada_potencia = _llamada_de_asignacion("alcanzo_potencia")
    assert isinstance(llamada_potencia.func, ast.Name)
    assert llamada_potencia.func.id == "_alcanzo_potencia"
    assert len(llamada_potencia.args) == 3 and not llamada_potencia.keywords
    assert _mismo_argumento(llamada_potencia.args[0], "pareado.n")
    assert _mismo_argumento(llamada_potencia.args[1], "n_delta_bajo")
    assert _mismo_argumento(llamada_potencia.args[2], "n_delta_alto")

    llamada_brecha_1 = _llamada_de_asignacion("texto_brecha_reeval_verdad")
    assert isinstance(llamada_brecha_1.func, ast.Name)
    assert llamada_brecha_1.func.id == "_texto_brecha"
    assert len(llamada_brecha_1.args) == 2 and not llamada_brecha_1.keywords
    assert _mismo_argumento(llamada_brecha_1.args[0], "resumen_brecha_reeval_verdad.ic_bajo")
    assert _mismo_argumento(llamada_brecha_1.args[1], "resumen_brecha_reeval_verdad.ic_alto")

    llamada_brecha_3 = _llamada_de_asignacion("texto_brecha_val_verdad")
    assert isinstance(llamada_brecha_3.func, ast.Name)
    assert llamada_brecha_3.func.id == "_texto_brecha"
    assert len(llamada_brecha_3.args) == 2
    assert _mismo_argumento(llamada_brecha_3.args[0], "resumen_brecha_val_verdad.ic_bajo")
    assert _mismo_argumento(llamada_brecha_3.args[1], "resumen_brecha_val_verdad.ic_alto")
    (cola_kw,) = llamada_brecha_3.keywords
    assert cola_kw.arg == "cola"
    assert isinstance(cola_kw.value, ast.Constant)
    assert cola_kw.value.value == "; para verla del todo hace falta un conjunto nuevo (pregunta 6)"

    llamada_teoria = _llamada_de_asignacion("comparacion_teoria")
    assert isinstance(llamada_teoria.func, ast.Name)
    assert llamada_teoria.func.id == "_comparacion_teoria"
    assert len(llamada_teoria.args) == 5 and not llamada_teoria.keywords
    assert _mismo_argumento(llamada_teoria.args[0], "ic_bajo_infl_puntos")
    assert _mismo_argumento(llamada_teoria.args[1], "ic_alto_infl_puntos")
    assert _mismo_argumento(llamada_teoria.args[2], "referencia_baja_puntos")
    assert _mismo_argumento(llamada_teoria.args[3], "referencia_alta_puntos")
    assert _mismo_argumento(llamada_teoria.args[4], "R")


# --- pruebas de las funciones puras del paso (mutan en las dos direcciones) ---


def test_veredicto_ic_positiva_cuando_el_ic_esta_entero_sobre_cero(veredicto_ic):
    assert veredicto_ic(0.01, 0.05) == "positiva"


def test_veredicto_ic_negativa_cuando_el_ic_esta_entero_bajo_cero(veredicto_ic):
    assert veredicto_ic(-0.05, -0.01) == "negativa"


def test_veredicto_ic_no_concluyente_cuando_el_ic_contiene_el_cero(veredicto_ic):
    assert veredicto_ic(-0.01, 0.01) == "no concluyente"


def test_veredicto_ic_usa_el_texto_no_concluyente_personalizado(veredicto_ic):
    assert veredicto_ic(-0.01, 0.01, no_concluyente="otro texto") == "otro texto"


def test_veredicto_ic_bordes_en_cero_caen_en_no_concluyente(veredicto_ic):
    # bajo == 0.0 no es "> 0.0", y alto == 0.0 no es "< 0.0": los dos bordes
    # caen en "no concluyente", no en "positiva"/"negativa" (mutar > por >= o
    # < por <= movería estos dos casos a la rama contraria).
    assert veredicto_ic(0.0, 0.05) == "no concluyente"
    assert veredicto_ic(-0.05, 0.0) == "no concluyente"


def test_distingue_de_una_semilla_false_cuando_d_es_la_media_exacta(distingue_de_una_semilla):
    pareado = _pareado_sintetico([0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005, -0.015, 0.01])
    assert distingue_de_una_semilla(pareado.diferencia_media, pareado) is False


def test_distingue_de_una_semilla_true_cuando_d_esta_muy_lejos(distingue_de_una_semilla):
    pareado = _pareado_sintetico([0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005, -0.015, 0.01])
    lejos = pareado.diferencia_media + 100 * pareado.de_dif
    assert distingue_de_una_semilla(lejos, pareado) is True


def test_distingue_de_una_semilla_usa_el_intervalo_de_prediccion_no_el_ic_de_la_media(
    distingue_de_una_semilla,
):
    """El intervalo de confianza de la media (ancho `t*de_dif/sqrt(n)`) y el
    intervalo de predicción (ancho `t*de_dif*sqrt(1+1/n)`, `sqrt(n+1)` veces
    más ancho) discrepan para un residuo entre los dos anchos: ese residuo
    cae "afuera" del intervalo de confianza de la media pero DENTRO del
    intervalo de predicción, así que la función debe usar el intervalo de
    predicción y declarar "NO se distingue".
    """
    pareado = _pareado_sintetico([0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005, -0.015, 0.01])
    margen_media = pareado.ic_alto - pareado.diferencia_media
    d = pareado.diferencia_media + 2.0 * margen_media
    regla_del_intervalo_de_la_media = (d > pareado.ic_alto) or (d < pareado.ic_bajo)
    assert regla_del_intervalo_de_la_media is True, (
        "premisa: este d queda fuera del intervalo de confianza de la media"
    )
    assert distingue_de_una_semilla(d, pareado) is False, (
        "el intervalo de predicción, más ancho, todavía contiene este residuo"
    )


def test_distingue_de_una_semilla_calibracion_sin_seleccion(distingue_de_una_semilla):
    """Sin ninguna selección (la semilla de la búsqueda es una observación MÁS,
    intercambiable con las de evaluación bajo H0), la tasa de "SÍ se distingue"
    debe acercarse a la tasa nominal del 5% (t de Student, dos colas): el
    intervalo de la media (ancho t·de_dif/sqrt(n)) sobre-declara selección frente al
    intervalo de predicción (ancho t·de_dif·sqrt(1+1/n)), que es el que usa
    `_distingue_de_una_semilla`.
    """
    rng = np.random.default_rng(20260926)
    n = 10
    total = 4000
    aciertos = 0
    for _ in range(total):
        diferencias = rng.normal(loc=0.0, scale=0.02, size=n + 1)
        d = float(diferencias[0])
        pareado = _pareado_sintetico(diferencias[1:].tolist())
        if distingue_de_una_semilla(d, pareado):
            aciertos += 1
    tasa = aciertos / total
    assert 0.03 <= tasa <= 0.07, f"tasa medida {tasa:.3f}, se esperaba cerca de 0.05"


def test_alcanzo_potencia_si_cuando_n_alcanza_el_extremo_alto(alcanzo_potencia):
    assert alcanzo_potencia(20, 10, 15) == "sí, las semillas alcanzaron la potencia planeada"


def test_alcanzo_potencia_si_en_el_borde_exacto(alcanzo_potencia):
    assert alcanzo_potencia(15, 10, 15) == "sí, las semillas alcanzaron la potencia planeada"


def test_alcanzo_potencia_no_cuando_n_no_llega_al_extremo_bajo(alcanzo_potencia):
    assert alcanzo_potencia(5, 10, 15) == "no, las semillas no alcanzaron la potencia planeada"


def test_alcanzo_potencia_no_concluyente_entre_los_dos_extremos(alcanzo_potencia):
    assert alcanzo_potencia(12, 10, 15) == "no concluyente (el n necesario va de 10 a 15)"


def test_texto_brecha_no_concluyente_cuando_el_ic_contiene_el_cero(texto_brecha):
    assert texto_brecha(-0.01, 0.01) == (
        "no concluyente: con estas semillas la brecha no se distingue de cero"
    )


def test_texto_brecha_distinta_de_cero_cuando_el_ic_no_contiene_el_cero(texto_brecha):
    assert texto_brecha(0.01, 0.03) == (
        "distinta de cero: esa parte (la suerte del conjunto de validación) no la ve "
        "re-evaluar con los mismos datos"
    )


def test_texto_brecha_agrega_la_cola_solo_en_la_rama_distinta_de_cero(texto_brecha):
    assert texto_brecha(-0.01, 0.01, cola="; nunca aparece") == (
        "no concluyente: con estas semillas la brecha no se distingue de cero"
    )
    assert texto_brecha(0.01, 0.03, cola="; siempre aparece") == (
        "distinta de cero: esa parte (la suerte del conjunto de validación) no la ve "
        "re-evaluar con los mismos datos; siempre aparece"
    )


def test_comparacion_teoria_negativa_cuando_el_ic_esta_bajo_cero(comparacion_teoria_fn):
    assert comparacion_teoria_fn(-2.0, -0.5, 1.0, 3.0, 6) == (
        "no: la inflación es negativa (intervalo por debajo de 0), contra lo que predice la teoría"
    )


def test_comparacion_teoria_no_atribuye_causa_cuando_el_ic_contiene_el_cero_y_se_solapa(
    comparacion_teoria_fn,
):
    """Un intervalo medido [-0.03, +1.86] que contiene el 0 pero NO queda
    entero a ningún lado de la cota [1.0, 3.0] (se solapa con ella) no puede
    atribuirse a "mezclar configuraciones con verdades distintas" -- esa
    causa exige que la media quede entera por debajo de la cota, algo que
    este intervalo no muestra.
    """
    assert comparacion_teoria_fn(-0.03, 1.86, 1.0, 3.0, 6) == (
        "no concluyente con R=6: el intervalo medido contiene el 0 y se solapa con el de la cota"
    )


def test_comparacion_teoria_si_con_cola_cuando_el_ic_contiene_el_cero_pero_queda_bajo_la_cota(
    comparacion_teoria_fn,
):
    """A diferencia del caso anterior, aquí el intervalo medido [-0.03, +1.86]
    SÍ queda entero por debajo de la cota [2.36, 3.99] (ic_alto=1.86 <
    referencia_baja=2.36): la posición relativa a la cota manda "sí", y como
    el intervalo también contiene el 0 se agrega la coletilla de que tampoco
    se distingue de 0.
    """
    assert comparacion_teoria_fn(-0.03, 1.86, 2.36, 3.99, 6) == (
        "sí: la media medida queda por debajo de la cota, consistente con que mezclar "
        "configuraciones con verdades distintas (la búsqueda compara configuraciones con "
        "verdades distintas, no N configuraciones de la misma verdad) solo puede empujar la "
        "inflación por debajo de la cota, nunca por encima; con R=6 la inflación tampoco se "
        "distingue de 0"
    )


def test_comparacion_teoria_no_concluyente_cuando_el_ic_sube_muy_por_encima_de_toda_la_cota(
    comparacion_teoria_fn,
):
    """Un intervalo medido [-0.5, +9.0] contiene el 0 Y se extiende muy por
    encima de TODA la cota [1.49, 4.60] (ic_alto=9.0 no queda por debajo de
    referencia_baja=1.49). La frase no puede decir "por debajo de la cota" en
    este caso, porque contradiría sus propios números; la posición relativa a
    la cota (que no es "por debajo") tiene que mandar sobre el mero contacto
    con el 0.
    """
    assert comparacion_teoria_fn(-0.5, 9.0, 1.49, 4.60, 2) == (
        "no concluyente con R=2: el intervalo medido contiene el 0 y se solapa con el de la cota"
    )


def test_comparacion_teoria_si_cuando_el_ic_esta_entero_bajo_la_cota(comparacion_teoria_fn):
    assert comparacion_teoria_fn(0.5, 0.8, 1.0, 3.0, 6).startswith("sí:")


def test_comparacion_teoria_no_cuando_el_ic_esta_entero_sobre_la_cota(comparacion_teoria_fn):
    assert comparacion_teoria_fn(3.5, 4.0, 1.0, 3.0, 6).startswith(
        "no: la media medida queda POR ENCIMA"
    )


def test_comparacion_teoria_no_concluyente_cuando_los_ic_se_solapan_sin_contener_el_cero(
    comparacion_teoria_fn,
):
    assert comparacion_teoria_fn(1.5, 3.5, 1.0, 3.0, 6) == (
        "no concluyente con R=6: el intervalo medido y el de la cota se solapan"
    )


@pytest.mark.gpu
def test_paso_06_reanudar_no_cambia_las_respuestas(corrida):
    """Con la MISMA `estudios/` de `corrida`, una segunda corrida no vuelve a
    optimizar (`optimizar` ya cuenta los 8 trials COMPLETE que dejó la
    primera) ni re-siembra la línea base (`sembrar_linea_base` ve una
    COMPLETE y no hace nada): las seis respuestas -- que solo dependen de esos
    trials y de las semillas fijas de las R repeticiones -- deben salir
    carácter por carácter iguales. Reusa `corrida` como "primera" en vez de
    correrla de nuevo (una corrida completa de menos).
    """
    segunda = _correr("humo", corrida["resultados"], corrida["estudios"])
    assert segunda.returncode == 0, segunda.stderr
    assert corrida["stdout"] == segunda.stdout
