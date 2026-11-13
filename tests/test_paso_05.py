"""Paso 05: además de que corra (eso ya lo cubre tests/test_pasos.py con
PRESUPUESTO=humo), esta prueba verifica el CONTENIDO -- que imprime las tres
respuestas numeradas, que sus veredictos y sus verbos ("le gana"/"pierde"/
"empata"/"no se distingue") son consistentes entre sí, y que el paso no puede
mentir sobre cuál trial eligió ni con qué algoritmo entrenó cada baseline sin
que alguna de estas pruebas lo note.

Patrón de fixture `scope="module"` copiado de `corrida_humo` en
`tests/test_paso_04.py`: corre el paso UNA vez por módulo, en vez de una vez
por test -- este paso paga R * N * 2 entrenamientos en la tarjeta de video
compartida de la máquina.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest
from scipy.stats import binomtest
from scipy.stats import t as t_student_scipy

from taller_hpo import estadistica

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "05_buscar_el_modelo.py"
ALGORITMOS_BUSCADOS = {"logistica", "bosque", "hgb", "mlp"}
VEREDICTOS_VALIDOS = {"mejor", "peor", "equivalente", "no concluyente"}

FLOAT = r"[+-]?\d\.\d+"
FLOAT_IC = r"\d\.\d\d"

_RE_REPLICA = re.compile(
    r"^replica (?P<r>\d+): dispositivo=(?P<dispositivo>cuda) \| "
    r"tpe exactitud_val\(elegido\)=(?P<val_elegido>\d\.\d+) "
    r"exactitud_val\(maximo_estudio\)=(?P<val_maximo>\d\.\d+) \| "
    r"verdadera tpe=(?P<verdad_tpe>\d\.\d+) random=(?P<verdad_random>\d\.\d+) "
    r"knn=(?P<verdad_knn>\d\.\d+) krr=(?P<verdad_krr>\d\.\d+) \| "
    r"algoritmo_ganador_tpe=(?P<algoritmo>\w+)$",
    re.MULTILINE,
)

_RE_R1 = re.compile(
    r"1\. TPE \(group=True\) vs Random, (?P<r_total>\d+) réplicas de (?P<n_total>\d+) "
    r"evaluaciones cada una \((?P<arranque>[^)]*(?:\([^)]*\)[^)]*)*)\): "
    rf"exactitud verdadera media (?P<media_tpe>\d\.\d+) vs (?P<media_random>\d\.\d+) "
    rf"\(diferencia (?P<diferencia>{FLOAT}), intervalo de confianza 95% "
    rf"\[(?P<ic_bajo>{FLOAT}), (?P<ic_alto>{FLOAT})\], veredicto "
    rf"'(?P<veredicto>[^']+)' con delta=(?P<delta>\d\.\d+)\)\.(?P<resto>[^\n]*)"
)

_RE_R2_UNICO = re.compile(
    r"2\. El algoritmo que TPE eligió como mejor con más frecuencia fue '(?P<top>\w+)' "
    r"\((?P<top_count>\d+) de (?P<total>\d+) réplicas, intervalo de confianza 95% Wilson de "
    rf"la proporción \[(?P<ic_bajo>{FLOAT_IC}), (?P<ic_alto>{FLOAT_IC})\]\); "
    r"conteo completo \(orden fijo\): (?P<conteo>\{[^}]*\})\."
)
_RE_R2_EMPATE = re.compile(
    r"2\. Empate entre (?P<lista>.+?) como el algoritmo que TPE eligió como mejor con más "
    r"frecuencia \((?P<top_count>\d+) de (?P<total>\d+) réplicas cada uno, intervalo de "
    r"confianza 95% Wilson de la "
    rf"proporción \[(?P<ic_bajo>{FLOAT_IC}), (?P<ic_alto>{FLOAT_IC})\]\); "
    r"conteo completo \(orden fijo\): (?P<conteo>\{[^}]*\})\."
)

_RE_R3 = re.compile(
    rf"3\. Baselines con sus defaults \(sin ninguna búsqueda\): knn "
    rf"(?P<media_knn>\d\.\d+), krr (?P<media_krr>\d\.\d+)\. La búsqueda con TPE "
    rf"\(exactitud verdadera media (?P<media_tpe>\d\.\d+)\) (?P<frase_knn>.+?) "
    rf"\(intervalo de confianza 95% \[(?P<ic_bajo_knn>{FLOAT}), (?P<ic_alto_knn>{FLOAT})\], "
    r"veredicto '(?P<veredicto_knn>[^']+)'\), y (?P<frase_krr>.+?) "
    rf"\(intervalo 95% \[(?P<ic_bajo_krr>{FLOAT}), (?P<ic_alto_krr>{FLOAT})\], "
    r"veredicto '(?P<veredicto_krr>[^']+)'\)\."
)

_RE_SEGUNDOS = re.compile(
    r"segundos por algoritmo \(mediana de ESTA corrida, ajuste \+ validación, sin oráculo; "
    r"dispositivo=(?P<dispositivo>cuda)\): (?P<dict>\{[^}]*\})"
)


def _codigo_paso() -> str:
    return PASO.read_text(encoding="utf-8")


def _arbol_paso() -> ast.Module:
    return ast.parse(_codigo_paso(), filename=str(PASO))


def _extraer_al_namespace(nombres: set[str]) -> dict:
    """Extrae del AST del paso las funciones/clases/asignaciones cuyo nombre
    esté en `nombres` y las ejecuta en un namespace nuevo con optuna/
    estadistica ya disponibles -- así se puede ejercer
    `_mejor_por_validacion`/`espacio`/`_RANGOS_ARBOL`/`_frase_veredicto`/
    `_frase_arranque`/`_frase_conteo` con datos sintéticos, sin pagar una
    corrida completa del paso en la tarjeta de video compartida. `R` queda con
    un valor fijo porque `_frase_veredicto` lee esa global del módulo en su
    rama "no concluyente". Las clases se extraen igual que las funciones --
    por `ast.ClassDef`, no solo `ast.FunctionDef`."""
    arbol = _arbol_paso()
    nodos = [
        nodo
        for nodo in arbol.body
        if (isinstance(nodo, (ast.FunctionDef, ast.ClassDef)) and nodo.name in nombres)
        or (
            isinstance(nodo, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in nombres for t in nodo.targets)
        )
    ]
    encontrados = {
        nodo.name if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)) else nodo.targets[0].id
        for nodo in nodos
    }
    faltan = nombres - encontrados
    assert not faltan, f"no se encontraron en el paso: {faltan}"
    ns: dict = {
        "optuna": optuna,
        "TrialState": optuna.trial.TrialState,
        "FrozenTrial": optuna.trial.FrozenTrial,
        "estadistica": estadistica,
        "binomtest": binomtest,
        "R": 5,
    }
    exec(compile(ast.Module(body=nodos, type_ignores=[]), str(PASO), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def corrida_humo(tmp_path_factory) -> tuple[str, str, Path]:
    resultados = tmp_path_factory.mktemp("resultados")
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        # Fijo (no heredado del entorno real, que no lo garantiza): así esta
        # corrida sirve también como el punto de comparación "hashseed=0" de
        # test_conteo_algoritmos_no_depende_de_pythonhashseed, sin pagar una
        # corrida completa extra en la tarjeta de video compartida solo para eso.
        "PYTHONHASHSEED": "0",
    }
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    return resultado.stdout, resultado.stderr, resultados


@pytest.fixture(scope="module")
def salida_humo(corrida_humo) -> str:
    return corrida_humo[0]


# --- caja negra: el paso corre, deja su figura, y las tres respuestas aparecen ---


@pytest.mark.gpu
def test_presupuesto_invalido_falla_con_mensaje(tmp_path):
    entorno = {**os.environ, "PRESUPUESTO": "gigante", "TALLER_RESULTADOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


@pytest.mark.gpu
def test_deja_su_figura(corrida_humo):
    _, _, resultados = corrida_humo
    figura = resultados / "05_buscar_el_modelo.png"
    assert figura.exists()
    assert figura.stat().st_size > 1000


@pytest.mark.gpu
def test_imprime_su_dispositivo_de_cash_paridad(salida_humo):
    m = re.search(r"cash_paridad: dispositivo=(cuda)(?: \([^)]*\))?", salida_humo)
    assert m, salida_humo


@pytest.mark.gpu
def test_imprime_las_tres_respuestas(salida_humo):
    assert re.search(r"^1\. TPE", salida_humo, re.MULTILINE), salida_humo
    assert re.search(r"^2\. (El algoritmo|Empate entre)", salida_humo, re.MULTILINE), salida_humo
    assert re.search(r"^3\. Baselines", salida_humo, re.MULTILINE), salida_humo


@pytest.mark.gpu
def test_humo_dice_que_tpe_no_actua_todavia(salida_humo):
    """PRESUPUESTOS['humo'] es (r=2, n=4): n=4 < TPESampler.n_startup_trials=10
    (Optuna 5.0.0), así que la respuesta 1 debe decirlo de forma explícita, no
    dejar que se lea como si TPE ya estuviera guiando la búsqueda."""
    assert "nunca llegó a actuar de forma guiada" in salida_humo, salida_humo


@pytest.mark.gpu
def test_aviso_experimental_se_explica_una_vez_y_no_llega_crudo_a_stderr(corrida_humo):
    """La nota usa `optuna.__version__` (no un texto fijo "Optuna 5.0.0"):
    `pyproject.toml` acepta `optuna>=5.0,<6`, así que con otra versión 5.x
    instalada un texto fijo diría una versión que no es la que de verdad
    corrió."""
    salida, error, _ = corrida_humo
    assert (
        salida.count(f"TPESampler(group=True) es experimental en Optuna {optuna.__version__}") == 1
    ), salida
    assert "ExperimentalWarning" not in error


# --- respuesta 1: TPE vs Random ---


@pytest.mark.gpu
def test_respuesta_1_numeros_tienen_sentido(salida_humo):
    m1 = _RE_R1.search(salida_humo)
    assert m1, salida_humo
    assert int(m1.group("r_total")) == 2, "PRESUPUESTOS['humo'] es (r=2, n=4)"
    assert int(m1.group("n_total")) == 4, "PRESUPUESTOS['humo'] es (r=2, n=4)"
    media_tpe, media_random = float(m1.group("media_tpe")), float(m1.group("media_random"))
    assert 0.0 <= media_tpe <= 1.0
    assert 0.0 <= media_random <= 1.0
    diferencia = float(m1.group("diferencia"))
    ic_bajo, ic_alto = float(m1.group("ic_bajo")), float(m1.group("ic_alto"))
    # 0.002: los tres números se redondean a 3 decimales por separado al imprimirse.
    assert ic_bajo - 0.002 <= diferencia <= ic_alto + 0.002
    assert m1.group("veredicto") in VEREDICTOS_VALIDOS
    assert float(m1.group("delta")) > 0.0


# --- respuesta 2: qué algoritmo ganó más veces (o empate), sin depender de PYTHONHASHSEED ---


@pytest.mark.gpu
def test_respuesta_2_numeros_tienen_sentido(salida_humo):
    m_unico = _RE_R2_UNICO.search(salida_humo)
    m_empate = _RE_R2_EMPATE.search(salida_humo)
    assert m_unico or m_empate, salida_humo

    if m_unico:
        total_replicas = int(m_unico.group("total"))
        assert total_replicas == 2
        assert m_unico.group("top") in ALGORITMOS_BUSCADOS
        conteo = ast.literal_eval(m_unico.group("conteo"))
        top_count = int(m_unico.group("top_count"))
        assert conteo[m_unico.group("top")] == top_count
        ic_bajo, ic_alto = float(m_unico.group("ic_bajo")), float(m_unico.group("ic_alto"))
        esperado_bajo, esperado_alto = estadistica.ic_proporcion(top_count, total_replicas)
        assert ic_bajo == pytest.approx(esperado_bajo, abs=0.005)
        assert ic_alto == pytest.approx(esperado_alto, abs=0.005)
    else:
        total_replicas = int(m_empate.group("total"))
        assert total_replicas == 2
        conteo = ast.literal_eval(m_empate.group("conteo"))
        nombres = re.findall(r"'(\w+)'", m_empate.group("lista"))
        assert len(nombres) >= 2, m_empate.group("lista")
        top_count = int(m_empate.group("top_count"))
        for nombre in nombres:
            assert nombre in ALGORITMOS_BUSCADOS
            assert conteo[nombre] == top_count
        ic_bajo, ic_alto = float(m_empate.group("ic_bajo")), float(m_empate.group("ic_alto"))
        esperado_bajo, esperado_alto = estadistica.ic_proporcion(top_count, total_replicas)
        assert ic_bajo == pytest.approx(esperado_bajo, abs=0.005)
        assert ic_alto == pytest.approx(esperado_alto, abs=0.005)

    assert set(conteo) == ALGORITMOS_BUSCADOS, "el conteo siempre trae las 4 claves (orden fijo)"
    assert sum(conteo.values()) == total_replicas
    assert list(conteo) == ["logistica", "bosque", "hgb", "mlp"], (
        "el conteo se imprime siempre en ALGORITMOS_ORDEN, no en el orden de un set/dict "
        "dependiente de PYTHONHASHSEED"
    )


@pytest.mark.gpu
def test_conteo_algoritmos_no_depende_de_pythonhashseed(corrida_humo, tmp_path_factory):
    """`max(conteo, key=conteo.get)` sobre un dict armado con `set(...)`
    desempataría según el orden de iteración del set, que depende de
    PYTHONHASHSEED -- este paso arma el conteo con `collections.Counter` sobre
    `ALGORITMOS_ORDEN` (orden fijo) precisamente para no depender de eso.
    Comparando `corrida_humo` (que ya fija PYTHONHASHSEED=0) contra una
    segunda corrida con PYTHONHASHSEED=1 (mismas semillas fijas, mismo
    PRESUPUESTO), la respuesta 2 debe salir carácter por carácter igual --
    una sola corrida extra, no dos: la de hashseed=0 ya la paga el fixture de
    módulo."""
    salida_hash0 = corrida_humo[0]
    resultados = tmp_path_factory.mktemp("resultados_hash1")
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "PYTHONHASHSEED": "1",
    }
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr
    salida_hash1 = resultado.stdout

    m0 = re.search(r"^2\..*$", salida_hash0, re.MULTILINE)
    m1 = re.search(r"^2\..*$", salida_hash1, re.MULTILINE)
    assert m0 and m1, (salida_hash0, salida_hash1)
    assert m0.group(0) == m1.group(0)


# --- líneas "replica N": el trial elegido es el máximo del ESTUDIO, no del oráculo ---


@pytest.mark.gpu
def test_lineas_por_replica_estan_completas(salida_humo):
    filas = list(_RE_REPLICA.finditer(salida_humo))
    assert len(filas) == 2, "una línea 'replica N' por cada una de las 2 réplicas de humo"
    for fila in filas:
        assert fila.group("algoritmo") in ALGORITMOS_BUSCADOS, fila.group(0)
        for clave in ("verdad_tpe", "verdad_random", "verdad_knn", "verdad_krr"):
            assert 0.0 <= float(fila.group(clave)) <= 1.0, fila.group(0)


@pytest.mark.gpu
def test_replica_elegido_coincide_con_el_maximo_del_estudio(salida_humo):
    """Chequeo de consistencia interna de caja NEGRA: `val_elegido` y
    `val_maximo` se imprimen desde dos expresiones distintas del paso
    (`elegido_tpe.value` y `max(t.value for t in completos_tpe)`), así que
    esta prueba detecta que las dos no se desincronizaron entre sí. NO
    detecta por sí sola que la selección sea por validación y no por el
    oráculo -- las dos expresiones podrían cambiar juntas a mirar
    `exactitud_verdadera` y seguirían coincidiendo. Esa protección es de caja
    BLANCA: ver test_mejor_por_validacion_elige_por_validacion_no_por_oraculo."""
    filas = list(_RE_REPLICA.finditer(salida_humo))
    assert filas
    for fila in filas:
        assert fila.group("val_elegido") == fila.group("val_maximo"), fila.group(0)


def test_mejor_por_validacion_elige_por_validacion_no_por_oraculo():
    """Caja BLANCA: extrae `_mejor_por_validacion` del AST del paso (sin
    correrlo) y la ejerce sobre un estudio armado a mano con
    `optuna.trial.create_trial`, donde el trial de mayor `value`
    (`exactitud_val`, lo que ve la búsqueda) NO es el de mayor
    `exactitud_verdadera` (el oráculo). Si la función alguna vez cambiara a
    elegir por el oráculo -- o por cualquier cosa que no sea `trial.value`--
    este assert lo nota; comparar los números que el paso imprime no lo
    detecta, porque los dos salen de la misma llamada a esta función (ver
    test_replica_elegido_coincide_con_el_maximo_del_estudio)."""
    ns = _extraer_al_namespace({"_mejor_por_validacion"})
    dist = {"algoritmo": optuna.distributions.CategoricalDistribution(["mlp"])}
    estudio = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    # value alto pero oráculo bajo, y viceversa: si _mejor_por_validacion
    # mirara el oráculo, elegiría el trial equivocado en las dos direcciones.
    estudio.add_trial(
        optuna.trial.create_trial(
            value=0.80,
            params={"algoritmo": "mlp"},
            distributions=dist,
            user_attrs={"exactitud_verdadera": 0.60},
        )
    )
    estudio.add_trial(
        optuna.trial.create_trial(
            value=0.70,
            params={"algoritmo": "mlp"},
            distributions=dist,
            user_attrs={"exactitud_verdadera": 0.90},
        )
    )
    elegido = ns["_mejor_por_validacion"](estudio)
    assert elegido.value == 0.80
    assert elegido.user_attrs["exactitud_verdadera"] == 0.60


def test_ast_elegido_usa_mejor_por_validacion_del_estudio_correspondiente():
    """Si `elegido_tpe`/`elegido_random` no vinieran de llamar
    `_mejor_por_validacion(study_tpe)`/`_mejor_por_validacion(study_random)`
    -- p. ej. si alguien inlineara `max(..., key=...)` con otro criterio --
    la prueba de caja blanca de arriba, que ejerce `_mejor_por_validacion`
    por su cuenta, no lo notaría: seguiría probando una función correcta que
    el paso ya no usa donde dice que la usa. Revisión estática de las dos
    asignaciones reales."""
    arbol = _arbol_paso()
    for nombre_var, nombre_estudio in (
        ("elegido_tpe", "study_tpe"),
        ("elegido_random", "study_random"),
    ):
        nodo = _asignacion_unica_de(arbol, nombre_var)
        llamada = nodo.value
        assert isinstance(llamada, ast.Call), ast.unparse(llamada)
        assert ast.unparse(llamada.func) == "_mejor_por_validacion", ast.unparse(llamada)
        assert len(llamada.args) == 1
        assert ast.unparse(llamada.args[0]) == nombre_estudio, ast.unparse(llamada)


@pytest.mark.gpu
def test_knn_y_krr_dan_valores_distintos(salida_humo):
    """knn (voto de vecinos) y krr (kernel ridge) son algoritmos distintos
    sobre los MISMOS datos de cada réplica: si `krr_por_replica` se entrenara
    por error como 'knn' (algoritmo equivocado en la llamada real a
    `cash_paridad.entrenar`), las dos series saldrían casi idénticas."""
    filas = list(_RE_REPLICA.finditer(salida_humo))
    assert filas
    for fila in filas:
        assert fila.group("verdad_knn") != fila.group("verdad_krr"), fila.group(0)


# --- recalcular medias y diferencia_pareada desde las líneas "replica N" ---


@pytest.mark.gpu
def test_recalculo_de_diferencia_pareada_coincide_con_lo_impreso(salida_humo):
    filas = list(_RE_REPLICA.finditer(salida_humo))
    tpe = {int(f.group("r")): float(f.group("verdad_tpe")) for f in filas}
    random_ = {int(f.group("r")): float(f.group("verdad_random")) for f in filas}
    knn = {int(f.group("r")): float(f.group("verdad_knn")) for f in filas}
    krr = {int(f.group("r")): float(f.group("verdad_krr")) for f in filas}

    comp = estadistica.diferencia_pareada(tpe, random_, delta=0.01, direccion="maximizar")
    comp_knn = estadistica.diferencia_pareada(tpe, knn, delta=0.01, direccion="maximizar")
    comp_krr = estadistica.diferencia_pareada(tpe, krr, delta=0.01, direccion="maximizar")

    m1 = _RE_R1.search(salida_humo)
    assert m1, salida_humo
    # abs=6e-4: los valores impresos en las líneas "replica N" ya vienen
    # redondeados a 4 decimales, así que la media/diferencia recalculada desde
    # ESOS textos puede diferir de la calculada en punto flotante completo por
    # hasta un par de redondeos de 4 decimales encadenados.
    assert float(m1.group("media_tpe")) == pytest.approx(sum(tpe.values()) / len(tpe), abs=6e-4)
    assert float(m1.group("media_random")) == pytest.approx(
        sum(random_.values()) / len(random_), abs=6e-4
    )
    assert float(m1.group("diferencia")) == pytest.approx(comp.diferencia_media, abs=6e-4)
    assert m1.group("veredicto") == comp.veredicto
    # abs=3e-3 (más ancho que el 6e-4 de las medias): el intervalo depende de de_dif al
    # cuadrado, y con R=2 (1 grado de libertad, t crítico ~12.7) ese redondeo a 4 decimales de
    # las líneas "replica N" se amplifica más que en una media simple -- medido
    # como el peor caso de encadenar ese redondeo por esta cuenta con R=2.
    assert float(m1.group("ic_bajo")) == pytest.approx(comp.ic_bajo, abs=3e-3)
    assert float(m1.group("ic_alto")) == pytest.approx(comp.ic_alto, abs=3e-3)

    m3 = _RE_R3.search(salida_humo)
    assert m3, salida_humo
    assert float(m3.group("media_tpe")) == pytest.approx(sum(tpe.values()) / len(tpe), abs=6e-4)
    # media_knn/media_krr, no solo media_tpe: si la respuesta 3 imprimiera las
    # medias de knn y krr cruzadas (la de krr donde dice knn y viceversa), la
    # única señal sería que estos dos números no coinciden con lo recalculado
    # aquí -- comparar solo el veredicto (abajo) no lo nota, porque un
    # veredicto puede seguir siendo el mismo aunque el número impreso a su
    # lado sea el de la otra comparación.
    assert float(m3.group("media_knn")) == pytest.approx(sum(knn.values()) / len(knn), abs=6e-4)
    assert float(m3.group("media_krr")) == pytest.approx(sum(krr.values()) / len(krr), abs=6e-4)
    assert m3.group("veredicto_knn") == comp_knn.veredicto
    assert m3.group("veredicto_krr") == comp_krr.veredicto
    assert float(m3.group("ic_bajo_knn")) == pytest.approx(comp_knn.ic_bajo, abs=3e-3)
    assert float(m3.group("ic_alto_knn")) == pytest.approx(comp_knn.ic_alto, abs=3e-3)
    assert float(m3.group("ic_bajo_krr")) == pytest.approx(comp_krr.ic_bajo, abs=3e-3)
    assert float(m3.group("ic_alto_krr")) == pytest.approx(comp_krr.ic_alto, abs=3e-3)


def test_frase_potencia_tres_ramas_con_pareado_sintetico():
    """Esta prueba, de caja blanca, extrae `_frase_potencia`
    (pasos/05_buscar_el_modelo.py) del AST del paso y la ejerce con un
    `estadistica.Pareado` sintético en sus cuatro veredictos posibles --
    'mejor' y 'peor' con de_dif > 0 (sin frase: la función solo arma la
    frase de refuerzo con veredicto 'no concluyente', nunca cuando la
    búsqueda ya ganó o perdió), 'equivalente' con de_dif > 0 (sin frase,
    mismo motivo) y 'no concluyente' en sus dos ramas -- de_dif > 0
    (recalcula el n de detección) y de_dif == 0 (nada que estimar) -- sin
    depender de que una corrida de humo con pocas réplicas visite las
    ramas por azar."""
    ns = _extraer_al_namespace({"_frase_potencia", "DELTA_EQUIVALENCIA"})
    base = dict(
        n=2,
        diferencia_media=0.02,
        ic_bajo=0.01,
        ic_alto=0.03,
        p_valor=0.2,
        wilcoxon_p=None,
        rho=0.5,
        delta=0.01,
        direccion="maximizar",
        confianza=0.95,
    )
    comp_equivalente = estadistica.Pareado(de_dif=0.01, veredicto="equivalente", **base)
    assert ns["_frase_potencia"](comp_equivalente, 2) == ""

    # Con de_dif > 0 pero veredicto 'mejor' o 'peor' (la búsqueda YA ganó o
    # perdió), no hay nada que "reforzar" con una potencia de detección: la
    # frase debe seguir vacía en las dos, igual que con 'equivalente'.
    comp_mejor = estadistica.Pareado(de_dif=0.05, veredicto="mejor", **base)
    assert ns["_frase_potencia"](comp_mejor, 2) == ""
    comp_peor = estadistica.Pareado(de_dif=0.05, veredicto="peor", **base)
    assert ns["_frase_potencia"](comp_peor, 2) == ""

    comp_no_concluyente = estadistica.Pareado(de_dif=0.05, veredicto="no concluyente", **base)
    n_esperado = estadistica.semillas_necesarias(0.05, 0.01)
    frase = ns["_frase_potencia"](comp_no_concluyente, 2)
    assert "Detectar" in frase, frase
    assert "decidir" not in frase, frase
    assert f"exigiría ~{n_esperado} réplicas" in frase, frase
    assert "con 1 grado de libertad" in frase, frase

    comp_de_dif_cero = estadistica.Pareado(de_dif=0.0, veredicto="no concluyente", **base)
    assert ns["_frase_potencia"](comp_de_dif_cero, 2) == (
        " (la desviación estándar de las diferencias midió 0: no hay variación entre "
        "réplicas que estimar)."
    )


@pytest.mark.gpu
def test_respuesta_1_frase_potencia_recalcula_semillas_necesarias(salida_humo):
    """Esta prueba afirma la rama que de verdad salió en ESTA corrida de
    humo, sin saltar ninguna: si el veredicto no fue 'no concluyente', la
    frase de refuerzo debe estar vacía
    (`test_frase_potencia_tres_ramas_con_pareado_sintetico` cubre esa rama y
    las otras dos con un `Pareado` sintético, así que esta prueba no
    necesita saltarse para no comprobar nada). Si sí salió 'no concluyente',
    debe decir "detectar" (nunca "decidir": semillas_necesarias da el n para
    DETECTAR delta con potencia 80%, no para decidir nada) y su n debe
    coincidir con `estadistica.semillas_necesarias(de_dif, delta)`
    recalculada desde las líneas 'replica N', no con una cifra fija."""
    m1 = _RE_R1.search(salida_humo)
    assert m1, salida_humo
    if m1.group("veredicto") != "no concluyente":
        assert m1.group("resto") == "", m1.group("resto")
        return
    assert "decidir" not in m1.group("resto"), m1.group("resto")

    filas = list(_RE_REPLICA.finditer(salida_humo))
    tpe = {int(f.group("r")): float(f.group("verdad_tpe")) for f in filas}
    random_ = {int(f.group("r")): float(f.group("verdad_random")) for f in filas}
    comp = estadistica.diferencia_pareada(tpe, random_, delta=0.01, direccion="maximizar")

    m_frase = re.search(r"exigiría ~(\d+) réplicas", m1.group("resto"))
    if comp.de_dif <= 0.0:
        assert m_frase is None, m1.group("resto")
        return
    assert m_frase, m1.group("resto")
    n_esperado = estadistica.semillas_necesarias(comp.de_dif, 0.01)
    # +-1: de_dif recalculada desde las líneas "replica N" (redondeadas a 4
    # decimales) puede diferir en su último dígito de la de_dif exacta que
    # usó el paso -- con R=2 (1 grado de libertad) eso a veces mueve el n
    # mínimo en 1. La protección de que el cálculo real es
    # semillas_necesarias(comparacion.de_dif, DELTA_EQUIVALENCIA) es de caja
    # blanca, ver test_ast_frase_potencia_usa_semillas_necesarias_correcto.
    assert abs(int(m_frase.group(1)) - n_esperado) <= 1, (m_frase.group(1), n_esperado)


def test_ast_frase_potencia_usa_semillas_necesarias_correcto():
    """Revisión estática de `n_necesario`: debe venir de
    `estadistica.semillas_necesarias(comparacion.de_dif, DELTA_EQUIVALENCIA)`
    -- ni una de_dif distinta ni un delta hardcodeado que se desincronice de
    `DELTA_EQUIVALENCIA` si algún día cambia."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "n_necesario")
    llamada = nodo.value
    assert isinstance(llamada, ast.Call)
    assert ast.unparse(llamada.func) == "estadistica.semillas_necesarias"
    assert [ast.unparse(a) for a in llamada.args] == [
        "comparacion.de_dif",
        "DELTA_EQUIVALENCIA",
    ]


def test_ast_comparaciones_usan_delta_equivalencia_y_direccion_maximizar():
    """Revisión estática de las tres llamadas a `estadistica.diferencia_pareada`
    de las respuestas 1 y 3 (`comparacion`, `comparacion_knn`,
    `comparacion_krr`): las tres deben compartir `delta=DELTA_EQUIVALENCIA` y
    `direccion='maximizar'` -- si una sola (p. ej. `comparacion_knn`) trajera
    un `delta` hardcodeado distinto, su veredicto 'equivalente'/'no
    concluyente' y la advertencia de zona de indiferencia que se imprimen al
    lado usarían un número distinto del que de verdad se sombrea en la
    figura (`DELTA_EQUIVALENCIA`), sin que ninguna corrida de humo (que casi
    siempre sale 'no concluyente' con R=2) lo note."""
    arbol = _arbol_paso()
    esperado = {
        "comparacion": ["tpe_por_replica", "random_por_replica"],
        "comparacion_knn": ["tpe_por_replica", "knn_por_replica"],
        "comparacion_krr": ["tpe_por_replica", "krr_por_replica"],
    }
    for nombre_var, args_esperados in esperado.items():
        nodo = _asignacion_unica_de(arbol, nombre_var)
        llamada = nodo.value
        assert isinstance(llamada, ast.Call)
        assert ast.unparse(llamada.func) == "estadistica.diferencia_pareada"
        assert [ast.unparse(a) for a in llamada.args] == args_esperados, ast.unparse(llamada)
        kw = {k.arg: ast.unparse(k.value) for k in llamada.keywords}
        assert kw.get("delta") == "DELTA_EQUIVALENCIA", (nombre_var, kw)
        assert kw.get("direccion") == "'maximizar'", (nombre_var, kw)


@pytest.mark.gpu
def test_respuesta_3_holm_se_imprime_en_rango_valido(salida_humo):
    """No se recalcula el p-valor exacto desde las líneas 'replica N' aquí:
    con R=2 (humo) la t pareada tiene 1 grado de libertad, y en esa zona el
    p-valor es tan sensible al redondeo a 4 decimales de esas líneas que
    recalcularlo desde ahí puede diferir del que el paso calculó con sus
    floats completos por varias milésimas (medido) -- no es un bug, es la
    propia forma de la t con 1 grado de libertad. La corrección real de que el cálculo es
    el correcto (mismos p_valor, mismo orden Random/knn/krr) es de caja
    BLANCA, ver test_ast_holm_se_calcula_sobre_los_tres_p_valores_en_orden;
    esta prueba solo verifica que los tres números impresos son p-valores
    válidos."""
    m = re.search(
        r"con Holm-Bonferroni: p=(?P<p1>\d\.\d+) vs Random, p=(?P<p2>\d\.\d+) vs knn, "
        r"p=(?P<p3>\d\.\d+) vs krr",
        salida_humo,
    )
    assert m, salida_humo
    for grupo in ("p1", "p2", "p3"):
        assert 0.0 <= float(m.group(grupo)) <= 1.0


def test_ast_comparaciones_holm_orden_fijo():
    """`_COMPARACIONES_HOLM` es la ÚNICA fuente de la que se derivan `p_holm`,
    `cambia_holm` y las etiquetas impresas (Random, knn, krr, en ese orden):
    etiqueta y comparación viven juntas en una sola tupla para que su orden
    no pueda desincronizarse entre sí sin que ninguna prueba de caja negra lo
    notara con solo 3 números en [0, 1]. Revisión estática de que es
    exactamente la tupla literal esperada, en este orden."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "_COMPARACIONES_HOLM")
    assert ast.unparse(nodo.value) == (
        "(('Random', comparacion), ('knn', comparacion_knn), ('krr', comparacion_krr))"
    ), ast.unparse(nodo.value)


def test_ast_p_holm_y_etiquetas_derivan_de_comparaciones_holm():
    """Revisión estática de que `p_holm` y `_etiquetas_holm` de verdad se
    calculan a partir de `_COMPARACIONES_HOLM` (y no de una lista/tupla
    aparte que pudiera desincronizarse de ella)."""
    arbol = _arbol_paso()

    nodo_p = _asignacion_unica_de(arbol, "p_holm")
    assert isinstance(nodo_p.value, ast.Call)
    assert ast.unparse(nodo_p.value.func) == "estadistica.holm"
    assert len(nodo_p.value.args) == 1
    lista = nodo_p.value.args[0]
    assert isinstance(lista, ast.ListComp), ast.unparse(nodo_p.value)
    assert ast.unparse(lista.elt) == "c.p_valor", ast.unparse(lista.elt)
    assert ast.unparse(lista.generators[0].iter) == "_COMPARACIONES_HOLM", ast.unparse(
        lista.generators[0].iter
    )

    nodo_etq = _asignacion_unica_de(arbol, "_etiquetas_holm")
    assert isinstance(nodo_etq.value, ast.Call)
    assert ast.unparse(nodo_etq.value.func) == "tuple"
    genexp = nodo_etq.value.args[0]
    assert isinstance(genexp, ast.GeneratorExp), ast.unparse(nodo_etq.value)
    assert ast.unparse(genexp.elt) == "nombre", ast.unparse(genexp.elt)
    assert ast.unparse(genexp.generators[0].iter) == "_COMPARACIONES_HOLM", ast.unparse(
        genexp.generators[0].iter
    )


# --- respuesta 3: el verbo impreso corresponde al veredicto ---


_VERBO_ESPERADO = {
    "mejor": "le gana a",
    "peor": "pierde contra",
    "equivalente": "empata con",
    "no concluyente": "no se distingue de",
}


@pytest.mark.gpu
def test_respuesta_3_el_verbo_corresponde_al_veredicto(salida_humo):
    m3 = _RE_R3.search(salida_humo)
    assert m3, salida_humo
    assert 0.0 <= float(m3.group("media_knn")) <= 1.0
    assert 0.0 <= float(m3.group("media_krr")) <= 1.0

    for nombre, frase_grupo, veredicto_grupo, ic_bajo_grupo, ic_alto_grupo in (
        ("knn", "frase_knn", "veredicto_knn", "ic_bajo_knn", "ic_alto_knn"),
        ("krr", "frase_krr", "veredicto_krr", "ic_bajo_krr", "ic_alto_krr"),
    ):
        veredicto = m3.group(veredicto_grupo)
        frase = m3.group(frase_grupo)
        assert veredicto in VEREDICTOS_VALIDOS, frase
        assert frase.startswith(f"{_VERBO_ESPERADO[veredicto]} {nombre}"), (veredicto, frase)
        ic_bajo, ic_alto = float(m3.group(ic_bajo_grupo)), float(m3.group(ic_alto_grupo))
        if veredicto == "mejor":
            assert ic_bajo > 0.0, frase
        elif veredicto == "peor":
            assert ic_alto < 0.0, frase
        else:
            assert ic_bajo <= 0.0 <= ic_alto, frase
        # La diferencia que trae la frase (para 'mejor'/'peor': "por +x"; para
        # 'equivalente'/'no concluyente': "diferencia +x") debe caer dentro del
        # intervalo que se imprime al lado -- si no, el texto en prosa y los
        # números de estadistica.diferencia_pareada se desincronizaron. Se
        # busca por la palabra que la precede ("por"/"diferencia"), no por la
        # última palabra de la frase: 'mejor' puede traer la advertencia
        # extra del intervalo cerca de delta al final, que ya no termina en
        # el número.
        m_diff = re.search(rf"(?:por|diferencia) ({FLOAT})", frase)
        assert m_diff, frase
        diferencia = float(m_diff.group(1))
        assert ic_bajo - 0.002 <= diferencia <= ic_alto + 0.002, (frase, ic_bajo, ic_alto)


# --- segundos por algoritmo: la línea existe y trae números positivos ---


@pytest.mark.gpu
def test_segundos_por_algoritmo_se_imprime_con_numeros_positivos(salida_humo):
    m = _RE_SEGUNDOS.search(salida_humo)
    assert m, salida_humo
    medianas = ast.literal_eval(m.group("dict"))
    assert medianas, "debe traer al menos un algoritmo con datos"
    assert set(medianas) <= (ALGORITMOS_BUSCADOS | {"knn", "krr"})
    for valor in medianas.values():
        assert valor > 0.0


# --- revisión estática (AST): defectos que una corrida de humo no garantiza detectar ---


def test_no_importa_mlp_paridad_directo():
    """Convención de pasos (docs/arquitectura.md): nunca importar
    mlp_paridad/paridad_torch directo. `cash_paridad.Divergencia`
    (importada de mlp_contrato dentro de cash_paridad) es la que este
    paso usa para capturar la divergencia del MLP."""
    arbol = _arbol_paso()
    prohibidos = {"mlp_paridad", "paridad_torch"}
    importados: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom) and nodo.module:
            importados.update(nodo.module.split("."))
            importados.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.Import):
            for alias in nodo.names:
                importados.update(alias.name.split("."))
    encontrados = importados & prohibidos
    assert not encontrados, encontrados


def _asignacion_unica_de(arbol: ast.Module, nombre: str) -> ast.Assign:
    candidatos = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == nombre for t in nodo.targets)
    ]
    assert len(candidatos) == 1, f"debe haber exactamente una asignación a {nombre}"
    return candidatos[0]


def test_ast_study_random_usa_randomsampler():
    """El brazo 'Random' de la comparación tiene que ser de verdad
    `RandomSampler`: si `study_random` terminara usando `TPESampler` por
    dentro, la respuesta 1 compararía TPE contra sí mismo sin que ninguna
    prueba de caja negra lo note (los números seguirían pareciendo válidos).
    Esta prueba busca la asignación real a `study_random`, sigue su
    `sampler=` hasta la variable que lo construye (mismo patrón indirecto que
    `study_tpe`, necesario ahí porque esa construcción va dentro de un
    `warnings.catch_warnings`) y verifica que ES una llamada a
    `optuna.samplers.RandomSampler` con `seed=BASE_RANDOM + r` (si la misma
    semilla de sampler se repitiera entre réplicas, dejaría de ser "números
    aleatorios comunes solo dentro de cada réplica"). También verifica
    `direction="maximize"` en el propio `optuna.create_study`: minimizar
    invertiría qué trial es "el mejor" sin que ningún número impreso, por sí
    solo, lo delate."""
    arbol = _arbol_paso()
    nodo_study = _asignacion_unica_de(arbol, "study_random")
    assert isinstance(nodo_study.value, ast.Call)
    assert ast.unparse(nodo_study.value.func) == "optuna.create_study"
    kw_sampler = next((kw for kw in nodo_study.value.keywords if kw.arg == "sampler"), None)
    assert kw_sampler is not None, "optuna.create_study(...) debe pasar sampler="
    kw_direction = next((kw for kw in nodo_study.value.keywords if kw.arg == "direction"), None)
    assert kw_direction is not None, "optuna.create_study(...) debe pasar direction="
    assert ast.unparse(kw_direction.value) == "'maximize'", ast.unparse(kw_direction.value)

    sampler_expr = kw_sampler.value
    if isinstance(sampler_expr, ast.Name):
        sampler_expr = _asignacion_unica_de(arbol, sampler_expr.id).value

    assert isinstance(sampler_expr, ast.Call), ast.unparse(kw_sampler.value)
    assert ast.unparse(sampler_expr.func) == "optuna.samplers.RandomSampler", ast.unparse(
        sampler_expr
    )
    kw_seed = next((kw for kw in sampler_expr.keywords if kw.arg == "seed"), None)
    assert kw_seed is not None, ast.unparse(sampler_expr)
    assert ast.unparse(kw_seed.value) == "BASE_RANDOM + r", ast.unparse(kw_seed.value)


def test_ast_ancho_de_mlp_depende_de_n_capas():
    """Si el `for` que sugiere `ancho_i` recorriera siempre `range(1, 4)` en
    vez de `range(1, n_capas + 1)`, un trial con `n_capas=1` sortearía de
    todos modos `ancho_2`/`ancho_3` -- rompe la condición del espacio
    (define-by-run) sin que un `humo` cualquiera lo note necesariamente. Se
    ubica, dentro de `espacio`, el `for` que sugiere `ancho_i` y se verifica
    que su rango depende de la variable `n_capas`."""
    arbol = _arbol_paso()
    funcion_espacio = next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "espacio"
    )
    fors_ancho = [
        f
        for f in ast.walk(funcion_espacio)
        if isinstance(f, ast.For)
        and any(
            isinstance(n, ast.Call)
            and ast.unparse(n.func) == "trial.suggest_int"
            and any("ancho_" in ast.unparse(a) for a in n.args)
            for n in ast.walk(f)
        )
    ]
    assert fors_ancho, "no se encontró el for que sugiere ancho_i dentro de espacio()"
    for bucle in fors_ancho:
        nombres_en_iter = {n.id for n in ast.walk(bucle.iter) if isinstance(n, ast.Name)}
        assert "n_capas" in nombres_en_iter, (
            "el rango del for de los anchos debe depender de n_capas, no de una constante fija",
            ast.unparse(bucle.iter),
        )


def _algoritmo_literal_de_entrenar(nombre_variable_resultado: str) -> str:
    arbol = _arbol_paso()
    asignaciones = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and nodo.targets[0].id == nombre_variable_resultado
    ]
    assert len(asignaciones) == 1, nombre_variable_resultado
    llamada = asignaciones[0].value
    assert isinstance(llamada, ast.Call)
    assert ast.unparse(llamada.func) == "cash_paridad.entrenar"
    primer_arg = llamada.args[0]
    assert isinstance(primer_arg, ast.Dict)
    for clave, valor in zip(primer_arg.keys, primer_arg.values, strict=True):
        if isinstance(clave, ast.Constant) and clave.value == "algoritmo":
            assert isinstance(valor, ast.Constant)
            return valor.value
    raise AssertionError(
        f"no se encontró la clave 'algoritmo' en la llamada de {nombre_variable_resultado}"
    )


def test_ast_knn_y_krr_entrenan_su_propio_algoritmo():
    """Si la llamada que arma `krr_por_replica` pasara `"algoritmo": "knn"`
    por error (copiar-pegar la llamada de knn sin cambiar el algoritmo),
    `knn_por_replica` y `krr_por_replica` saldrían entrenados igual sin que el
    texto impreso lo delate. Revisión estática de la llamada real;
    test_knn_y_krr_dan_valores_distintos lo cruza en tiempo de ejecución."""
    assert _algoritmo_literal_de_entrenar("resultado_knn") == "knn"
    assert _algoritmo_literal_de_entrenar("resultado_krr") == "krr"


def _segundo_arg_de_entrenar(nombre_variable_resultado: str) -> str:
    arbol = _arbol_paso()
    asignaciones = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Name)
        and nodo.targets[0].id == nombre_variable_resultado
    ]
    assert len(asignaciones) == 1, nombre_variable_resultado
    llamada = asignaciones[0].value
    assert isinstance(llamada, ast.Call)
    assert ast.unparse(llamada.func) == "cash_paridad.entrenar"
    assert len(llamada.args) == 2, ast.unparse(llamada)
    return ast.unparse(llamada.args[1])


def test_ast_baselines_usan_semillas_r():
    """Si `resultado_knn`/`resultado_krr` se entrenaran con otra `Semillas`
    (no `semillas_r`, la misma de la réplica), la respuesta 3 dejaría de ser
    una comparación pareada de verdad -- y test_knn_y_krr_dan_valores_distintos
    no lo notaría: dos series entrenadas con otra semilla igual de consistente
    entre sí siguen siendo distintas entre ellas. Revisión estática del
    segundo argumento posicional de las dos llamadas reales."""
    assert _segundo_arg_de_entrenar("resultado_knn") == "semillas_r"
    assert _segundo_arg_de_entrenar("resultado_krr") == "semillas_r"


def test_ast_objetivo_r_usa_semillas_r():
    """`objetivo_r` es el único objective que arma este paso, y lo reusan
    TANTO `study_tpe` como `study_random` (ver
    test_ast_optimizar_usa_mismo_objetivo_y_n_total): si se construyera con
    otra cosa que `semillas_r`, las dos búsquedas dejarían de compartir los
    números aleatorios comunes de la réplica sin que ningún número impreso lo
    delate por sí solo."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "objetivo_r")
    llamada = nodo.value
    assert isinstance(llamada, ast.Call)
    assert ast.unparse(llamada.func) == "en_proceso"
    kw_semillas = next((kw for kw in llamada.keywords if kw.arg == "semillas"), None)
    assert kw_semillas is not None, "en_proceso(...) debe pasar semillas="
    assert ast.unparse(kw_semillas.value) == "semillas_r"


def test_ast_optimizar_usa_mismo_objetivo_y_n_total():
    """Dos defectos que una corrida de humo no garantiza detectar: Random con
    `n_total=2*N` (rompe "el MISMO presupuesto" entre TPE y Random) y Random
    optimizando sobre un objective DISTINTO de `objetivo_r` (rompe los datos
    CRN dentro de la réplica, ver test_ast_objetivo_r_usa_semillas_r).
    Revisión estática: las dos llamadas a `optimizar` deben pasar el mismo
    `objetivo_r` y el mismo `n_total=N`."""
    arbol = _arbol_paso()
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and ast.unparse(nodo.func) == "optimizar"
    ]
    assert len(llamadas) == 2, llamadas
    estudios_vistos = set()
    for llamada in llamadas:
        assert len(llamada.args) == 2, ast.unparse(llamada)
        nombre_estudio = ast.unparse(llamada.args[0])
        assert nombre_estudio in ("study_tpe", "study_random"), ast.unparse(llamada)
        estudios_vistos.add(nombre_estudio)
        assert ast.unparse(llamada.args[1]) == "objetivo_r", ast.unparse(llamada)
        kw_n_total = next((kw for kw in llamada.keywords if kw.arg == "n_total"), None)
        assert kw_n_total is not None, ast.unparse(llamada)
        assert ast.unparse(kw_n_total.value) == "N", ast.unparse(llamada)
    assert estudios_vistos == {"study_tpe", "study_random"}


def test_espacio_es_condicional_por_algoritmo():
    """Caja BLANCA con `optuna.trial.FixedTrial`: `_suggest` (verificado en
    el código fuente instalado, optuna/trial/_fixed.py) lanza ValueError si
    se pide un parámetro que no está en su diccionario. Si `espacio` alguna
    vez sugiriera un parámetro de OTRO algoritmo sin condicionarlo primero
    (p. ej. siempre `logistica_C` y `hgb_learning_rate`, sin importar el
    algoritmo sorteado), construir el FixedTrial con SOLO los parámetros del
    algoritmo elegido hace que esa llamada de más falle -- y ninguna prueba
    de caja negra lo garantiza, porque el espacio real (Optuna real, no
    FixedTrial) sí sugeriría esos parámetros de más sin protestar."""
    ns = _extraer_al_namespace({"espacio", "_RANGOS_ARBOL"})
    casos = {
        "logistica": {"algoritmo": "logistica", "logistica_C": 1.0},
        "bosque": {
            "algoritmo": "bosque",
            "bosque_n_arboles": 50,
            "bosque_profundidad_max": 5,
            "bosque_min_muestras_hoja": 2,
        },
        "hgb": {
            "algoritmo": "hgb",
            "hgb_learning_rate": 0.1,
            "hgb_max_hojas": 31,
            "hgb_max_iter": 50,
        },
        "mlp_1capa": {
            "algoritmo": "mlp",
            "n_capas": 1,
            "ancho_1": 16,
            "mlp_learning_rate_init": 1e-3,
            "mlp_alpha": 1e-4,
        },
        "mlp_2capas": {
            "algoritmo": "mlp",
            "n_capas": 2,
            "ancho_1": 16,
            "ancho_2": 32,
            "mlp_learning_rate_init": 1e-3,
            "mlp_alpha": 1e-4,
        },
    }
    for nombre, params in casos.items():
        trial = optuna.trial.FixedTrial(dict(params))
        config = ns["espacio"](trial)
        assert set(config) == set(params), (nombre, config, params)


def _clave_subscript_de(nombre_dict: str) -> str:
    arbol = _arbol_paso()
    asignaciones = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Subscript)
        and isinstance(nodo.targets[0].value, ast.Name)
        and nodo.targets[0].value.id == nombre_dict
    ]
    assert len(asignaciones) == 1, f"debe haber una única asignación a {nombre_dict}[r]"
    nodo = asignaciones[0]
    assert isinstance(nodo.value, ast.Subscript), f"{nombre_dict}[r] debe leer una clave de dict"
    clave = nodo.value.slice
    assert isinstance(clave, ast.Constant), ast.unparse(nodo.value)
    return clave.value


def test_ast_knn_y_krr_se_miden_con_exactitud_verdadera():
    """Si `knn_por_replica`/`krr_por_replica` leyeran `exactitud_val` en vez de
    `exactitud_verdadera`, la comparación de la respuesta 3 dejaría de ser
    honesta (compararía contra la métrica que ve la propia validación, no
    contra el oráculo) sin que compararlos entre sí lo note por sí solo:
    los dos podrían estar mal de la misma manera. Revisión estática de la
    clave que lee cada asignación."""
    assert _clave_subscript_de("knn_por_replica") == "exactitud_verdadera"
    assert _clave_subscript_de("krr_por_replica") == "exactitud_verdadera"


def test_ast_dispositivo_se_resuelve_desde_defaults_y_fijos_mergeados():
    """Resolver el dispositivo impreso desde `cash_paridad.DEFAULTS` puro
    (sin mergear `FIJOS`) imprimiría un dispositivo que podría no ser el que
    de verdad se entrena, el día que `FIJOS` trajera su propia clave
    'dispositivo'."""
    assert "cash_paridad.DEFAULTS, **FIJOS" in _codigo_paso(), (
        "la resolución del dispositivo impreso debe mergear FIJOS sobre DEFAULTS"
    )


# --- métrica honesta: las series de TPE y Random también deben leer el oráculo ---


def _rhs_de_asignacion_subscript(nombre_dict: str) -> ast.expr:
    arbol = _arbol_paso()
    candidatos = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Assign)
        and len(nodo.targets) == 1
        and isinstance(nodo.targets[0], ast.Subscript)
        and isinstance(nodo.targets[0].value, ast.Name)
        and nodo.targets[0].value.id == nombre_dict
    ]
    assert len(candidatos) == 1, f"debe haber una única asignación a {nombre_dict}[r]"
    return candidatos[0].value


def test_ast_series_tpe_y_random_leen_el_oraculo():
    """Caja BLANCA que protege exactamente lo que `knn_por_replica`/
    `krr_por_replica` ya protegen (test_ast_knn_y_krr_se_miden_con_exactitud_verdadera),
    pero para los otros dos brazos: si `tpe_por_replica[r]`/
    `random_por_replica[r]` leyeran `elegido_tpe.value`/`elegido_random.value`
    (`exactitud_val`, lo que SÍ ve la búsqueda) en vez de
    `user_attrs['exactitud_verdadera']` (el oráculo), la respuesta 1
    compararía validaciones infladas entre sí, y la respuesta 3 compararía la
    validación de TPE contra el oráculo de knn/krr -- una comparación honesta
    de un lado y no del otro, sin que ningún número impreso lo delate por sí
    solo (los dos números seguirían estando en [0, 1])."""
    for nombre_dict, nombre_elegido in (
        ("tpe_por_replica", "elegido_tpe"),
        ("random_por_replica", "elegido_random"),
    ):
        rhs = _rhs_de_asignacion_subscript(nombre_dict)
        assert ast.unparse(rhs) == f"{nombre_elegido}.user_attrs['exactitud_verdadera']", (
            ast.unparse(rhs)
        )


# --- respuesta 1/2/3: las frases en prosa se ejercen en sus 4 ramas, no solo la de humo ---


def test_ast_frase_veredicto_cuatro_ramas():
    """Caja BLANCA: extrae `_frase_veredicto` del AST del paso y la ejerce con
    un `estadistica.Pareado` sintético para cada uno de los cuatro veredictos
    posibles. `test_respuesta_3_el_verbo_corresponde_al_veredicto` (caja
    negra) solo ejerce la rama que salió en ESA corrida de humo -- con solo
    2 réplicas, ningún humo garantiza pasar por las cuatro; si el verbo de
    una rama que humo no visita se desincronizara del veredicto (p. ej.
    'mejor' devolviendo "pierde contra..."), esa caja negra no lo notaría."""
    ns = _extraer_al_namespace({"_frase_veredicto"})
    esperado = {
        "mejor": "le gana a knn",
        "peor": "pierde contra knn",
        "equivalente": "empata con knn",
        "no concluyente": "no se distingue de knn",
    }
    for veredicto, prefijo in esperado.items():
        comp = estadistica.Pareado(
            n=5,
            diferencia_media=0.02,
            de_dif=0.01,
            ic_bajo=0.01,
            ic_alto=0.03,
            p_valor=0.01,
            wilcoxon_p=None,
            rho=0.5,
            veredicto=veredicto,
            delta=0.01,
            direccion="maximizar",
            confianza=0.95,
        )
        frase = ns["_frase_veredicto"]("knn", comp)
        assert frase.startswith(prefijo), (veredicto, frase)
        # La diferencia impresa (+0.020, de diferencia_media=0.02) debe aparecer
        # en las cuatro ramas, y "no concluyente" además debe nombrar cuántas
        # réplicas hay (R=5 en este namespace, ver _extraer_al_namespace):
        # comprobar solo el verbo dejaría pasar una rama que dijera "los datos
        # no alcanzan" sin nombrar ni las réplicas ni la diferencia real.
        assert "+0.020" in frase, frase
        if veredicto == "no concluyente":
            assert "con 5 réplicas" in frase, frase


def test_frase_veredicto_mejor_advierte_si_ic_no_descarta_ventaja_menor_que_delta():
    """Caja BLANCA: dos `Pareado` con veredicto 'mejor' pero con el borde
    inferior del intervalo en lados distintos de `delta` -- solo el que cae
    DENTRO de la zona de indiferencia debe llevar la advertencia; el otro
    (ganancia ya mayor que delta) no debe decir que el intervalo "no
    descarta" nada, porque sí lo descarta."""
    ns = _extraer_al_namespace({"_frase_veredicto"})
    base = dict(
        n=5,
        diferencia_media=0.02,
        de_dif=0.01,
        ic_alto=0.03,
        p_valor=0.01,
        wilcoxon_p=None,
        rho=0.5,
        veredicto="mejor",
        delta=0.01,
        direccion="maximizar",
        confianza=0.95,
    )
    comp_dentro = estadistica.Pareado(ic_bajo=0.005, **base)
    comp_fuera = estadistica.Pareado(ic_bajo=0.015, **base)
    frase_dentro = ns["_frase_veredicto"]("knn", comp_dentro)
    frase_fuera = ns["_frase_veredicto"]("knn", comp_fuera)
    assert "el intervalo no descarta una ventaja menor que delta" in frase_dentro, frase_dentro
    assert "el intervalo no descarta" not in frase_fuera, frase_fuera


def test_ast_frase_arranque_dos_ramas():
    """Caja BLANCA: extrae `_frase_arranque` del AST del paso y la ejerce con
    los dos casos reales del taller (`humo`: n=4 < n_startup=10, TPE nunca
    actúa guiado; `taller`: n=15 -- PRESUPUESTOS['taller'] real -- > n_startup=10,
    actúa guiado en el resto). `humo` siempre cae en la primera rama -- si la
    rama de `taller` dijera "los {n} restantes" en vez de "los {n -
    n_aleatorios} restantes" (mintiendo sobre cuántos trials arrancaron
    guiados), ninguna prueba que solo corra `humo` lo notaría."""
    ns = _extraer_al_namespace({"_frase_arranque"})
    frase_humo = ns["_frase_arranque"](4, 10)
    assert "nunca llegó a actuar de forma guiada" in frase_humo, frase_humo
    # "evaluaciones completAS"/"trials completOS": la frase cuenta solo los
    # trials que SÍ completaron -- si un trial divergiera y terminara FAIL, el
    # estudio real tendría más intentos que `n`, y esta frase no debe leerse
    # como si contara ese total.
    assert "evaluaciones completas" in frase_humo, frase_humo
    frase_taller = ns["_frase_arranque"](15, 10)
    assert "trials completos" in frase_taller, frase_taller
    assert "los primeros 10 fueron" in frase_taller, frase_taller
    assert "los 5 restantes usaron el modelo" in frase_taller, frase_taller


def test_ast_frase_arranque_se_llama_con_n_y_n_startup_reales():
    """Revisión estática de la única asignación real: `frase_arranque` debe
    venir de `_frase_arranque(N, N_STARTUP_TRIALS_TPE)` -- las dos globales
    reales del paso, no dos literales copiados a mano. La prueba de caja
    blanca de arriba ejerce `_frase_arranque` por su cuenta con (4, 10) y
    (15, 10) y no nota si el paso, en la línea real, la llamara con otros
    argumentos (p. ej. `_frase_arranque(N, N)`, que en taller haría creer que
    TPE nunca actuó guiado con N=15): ese error deja pasar cualquier
    prueba que solo ejerza la función ya extraída."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "frase_arranque")
    assert ast.unparse(nodo.value) == "_frase_arranque(N, N_STARTUP_TRIALS_TPE)", ast.unparse(
        nodo.value
    )


def test_ast_frase_conteo_dos_ramas():
    """Caja BLANCA: extrae `_frase_conteo` del AST del paso y la ejerce con un
    conteo sintético de ganador único y otro de empate. Si la rama de ganador
    único sumara +1 al conteo impreso (mintiendo sobre cuántas réplicas ganó
    en realidad), esta prueba lo nota sin depender de qué algoritmo ganó en
    la corrida real de humo (con R=2, esa corrida casi siempre empata). El
    intervalo de confianza 95% Wilson que acompaña el conteo, y el p-valor
    cara a cara contra el segundo lugar, se recalculan aquí con
    `estadistica.ic_proporcion` y
    `scipy.stats.binomtest` directo -- nunca un número fijo, que se
    desincronizaría si `_frase_conteo` cambiara de método (ver
    test_frase_conteo_ganador_unico_compara_contra_segundo_lugar para las dos
    ramas -- 'sí'/'no se distingue' -- de esa comparación)."""
    ns = _extraer_al_namespace({"_frase_conteo"})
    conteo_unico = {"logistica": 0, "bosque": 1, "hgb": 3, "mlp": 1}
    frase_unico = ns["_frase_conteo"](conteo_unico, 5)
    ic_bajo, ic_alto = estadistica.ic_proporcion(3, 5)
    assert (
        f"'hgb' (3 de 5 réplicas, intervalo de confianza 95% Wilson de la proporción "
        f"[{ic_bajo:.2f}, {ic_alto:.2f}])" in frase_unico
    ), frase_unico
    assert str(conteo_unico) in frase_unico, frase_unico
    # El segundo lugar real ('bosque' y 'mlp', ambos con 1 de 5) se nombra y
    # se compara cara a cara con el ganador (prueba de signos binomial).
    ic2_bajo, ic2_alto = estadistica.ic_proporcion(1, 5)
    p_cara_a_cara = binomtest(3, 3 + 1).pvalue
    assert "'bosque' y 'mlp'" in frase_unico, frase_unico
    assert f"[{ic2_bajo:.2f}, {ic2_alto:.2f}]" in frase_unico, frase_unico
    assert f"p={p_cara_a_cara:.4f}" in frase_unico, frase_unico
    assert (p_cara_a_cara < 0.05) == ("no se distingue" not in frase_unico), frase_unico

    conteo_empate = {"logistica": 2, "bosque": 2, "hgb": 0, "mlp": 0}
    frase_empate = ns["_frase_conteo"](conteo_empate, 4)
    ic_bajo_e, ic_alto_e = estadistica.ic_proporcion(2, 4)
    assert "Empate entre 'logistica' y 'bosque'" in frase_empate, frase_empate
    assert (
        f"(2 de 4 réplicas cada uno, intervalo de confianza 95% Wilson de la proporción "
        f"[{ic_bajo_e:.2f}, {ic_alto_e:.2f}])" in frase_empate
    ), frase_empate
    # La rama de empate no compara "segundo lugar": los dos ya son el máximo.
    assert "distingue" not in frase_empate, frase_empate


def test_frase_conteo_ganador_unico_compara_contra_segundo_lugar():
    """Caja BLANCA: dos casos sintéticos de ganador único, uno donde la
    prueba de signos binomial cara a cara NO distingue al ganador del segundo
    lugar (3 de 5 contra 2 de 5: `binomtest(3, 5)` da un p-valor muy por
    encima de 0.05) y otro donde SÍ lo distingue (20 de 20 contra 0 de 20:
    `binomtest(20, 20)` da un p-valor minúsculo) -- si `_frase_conteo` dejara
    de comparar contra el segundo lugar real (p. ej. comparando siempre
    contra 0, o comparando por intervalo de Wilson en vez de la prueba real
    -- ver el docstring del módulo para por qué eso no es una prueba), esta
    prueba lo nota en las dos direcciones. Con R=5 y un 5 a 0, la comparación
    por intervalo de Wilson NUNCA se traslapaba ([0.57, 1.00] contra [0.00,
    0.43]) y aun así la prueba de signos exacta da p=0.0625 -- por eso la
    decisión real usa `binomtest`, no el traslape de los dos intervalos."""
    ns = _extraer_al_namespace({"_frase_conteo"})

    conteo_cercano = {"logistica": 3, "bosque": 2, "hgb": 0, "mlp": 0}
    frase_cercano = ns["_frase_conteo"](conteo_cercano, 5)
    p_cercano = binomtest(3, 3 + 2).pvalue
    assert p_cercano >= 0.05, "este caso sintético debe NO distinguirse"
    assert f"p={p_cercano:.4f}" in frase_cercano, frase_cercano
    assert "no se distingue con 5 réplicas" in frase_cercano, frase_cercano

    conteo_lejano = {"logistica": 20, "bosque": 0, "hgb": 0, "mlp": 0}
    frase_lejano = ns["_frase_conteo"](conteo_lejano, 20)
    p_lejano = binomtest(20, 20 + 0).pvalue
    assert p_lejano < 0.05, "este caso sintético SÍ debe distinguirse"
    assert f"p={p_lejano:.4f}" in frase_lejano, frase_lejano
    assert "sí se distingue" in frase_lejano, frase_lejano
    assert "no se distingue" not in frase_lejano, frase_lejano


def test_prueba_de_signos_5_a_0_no_es_significativa_con_r5():
    """Con R=5, un resultado 5 a 0 (el caso más extremo posible con solo 5
    réplicas) da p=0.0625 en la prueba de signos exacta -- por encima de
    0.05 -- así que `_frase_conteo` NUNCA debe imprimir "sí se distingue" con
    solo 5 réplicas, sin importar qué tan lopsided sea el conteo: el traslape
    de dos intervalos de Wilson SÍ se separa en ese caso ([0.57, 1.00] contra
    [0.00, 0.43]), y por eso la decisión real usa `binomtest`, nunca ese
    traslape."""
    ns = _extraer_al_namespace({"_frase_conteo"})
    conteo_5_a_0 = {"logistica": 5, "bosque": 0, "hgb": 0, "mlp": 0}
    frase = ns["_frase_conteo"](conteo_5_a_0, 5)
    p = binomtest(5, 5).pvalue
    assert p == pytest.approx(0.0625), p
    assert "no se distingue" in frase, frase


# --- el brazo TPE también tiene su configuración protegida, no solo el brazo Random ---


def test_ast_sampler_tpe_usa_group_y_la_constante_impresa():
    """Caja BLANCA, simétrica a `test_ast_study_random_usa_randomsampler`
    (que protege el brazo Random): si `sampler_tpe` se construyera con un
    `n_startup_trials` real distinto de `N_STARTUP_TRIALS_TPE` -- la misma
    constante que la respuesta 1 usa para describir el arranque -- la
    respuesta 1 describiría un sampler distinto del que de verdad corrió,
    sin que ningún número impreso lo note. Lo mismo si faltara
    `multivariate=True, group=True` (lo que hace que TPE entienda el espacio
    condicional, ver el markdown de arriba), si la semilla del sampler no
    dependiera de `r` (rompería "semillas de sampler distintas" entre
    réplicas), o si `study_tpe` terminara usando otro sampler que no sea
    `sampler_tpe`."""
    arbol = _arbol_paso()
    nodo_sampler = _asignacion_unica_de(arbol, "sampler_tpe")
    llamada = nodo_sampler.value
    assert isinstance(llamada, ast.Call)
    assert ast.unparse(llamada.func) == "optuna.samplers.TPESampler", ast.unparse(llamada)
    kw = {k.arg: ast.unparse(k.value) for k in llamada.keywords}
    assert kw.get("n_startup_trials") == "N_STARTUP_TRIALS_TPE", kw
    assert kw.get("multivariate") == "True", kw
    assert kw.get("group") == "True", kw
    assert kw.get("seed") == "BASE_TPE + r", kw

    nodo_study = _asignacion_unica_de(arbol, "study_tpe")
    assert isinstance(nodo_study.value, ast.Call)
    assert ast.unparse(nodo_study.value.func) == "optuna.create_study"
    kw_sampler_study = next((kw for kw in nodo_study.value.keywords if kw.arg == "sampler"), None)
    assert kw_sampler_study is not None, "optuna.create_study(...) debe pasar sampler="
    assert ast.unparse(kw_sampler_study.value) == "sampler_tpe", ast.unparse(kw_sampler_study.value)
    kw_direction_study = next(
        (kw for kw in nodo_study.value.keywords if kw.arg == "direction"), None
    )
    assert kw_direction_study is not None, "optuna.create_study(...) debe pasar direction="
    assert ast.unparse(kw_direction_study.value) == "'maximize'", ast.unparse(
        kw_direction_study.value
    )


# --- espacio: solo los cuatro algoritmos buscados, con un Trial real (no FixedTrial) ---


def test_espacio_solo_ofrece_los_cuatro_buscados():
    """Caja BLANCA con un `Trial` REAL (no `FixedTrial`, que no valida las
    opciones de un `suggest_categorical` -- construir el `FixedTrial` con
    solo 4 claves no impide que `espacio()` internamente ofrezca una quinta
    opción): si `espacio()` ofreciera 'knn' o 'krr' -- que deben ser solo
    baselines sin búsqueda -- como opción de `algoritmo`, esta prueba lo nota
    mirando `trial.distributions` directamente.
    `test_espacio_es_condicional_por_algoritmo` no lo detecta: acepta
    cualquier lista de opciones mientras el valor fijado esté entre ellas."""
    ns = _extraer_al_namespace({"espacio", "_RANGOS_ARBOL"})
    estudio = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = estudio.ask()
    ns["espacio"](trial)
    choices = trial.distributions["algoritmo"].choices
    assert tuple(choices) == ("logistica", "bosque", "hgb", "mlp"), choices


# --- baselines: el primer argumento completo a cash_paridad.entrenar, no solo 'algoritmo' ---


def test_ast_baselines_reciben_exactamente_fijos_y_su_algoritmo():
    """`_algoritmo_literal_de_entrenar` (arriba, usada por
    test_ast_knn_y_krr_entrenan_su_propio_algoritmo) solo mira la clave
    'algoritmo' del diccionario: un baseline con una clave EXTRA (p. ej.
    `knn_n_vecinos=1`, que ya no serían los defaults de cash_paridad) pasaría
    esa prueba sin que se note. Esta compara el primer
    argumento COMPLETO, como texto, contra exactamente
    `{**FIJOS, 'algoritmo': '<alg>'}`."""
    arbol = _arbol_paso()
    for nombre_variable, alg in (("resultado_knn", "knn"), ("resultado_krr", "krr")):
        nodo = _asignacion_unica_de(arbol, nombre_variable)
        llamada = nodo.value
        assert isinstance(llamada, ast.Call)
        primer_arg = ast.unparse(llamada.args[0])
        assert primer_arg == f"{{**FIJOS, 'algoritmo': '{alg}'}}", primer_arg


# --- Holm-Bonferroni: la nota dice si ajustar por multiplicidad cambiaría algún veredicto ---


@pytest.mark.gpu
def test_respuesta_holm_dice_si_algun_veredicto_cambiaria(salida_humo):
    """La nota de Holm-Bonferroni debe decir explícitamente si ajustar por
    multiplicidad cambiaría el llamado "mejor"/"peor" de alguna comparación,
    en vez de una frase fija que el código no calcula. La protección de que
    el cálculo real es el correcto es de caja BLANCA, ver
    test_ast_cambia_holm_se_calcula_desde_p_holm_y_veredictos."""
    m = re.search(
        r"con Holm-Bonferroni: p=\d\.\d+ vs Random, p=\d\.\d+ vs knn, "
        r"p=\d\.\d+ vs krr -- (?P<frase>[^\n]+)\)\s*$",
        salida_humo,
        re.MULTILINE,
    )
    assert m, salida_humo
    frase = m.group("frase")
    assert ("cambiaría contra" in frase) or ("no cambia ningún veredicto" in frase), frase


def test_ast_lineas_p_holm_empareja_etiqueta_y_p_por_zip():
    """Revisión estática de `_lineas_p_holm` (lo que arma cada 'p=... vs
    ...' de la línea impresa): debe leer `_etiquetas_holm` y `p_holm` EN
    PARALELO con `zip(..., strict=True)`, nunca por índice literal
    (`p_holm[1]` bajo la etiqueta fija 'knn'). Con índices literales, invertir
    el orden de `_COMPARACIONES_HOLM` (o solo trasponer dos índices en la
    f-string, como cruzar `p_holm[1]`/`p_holm[2]` bajo 'knn'/'krr') deja cada
    número bajo la etiqueta de la comparación vecina sin que ningún test de
    caja negra lo note -- los tres siguen siendo p-valores válidos en
    [0, 1]."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "_lineas_p_holm")
    assert isinstance(nodo.value, ast.ListComp), ast.unparse(nodo.value)
    assert ast.unparse(nodo.value.elt) == "f'p={p:.3f} vs {etiqueta}'", ast.unparse(nodo.value.elt)
    assert len(nodo.value.generators) == 1
    generador = nodo.value.generators[0]
    assert ast.unparse(generador.target) == "(etiqueta, p)", ast.unparse(generador.target)
    assert ast.unparse(generador.iter) == "zip(_etiquetas_holm, p_holm, strict=True)", ast.unparse(
        generador.iter
    )


def test_ast_cambia_holm_se_calcula_desde_p_holm_y_veredictos():
    """Revisión estática: `cambia_holm` debe comparar, para las tres
    comparaciones en el mismo orden que `p_holm` (Random, knn, krr), si el
    veredicto es 'mejor'/'peor' contra si el p ajustado es < 0.05 -- nunca
    una lista fija ni un booleano hardcodeado, que dejaría la nota
    describiendo algo que el código no calculó."""
    arbol = _arbol_paso()
    nodo = _asignacion_unica_de(arbol, "cambia_holm")
    assert isinstance(nodo.value, ast.ListComp), ast.unparse(nodo.value)
    elemento = ast.unparse(nodo.value.elt)
    # Comparación EXACTA (no solo "contiene 'veredicto in'"): un `!=` volteado a
    # `==` invertiría qué comparaciones se marcan como "cambiaría" sin dejar de
    # mencionar "veredicto in"/"0.05", así que una sub-cadena sola no alcanza.
    assert elemento == "(c.veredicto in ('mejor', 'peor')) != (p < 0.05)", elemento
    generador = nodo.value.generators[0]
    assert ast.unparse(generador.iter) == "zip(_COMPARACIONES_HOLM, p_holm, strict=True)", (
        ast.unparse(generador.iter)
    )


# --- _frase_holm: sus dos ramas, y que la frase impresa de verdad venga de ella ---


def test_ast_frase_holm_dos_ramas():
    """Caja BLANCA, mismo patrón que test_ast_frase_arranque_dos_ramas /
    test_ast_frase_conteo_dos_ramas: extrae `_frase_holm` del AST del paso y
    la ejerce con listas sintéticas. Invertir `any(cambia)` a `not any(...)`
    (o vaciar la lista de afectados) seguiría imprimiendo una frase con las
    palabras clave correctas pero al revés o vacía, y con solo 2 réplicas de
    humo ninguna corrida real garantiza pasar por la rama de 'sí cambia'."""
    ns = _extraer_al_namespace({"_frase_holm"})
    frase_ninguno = ns["_frase_holm"]([False, False, False], ("Random", "knn", "krr"))
    assert frase_ninguno == (
        "ajustar por Holm-Bonferroni no cambia ningún veredicto 'mejor'/'peor' con estos datos"
    ), frase_ninguno

    frase_alguno = ns["_frase_holm"]([False, True, True], ("Random", "knn", "krr"))
    assert frase_alguno == (
        "al ajustar por Holm-Bonferroni el veredicto 'mejor'/'peor' cambiaría contra: knn, krr"
    ), frase_alguno


@pytest.mark.gpu
def test_respuesta_holm_frase_coincide_con_cambia_recalculado(salida_humo):
    """Recalcula, desde los tres veredictos impresos (respuestas 1 y 3) y los
    tres p ajustados de la línea de Holm-Bonferroni, exactamente qué
    comparaciones "cambiarían" (`(veredicto in ('mejor','peor')) != (p < 0.05)`,
    la misma fórmula que protege test_ast_cambia_holm_se_calcula_desde_p_holm_y_veredictos
    de caja blanca), y exige que la frase impresa sea la que `_frase_holm`
    produciría con esa lista -- no solo que mencione "cambiaría"/"no cambia"
    (eso ya lo cubre test_respuesta_holm_dice_si_algun_veredicto_cambiaria).
    Si `cambia_holm` invirtiera su `!=` a `==`, esta prueba lo nota aunque la
    corrida real de humo no cambie ningún veredicto: la lista recalculada
    (con veredictos y p reales) puede diferir de la lista invertida sin que
    ninguna quede vacía."""
    m1 = _RE_R1.search(salida_humo)
    m3 = _RE_R3.search(salida_humo)
    m_holm = re.search(
        r"con Holm-Bonferroni: p=(?P<p1>\d\.\d+) vs Random, p=(?P<p2>\d\.\d+) vs knn, "
        r"p=(?P<p3>\d\.\d+) vs krr -- (?P<frase>[^\n]+)\)\s*$",
        salida_humo,
        re.MULTILINE,
    )
    assert m1 and m3 and m_holm, salida_humo

    veredictos = (m1.group("veredicto"), m3.group("veredicto_knn"), m3.group("veredicto_krr"))
    p_holm = [float(m_holm.group(g)) for g in ("p1", "p2", "p3")]
    cambia = [
        (v in ("mejor", "peor")) != (p < 0.05) for v, p in zip(veredictos, p_holm, strict=True)
    ]

    ns = _extraer_al_namespace({"_frase_holm"})
    esperado = ns["_frase_holm"](cambia, ("Random", "knn", "krr"))
    assert m_holm.group("frase") == esperado, (m_holm.group("frase"), esperado, cambia)


# --- respuesta 2: el conteo cuenta ganadores de TPE, nunca de Random ---


def test_ast_algoritmo_ganador_tpe_usa_params_de_elegido_tpe():
    """Si `algoritmo_ganador_tpe.append(...)` recibiera
    `elegido_random.params['algoritmo']` en vez de `elegido_tpe.params[...]`
    (copiar-pegar entre las dos ramas, después de definir `elegido_random`), la
    respuesta 2 contaría los ganadores de RANDOM en vez de los de TPE --
    ninguna prueba de caja negra lo nota, porque el número seguiría siendo un
    algoritmo válido de los cuatro buscados y el conteo seguiría sumando `R`.
    Revisión estática de la única llamada real."""
    arbol = _arbol_paso()
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr == "append"
        and isinstance(nodo.func.value, ast.Name)
        and nodo.func.value.id == "algoritmo_ganador_tpe"
    ]
    assert len(llamadas) == 1, llamadas
    assert len(llamadas[0].args) == 1, ast.unparse(llamadas[0])
    assert ast.unparse(llamadas[0].args[0]) == "elegido_tpe.params['algoritmo']", ast.unparse(
        llamadas[0]
    )


# --- FIJOS: literal vacío de verdad, y se pasa tal cual a en_proceso ---


def test_ast_fijos_es_literal_vacio_y_se_pasa_tal_cual_a_en_proceso():
    """`FIJOS` debe ser el diccionario vacío -- sin ninguna clave de problema
    (`n_train`, `n_val`, ...), de oráculo (`n_oraculo`) ni `dispositivo` -- y
    `en_proceso(...)` debe recibir exactamente esa variable en `fijos=`, no una
    copia con overrides (p. ej. `{**FIJOS, "dispositivo": "cuda:1"}`, que
    entrenaría toda la búsqueda en otra tarjeta mientras `_DISPOSITIVO_CASH` --
    lo que de verdad se imprime -- sigue siendo el resuelto sobre `FIJOS`
    solo, sin que ningún número lo delate). test_paso_usa_realmente_la_tarjeta_de_video
    es la protección de caja NEGRA equivalente, ejercitando la corrida real."""
    arbol = _arbol_paso()
    nodo_fijos = _asignacion_unica_de(arbol, "FIJOS")
    assert isinstance(nodo_fijos.value, ast.Dict), ast.unparse(nodo_fijos.value)
    assert nodo_fijos.value.keys == [], ast.unparse(nodo_fijos.value)

    llamada_en_proceso = next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and ast.unparse(nodo.func) == "en_proceso"
    )
    kw_fijos = next((kw for kw in llamada_en_proceso.keywords if kw.arg == "fijos"), None)
    assert kw_fijos is not None, "en_proceso(...) debe pasar fijos="
    assert ast.unparse(kw_fijos.value) == "FIJOS", ast.unparse(kw_fijos.value)


@pytest.mark.gpu
def test_paso_usa_realmente_la_tarjeta_de_video(tmp_path):
    """Protege contra "el texto dice dispositivo=cuda pero ningún trial entrenó en la
    tarjeta de video". Envuelve las dos funciones reales de entrenamiento de
    `cash_paridad`/`paridad_torch` con un contador y también `cash_paridad.entrenar`, para
    contar por `config["algoritmo"]`.

    `knn` y `krr` entrenan siempre en la tarjeta, así que el conteo agregado no basta: la
    aserción fuerte es que `logistica + bosque + hgb + mlp`, los algoritmos que sortea la
    búsqueda, sumen al menos `2*R*N` llamadas (`>=` porque un trial que diverge se
    reintenta). El paso corre con `runpy.run_path` en un subproceso nuevo, para no dejar
    parches ni el contexto CUDA en el proceso de pytest."""
    registro = tmp_path / "conteo.json"
    resultados = tmp_path / "resultados"
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_REGISTRO_CONTEO": str(registro),
    }
    envoltura = (
        "import json\n"
        "import os\n"
        "import runpy\n"
        "from collections import Counter\n"
        "\n"
        "from taller_hpo.problemas import cash_paridad, paridad_torch\n"
        "\n"
        "_conteo = Counter()\n"
        "\n"
        "\n"
        "def _envolver(nombre, original):\n"
        "    def _envuelta(*args, **kwargs):\n"
        "        _conteo[nombre] += 1\n"
        "        return original(*args, **kwargs)\n"
        "\n"
        "    return _envuelta\n"
        "\n"
        "\n"
        "_entrenar_original = cash_paridad.entrenar\n"
        "\n"
        "\n"
        "def _entrenar_contado(config, *args, **kwargs):\n"
        "    _conteo[config['algoritmo']] += 1\n"
        "    return _entrenar_original(config, *args, **kwargs)\n"
        "\n"
        "\n"
        "cash_paridad._ajustar_baseline_gpu = _envolver(\n"
        "    'gpu_baseline', cash_paridad._ajustar_baseline_gpu\n"
        ")\n"
        "paridad_torch.entrenar_red = _envolver('gpu_mlp', paridad_torch.entrenar_red)\n"
        "cash_paridad.entrenar = _entrenar_contado\n"
        f"_ns = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "_registro = {'conteo': dict(_conteo), 'presupuesto_humo': _ns['PRESUPUESTOS']['humo']}\n"
        "with open(os.environ['TALLER_REGISTRO_CONTEO'], 'w', encoding='utf-8') as _f:\n"
        "    json.dump(_registro, _f)\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", envoltura], env=entorno, capture_output=True, text=True, timeout=110
    )
    assert resultado.returncode == 0, resultado.stderr

    registrado = json.loads(registro.read_text(encoding="utf-8"))
    conteo = registrado["conteo"]
    m = re.search(r"cash_paridad: dispositivo=(cuda)", resultado.stdout)
    assert m, resultado.stdout

    # r_humo, n_humo salen del propio PRESUPUESTOS['humo'] del paso (leído del
    # namespace que runpy.run_path devuelve, arriba), nunca de un par fijo a
    # mano: si PRESUPUESTOS['humo'] cambiara (por ejemplo, más réplicas o más
    # evaluaciones por sampler), el piso de esta prueba crece con él en vez de
    # quedar atrás y dejar pasar una búsqueda que entrenó de menos.
    # 2*R*N: TPE + Random, cada una N evaluaciones por réplica; 2*R: knn + krr,
    # una vez cada una por réplica.
    r_humo, n_humo = registrado["presupuesto_humo"]
    llamadas_busqueda = sum(conteo.get(a, 0) for a in ("logistica", "bosque", "hgb", "mlp"))
    assert llamadas_busqueda >= 2 * r_humo * n_humo, conteo
    assert conteo.get("gpu_baseline", 0) + conteo.get("gpu_mlp", 0) >= (
        2 * r_humo * n_humo + 2 * r_humo
    ), conteo


# --- _frase_tamano_problema: computada de verdad, no una frase fija ---


def test_frase_tamano_problema_dos_ramas():
    """Caja BLANCA: `_frase_tamano_problema` debe comparar de verdad
    `n_train`/`n_val` contra `taller_hpo.problemas.juguete.DEFAULTS` en vez de
    afirmar una frase fija -- se ejerce con una config sintética igual y otras
    distintas (por `n_train`, y por separado por `n_val`: si la comparación
    solo mirara `n_train`, un cambio de `n_val` seguiría diciendo "el mismo
    tamaño que el resto del taller"). `rotar` se compara APARTE (ver también
    test_frase_tamano_problema_rotar_siempre_se_reporta): un cambio de
    n_train/n_val no debe afectar la frase de rotar, y viceversa."""
    ns = _extraer_al_namespace({"_frase_tamano_problema"})
    defaults_juguete = {"n_train": 2000, "n_val": 500, "rotar": False}

    igual = ns["_frase_tamano_problema"](
        {"n_train": 2000, "n_val": 500, "rotar": False}, defaults_juguete
    )
    assert igual.startswith("el mismo tamaño que el resto del taller"), igual

    distinto = ns["_frase_tamano_problema"](
        {"n_train": 600, "n_val": 500, "rotar": False}, defaults_juguete
    )
    assert "un tamaño distinto al resto del taller" in distinto, distinto
    assert "n_train=600" in distinto, distinto
    assert "el resto del taller usa 2000" in distinto, distinto

    distinto_val = ns["_frase_tamano_problema"](
        {"n_train": 2000, "n_val": 300, "rotar": False}, defaults_juguete
    )
    assert "un tamaño distinto al resto del taller" in distinto_val, distinto_val
    assert "n_val=300" in distinto_val, distinto_val
    assert "el resto del taller usa 500" in distinto_val, distinto_val


def test_frase_tamano_problema_rotar_siempre_se_reporta():
    """Caja BLANCA: `rotar` no es un tamaño -- es una transformación del
    problema -- y `cash_paridad.DEFAULTS` usa `rotar=True` mientras
    `taller_hpo.problemas.juguete.DEFAULTS` usa `rotar=False` (ver el
    docstring de `cash_paridad`): la config real de este paso SIEMPRE cae en
    la rama "rotado", y esta prueba lo ejerce en las dos direcciones (rotar
    igual y rotar distinto) para que un `_frase_tamano_problema` que dejara
    de comparar 'rotar' no pase en silencio."""
    ns = _extraer_al_namespace({"_frase_tamano_problema"})
    defaults_juguete = {"n_train": 2000, "n_val": 500, "rotar": False}

    igual_rotar = ns["_frase_tamano_problema"](
        {"n_train": 2000, "n_val": 500, "rotar": False}, defaults_juguete
    )
    assert "rotar=False, igual que el resto del taller" in igual_rotar, igual_rotar

    rotado = ns["_frase_tamano_problema"](
        {"n_train": 2000, "n_val": 500, "rotar": True}, defaults_juguete
    )
    assert "rotado (rotar=True)" in rotado, rotado
    assert "a diferencia del resto del taller (rotar=False)" in rotado, rotado


# --- réplicas: los datos Y las semillas de sampler cambian entre ellas, no solo dentro ---


def test_ast_bucle_replicas_itera_semillas_de_sin_fijar_datos():
    """El `for` que arma cada réplica debe iterar exactamente sobre
    `enumerate(semillas_de(BASE_DATOS, R))`, SIN pasar `datos=`: fijar los
    datos entre réplicas (en vez de solo compartirlos dentro de cada una,
    entre TPE y Random) rompería "entre réplicas, tanto los datos como las
    semillas de cada sampler cambian" (ver el markdown de arriba) sin que
    ninguna prueba de caja negra lo note por sí sola -- knn es determinista
    dado el dataset, así que con datos fijos entre réplicas sus valores
    saldrían idénticos (ver test_knn_difiere_entre_replicas_de_humo, la
    protección de caja negra complementaria)."""
    arbol = _arbol_paso()
    fors_replica = [
        nodo
        for nodo in arbol.body
        if isinstance(nodo, ast.For)
        and isinstance(nodo.target, ast.Tuple)
        and [ast.unparse(e) for e in nodo.target.elts] == ["r", "semillas_r"]
    ]
    assert len(fors_replica) == 1, "debe haber un único for que arme r, semillas_r"
    iter_expr = fors_replica[0].iter
    assert isinstance(iter_expr, ast.Call)
    assert ast.unparse(iter_expr.func) == "enumerate", ast.unparse(iter_expr)
    assert len(iter_expr.args) == 1
    llamada_semillas = iter_expr.args[0]
    assert isinstance(llamada_semillas, ast.Call)
    assert ast.unparse(llamada_semillas.func) == "semillas_de", ast.unparse(llamada_semillas)
    assert [ast.unparse(a) for a in llamada_semillas.args] == ["BASE_DATOS", "R"], ast.unparse(
        llamada_semillas
    )
    assert not llamada_semillas.keywords, (
        "semillas_de(...) no debe pasar datos= aquí: fijaría el dataset entre réplicas",
        ast.unparse(llamada_semillas),
    )


@pytest.mark.gpu
def test_knn_difiere_entre_replicas_de_humo(salida_humo):
    """Caja NEGRA complementaria a la de arriba: `knn` es determinista dado
    el dataset (no lee `semillas.init`, ver el docstring de `cash_paridad`),
    así que si las réplicas compartieran los mismos datos (el error que
    fijaría `datos=` en `semillas_de`), las dos líneas 'replica N' de humo
    (R=2) imprimirían el MISMO `verdad_knn` -- esta prueba lo nota sin
    necesitar leer el AST."""
    filas = list(_RE_REPLICA.finditer(salida_humo))
    assert len(filas) == 2
    valores_knn = {f.group("verdad_knn") for f in filas}
    assert len(valores_knn) == 2, ("las dos réplicas de humo dieron el mismo knn", filas)


# --- _ic_tost: la fórmula real del TOST, protegida contra confundir alfa con alfa/2 ---


def test_ic_tost_formula_correcta_no_confunde_ic90_con_ic95():
    """Caja BLANCA: extrae `_ic_tost` del AST del paso y la ejerce con un
    `estadistica.Pareado` sintético. La fórmula real usa el t crítico de
    nivel `1 - alfa` (TOST, MÁS ANGOSTO que el intervalo 95% -- 90% con los
    defaults de `estadistica.diferencia_pareada`): si se usara `alfa/2` en
    vez de `alfa` (dibujando el intervalo 95% como si fuera el intervalo
    90%), el intervalo saldría más ancho de lo que corresponde. Se compara
    el resultado real contra el
    cálculo de referencia hecho aquí con scipy.stats directamente, y también
    contra el resultado de la fórmula mutada (que NO debe coincidir)."""
    ns = _extraer_al_namespace({"_ic_tost"})
    ns["math"] = math
    ns["t_student"] = t_student_scipy
    comp = estadistica.Pareado(
        n=5,
        diferencia_media=0.02,
        de_dif=0.01,
        ic_bajo=0.0,
        ic_alto=0.04,
        p_valor=0.03,
        wilcoxon_p=None,
        rho=0.5,
        veredicto="equivalente",
        delta=0.01,
        direccion="maximizar",
        confianza=0.95,
    )
    ic_bajo, ic_alto = ns["_ic_tost"](comp)

    alfa = 1.0 - comp.confianza
    ee = comp.de_dif / math.sqrt(comp.n)
    t_critico = float(t_student_scipy.ppf(1.0 - alfa, df=comp.n - 1))
    esperado_bajo = comp.diferencia_media - t_critico * ee
    esperado_alto = comp.diferencia_media + t_critico * ee
    assert ic_bajo == pytest.approx(esperado_bajo)
    assert ic_alto == pytest.approx(esperado_alto)

    # Con alfa/2 se dibujaría el intervalo 95% (más ancho) como si fuera el
    # intervalo 90% (TOST) -- el resultado real NO debe coincidir con eso.
    t_critico_mutado = float(t_student_scipy.ppf(1.0 - alfa / 2.0, df=comp.n - 1))
    ic_alto_mutado = comp.diferencia_media + t_critico_mutado * ee
    assert ic_alto != pytest.approx(ic_alto_mutado)


def test_ic_tost_de_dif_no_positivo_colapsa_a_un_punto():
    """Segunda rama de `_ic_tost`: con `de_dif <= 0.0` (todas las diferencias
    idénticas) el intervalo colapsa a un punto (`diferencia_media`,
    `diferencia_media`), en vez de dividir por cero."""
    ns = _extraer_al_namespace({"_ic_tost"})
    ns["math"] = math
    ns["t_student"] = t_student_scipy
    comp = estadistica.Pareado(
        n=5,
        diferencia_media=0.01,
        de_dif=0.0,
        ic_bajo=0.01,
        ic_alto=0.01,
        p_valor=0.0,
        wilcoxon_p=None,
        rho=float("nan"),
        veredicto="mejor",
        delta=0.01,
        direccion="maximizar",
        confianza=0.95,
    )
    assert ns["_ic_tost"](comp) == (0.01, 0.01)
