"""Paso 02: el paso corre de punta a punta y sus respuestas dicen lo que miden.

`tests/test_pasos.py` ya cubre exit 0 y el PNG para TODOS los pasos con
PRESUPUESTO=humo; aquí se verifica además el CONTENIDO propio del paso 02: las
cinco respuestas numeradas aparecen, sus números tienen sentido (p.ej. el mejor
por validación nunca puede ser peor que la línea base: los dos son trials del
MISMO estudio con la MISMA dirección de optimización), reanudar entre dos
sesiones no repite trabajo ni vuelve a sembrar la línea base, y un trial
huérfano no tumba el paso con StopIteration.

Patrón de fixture copiado de tests/test_paso_01.py::salida_humo (scope="module",
subprocess con PRESUPUESTO=humo, stdout compartido entre pruebas): un estudio
de Optuna completo es caro, y la salida es determinista con semillas fijas,
así que correr un subproceso por prueba de contenido multiplicaría el costo
de la suite sin agregar cobertura -- el mismo argumento que ya usa
`salida_humo`. Aquí `corrida` es la única `_correr("humo", ...)` que paga ese
costo; todas las pruebas de contenido leen su stdout compartido.
"""

from __future__ import annotations

import ast
import dataclasses
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
from taller_hpo.problemas import paridad
from taller_hpo.semillas import semillas_de

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "02_primer_estudio.py"

FLOAT = r"([0-9]+\.[0-9]+)"


def _correr(
    presupuesto: str,
    resultados: Path,
    estudios: Path,
    timeout: float = 90,
    *,
    esperar_exito: bool = True,
    entorno_extra: dict | None = None,
):
    entorno = {
        **os.environ,
        "PRESUPUESTO": presupuesto,
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_ESTUDIOS": str(estudios),
        **(entorno_extra or {}),
    }
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=timeout
    )
    if esperar_exito:
        assert resultado.returncode == 0, resultado.stderr
        # Convención del taller: cero warnings. Un UserWarning de Optuna (p.ej.
        # reabrir con una instancia de sampler sembrada) o un aviso experimental
        # deben verse aquí, no perderse en un subproceso que nadie revisa.
        assert resultado.stderr == "", resultado.stderr
    return resultado


def _parse_lista(texto: str) -> list[str]:
    """'ninguna' -> [] ; "['alpha', 'ancho']" -> ['alpha', 'ancho'] (ast, no eval)."""
    return [] if texto == "ninguna" else ast.literal_eval(texto)


def _numero(patron: str, texto: str) -> float:
    coincidencia = re.search(patron, texto)
    assert coincidencia, f"no se encontró {patron!r} en:\n{texto}"
    return float(coincidencia.group(1))


def _campo(patron: str, texto: str) -> str:
    """Como `_numero`, pero devuelve el texto crudo del primer grupo (para
    campos que no son floats simples, como el signo encima/debajo)."""
    coincidencia = re.search(patron, texto)
    assert coincidencia, f"no se encontró {patron!r} en:\n{texto}"
    return coincidencia.group(1)


def _extraer_del_paso(nombres_funciones: set[str], nombres_constantes: set[str] = frozenset()):
    """Extrae del AST de PASO las funciones y constantes module-level pedidas
    y las ejecuta en un namespace aislado: prueba las funciones REALES que
    imprime el paso (un error real en `_frase_z`, por ejemplo, sí hace
    fallar las aserciones de más abajo), no una copia de su lógica escrita
    aparte en esta prueba -- copiar la fórmula es justo lo que no detecta un
    error en la fórmula original.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"))
    nodos = []
    vistos_fn: set[str] = set()
    vistos_const: set[str] = set()
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name in nombres_funciones:
            nodos.append(nodo)
            vistos_fn.add(nodo.name)
        elif isinstance(nodo, ast.Assign) and len(nodo.targets) == 1:
            objetivo = nodo.targets[0]
            if isinstance(objetivo, ast.Name) and objetivo.id in nombres_constantes:
                nodos.append(nodo)
                vistos_const.add(objetivo.id)
    faltan = (nombres_funciones - vistos_fn) | (nombres_constantes - vistos_const)
    assert not faltan, f"no se encontraron en {PASO.name}: {faltan}"
    modulo = ast.Module(body=nodos, type_ignores=[])
    ast.fix_missing_locations(modulo)
    # Todo lo que las funciones/constantes reales del paso pueden referenciar
    # como global: math y scipy_stats (usados por `_z_semilla`/`_umbral_z`),
    # optuna/dataclasses/semillas_de/juguete (usados al armar PROTOCOLO,
    # DISTRIBUCIONES y SEMILLA_UNICA), estadistica (el type hint
    # `estadistica.Resumen` de `_frase_sesgo_config`/`_frase_bayes_gana` se
    # evalúa al definir la función, no solo al llamarla). Sin esto, extraer
    # esas funciones/constantes por AST fallaría con NameError apenas se
    # ejecutara el nodo reducido.
    espacio: dict = {
        "math": math,
        "scipy_stats": scipy_stats,
        "optuna": optuna,
        "dataclasses": dataclasses,
        "juguete": juguete,
        "semillas_de": semillas_de,
        "estadistica": estadistica,
    }
    exec(compile(modulo, str(PASO), "exec"), espacio)
    return espacio


def _protocolo_y_distribuciones() -> tuple[dict, dict]:
    """PROTOCOLO y DISTRIBUCIONES reales de pasos/02_primer_estudio.py, extraídos
    por AST: una sola fuente de verdad para las pruebas que arman un estudio
    presembrado con este protocolo (si no coincide byte a byte con el del
    paso, `crear_estudio` lo rechaza con ValueError al reabrir). Copiar estos
    dos dicts a mano en cada prueba deja varios lugares que un cambio del
    protocolo real podría desincronizar sin que ninguna prueba lo note;
    extraerlos del propio módulo hace que ese desincronizado sea imposible.
    """
    espacio = _extraer_del_paso(
        set(),
        {
            "SEMILLA_BASE_SAMPLER",
            "N_STARTUP_TRIALS",
            "SEMILLA_UNICA",
            "PROTOCOLO",
            "DISTRIBUCIONES",
        },
    )
    return espacio["PROTOCOLO"], espacio["DISTRIBUCIONES"]


# SEMILLA_BASE_REEVAL y N_REEVAL_POR_PRESUPUESTO reales de
# pasos/02_primer_estudio.py, extraídos por AST (mismo mecanismo que
# `_protocolo_y_distribuciones`): una sola fuente de verdad para las pruebas
# que reconstruyen la re-evaluación de la respuesta 2 -- copiar estos dos
# valores a mano deja un lugar más que un cambio en el paso podría
# desincronizar sin que ninguna prueba lo note.
_VALORES_REEVAL = _extraer_del_paso(set(), {"SEMILLA_BASE_REEVAL", "N_REEVAL_POR_PRESUPUESTO"})
SEMILLA_BASE_REEVAL = _VALORES_REEVAL["SEMILLA_BASE_REEVAL"]
N_REEVAL_POR_PRESUPUESTO = _VALORES_REEVAL["N_REEVAL_POR_PRESUPUESTO"]


def _fabrica_sampler_prueba():
    """El mismo sampler que arma `fabrica_sampler` en el paso -- misma semilla
    base y mismo n_startup_trials, extraídos por AST -- para presembrar un
    estudio en las pruebas sin copiar esos dos números a mano en cada una.
    """
    valores = _extraer_del_paso(set(), {"SEMILLA_BASE_SAMPLER", "N_STARTUP_TRIALS"})
    base = valores["SEMILLA_BASE_SAMPLER"]
    startup = valores["N_STARTUP_TRIALS"]
    return lambda n: optuna.samplers.TPESampler(
        seed=base + n, n_startup_trials=startup, multivariate=True, constant_liar=False
    )


# Une las cuatro cifras de la respuesta 1 en una sola pasada: con regex
# separadas y `.*` codicioso, "Línea base:.*verdadera X" hace backtracking
# hasta la ÚLTIMA "verdadera" de la línea (la del mejor, no la de la línea
# base) porque las dos frases viven en el mismo print() sin salto de línea.
# El grupo del empate es opcional: solo aparece cuando otro trial iguala
# exactamente el valor del mejor (ver `test_paso_02_reporta_empate...`).
PATRON_RESPUESTA_1 = re.compile(
    r"Línea base: validación (?P<base_val>[0-9]+\.[0-9]+) "
    r"\(verdadera (?P<base_verdad>[0-9]+\.[0-9]+)\)\. "
    r"Mejor por validación \(trial \d+\)(?: \(empatado con los trials \[[^\]]*\]\))?: "
    r"validación (?P<mejor_val>[0-9]+\.[0-9]+) "
    r"\(verdadera (?P<mejor_verdad>[0-9]+\.[0-9]+)\)\. "
    r"Diferencia de validación: (?P<dif_val>[+-][0-9]+\.[0-9]+) puntos\. "
    r"Mejora verdadera: (?P<mejora_verdad>[+-][0-9]+\.[0-9]+) puntos\."
)

# La respuesta 2 intercala números "limpios" con frases largas que traen sus
# propios paréntesis (`_frase_sesgo_config`, `_frase_bayes_gana`, `_frase_z`,
# `_frase_dificultad_conjunto`, `_frase_spearman`): cada frase se acota con
# `.+?` no goloso hasta el próximo literal fijo del f-string (nunca hasta un
# paréntesis, que la frase ya usa), así que el punto de corte es el mismo
# texto fijo que imprime el paso, no una cuenta de paréntesis. `frase_sesgo` y
# `frase_bayes` son dos frases INDEPENDIENTES (`_frase_sesgo_config` decide si
# la brecha es atribuible al conjunto; `_frase_bayes_gana` decide, aparte, si
# corresponde leerle al clasificador de Bayes una fracción de esa brecha o un
# número neutro), así que ninguna de las dos tiene una forma fija: los números
# EXACTOS que decidieron cada una (nunca los redondeados de aquí) se verifican
# aparte, contra las funciones reales extraídas por AST, en
# test_paso_02_estudio_persistido_tiene_los_trials_y_el_protocolo_esperados.
PATRON_RESPUESTA_2 = re.compile(
    r"El ganador dice (?P<ganador>[0-9]+\.[0-9]+) y su verdad es (?P<verdad>[0-9]+\.[0-9]+) "
    r"\((?P<inflado>[+-][0-9]+\.[0-9]+) pts: (?P<frase_inflado>[a-záéíóúñ ]+)\)\. "
    r"Re-evaluando la MISMA config con (?P<n_reeval>[0-9]+) semillas nuevas de orden/init sobre "
    r"los mismos datos \(método del paso 06\): "
    r"(?P<frase_sesgo>.+?); (?P<frase_bayes>.+?)\. Frente a esa "
    r"misma reevaluación, la semilla de la búsqueda quedó "
    r"(?P<e_ganador>[+-][0-9]+\.[0-9]+) pts fuera de ese promedio "
    r"\(z=(?P<z_ganador>[+-][0-9]+\.[0-9]+), umbral (?P<umbral>[0-9]+\.[0-9]+) con "
    r"n=(?P<n_reeval_z>[0-9]+)\) y la línea base \(control, nunca elegida\) quedó "
    r"(?P<e_base>[+-][0-9]+\.[0-9]+) pts fuera del suyo \(z=(?P<z_base>[+-][0-9]+\.[0-9]+)\): "
    r"(?P<frase_z>.+?)\. Sin oráculo, lo único visible es que la validación del ganador difiere "
    r"(?P<visible>[+-][0-9]+\.[0-9]+) pts del promedio de sus propias re-evaluaciones; "
    r"(?P<frase_dificultad>.+?)\. "
    r"\(Revelación del oráculo: la mediana de los demás (?P<n_otros>[0-9]+) trials es "
    r"(?P<brecha>[+-][0-9]+\.[0-9]+) pts -- (?P<frase_spearman>.+?)\.\)"
)

PATRON_RESPUESTA_4 = re.compile(
    r"4\. (?P<n_trials>\d+) trials, (?P<n_top>\d+) sobre el cuantil (?P<cuantil>\d+)% "
    r"\(placebo=(?P<piso>[0-9]+\.[0-9]+)\): (?P<nota>.+?)\. "
    r"Perillas sobre el piso del placebo: (?P<lista>.+?)\."
)


@pytest.fixture(scope="module")
def corrida(tmp_path_factory) -> dict:
    resultados = tmp_path_factory.mktemp("resultados_humo")
    estudios = tmp_path_factory.mktemp("estudios_humo")
    resultado = _correr("humo", resultados, estudios)
    return {"stdout": resultado.stdout, "resultados": resultados, "estudios": estudios}


@pytest.mark.gpu
def test_paso_02_humo_respuestas_numeradas_tienen_sentido(corrida):
    stdout = corrida["stdout"]

    for inicio in (
        "1. Línea base",
        "2. El ganador dice",
        "3. Los",
        "4. ",
        "5. Para explorar el estudio a mano",
    ):
        assert inicio in stdout, stdout

    # 1. la línea base y el mejor encontrado, validación y verdadera, en [0, 1].
    coincidencia = PATRON_RESPUESTA_1.search(stdout)
    assert coincidencia, stdout
    base_val = float(coincidencia["base_val"])
    base_verdad = float(coincidencia["base_verdad"])
    mejor_val = float(coincidencia["mejor_val"])
    mejor_verdad = float(coincidencia["mejor_verdad"])
    dif_val = float(coincidencia["dif_val"])
    mejora_verdad = float(coincidencia["mejora_verdad"])
    for valor in (base_val, base_verdad, mejor_val, mejor_verdad):
        assert 0.0 <= valor <= 1.0, valor
    # Invariante fuerte, no solo probable: la línea base es UN trial más del
    # mismo estudio (maximize), así que el mejor por validación nunca puede
    # valer menos -- sin importar semillas ni presupuesto.
    assert mejor_val >= base_val
    assert dif_val == pytest.approx(100 * (mejor_val - base_val), abs=0.05)
    assert mejora_verdad == pytest.approx(100 * (mejor_verdad - base_verdad), abs=0.05)

    # 2. la respuesta 2 se lee en una sola pasada con PATRON_RESPUESTA_2: los
    # anclajes numéricos "limpios" y las frases largas (`frase_sesgo`,
    # `frase_z`, `frase_dificultad`, `frase_spearman`), cada una acotada por
    # el texto fijo que la sigue en el f-string, nunca por sus propios
    # paréntesis internos. Aquí solo se verifican RANGOS e invariantes
    # estructurales (nunca la condición exacta de una rama, recomputada a
    # partir de floats YA redondeados a 1-2 decimales: cerca de un umbral, o
    # con -0.0, esa reconstrucción puede discrepar del valor exacto que el
    # paso realmente usó para decidir). La igualdad EXACTA de cada frase
    # contra la función real que la produjo, evaluada en los valores sin
    # redondear recalculados sobre el estudio persistido, vive en
    # `test_paso_02_estudio_persistido_tiene_los_trials_y_el_protocolo_esperados`;
    # la cobertura de TODAS las ramas con valores exactos y sus bordes vive en
    # `test_paso_02_frases_cubren_las_ramas_con_valores_exactos`.
    m2 = PATRON_RESPUESTA_2.search(stdout)
    assert m2, stdout
    ganador = float(m2["ganador"])
    verdad = float(m2["verdad"])
    inflado = float(m2["inflado"])
    e_ganador = float(m2["e_ganador"])
    z_ganador = float(m2["z_ganador"])
    umbral = float(m2["umbral"])
    e_base = float(m2["e_base"])
    z_base = float(m2["z_base"])
    visible = float(m2["visible"])
    # `suerte` (=100*media) y `bayes_gana` viven DENTRO de `frase_sesgo` y
    # `frase_bayes`: cada una los redacta distinto según su propia rama (ver
    # `_frase_sesgo_config`/`_frase_bayes_gana`), así que no hay un ancla fija
    # que exista en las dos a la vez -- se leen del primer número con signo de
    # cada frase.
    suerte = float(re.search(r"([+-][0-9]+\.[0-9]+) pts", m2["frase_sesgo"]).group(1))
    bayes_gana = float(re.search(r"gana ([+-][0-9]+\.[0-9]+)", m2["frase_bayes"]).group(1))
    m_ic = re.search(
        r"intervalo de confianza 95% \[([+-][0-9]+\.[0-9]+), ([+-][0-9]+\.[0-9]+)\]",
        m2["frase_sesgo"],
    )
    assert m_ic, m2["frase_sesgo"]
    assert float(m_ic.group(1)) <= float(m_ic.group(2))
    assert ganador == pytest.approx(mejor_val, abs=0.0005)
    assert verdad == pytest.approx(mejor_verdad, abs=0.0005)
    assert inflado == pytest.approx(100 * (ganador - verdad), abs=0.05)
    assert int(m2["n_reeval"]) == N_REEVAL_POR_PRESUPUESTO["humo"]
    # Los dos "n=" de la respuesta 2 (el de la re-evaluación y el del umbral
    # impreso junto al z-score) son el MISMO N_REEVAL: si divergieran, alguno
    # de los dos se calculó con un valor viejo o de otro presupuesto.
    assert int(m2["n_reeval_z"]) == int(m2["n_reeval"])
    assert umbral > 0.0
    assert math.isfinite(bayes_gana)
    # e_ganador = inflado - suerte (por construcción); redondeados por
    # separado (":+.1f" cada uno), así que la resta de los dos ya redondeados
    # puede diferir del tercero redondeado hasta en 0.15 (tres redondeos de a
    # 0.05). `visible` no tiene esa relación algebraica exacta con inflado
    # (usa el promedio de validación de la reevaluación, no el de verdad),
    # así que solo se exige que sea un número finito de magnitud razonable.
    assert e_ganador == pytest.approx(inflado - suerte, abs=0.15)
    assert math.isfinite(visible)
    assert abs(visible) < 20.0
    assert math.isfinite(e_base)
    assert math.isfinite(z_ganador)
    assert math.isfinite(z_base)

    # Chequeo de FORMA, no de rama exacta (esa exactitud vive en las dos
    # pruebas citadas arriba): cada frase es una de las que puede producir la
    # función real que la escribe, nunca texto vacío ni de otra respuesta.
    assert m2["frase_inflado"] in (
        "la validación sobreestima la verdad",
        "la validación subestima la verdad",
    )
    assert m2["frase_sesgo"].startswith("con ")
    assert m2["frase_bayes"]  # no vacía; el contenido exacto se cubre aparte.
    assert m2["frase_z"]  # no vacía; el contenido exacto se cubre aparte.
    assert m2["frase_dificultad"].startswith("este conjunto de validación es")
    assert m2["frase_spearman"]

    # Campos detrás de paréntesis anidados: cada uno por su propio ancla
    # literal, que no depende de acotar un `.+?` a través de un paréntesis
    # que la propia frase ya usa (Bayes, Spearman).
    acc_bayes = float(_campo(r"el clasificador de Bayes acierta " + FLOAT + r" en él", stdout))
    signo_techo = _campo(r"por (encima|debajo) del techo de ", stdout)
    techo = float(_campo(r"por (?:encima|debajo) del techo de " + FLOAT, stdout))
    assert 0.0 <= acc_bayes <= 1.0
    assert 0.0 <= techo <= 1.0
    assert (signo_techo == "encima") == (acc_bayes >= techo)

    n_otros = int(_campo(r"la mediana de los demás (\d+) trials es", stdout))
    brecha = float(_campo(r"trials es ([+-][0-9]+\.[0-9]+) pts -- ", stdout))
    assert n_otros == 9  # PRESUPUESTO=humo siempre da 10 trials (ver la sección "3." abajo).
    assert math.isfinite(brecha)
    # n_otros y brecha ya salen de PATRON_RESPUESTA_2 (m2): misma cifra por
    # dos caminos de lectura distintos (ancla propia vs. grupo del patrón
    # grande) es una guarda contra que uno de los dos quede desincronizado.
    assert int(m2["n_otros"]) == n_otros
    assert float(m2["brecha"]) == pytest.approx(brecha, abs=0.05)
    rho = float(_campo(r"Spearman\(verdad, brecha\)=(-?[0-9]+\.[0-9]+)", stdout))
    p_spearman = float(
        _campo(r"Spearman\(verdad, brecha\)=-?[0-9]+\.[0-9]+, p=([0-9.eE+-]+)", stdout)
    )
    assert -1.0 <= rho <= 1.0
    assert 0.0 <= p_spearman <= 1.0

    # 3. con PRESUPUESTO=humo (10 trials) no hay colisiones: 10 configuraciones únicas.
    n_trials = int(_numero(r"Los ([0-9]+) trials completados", stdout))
    n_unicas = int(_numero(r"usan ([0-9]+) configuraciones distintas", stdout))
    assert n_trials == 10
    assert n_unicas == n_trials

    # 4. con solo 10 trials (n_top = ceil(0.1*10) = 1 < 5), PED-ANOVA se marca
    # explícitamente como no confiable, y las 6 perillas del espacio se imprimen.
    m4 = PATRON_RESPUESTA_4.search(stdout)
    assert m4, stdout
    assert int(m4["n_trials"]) == 10
    assert int(m4["n_top"]) == 1
    assert "recomienda al menos" in m4["nota"]
    assert float(m4["piso"]) >= 0.0
    for parametro in (
        "learning_rate_init",
        "alpha",
        "ancho",
        "n_capas",
        "batch_size",
        "placebo",
    ):
        importancia = _numero(rf"{parametro}: importancia " + FLOAT, stdout)
        assert importancia >= 0.0

    # 5. el comando del dashboard apunta al MISMO archivo de estudio que se usó.
    destino_esperado = corrida["estudios"] / "02_primer_estudio_humo.log"
    assert f"uv run optuna-dashboard {destino_esperado}" in stdout

    assert list(corrida["resultados"].glob("02_primer_estudio.png")), "el paso no guardó su figura"


@pytest.mark.gpu
def test_paso_02_dos_sesiones_reabren_con_los_trials_previos_medidos(corrida):
    """Con humo, N_TOTAL=10 y mitad=5: la sesión 1 se crea desde cero (0 previos,
    semilla 314159) y la sesión 2 reabre viendo los 5 que dejó la sesión 1
    (semilla 314164 = 314159 + 5). Si las dos sesiones colapsaran en una sola
    (sin reabrir de verdad), la sesión 2 no vería ningún trial previo -- esta
    prueba distingue ambos casos con números medidos, no con la sola presencia
    de la palabra "reabierta".
    """
    stdout = corrida["stdout"]
    assert "Sesión 1: creada con 0 trials previos, semilla del sampler 314159." in stdout
    assert "Sesión 2: reabierta con 5 trials previos, semilla del sampler 314164." in stdout
    assert "Huérfanos recuperados al abrir la sesión 1: 0." in stdout
    assert "Sesión 1 completa: 5 trials terminados (COMPLETE + PRUNED)." in stdout
    assert "Sesión 2 completa: 10 trials terminados en total." in stdout


@pytest.mark.gpu
def test_paso_02_marcas_de_importancia_son_consistentes_con_el_piso(corrida):
    """`sobre_el_piso` se define como `importancia > piso` (piso = importancia
    del propio placebo, `taller_hpo.estudio.importancia_con_piso`). Dos
    invariantes:

    (a) El placebo se compara contra sí mismo, así que SIEMPRE debe marcarse
        "en el piso" (nunca puede superarse a sí mismo) -- exacto por
        construcción, sin importar redondeo: si las marcas salieran
        invertidas, esto solo fallaría de inmediato.
    (b) Cualquier otra perilla debe coincidir con `importancia > piso`
        recalculado directamente sobre el estudio persistido (con el MISMO
        evaluador y `target_quantile` que usa el paso): comparar contra los
        floats exactos, no contra el texto redondeado a 3 decimales que
        imprime el paso, evita falsos negativos cuando una importancia
        queda a menos de 0.001 del piso.
    """
    stdout = corrida["stdout"]
    m_placebo = re.search(r"placebo: importancia " + FLOAT + r" \((.+?)\)", stdout)
    assert m_placebo, stdout
    assert m_placebo.group(2) == "en el piso"

    destino = corrida["estudios"] / "02_primer_estudio_humo.log"
    study = optuna.load_study(
        study_name="02_primer_estudio", storage=estudio.abrir_storage(destino)
    )
    evaluador = optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.1)
    importancias = estudio.importancia_con_piso(study, evaluador=evaluador, placebo="placebo")

    for parametro in ("learning_rate_init", "alpha", "ancho", "n_capas", "batch_size"):
        m = re.search(rf"{parametro}: importancia " + FLOAT + r" \((.+?)\)", stdout)
        assert m, stdout
        marca = m.group(2)
        assert marca in ("sobre el piso", "en el piso")
        assert (marca == "sobre el piso") == importancias[parametro]["sobre_el_piso"]

    # La lista de "Perillas sobre el piso del placebo" (respuesta 4) y las
    # marcas individuales de más abajo vienen de dos comprehensions distintas
    # sobre el MISMO `importancias`: si una de las dos quedara invertida
    # frente a la otra, esto lo detecta -- una lista global y una marca por
    # parámetro pueden desincronizarse sin que ninguna prueba lo note si solo
    # se verifica una de las dos por separado.
    m4 = PATRON_RESPUESTA_4.search(stdout)
    assert m4, stdout
    sobre_piso_por_marca = sorted(
        p
        for p in ("learning_rate_init", "alpha", "ancho", "n_capas", "batch_size")
        if re.search(rf"{p}: importancia " + FLOAT + r" \((.+?)\)", stdout).group(2)
        == "sobre el piso"
    )
    assert _parse_lista(m4["lista"]) == sobre_piso_por_marca


@pytest.mark.gpu
def test_paso_02_robustez_de_importancia_entre_evaluadores(corrida):
    """La respuesta 4 promete una sola importancia con PED-ANOVA
    (target_quantile=0.1); además imprime, cuantil por cuantil, qué perillas
    superan al placebo con PED-ANOVA q=0.1, q=0.2 y q=0.3, y la intersección
    de los tres -- sin eso, un tesista con 10 o 60 trials creería que una
    perilla "importa" solo porque UN cuantil se lo dijo: en este mismo
    estudio persistido, PED-ANOVA con target_quantile=0.2 puede no encontrar
    ninguna perilla sobre el piso mientras que q=0.1 sí marca alguna, así que
    un solo cuantil no alcanza para una conclusión. Cada lista se recalcula
    directamente sobre el estudio persistido (nunca sobre el texto redondeado
    a 3 decimales).
    """
    stdout = corrida["stdout"]
    destino = corrida["estudios"] / "02_primer_estudio_humo.log"
    study = optuna.load_study(
        study_name="02_primer_estudio", storage=estudio.abrir_storage(destino)
    )

    evaluadores = {
        "PED-ANOVA q=0.1": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.1),
        "PED-ANOVA q=0.2": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.2),
        "PED-ANOVA q=0.3": optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.3),
    }
    listas_esperadas: dict[str, set[str]] = {}
    for nombre, evaluador in evaluadores.items():
        importancias_e = estudio.importancia_con_piso(study, evaluador=evaluador, placebo="placebo")
        listas_esperadas[nombre] = {
            p for p, v in importancias_e.items() if p != "placebo" and v["sobre_el_piso"]
        }
        m = re.search(rf"   {re.escape(nombre)}: sobre el piso (.+)", stdout)
        assert m, stdout
        assert set(_parse_lista(m.group(1))) == listas_esperadas[nombre]

    robustas_esperadas = set.intersection(*listas_esperadas.values())
    n_completos = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
    # Mismo criterio que usa el paso para decidir `prefijo_robustez`
    # (TARGET_QUANTILE=0.1, N_TOP_MINIMO=5): con pocos trials sobre el cuantil,
    # la conclusión se marca "orientativo: ". Esta prueba recalcula el
    # prefijo aquí (en vez de asumir que el prefijo esperado siempre está
    # vacío) y lo ancla al INICIO de la línea con `re.M`: así un error que
    # vacíe `prefijo_robustez` sin tocar el resto del texto (que sigue siendo
    # una subcadena válida) sí lo detecta.
    prefijo = "" if math.ceil(0.1 * n_completos) >= 5 else "orientativo: "
    if robustas_esperadas:
        conclusion_esperada = (
            f"Solo {sorted(robustas_esperadas)} supera al placebo con los tres cuantiles "
            "de PED-ANOVA."
        )
    else:
        conclusion_esperada = (
            "Ninguna perilla supera al placebo con los tres cuantiles de PED-ANOVA: en "
            f"esta corrida ({n_completos} trials), la importancia no es una conclusión."
        )
    assert re.search(rf"^   {re.escape(prefijo + conclusion_esperada)}$", stdout, re.M), stdout


def test_paso_02_solo_usa_ped_anova_en_optuna_importance():
    """Esta prueba recorre el AST de pasos/02_primer_estudio.py, sin correr el paso y sin
    necesitar tarjeta de video: toda llamada `optuna.importance.<X>(...)` tiene que
    instanciar `PedAnovaImportanceEvaluator`, y ningún nombre del archivo puede mencionar
    `Fanova` ni `MeanDecreaseImpurity`, los otros dos evaluadores de `optuna.importance`.
    El paso no ajusta el `RandomForestRegressor` de fANOVA (un bosque de scikit-learn,
    nunca de la tarjeta de video); esta prueba falla en la integración continua si
    alguien vuelve a agregar `FanovaImportanceEvaluator`.
    """
    fuente = PASO.read_text(encoding="utf-8")
    arbol = ast.parse(fuente, filename=str(PASO))
    llamadas_importance = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and isinstance(nodo.func.value, ast.Attribute)
        and isinstance(nodo.func.value.value, ast.Name)
        and nodo.func.value.value.id == "optuna"
        and nodo.func.value.attr == "importance"
    ]
    assert llamadas_importance, "no se encontró ninguna llamada a optuna.importance.<algo>(...)"
    for llamada in llamadas_importance:
        assert llamada.func.attr == "PedAnovaImportanceEvaluator", ast.dump(llamada)
    assert "Fanova" not in fuente, fuente
    assert "MeanDecreaseImpurity" not in fuente, fuente


@pytest.mark.gpu
def test_paso_02_humo_si_carga_sklearn_por_optuna_importance(tmp_path_factory):
    """El paso 02 carga scikit-learn y threadpoolctl aunque ningún evaluador sea de fANOVA:
    `EVALUADORES_ROBUSTEZ` solo usa `PedAnovaImportanceEvaluator`, pero
    `optuna/importance/__init__.py` importa `FanovaImportanceEvaluator` sin condición, y ese
    módulo importa `RandomForestRegressor`. La prueba corre el paso en un subproceso limpio y
    exige que todo evaluador de importancia del módulo sea `PedAnovaImportanceEvaluator`. Si
    una versión futura de Optuna aísla fANOVA, deja de pasar y avisa que hay que revisar
    esta excepción.
    """
    resultados = tmp_path_factory.mktemp("resultados_sklearn")
    estudios = tmp_path_factory.mktemp("estudios_sklearn")
    codigo = (
        "import runpy\n"
        "import sys\n"
        "import optuna\n"
        f"g = runpy.run_path({str(PASO)!r})\n"
        "assert 'sklearn' in sys.modules\n"
        "assert 'threadpoolctl' in sys.modules\n"
        "evaluadores = [\n"
        "    v for v in g.values()\n"
        "    if isinstance(v, optuna.importance.BaseImportanceEvaluator)\n"
        "]\n"
        "assert evaluadores, 'no quedó ningún BaseImportanceEvaluator en el módulo del paso'\n"
        "assert all(\n"
        "    type(v) is optuna.importance.PedAnovaImportanceEvaluator for v in evaluadores\n"
        "), evaluadores\n"
        "robustez = g['EVALUADORES_ROBUSTEZ']\n"
        "assert robustez, 'EVALUADORES_ROBUSTEZ quedó vacío'\n"
        "assert all(\n"
        "    type(v) is optuna.importance.PedAnovaImportanceEvaluator\n"
        "    for v in robustez.values()\n"
        "), robustez\n"
        "print('OK')\n"
    )
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(resultados),
        "TALLER_ESTUDIOS": str(estudios),
    }
    resultado = subprocess.run(
        [sys.executable, "-c", codigo], env=entorno, capture_output=True, text=True, timeout=90
    )
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stderr == "", resultado.stderr
    assert "OK" in resultado.stdout, resultado.stdout


@pytest.mark.gpu
def test_paso_02_reanudar_no_repite_trabajo_ni_la_linea_base(corrida):
    """La corrida de `corrida` YA dejó el estudio completo (10 trials, línea
    base sembrada): reabrirlo desde cero no debe repetir ningún trabajo. Se
    reusa esa corrida como "primera" en vez de correrla de nuevo aquí -- una
    corrida completa de menos que relanzar el módulo sin necesidad.
    """
    assert "Línea base sembrada en esta corrida: True" in corrida["stdout"]

    segunda = _correr("humo", corrida["resultados"], corrida["estudios"])
    # No hay una segunda línea base: sembrar_linea_base ve la primera (COMPLETE) y no hace nada.
    assert "Línea base sembrada en esta corrida: False" in segunda.stdout
    # Reanudar sin trabajo pendiente no vuelve a correr trials: mismo total, mismas únicas.
    assert "usan 10 configuraciones distintas" in segunda.stdout
    # Con los 10 trials ya completos antes de abrir, las dos sesiones de la
    # segunda corrida ven el mismo total de previos: ninguna optimiza de más.
    assert "Sesión 1: reabierta con 10 trials previos" in segunda.stdout
    assert "Sesión 2: reabierta con 10 trials previos" in segunda.stdout

    # Y las respuestas 1 y 2 son EXACTAMENTE reproducibles: mismas semillas CRN,
    # mismo protocolo, ningún trial nuevo entre las dos corridas. Cada
    # respuesta vive en su propia línea de stdout (el f-string no mete saltos
    # de línea propios), así que ".*" sin DOTALL ya la acota sin ambigüedad,
    # sin depender de cuál sea la frase final de cada rama.
    patron_resumen = re.compile(r"1\. Línea base:.*puntos\.")
    resumen_1 = patron_resumen.search(corrida["stdout"]).group(0)
    resumen_2 = patron_resumen.search(segunda.stdout).group(0)
    assert resumen_1 == resumen_2
    patron_respuesta_2 = re.compile(r"2\. El ganador dice.*")
    r2_1 = patron_respuesta_2.search(corrida["stdout"]).group(0)
    r2_2 = patron_respuesta_2.search(segunda.stdout).group(0)
    assert r2_1 == r2_2


@pytest.mark.gpu
def test_paso_02_recupera_un_huerfano_de_la_linea_base_sin_morir(tmp_path):
    """Sin `estudio.recuperar_huerfanos`, un trial RUNNING huérfano (kill -9,
    corte de luz -- JournalStorage no tiene heartbeat) deja la línea base
    RUNNING para siempre: `candidatos_base = [t for t in trials if
    user_attrs['rol'] == 'linea_base']` sobre trials COMPLETE/PRUNED queda
    vacía y el paso levanta `RuntimeError` explícito (no un `StopIteration`
    ciego). Esta prueba crea el huérfano con `Study.add_trial` ANTES de correr
    el paso, con el MISMO protocolo que arma pasos/02_primer_estudio.py (extraído
    del propio archivo por `_protocolo_y_distribuciones()`: si `crear_estudio`
    lo rechaza con ValueError al reabrir, es que hay un desincronizado real en
    el paso, no en esta prueba).
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    destino = estudios / "02_primer_estudio_humo.log"

    protocolo, distribuciones = _protocolo_y_distribuciones()
    linea_base = {clave: juguete.DEFAULTS[clave] for clave in distribuciones}

    estudio_previo = estudio.crear_estudio(
        "02_primer_estudio",
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler_prueba(),
        pruner=optuna.pruners.NopPruner(),
        protocolo=protocolo,
    )
    huerfano = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        params=linea_base,
        distributions=distribuciones,
        user_attrs={"rol": "linea_base"},
    )
    estudio_previo.add_trial(huerfano)
    assert (
        len(estudio_previo.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)))
        == 1
    )

    salida = _correr("humo", resultados, estudios)
    assert "Huérfanos recuperados al abrir la sesión 1: 1." in salida.stdout
    # El huérfano recuperado ya cuenta como línea base activa (reencolada
    # WAITING con el mismo rol): no se siembra una segunda.
    assert "Línea base sembrada en esta corrida: False" in salida.stdout
    # Y sí llega a completarse (si no, "1. Línea base:" ni se imprimiría:
    # la búsqueda de la línea base habría reventado con RuntimeError/StopIteration).
    assert "1. Línea base:" in salida.stdout


@pytest.mark.gpu
def test_paso_02_mejor_es_por_validacion_no_por_verdad_y_detecta_repetidas(tmp_path):
    """Estudio presembrado (mismo protocolo, `Study.add_trial` antes de correr el paso) con
    tres trials COMPLETE sintéticos: dos con la config P (validación 0.70, "verdad" falsa
    0.9999) y uno con Q (validación 0.99, "verdad" falsa 0.999). Sobre esos tres más los que
    complete la búsqueda real:

    - El ganador por validación debe ser Q, nunca un P: si el paso eligiera por
      `exactitud_verdadera`, elegiría un P.
    - P se repite dos veces: `_contar_repetidas` sobre los trials reales debe contar
      exactamente 1 repetida.
    - Q gana con una validación por debajo de su "verdad" falsa, lo que ejercita la rama
      negativa de la respuesta 2 ("la validación subestima la verdad").
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    destino = estudios / "02_primer_estudio_humo.log"

    protocolo, distribuciones = _protocolo_y_distribuciones()

    estudio_previo = estudio.crear_estudio(
        "02_primer_estudio",
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler_prueba(),
        pruner=optuna.pruners.NopPruner(),
        protocolo=protocolo,
    )
    # Distinta de LINEA_BASE (DEFAULTS con placebo=0.0) para no crear una
    # repetida involuntaria contra la línea base que el paso siembra después.
    p_params = {
        "learning_rate_init": 0.05,
        "alpha": 0.01,
        "ancho": 16,
        "n_capas": 2,
        "batch_size": 128,
        "placebo": 0.9,
    }
    q_params = {
        "learning_rate_init": 1e-3,
        "alpha": 1e-4,
        "ancho": 64,
        "n_capas": 1,
        "batch_size": 64,
        "placebo": 0.1,
    }
    for params, valor, verdad_falsa in (
        (p_params, 0.70, 0.9999),
        (p_params, 0.70, 0.9999),
        (q_params, 0.99, 0.999),
    ):
        estudio_previo.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=valor,
                params=params,
                distributions=distribuciones,
                user_attrs={"exactitud_verdadera": verdad_falsa},
            )
        )

    salida = _correr("humo", resultados, estudios)
    assert "Mejor por validación (trial 2)" in salida.stdout
    assert "3. De 10 trials, 1 repiten una configuración ya vista (9 únicas)" in salida.stdout

    m2 = PATRON_RESPUESTA_2.search(salida.stdout)
    assert m2, salida.stdout
    assert float(m2["ganador"]) == pytest.approx(0.99, abs=0.0005)
    assert float(m2["verdad"]) == pytest.approx(0.999, abs=0.0005)
    inflado = float(m2["inflado"])
    assert inflado < 0.0
    assert m2["frase_inflado"] == "la validación subestima la verdad"


@pytest.mark.gpu
def test_paso_02_reporta_empate_en_el_mejor_cuando_lo_hay(tmp_path):
    """Dos trials pueden empatar EXACTO en validación (Optuna no desempata por
    verdad, que no ve): con dos configs distintas sembradas al mismo valor
    -- tan alto (0.999) que ningún trial real de la búsqueda lo va a alcanzar
    ni por suerte, dado el techo de Bayes de este problema (0.825) -- el
    ganador reportado debe seguir siendo el de MENOR número (el mismo
    criterio que usa `Study.best_trial`), y el paso debe avisar el empate en
    vez de esconderlo detrás de "Mejor por validación (trial N)".
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    destino = estudios / "02_primer_estudio_humo.log"

    protocolo, distribuciones = _protocolo_y_distribuciones()
    estudio_previo = estudio.crear_estudio(
        "02_primer_estudio",
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler_prueba(),
        pruner=optuna.pruners.NopPruner(),
        protocolo=protocolo,
    )
    a_params = {
        "learning_rate_init": 0.02,
        "alpha": 0.02,
        "ancho": 32,
        "n_capas": 2,
        "batch_size": 32,
        "placebo": 0.4,
    }
    b_params = {
        "learning_rate_init": 5e-3,
        "alpha": 5e-3,
        "ancho": 16,
        "n_capas": 3,
        "batch_size": 128,
        "placebo": 0.6,
    }
    for params in (a_params, b_params):
        estudio_previo.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0.999,
                params=params,
                distributions=distribuciones,
                user_attrs={"exactitud_verdadera": 0.8},
            )
        )

    salida = _correr("humo", resultados, estudios)
    assert "Mejor por validación (trial 0) (empatado con los trials [1])" in salida.stdout


@pytest.mark.gpu
def test_paso_02_presupuesto_invalido_falla_con_mensaje(tmp_path):
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    salida = _correr("gigante", resultados, estudios, timeout=30, esperar_exito=False)
    assert salida.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in salida.stderr


@pytest.mark.gpu
def test_paso_02_estudio_persistido_tiene_los_trials_y_el_protocolo_esperados(corrida):
    """Abre el estudio persistido (no solo el stdout) y verifica lo que
    `pasos/02_primer_estudio.py` promete: el trial que reporta como "mejor"
    es de verdad `best_trial`, hay exactamente un COMPLETE con
    rol=linea_base y sus params son los DEFAULTS de las 6 claves del
    espacio, el protocolo guardado documenta CRN con sus semillas, hay 10
    trials COMPLETE en total, y la descomposición de la respuesta 2 (suerte,
    residuos, z, lo visible sin oráculo, la dificultad del conjunto vía
    Bayes) coincide con recalcularla de cero sobre el estudio persistido, con
    las MISMAS funciones que usa el paso (`honestidad.reevaluar`,
    `estadistica.resumen`, `paridad.posterior`), nunca con una copia de sus
    fórmulas escrita aparte.
    """
    destino = corrida["estudios"] / "02_primer_estudio_humo.log"
    storage = estudio.abrir_storage(destino)
    study = optuna.load_study(study_name="02_primer_estudio", storage=storage)

    completos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    assert len(completos) == 10

    con_rol_base = [t for t in completos if t.user_attrs.get("rol") == "linea_base"]
    assert len(con_rol_base) == 1
    claves_espacio = (
        "learning_rate_init",
        "alpha",
        "ancho",
        "n_capas",
        "batch_size",
        "placebo",
    )
    assert con_rol_base[0].params == {k: juguete.DEFAULTS[k] for k in claves_espacio}
    linea_base_trial = con_rol_base[0]

    m = re.search(r"Mejor por validación \(trial (\d+)\)", corrida["stdout"])
    assert m, corrida["stdout"]
    mejor = study.best_trial
    assert int(m.group(1)) == mejor.number

    protocolo = study.user_attrs["protocolo"]
    assert protocolo["politica_semillas"]["tipo"] == "CRN"
    assert protocolo["politica_semillas"]["base"] == 0
    assert protocolo["politica_semillas"]["semillas"] == dataclasses.asdict(semillas_de(0, 1)[0])
    assert protocolo["motor"] == juguete.MOTOR
    assert protocolo["dispositivo"] == juguete.DISPOSITIVO.split(":", 1)[0]

    # Ground truth de la respuesta 2, recalculada directamente sobre el
    # estudio persistido: atrapa tanto una mediana hardcodeada a 0.0 como una
    # que promedie incluyendo al ganador. `mejor.params`, la línea base y las
    # semillas de re-evaluación son las MISMAS que usa pasos/02_primer_estudio.py.
    otros = [
        t
        for t in study.get_trials(
            deepcopy=False,
            states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
        )
        if t.number != mejor.number
    ]
    otros = sorted(otros, key=lambda t: t.number)
    brecha_tipica_esperada = st.median(t.value - t.user_attrs["exactitud_verdadera"] for t in otros)
    verdad_otros_esperado = [t.user_attrs["exactitud_verdadera"] for t in otros]
    brechas_otros_esperado = [t.value - t.user_attrs["exactitud_verdadera"] for t in otros]
    rho_esperado, p_esperado = scipy_stats.spearmanr(verdad_otros_esperado, brechas_otros_esperado)
    semilla_datos = semillas_de(0, 1)[0].datos
    semillas_reeval = semillas_de(
        SEMILLA_BASE_REEVAL, N_REEVAL_POR_PRESUPUESTO["humo"], datos=semilla_datos
    )
    reevaluaciones = honestidad.reevaluar(
        {"ganador": dict(mejor.params), "linea_base": dict(linea_base_trial.params)},
        juguete.entrenar,
        semillas_reeval,
    )

    def _gaps(metricas):
        return [fila["exactitud_val"] - fila["exactitud_verdadera"] for fila in metricas]

    resumen_ganador = estadistica.resumen(_gaps(reevaluaciones["ganador"]))
    resumen_base = estadistica.resumen(_gaps(reevaluaciones["linea_base"]))
    inflado_ganador = mejor.value - mejor.user_attrs["exactitud_verdadera"]
    inflado_base = linea_base_trial.value - linea_base_trial.user_attrs["exactitud_verdadera"]
    e_ganador_esperado = inflado_ganador - resumen_ganador.media
    e_base_esperado = inflado_base - resumen_base.media
    visible_esperado = mejor.value - st.mean(
        fila["exactitud_val"] for fila in reevaluaciones["ganador"]
    )

    m2 = PATRON_RESPUESTA_2.search(corrida["stdout"])
    assert m2, corrida["stdout"]
    brecha_stdout = float(_campo(r"trials es ([+-][0-9]+\.[0-9]+) pts -- ", corrida["stdout"]))
    assert brecha_stdout == pytest.approx(100 * brecha_tipica_esperada, abs=0.06)
    assert float(m2["e_ganador"]) == pytest.approx(100 * e_ganador_esperado, abs=0.06)
    assert float(m2["e_base"]) == pytest.approx(100 * e_base_esperado, abs=0.06)
    assert float(m2["visible"]) == pytest.approx(100 * visible_esperado, abs=0.06)
    if resumen_ganador.de > 0:
        assert float(m2["z_ganador"]) == pytest.approx(
            e_ganador_esperado / resumen_ganador.de, abs=0.02
        )
    if resumen_base.de > 0:
        assert float(m2["z_base"]) == pytest.approx(e_base_esperado / resumen_base.de, abs=0.02)

    # La dificultad del conjunto de validación (revelación del oráculo, vía el
    # clasificador de Bayes) se recalcula igual que el paso: `paridad.generar`
    # + `paridad.posterior` sobre el mismo split (filas después de n_train).
    problema = {clave: juguete.DEFAULTS[clave] for clave in ("d", "a", "k_ruido", "rotar")}
    techo_esperado = 1.0 - paridad.error_bayes(problema["d"], problema["a"])
    n_train = juguete.DEFAULTS["n_train"]
    x_split, y_split = paridad.generar(
        n_train + juguete.DEFAULTS["n_val"], semilla=semilla_datos, **problema
    )
    p_bayes = paridad.posterior(
        x_split[n_train:], d=problema["d"], a=problema["a"], rotar=problema["rotar"]
    )
    acc_bayes_esperada = float(np.mean((p_bayes >= 0.5).astype(int) == y_split[n_train:]))
    ganancia_bayes_esperada = acc_bayes_esperada - techo_esperado
    acc_bayes_stdout = float(
        _campo(r"el clasificador de Bayes acierta " + FLOAT + r" en él", corrida["stdout"])
    )
    techo_stdout = float(_campo(r"por (?:encima|debajo) del techo de " + FLOAT, corrida["stdout"]))
    assert acc_bayes_stdout == pytest.approx(acc_bayes_esperada, abs=0.0005)
    assert techo_stdout == pytest.approx(techo_esperado, abs=0.0005)

    # Las frases de la respuesta 2, comparadas por IGUALDAD EXACTA contra las
    # funciones REALES que las escriben (extraídas por AST del propio paso),
    # evaluadas en los valores SIN REDONDEAR recalculados arriba -- nunca
    # contra una copia de la fórmula escrita aparte, ni contra floats ya
    # redondeados a 1-2 decimales leídos de stdout: cerca de un umbral, o con
    # un valor que redondea a "-0.0", esa reconstrucción puede discrepar del
    # valor exacto que el paso usó para decidir la rama.
    espacio = _extraer_del_paso(
        {
            "_frase_inflado",
            "_frase_sesgo_config",
            "_frase_bayes_gana",
            "_frase_z",
            "_frase_dificultad_conjunto",
            "_frase_spearman",
            "_z_semilla",
            "_umbral_z",
        }
    )
    umbral_esperado = espacio["_umbral_z"](N_REEVAL_POR_PRESUPUESTO["humo"])
    z_ganador_esperado = espacio["_z_semilla"](e_ganador_esperado, resumen_ganador.de)
    z_base_esperado = espacio["_z_semilla"](e_base_esperado, resumen_base.de)

    assert float(m2["umbral"]) == pytest.approx(umbral_esperado, abs=0.005)
    assert m2["frase_inflado"] == espacio["_frase_inflado"](inflado_ganador)
    assert m2["frase_sesgo"] == espacio["_frase_sesgo_config"](resumen_ganador)
    assert m2["frase_bayes"] == espacio["_frase_bayes_gana"](
        resumen_ganador, ganancia_bayes_esperada
    )
    assert m2["frase_z"] == espacio["_frase_z"](
        z_ganador_esperado, z_base_esperado, e_base_esperado, umbral_esperado
    )
    assert m2["frase_dificultad"] == espacio["_frase_dificultad_conjunto"](
        acc_bayes_esperada, techo_esperado
    )
    assert m2["frase_spearman"] == espacio["_frase_spearman"](rho_esperado, p_esperado, len(otros))


def test_paso_02_frases_cubren_las_ramas_con_valores_exactos():
    """Las funciones que redactan la respuesta 2 se prueban con SUS valores
    exactos (nunca con floats ya redondeados a 1-2 decimales leídos de
    stdout, que cerca de un umbral pueden discrepar del valor real que usó el
    paso para decidir la rama) y en las dos direcciones de cada rama,
    incluidos los bordes exactos del umbral -- donde un `<` en vez de un
    `<=` (o viceversa) sí cambiaría el resultado -- y el caso z_b fuertemente
    NEGATIVO en la rama "ganador afortunado", que distingue `abs(z_b) <=
    umbral` de `z_b <= umbral`: con esa segunda condición (mutada), una línea
    base que se desvía fuerte HACIA ABAJO caería igual en "el ganador tuvo
    una semilla más afortunada... y la línea base no", cuando en realidad el
    control también se movió tanto como el ganador.
    """
    espacio = _extraer_del_paso(
        {
            "_frase_inflado",
            "_frase_sesgo_config",
            "_frase_bayes_gana",
            "_frase_z",
            "_frase_dificultad_conjunto",
            "_frase_spearman",
            "_z_semilla",
            "_umbral_z",
        }
    )
    frase_inflado = espacio["_frase_inflado"]
    frase_sesgo_config = espacio["_frase_sesgo_config"]
    frase_bayes_gana = espacio["_frase_bayes_gana"]
    frase_z = espacio["_frase_z"]
    frase_dificultad = espacio["_frase_dificultad_conjunto"]
    frase_spearman = espacio["_frase_spearman"]
    z_semilla = espacio["_z_semilla"]
    umbral_z = espacio["_umbral_z"]

    def _resumen(n: int, media: float, ic_bajo: float, ic_alto: float) -> estadistica.Resumen:
        # `de`/`ee` no los usan `_frase_sesgo_config` ni `_frase_bayes_gana`
        # (solo miran `n`, `media`, `ic_bajo`, `ic_alto`): 0.0 es un
        # placeholder inerte, no un valor que decida ninguna rama.
        return estadistica.Resumen(
            n=n, media=media, de=0.0, ee=0.0, ic_bajo=ic_bajo, ic_alto=ic_alto, confianza=0.95
        )

    # `_umbral_z` (intervalo de PREDICCIÓN de una observación nueva, no de la
    # media): depende de n_reeval, nunca una constante fija. Los tres
    # presupuestos del paso (humo=2, taller=8, completo=16 reevaluaciones).
    assert umbral_z(2) == pytest.approx(7.73, abs=0.01)
    assert umbral_z(8) == pytest.approx(2.01, abs=0.01)
    assert umbral_z(16) == pytest.approx(1.81, abs=0.01)

    # Umbral fijo solo para ejercitar las cuatro ramas de `_frase_z` abajo
    # (independiente de qué presupuesto use `_umbral_z` de verdad): `_frase_z`
    # recibe el umbral como parámetro explícito, así que cualquier valor sirve
    # para cubrir sus ramas.
    UMBRAL_PRUEBA = 2.0

    # _frase_inflado: las dos direcciones y el borde en cero (>= es "sobreestima").
    assert frase_inflado(0.031) == "la validación sobreestima la verdad"
    assert frase_inflado(0.0) == "la validación sobreestima la verdad"
    assert frase_inflado(-0.002) == "la validación subestima la verdad"

    # _frase_sesgo_config: tres ramas por el intervalo de confianza al 95%, no
    # por el signo de la media cruda. (a) ic_bajo > 0: "mejor". (a') ic_alto <
    # 0: "peor". (b) el intervalo cruza cero (incluido el borde ic_bajo ==
    # 0.0, que NO cuenta como > 0, y el caso n=2, donde el intervalo sale más
    # ancho): "no se distingue de cero".
    resumen_mejor = _resumen(8, 0.02, 0.001, 0.035)
    resumen_peor = _resumen(8, -0.01, -0.02, -0.001)
    resumen_borde_cero = _resumen(5, 0.01, 0.0, 0.02)
    resumen_cruza_n2 = _resumen(2, 0.02, -0.05, 0.09)
    assert frase_sesgo_config(resumen_mejor) == (
        "con ESTOS datos de validación, esta config se ve en promedio +2.0 pts mejor que su "
        "verdad (intervalo de confianza 95% [+0.1, +3.5]): esa parte no depende de la semilla "
        "de orden/init -- re-evaluar con los mismos datos no la quita -- es el conjunto de "
        "validación, no la config"
    )
    assert frase_sesgo_config(resumen_peor) == (
        "con ESTOS datos de validación, esta config se ve en promedio -1.0 pts peor que su "
        "verdad (intervalo de confianza 95% [-2.0, -0.1]): esa parte no depende de la semilla "
        "de orden/init -- re-evaluar con los mismos datos no la quita -- es el conjunto de "
        "validación, no la config"
    )
    assert frase_sesgo_config(resumen_borde_cero) == (
        "con 5 semillas, la brecha media de esta config (+1.0 pts, intervalo de confianza 95% "
        "[+0.0, +2.0]) no se distingue de cero: no alcanza para atribuirla al conjunto de "
        "validación"
    )
    assert frase_sesgo_config(resumen_cruza_n2) == (
        "con 2 semillas, la brecha media de esta config (+2.0 pts, intervalo de confianza 95% "
        "[-5.0, +9.0]) no se distingue de cero: no alcanza para atribuirla al conjunto de "
        "validación"
    )

    # _frase_bayes_gana: "de esos X pts" SOLO cuando la brecha es atribuible
    # (rama (a) de arriba) Y Bayes se mueve en el MISMO sentido; en cualquier
    # otro caso (signo contrario, o brecha no distinguible de cero) el texto
    # neutro, que no le atribuye nada a nadie.
    assert frase_bayes_gana(resumen_mejor, 0.025) == (
        "de esos +2.0 pts, el clasificador de Bayes también gana +2.5 en este conjunto"
    )
    assert frase_bayes_gana(resumen_mejor, -0.03) == (
        "en este mismo conjunto el clasificador de Bayes gana -3.0 pts"
    )
    assert frase_bayes_gana(resumen_cruza_n2, 0.025) == (
        "en este mismo conjunto el clasificador de Bayes gana +2.5 pts"
    )

    # _z_semilla: división normal y las dos direcciones del caso de==0.
    assert z_semilla(0.01, 0.02) == pytest.approx(0.5)
    assert z_semilla(-0.01, 0.02) == pytest.approx(-0.5)
    assert z_semilla(0.0, 0.0) == 0.0
    assert z_semilla(0.03, 0.0) == math.inf
    assert z_semilla(-0.03, 0.0) == -math.inf

    # _frase_z: las cuatro ramas, incluido el borde exacto z_gan == umbral
    # (cae en "no se distingue", por el `<=`).
    assert frase_z(UMBRAL_PRUEBA, 0.0, -0.008, UMBRAL_PRUEBA) == (
        "no se distingue de la suerte de una semilla cualquiera (la línea base, que "
        "nadie eligió, se desvía -0.8 pts de su propio promedio)"
    )
    assert frase_z(1.5, 5.0, 0.01, UMBRAL_PRUEBA) == (
        "no se distingue de la suerte de una semilla cualquiera (la línea base, que "
        "nadie eligió, se desvía +1.0 pts de su propio promedio)"
    )
    assert frase_z(2.06, -0.17, -0.003, UMBRAL_PRUEBA) == (
        "el ganador tuvo una semilla de búsqueda más afortunada de lo que su config da "
        "en promedio y la línea base no: es lo esperable de la maldición del ganador, y "
        "con UN ganador es una anécdota (el paso 06 repite la búsqueda para medirla como "
        "distribución)"
    )
    assert frase_z(-2.5, 0.0, 0.0, UMBRAL_PRUEBA) == (
        "el ganador tuvo una semilla de búsqueda MENOS afortunada de lo que su config da "
        "en promedio: la búsqueda lo escogió pese a esa mala suerte, no por ella"
    )
    assert frase_z(3.0, 2.5, 0.021, UMBRAL_PRUEBA) == (
        "la línea base, que nadie eligió, también quedó fuera de su intervalo "
        "(|z|=2.50 > 2.00, en el mismo sentido): si un control no elegido también se sale, el "
        "intervalo no basta para atribuir la del ganador a la selección"
    )
    # Mismo borde de la rama "el control también se sale", pero con la línea
    # base desviada fuerte hacia ABAJO (z_b muy negativo, no positivo como
    # arriba): esta prueba mata un error que cambie `abs(z_b) <= umbral` por
    # `z_b <= umbral` en la condición de la rama 2 -- ese error deja pasar
    # z_b = -2.5 (que sí es <= 2.0) como si la línea base NO se hubiera
    # movido, e imprimiría la rama de "maldición del ganador" en vez de esta.
    # El texto distingue el sentido (z_b > 0 -> "mismo sentido"; si no,
    # "sentido contrario"), en vez de afirmar una igualdad de magnitud que
    # nadie comparó.
    assert frase_z(3.0, -2.5, -0.021, UMBRAL_PRUEBA) == (
        "la línea base, que nadie eligió, también quedó fuera de su intervalo "
        "(|z|=2.50 > 2.00, en sentido contrario): si un control no elegido también se sale, el "
        "intervalo no basta para atribuir la del ganador a la selección"
    )

    # _frase_dificultad_conjunto: el borde exacto acc == techo cuenta como "fácil" (`>=`).
    assert frase_dificultad(0.850, 0.825) == (
        "este conjunto de validación es fácil para los modelos buenos (el clasificador "
        "de Bayes acierta 0.850 en él, por encima del techo de 0.825 sobre toda la "
        "distribución)"
    )
    assert frase_dificultad(0.825, 0.825) == (
        "este conjunto de validación es fácil para los modelos buenos (el clasificador "
        "de Bayes acierta 0.825 en él, por encima del techo de 0.825 sobre toda la "
        "distribución)"
    )
    assert frase_dificultad(0.80, 0.825) == (
        "este conjunto de validación es más duro que el promedio (el clasificador de "
        "Bayes acierta 0.800 en él, por debajo del techo de 0.825 sobre toda la "
        "distribución)"
    )

    # _frase_spearman: p < 0.05 y rho > 0 es la ÚNICA rama que afirma relación;
    # el borde p == 0.05 y el borde rho == 0.0 caen en la rama conservadora.
    assert frase_spearman(0.505, 4.4e-05, 59) == (
        "la brecha crece con la verdad (Spearman(verdad, brecha)=0.505, p=4.4e-05 con "
        "59 trials): esa mediana subestima la de un modelo tan bueno como el ganador"
    )
    assert frase_spearman(0.217, 0.05, 9) == (
        "con 9 trials no hay relación detectable (Spearman(verdad, brecha)=0.217, "
        "p=5.0e-02): la mediana es solo una referencia gruesa"
    )
    assert frase_spearman(0.0, 0.001, 9) == (
        "con 9 trials no hay relación detectable (Spearman(verdad, brecha)=0.000, "
        "p=1.0e-03): la mediana es solo una referencia gruesa"
    )
    assert frase_spearman(-0.3, 0.001, 9) == (
        "con 9 trials no hay relación detectable (Spearman(verdad, brecha)=-0.300, "
        "p=1.0e-03): la mediana es solo una referencia gruesa"
    )


def test_paso_02_no_importa_motores_directos_y_usa_juguete_entrenar():
    """Guarda estática (AST): el juguete se entrena SIEMPRE a través de
    `taller_hpo.problemas.juguete` (nunca importando `mlp_paridad` ni
    `paridad_torch` directo, la convención de docs/arquitectura.md), y las
    dos llamadas que de verdad entrenan -- `en_proceso` para la búsqueda,
    `honestidad.reevaluar` para la re-evaluación -- reciben `juguete.entrenar`,
    no una copia.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"))

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            nombres = {alias.name for alias in nodo.names}
        elif isinstance(nodo, ast.ImportFrom):
            nombres = {nodo.module or ""} | {f"{nodo.module}.{alias.name}" for alias in nodo.names}
        else:
            continue
        prohibidos = {n for n in nombres if "mlp_paridad" in n or "paridad_torch" in n}
        assert not prohibidos, f"el paso no debe importar motores directos: {prohibidos}"

    llamadas_entrenar = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        if isinstance(nodo.func, ast.Name) and nodo.func.id == "en_proceso":
            llamadas_entrenar.append(("en_proceso", nodo.args[0]))
        if (
            isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "reevaluar"
            and isinstance(nodo.func.value, ast.Name)
            and nodo.func.value.id == "honestidad"
        ):
            llamadas_entrenar.append(("honestidad.reevaluar", nodo.args[1]))

    assert len(llamadas_entrenar) == 2, llamadas_entrenar
    for nombre, arg in llamadas_entrenar:
        es_juguete_entrenar = (
            isinstance(arg, ast.Attribute)
            and arg.attr == "entrenar"
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "juguete"
        )
        assert es_juguete_entrenar, f"{nombre} debe recibir juguete.entrenar, no {ast.dump(arg)}"

    # Guarda estática adicional: el umbral que decide la rama de `_frase_z` y
    # el umbral que la respuesta 2 imprime junto al z-score deben ser el
    # MISMO nombre (`UMBRAL_Z`), no dos expresiones que solo coinciden por
    # casualidad -- de otro modo, un texto que imprime un umbral distinto del
    # que de verdad decidió la rama pasaría inadvertido con presupuestos
    # donde las dos ramas caen del mismo lado.
    llamadas_frase_z = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "_frase_z"
    ]
    assert llamadas_frase_z, "no se encontró ninguna llamada a _frase_z en el paso"
    for llamada in llamadas_frase_z:
        assert len(llamada.args) >= 4, ast.dump(llamada)
        cuarto = llamada.args[3]
        assert isinstance(cuarto, ast.Name) and cuarto.id == "UMBRAL_Z", (
            "_frase_z debe recibir UMBRAL_Z como cuarto argumento (la misma constante que se "
            f"imprime junto al z-score), no {ast.dump(cuarto)}"
        )
