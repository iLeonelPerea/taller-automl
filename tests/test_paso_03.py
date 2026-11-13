"""Paso 03: poda contra early stopping.

`tests/test_pasos.py` ya corre `pasos/03_poda.py` con PRESUPUESTO=humo. Este archivo
verifica además el contenido: que las cinco respuestas numeradas se imprimen, que sus
números tienen sentido y que lo impreso coincide con lo que guardó el estudio persistido,
no solo con el propio texto de la corrida. Una fixture de `scope="module"` corre el paso
una sola vez y reparte el resultado entre las pruebas de contenido.
"""

from __future__ import annotations
import __future__

import ast
import dataclasses
import os
import re
import statistics
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import optuna
import pytest

from taller_hpo import estudio
from taller_hpo.estadistica import diferencia_pareada, ic_proporcion, resumen
from taller_hpo.problemas import paridad_torch
from taller_hpo.semillas import Semillas, semillas_de

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "03_poda.py"

ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)

# 2 * 0.0012: dos cotas de error Monte Carlo del oráculo (n_oraculo = 200_000,
# documentada en taller_hpo.problemas.paridad.riesgo_verdadero), el mismo
# margen de empate que usa pasos/03_poda.py::EMPATE_ORACULO -- una sola
# fuente de verdad para las pruebas que replican esa comparación.
EMPATE_ORACULO = 2 * 0.0012


def _cargar_funcion_pura(nombre: str):
    """Extrae y compila SOLO la `FunctionDef` `nombre` de `pasos/03_poda.py`
    -- más cualquier otra función de nivel de módulo que su cuerpo llame por
    nombre, de forma transitiva (así `conclusion_por_ic` arrastra a
    `categoria_por_ic` sin arrastrar todo el paso, que necesitaría objetos
    que solo existen al importar Optuna o al resolver el motor real) --
    nunca el módulo completo, que entrenaría trials de verdad al
    importarse: mismo motivo que ya obliga a
    `test_paso_no_importa_mlp_paridad_ni_paridad_torch_directo` a un
    análisis estático en vez de importar el paso -- los pasos son scripts,
    no módulos importables sin correr todo lo que hacen al importarlos. Se
    usa para ejercitar las funciones puras del paso (`categoria_por_ic`,
    `conclusion_por_ic`) con entradas sintéticas, sin correr ningún
    entrenamiento.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"), filename=str(PASO))
    funciones = {nodo.name: nodo for nodo in arbol.body if isinstance(nodo, ast.FunctionDef)}
    if nombre not in funciones:
        raise AssertionError(f"no se encontró la función {nombre!r} en {PASO.name}")

    necesarias: dict[str, ast.FunctionDef] = {}
    pendientes = [nombre]
    while pendientes:
        actual = pendientes.pop()
        if actual in necesarias:
            continue
        necesarias[actual] = funciones[actual]
        for llamada in ast.walk(funciones[actual]):
            if (
                isinstance(llamada, ast.Call)
                and isinstance(llamada.func, ast.Name)
                and llamada.func.id in funciones
                and llamada.func.id not in necesarias
            ):
                pendientes.append(llamada.func.id)

    modulo = ast.Module(body=list(necesarias.values()), type_ignores=[])
    ast.fix_missing_locations(modulo)
    espacio_nombres: dict = {}
    # flags=...compiler_flag, dont_inherit=True: sin esto, `compile` hereda el
    # `from __future__ import annotations` de ESTE archivo de prueba en vez de
    # decidirlo por sí mismo -- con eso, una anotación como `Sequence[...]` en
    # una función del paso (que no importa `Sequence`) solo compila porque el
    # future flag la deja como cadena sin evaluar, un acoplamiento oculto con
    # el import de este archivo, no con el módulo que de verdad se compila.
    codigo = compile(
        modulo,
        filename=str(PASO),
        mode="exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(codigo, espacio_nombres)
    return espacio_nombres[nombre]


# El espacio completo del paso 02 (sin `placebo`), tal como lo pide
# `pasos/03_poda.py::espacio` -- SIN recortar `ancho` ni `n_capas`.
DISTRIBUCIONES_ESPACIO = {
    "n_capas": optuna.distributions.IntDistribution(1, 3),
    "ancho": optuna.distributions.IntDistribution(8, 128, log=True),
    "learning_rate_init": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": optuna.distributions.FloatDistribution(1e-6, 1.0, log=True),
    "batch_size": optuna.distributions.CategoricalDistribution([32, 64, 128]),
}

# La misma política de semillas que debe quedar en el protocolo persistido:
# UNA Semillas para los tres estudios (ver pasos/03_poda.py::SEMILLA_BUSQUEDA).
SEMILLA_BUSQUEDA_ESPERADA = semillas_de(20_260_301, 1)[0]

PRUNER_KWARGS_ESPERADOS = {
    "nop": {},
    "median": {"n_startup_trials": 5, "n_warmup_steps": 5},
    "hyperband": {
        "min_resource": 3,
        "max_resource": paridad_torch.DEFAULTS["epocas"],
        "reduction_factor": 3,
    },
}

# Una config REAL (dentro del espacio de búsqueda) para sembrar trials PODADOS
# controlados: `pasos/03_poda.py` la re-entrena de verdad, así que su
# validación y verdad quedan dentro del rango calibrado del problema
# (`taller_hpo.problemas.mlp_paridad.DEFAULTS`: media 0.7024, rango
# 0.6283-0.7248 con DEFAULTS), lejos de los umbrales 0.50/0.99 que usan las
# pruebas de las cuatro ramas del veredicto más abajo.
CONFIG_REAL_CONTROLADA = {
    "n_capas": paridad_torch.DEFAULTS["n_capas"],
    "ancho": paridad_torch.DEFAULTS["ancho"],
    "learning_rate_init": paridad_torch.DEFAULTS["learning_rate_init"],
    "alpha": paridad_torch.DEFAULTS["alpha"],
    "batch_size": paridad_torch.DEFAULTS["batch_size"],
}

PATRON_RESPUESTA_3 = re.compile(
    r"3\. Median podó (?P<podados>\d+) trials; re-entrenados completos, "
    r"(?P<gan_val>\d+) habrían superado la mejor VALIDACIÓN \((?P<mejor_val>\d\.\d+)\) y "
    r"(?P<gan_verdad>\d+) la verdad del ganador de Median \((?P<mejor_verdad>\d\.\d+)\)\. "
    r"Entre estas mismas configuraciones, sin podar, el ganador de Median habría sido el trial "
    r"(?P<num_sin_podar>\d+) \(validación (?P<val_sin_podar>\d\.\d+), verdad "
    r"(?P<verdad_sin_podar>\d\.\d+)\) frente al ganador real, trial (?P<num_real>\d+) "
    r"\(validación (?P<val_real>\d\.\d+), verdad (?P<verdad_real>\d\.\d+)\)\. "
    r"(?P<veredicto>.+?) Nota: esta es UNA sola realización"
)

# Un solo grupo "cuerpo" para el detalle + la conclusión condicional (ver
# pasos/03_poda.py::detalle_cortes/frase_ahorro): sus fronteras no tienen un
# separador fijo entre sí, así que las pruebas afirman substrings dentro de
# "cuerpo"/"frase_ahorro" en vez de partirlos con más grupos con `.+?`
# ambiguos.
PATRON_RESPUESTA_4 = re.compile(
    r"4\. Con paciencia=(?P<paciencia>\d+) y el presupuesto real del problema "
    r"\((?P<epocas_reales>\d+) épocas\) sobre (?P<n_semillas>\d+) semillas nuevas, cortó en "
    r"(?P<cortes>\d+) de (?P<n_semillas2>\d+)\. (?P<cuerpo>.+?) Cambio en verdad promedio sobre "
    r"las (?P<n_semillas3>\d+) semillas \(parada - completo\): "
    r"(?P<cambio_verdad>[+-]\d+\.\d+)\. (?P<frase_ahorro>.+)"
)

# Cada corte, dentro de "cuerpo", con su intervalo de Wilson (pasos/03_poda.py::
# lista_cortes). `resto` recoge la conclusión condicional y, con cortes >= 2,
# el texto de la prueba pareada.
PATRON_DETALLE_CORTE = re.compile(
    r"^En las que cortó \((?P<lista>.+?)\), validación media (?P<val_corte>\d\.\d+) y verdad "
    r"media (?P<verdad_corte>\d\.\d+) contra (?P<verdad_completo>\d\.\d+) del completo "
    r"\(pérdida de (?P<perdida>-?\d+\.\d+) puntos porcentuales de exactitud verdadera, completo "
    r"- parada\)\.(?P<resto>.*)$"
)
PATRON_CORTE_ITEM = re.compile(
    r"época (?P<epoca>\d+) val (?P<val>\d\.\d+) "
    r"\((?:intervalo de confianza|intervalo) 95% "
    r"\[(?P<ic_bajo>\d\.\d+), (?P<ic_alto>\d\.\d+)\]\)"
)
PATRON_PAREADA_PACIENCIA = re.compile(
    r"Prueba pareada \(parada contra completo, delta=(?P<delta>\d+\.\d+)\): diferencia media "
    r"(?P<dif>[+-]\d+\.\d+) \(intervalo (?P<confianza>\d+)% "
    r"\[(?P<ic_bajo>-?\d+\.\d+), (?P<ic_alto>-?\d+\.\d+)\]\), veredicto: "
    # Los 4 veredictos reales de taller_hpo.estadistica.diferencia_pareada:
    # dos de ellos tienen un espacio ("no concluyente"), así que la
    # alternancia enumera los cuatro literales en vez de usar "\w+" (que
    # nunca matchea un espacio y dejaría a "no concluyente" sin capturar).
    r"(?P<veredicto>mejor|peor|equivalente|no concluyente)\."
)
PATRON_AHORRO_PACIENCIA = re.compile(
    r"Ahorró (?P<ahorradas>\d+) de (?P<posibles>\d+) épocas posibles \((?P<pct>\d+\.\d+)%\)\."
)


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


@pytest.fixture(scope="module")
def corrida(tmp_path_factory) -> dict:
    resultados = tmp_path_factory.mktemp("resultados_humo")
    estudios = tmp_path_factory.mktemp("estudios_humo")
    resultado = _correr("humo", resultados, estudios)
    assert resultado.returncode == 0, resultado.stderr
    # Nada de warnings sin explicar: este paso no usa ninguna API experimental
    # de Optuna a propósito, así que un "Warning" en stderr sería una
    # regresión silenciosa (p. ej. reabrir con una instancia de sampler
    # sembrada en vez de la fábrica) y no un aviso esperado.
    assert "Warning" not in resultado.stderr, resultado.stderr
    return {"stdout": resultado.stdout, "resultados": resultados, "estudios": estudios}


@pytest.fixture(scope="module")
def salida_humo(corrida) -> str:
    return corrida["stdout"]


@pytest.fixture(scope="module")
def estudios_persistidos(corrida) -> dict[str, optuna.Study]:
    resultado = {}
    for pruner in ("nop", "median", "hyperband"):
        destino = corrida["estudios"] / f"03_poda_{pruner}_humo.log"
        resultado[pruner] = optuna.load_study(
            study_name=f"03_poda_{pruner}", storage=estudio.abrir_storage(destino)
        )
    return resultado


@pytest.mark.gpu
def test_deja_su_propio_png_con_contenido(corrida):
    _, resultados = corrida["stdout"], corrida["resultados"]
    figura = resultados / "03_poda.png"
    assert figura.exists()
    # No solo "existe": un PNG vacío o truncado también "existiría".
    assert figura.stat().st_size > 1000


@pytest.mark.gpu
def test_imprime_el_motor_de_juguete_al_arrancar(salida_humo):
    # El proceso de pytest corre con el mismo entorno que el subproceso (`_correr`
    # parte de `os.environ`), así que `juguete.DESCRIPCION` calculado aquí debe
    # ser byte a byte el que imprimió el paso -- no solo el prefijo "Motor: ".
    assert salida_humo.startswith(f"Motor: {juguete.DESCRIPCION}")


@pytest.mark.gpu
def test_imprime_las_cinco_respuestas_numeradas(salida_humo):
    for inicio in (
        "1. Sin podar",
        "2. Mejor validación",
        "3. Median podó",
        "4. Con paciencia=5",
        "5. Trial.report en un estudio multiobjetivo levanta",
    ):
        assert inicio in salida_humo, salida_humo


def test_paso_no_importa_mlp_paridad_ni_paridad_torch_directo():
    """Estático (ast, sin ejecutar nada): el juguete se entrena SIEMPRE vía
    `taller_hpo.problemas.juguete` (convención de docs/arquitectura.md). Mismo
    chequeo que `test_pasos_con_juguete_no_entrenan_via_mlp_paridad_directo`
    en tests/test_pasos.py, repetido aquí para este paso en concreto.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"), filename=str(PASO))
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
    assert not encontrados, f"{PASO.name} importa {sorted(encontrados)} directo"


def test_contrafactual_atrapa_divergencia_al_reentrenar_un_podado():
    """El re-entrenamiento de la pregunta 3 debe atrapar `Divergencia`
    alrededor de la llamada a `entrenar` (nunca un `except` desnudo ni un
    `except Exception` amplio): un podado que diverge después de su época de
    poda no debe tirar el paso entero por una sola configuración mala, igual
    que `atrapar=(Divergencia,)` ya hace durante la búsqueda real.

    Estático (ast), no dinámico: verifica la MECÁNICA del `try/except` en el
    código (que exista, que no sea un `except` desnudo ni `except Exception`,
    que atrape `Divergencia`). La rama SÍ se ejercita de verdad, con una
    config que diverge (`learning_rate_init=1e200`, medido: `Divergencia:
    pérdida o salidas no finitas en la época 0 con learning_rate_init=1e+200`)
    en `test_contrafactual_reentrenar_podado_que_diverge_se_excluye_del_paso`.
    """
    arbol = ast.parse(PASO.read_text(encoding="utf-8"), filename=str(PASO))
    for nodo in ast.walk(arbol):
        if not (isinstance(nodo, ast.For) and isinstance(nodo.iter, ast.Name)):
            continue
        if nodo.iter.id != "podados_median":
            continue
        intentos = [n for n in ast.walk(nodo) if isinstance(n, ast.Try)]
        assert intentos, "el for sobre podados_median no tiene ningún try/except"
        for intento in intentos:
            for handler in intento.handlers:
                assert handler.type is not None, (
                    "except desnudo alrededor del re-entrenamiento del contrafactual"
                )
                assert not (
                    isinstance(handler.type, ast.Name) and handler.type.id == "Exception"
                ), "except Exception amplio alrededor del re-entrenamiento del contrafactual"
        atrapa_divergencia = any(
            isinstance(handler.type, ast.Name) and handler.type.id == "Divergencia"
            for intento in intentos
            for handler in intento.handlers
        )
        assert atrapa_divergencia, "el try del contrafactual no atrapa Divergencia"
        return
    raise AssertionError("no se encontró el for sobre podados_median en pasos/03_poda.py")


@pytest.mark.gpu
def test_contrafactual_reentrenar_podado_que_diverge_se_excluye_del_paso(tmp_path):
    """Ejercita de verdad la rama `except Divergencia` del contrafactual:
    `entrenar({**CONFIG_REAL_CONTROLADA, 'learning_rate_init': 1e200}, ...)`
    diverge (medido: `Divergencia: pérdida o salidas no finitas en la época 0
    con learning_rate_init=1e+200`, con 1e3/1e8/1e30 no diverge). Se siembra
    UN podado con esa config exacta en `user_attrs['config']` (los demás con
    `CONFIG_REAL_CONTROLADA`, que no diverge): el paso debe imprimir
    're-entrenado trial N: diverge sin podar (excluido)' para ese trial,
    salir con 0, y dejar a N fuera del argmax `ganador_sin_podar`.
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    n_total = 8
    podados_semillas = semillas_de(20_260_777, n_total - 1)
    config_divergente = {**CONFIG_REAL_CONTROLADA, "learning_rate_init": 1e200}
    configs = [CONFIG_REAL_CONTROLADA] * (len(podados_semillas) - 1) + [config_divergente]
    _sembrar_estudio_median_controlado(
        estudios,
        n_total=n_total,
        val_ganador=0.5,
        verdad_ganador=0.5,
        podados=podados_semillas,
        configs=configs,
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr

    # El ganador sembrado ocupa el trial 0; los podados son 1..n_total-1 en
    # el mismo orden que `podados_semillas`/`configs`, así que el último
    # (la config divergente) es el trial n_total - 1.
    trial_divergente = n_total - 1
    assert f"re-entrenado trial {trial_divergente}: diverge sin podar (excluido)" in salida.stdout

    m = PATRON_RESPUESTA_3.search(salida.stdout)
    assert m, salida.stdout
    assert int(m["num_sin_podar"]) != trial_divergente, salida.stdout
    assert int(m["podados"]) == n_total - 1


@pytest.mark.gpu
def test_distribuciones_de_cada_trial_son_el_espacio_completo_sin_recortar(estudios_persistidos):
    """Ningún trial (de ningún pruner) debe tener un espacio recortado: `ancho`
    sigue en [8, 128] (log) y `n_capas` en [1, 3], exactamente el espacio del
    paso 02 sin `placebo` -- nunca [8, 64] / [1, 2] recortado para ahorrar tiempo.
    """
    for pruner, study in estudios_persistidos.items():
        trials = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
        assert trials, f"el estudio {pruner!r} no tiene trials terminados"
        for trial in trials:
            assert trial.distributions == DISTRIBUCIONES_ESPACIO, (
                f"{pruner} trial {trial.number}: {trial.distributions}"
            )


@pytest.mark.gpu
def test_protocolo_persistido_documenta_sampler_pruner_y_politica_de_semillas(
    estudios_persistidos,
):
    """El protocolo debe bastar para saber CÓMO se buscó sin abrir un
    dashboard (docs/arquitectura.md: "protocolo: sampler y sus kwargs,
    pruner, política de semillas, métrica"): no solo el NOMBRE del sampler y
    el pruner, también sus kwargs completos y la política de semillas (CRN,
    UNA Semillas para los tres estudios).
    """
    semillas_esperadas = dataclasses.asdict(SEMILLA_BUSQUEDA_ESPERADA)
    for pruner, study in estudios_persistidos.items():
        protocolo = study.user_attrs["protocolo"]
        assert protocolo["pruner"] == pruner
        assert protocolo["pruner_kwargs"] == PRUNER_KWARGS_ESPERADOS[pruner]
        assert protocolo["sampler"] == "TPESampler"
        assert protocolo["sampler_kwargs"] == {
            "seed_base": 20_260_302,
            "seed": "seed_base + n_previos",
            "n_startup_trials": 10,
            "multivariate": True,
            # False, igual que pasos/02_primer_estudio.py: con n_jobs=1 no hay
            # trials concurrentes a los que "mentirles" con un valor provisional.
            "constant_liar": False,
        }
        assert protocolo["politica_semillas"] == {
            "tipo": "CRN",
            "base": 20_260_301,
            "n": 1,
            "semillas": semillas_esperadas,
        }
        assert protocolo["metrica"] == "exactitud_val"
        assert protocolo["n_total"] == 8
        assert protocolo["motor"] == juguete.MOTOR
        assert protocolo["dispositivo"] == juguete.DISPOSITIVO.split(":", 1)[0]


@pytest.mark.gpu
def test_los_tres_estudios_usan_la_misma_semillas_en_todos_sus_trials(estudios_persistidos):
    """CRN de verdad: TODOS los trials, de los TRES estudios, deben compartir
    la MISMA `Semillas` (no una por trial). Con una `Semillas` distinta por
    trial, Median compararía curvas de aprendizaje medidas sobre conjuntos de
    VALIDACIÓN distintos entre trials -- ruido ajeno al pruner mezclado con el
    efecto real de podar.
    """
    esperada = dataclasses.asdict(SEMILLA_BUSQUEDA_ESPERADA)
    for pruner, study in estudios_persistidos.items():
        trials = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
        for trial in trials:
            assert trial.user_attrs["semillas"] == esperada, (
                f"{pruner} trial {trial.number}: {trial.user_attrs['semillas']}"
            )


@pytest.mark.gpu
def test_pruners_respetan_sus_propios_kwargs_persistidos(estudios_persistidos):
    """Ningún PRUNED puede violar los kwargs que su propio protocolo declara:
    Median no poda antes de `n_startup_trials` trials completados ni antes de
    reportar `n_warmup_steps` épocas, y Hyperband no poda antes de
    `min_resource`. Con los pruners construidos DESDE `PRUNER_KWARGS` (única
    fuente de verdad en pasos/03_poda.py) el protocolo persistido y el pruner
    que de verdad corrió no pueden desincronizarse entre sí.
    """
    median = estudios_persistidos["median"]
    podados_median = median.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    assert podados_median
    for trial in podados_median:
        assert trial.number >= PRUNER_KWARGS_ESPERADOS["median"]["n_startup_trials"]
        assert max(trial.intermediate_values) >= PRUNER_KWARGS_ESPERADOS["median"]["n_warmup_steps"]

    hyperband = estudios_persistidos["hyperband"]
    podados_hb = hyperband.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    assert podados_hb
    min_resource = PRUNER_KWARGS_ESPERADOS["hyperband"]["min_resource"]
    for trial in podados_hb:
        assert max(trial.intermediate_values) >= min_resource


@pytest.mark.gpu
def test_respuesta_1_epocas_y_ahorro_coinciden_con_lo_persistido(salida_humo, estudios_persistidos):
    m_nop = re.search(r"consumieron (\d+) épocas", salida_humo)
    assert m_nop, salida_humo
    m = re.search(
        r"Median usó (\d+): ahorró (\d+\.\d+)% \((\d+) épocas\)\. "
        r"Hyperband usó (\d+): ahorró (\d+\.\d+)% \((\d+) épocas\)\.",
        salida_humo,
    )
    assert m, salida_humo
    epocas_nop_txt = int(m_nop.group(1))
    epocas_median, ahorro_median, ahorradas_median = m.group(1, 2, 3)
    epocas_hb, ahorro_hb, ahorradas_hb = m.group(4, 5, 6)
    epocas_median, ahorro_median, ahorradas_median = (
        int(epocas_median),
        float(ahorro_median),
        int(ahorradas_median),
    )
    epocas_hb, ahorro_hb, ahorradas_hb = int(epocas_hb), float(ahorro_hb), int(ahorradas_hb)

    def epocas_reales(pruner: str) -> int:
        trials = estudios_persistidos[pruner].get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
        return int(sum(t.user_attrs["epocas"] for t in trials))

    epocas_nop = epocas_reales("nop")
    assert epocas_nop_txt == epocas_nop
    assert epocas_median == epocas_reales("median")
    assert epocas_hb == epocas_reales("hyperband")

    # El ahorro es 100*(1 - podado/nop) y las épocas ahorradas son la resta:
    # con el signo invertido (la fracción CONSUMIDA en vez de la ahorrada)
    # estas dos igualdades no se sostienen -- por eso prueban que la fórmula
    # impresa es la correcta, no solo consistente consigo misma.
    assert ahorro_median == pytest.approx(100 * (1 - epocas_median / epocas_nop), abs=0.05)
    assert ahorro_hb == pytest.approx(100 * (1 - epocas_hb / epocas_nop), abs=0.05)
    assert ahorradas_median == epocas_nop - epocas_median
    assert ahorradas_hb == epocas_nop - epocas_hb

    # NopPruner nunca poda: es la cota superior de épocas. Median y Hyperband
    # solo pueden cortar ANTES, nunca correr de más.
    assert 0.0 <= ahorro_median < 100.0
    assert 0.0 <= ahorro_hb < 100.0


@pytest.mark.gpu
def test_respuesta_1_median_e_hyperband_podan_y_nop_nunca(estudios_persistidos):
    podados = {
        pruner: len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,)))
        for pruner, study in estudios_persistidos.items()
    }
    assert podados["nop"] == 0
    assert podados["median"] >= 1
    assert podados["hyperband"] >= 1


@pytest.mark.gpu
def test_respuesta_2_las_verdades_impresas_coinciden_con_el_best_trial_persistido(
    salida_humo, estudios_persistidos
):
    m = re.search(
        r"Mejor validación: nop (\d\.\d+), median (\d\.\d+), hyperband (\d\.\d+)\. "
        r"Su exactitud verdadera: nop (\d\.\d+), median (\d\.\d+) \(([+-]\d\.\d+) vs\. nop\), "
        r"hyperband (\d\.\d+) \(([+-]\d\.\d+) vs\. nop\)\.",
        salida_humo,
    )
    assert m, salida_humo
    val_nop, val_median, val_hb = (float(x) for x in m.group(1, 2, 3))
    verdad_nop, verdad_median, delta_median = m.group(4, 5, 6)
    verdad_hb, delta_hb = m.group(7, 8)
    verdad_nop, verdad_median, delta_median = (
        float(verdad_nop),
        float(verdad_median),
        float(delta_median),
    )
    verdad_hb, delta_hb = float(verdad_hb), float(delta_hb)

    for pruner, val_txt, verdad_txt in (
        ("nop", val_nop, verdad_nop),
        ("median", val_median, verdad_median),
        ("hyperband", val_hb, verdad_hb),
    ):
        mejor = estudios_persistidos[pruner].best_trial
        assert val_txt == pytest.approx(mejor.value, abs=0.0005)
        # La verdad impresa debe ser la del ORÁCULO (user_attrs), no la
        # validación otra vez: comparar contra el user_attr directo, no
        # contra `mejor.value`, es lo que detecta "la verdad impresa es en
        # realidad la validación".
        assert verdad_txt == pytest.approx(mejor.user_attrs["exactitud_verdadera"], abs=0.0005)

    assert delta_median == pytest.approx(verdad_median - verdad_nop, abs=0.0005)
    assert delta_hb == pytest.approx(verdad_hb - verdad_nop, abs=0.0005)

    # La frase depende del signo y del tamaño medidos, nunca de un texto fijo
    # que no dependa del resultado: sin réplicas el paso no puede afirmar que
    # una diferencia sea peor o mejor, así que en vez de esa comparación
    # imprime que hacen falta réplicas y remite a la prueba pareada del
    # paso 06.
    assert "no tiene por qué significar un peor" not in salida_humo
    assert "hacen falta réplicas: repetir los tres estudios" in salida_humo
    assert "compararlos pareados (el método del paso 06)" in salida_humo


@pytest.mark.gpu
def test_respuesta_2_delta_verdad_tiene_el_signo_correcto(tmp_path):
    """Sin sembrar semillas, en una corrida real los tres pruners con el
    mismo TPE y el mismo CRN suelen elegir el mismo trial ganador, así que el
    delta impreso sale +0.000 y no distingue si la resta `verdad_median -
    verdad_nop` tiene el signo correcto. Sembrando nop y median con verdades
    DISTINTAS (0.70 y 0.75) el delta impreso debe reflejar ese signo.
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    n_total = 8
    _sembrar_estudio_ganador_controlado(
        estudios / "03_poda_nop_humo.log",
        pruner="nop",
        n_total=n_total,
        val_ganador=0.9,
        verdad_ganador=0.70,
    )
    _sembrar_estudio_ganador_controlado(
        estudios / "03_poda_median_humo.log",
        pruner="median",
        n_total=n_total,
        val_ganador=0.9,
        verdad_ganador=0.75,
    )
    _sembrar_estudio_terminado(
        estudios / "03_poda_hyperband_humo.log", pruner="hyperband", n_total=n_total
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr
    assert "(+0.050 vs. nop)" in salida.stdout, salida.stdout


@pytest.mark.gpu
def test_respuesta_3_contrafactual_consistente_con_lo_reentrenado_y_lo_persistido(
    salida_humo, estudios_persistidos
):
    m = PATRON_RESPUESTA_3.search(salida_humo)
    assert m, salida_humo
    podados_txt = int(m["podados"])
    ganadores_val_txt = int(m["gan_val"])
    mejor_val_txt = float(m["mejor_val"])
    ganadores_verdad_txt = int(m["gan_verdad"])
    mejor_verdad_txt = float(m["mejor_verdad"])
    veredicto = m["veredicto"]

    median = estudios_persistidos["median"]
    podados = median.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    assert podados_txt == len(podados)

    ganador_real = median.best_trial
    verdad_real = ganador_real.user_attrs["exactitud_verdadera"]
    assert mejor_val_txt == pytest.approx(ganador_real.value, abs=0.0005)
    assert mejor_verdad_txt == pytest.approx(verdad_real, abs=0.0005)
    assert int(m["num_real"]) == ganador_real.number
    assert float(m["val_real"]) == pytest.approx(ganador_real.value, abs=0.0005)
    assert float(m["verdad_real"]) == pytest.approx(
        ganador_real.user_attrs["exactitud_verdadera"], abs=0.0005
    )

    reentrenos = re.findall(
        r"re-entrenado trial (\d+): (\d+) épocas completas \(sin podar\), "
        r"validación (\d\.\d+), verdadera (\d\.\d+)",
        salida_humo,
    )
    # Un podado que diverge al re-entrenar se imprime distinto y se excluye
    # de las categorías (ver pasos/03_poda.py): la unión de re-entrenados y
    # divergentes, no solo `reentrenos`, debe agotar los podados.
    divergentes = re.findall(
        r"re-entrenado trial (\d+): diverge sin podar \(excluido\)", salida_humo
    )
    numeros_podados = {t.number for t in podados}
    numeros_reentrenados = {int(n) for n, _, _, _ in reentrenos}
    numeros_divergentes = {int(n) for n in divergentes}
    assert numeros_reentrenados | numeros_divergentes == numeros_podados, salida_humo
    assert len(reentrenos) + len(divergentes) == len(podados), salida_humo

    # Tres categorías mutuamente excluyentes, según cómo se compara cada
    # re-entrenado contra el ganador REAL (no contra "la mejor verdad del
    # estudio"): un ganador tardío de verdad exige que la búsqueda TAMBIÉN lo
    # hubiera preferido (validación), no solo que el oráculo lo prefiera.
    tardios, solo_verdad, solo_val = [], [], []
    candidatos_sin_podar = [
        (ganador_real.number, ganador_real.value, ganador_real.user_attrs["exactitud_verdadera"])
    ]
    for numero, epocas_txt, val_txt, verdad_txt in reentrenos:
        # Contrafactual "completo": SIN podar, así que corre las EPOCAS del
        # problema enteras -- nunca corta a mitad de camino.
        assert int(epocas_txt) == juguete.DEFAULTS["epocas"]
        val, verdad = float(val_txt), float(verdad_txt)
        candidatos_sin_podar.append((int(numero), val, verdad))
        # + EMPATE_ORACULO en la comparación de verdad (nunca en la de
        # validación): dos verdades que difieren por menos que el margen de
        # empate del oráculo no se pueden ordenar de forma confiable (ver
        # pasos/03_poda.py::EMPATE_ORACULO), así que "superó la verdad" exige
        # superarla más allá de ese margen, no por un solo epsilon de ruido
        # Monte Carlo del oráculo.
        if val > mejor_val_txt and verdad > mejor_verdad_txt + EMPATE_ORACULO:
            tardios.append(int(numero))
        elif verdad > mejor_verdad_txt + EMPATE_ORACULO:
            solo_verdad.append(int(numero))
        elif val > mejor_val_txt:
            solo_val.append(int(numero))

    assert ganadores_val_txt == len(tardios) + len(solo_val)
    assert ganadores_verdad_txt == len(tardios) + len(solo_verdad)
    # Los "ganadores" salen de RE-ENTRENAR los trials podados: no puede haber
    # más que trials podados, en ninguna métrica.
    assert 0 <= ganadores_val_txt <= podados_txt
    assert 0 <= ganadores_verdad_txt <= podados_txt

    # "Sin podar": el ganador entre los COMPLETE reales de Median y los
    # podados re-entrenados, desempatando por el número de trial más bajo
    # (mismo criterio que Study.best_trial).
    completos = median.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in completos:
        if t.number != ganador_real.number:
            candidatos_sin_podar.append((t.number, t.value, t.user_attrs["exactitud_verdadera"]))
    ganador_sin_podar_esperado = max(sorted(candidatos_sin_podar), key=lambda c: c[1])
    numero_sin_podar_esperado = ganador_sin_podar_esperado[0]
    assert int(m["num_sin_podar"]) == numero_sin_podar_esperado
    assert float(m["val_sin_podar"]) == pytest.approx(ganador_sin_podar_esperado[1], abs=0.0005)
    assert float(m["verdad_sin_podar"]) == pytest.approx(ganador_sin_podar_esperado[2], abs=0.0005)

    # El veredicto sale del argmax de validación ya verificado arriba contra
    # lo persistido (`ganador_sin_podar_esperado`), NUNCA de
    # `tardios`/`solo_verdad`/`solo_val`: esas listas son descriptivas.
    # "Sí mató un ganador tardío" exige que ESE argmax sea un podado Y que su
    # verdad supere a la del ganador real por encima del error del oráculo;
    # dentro de esa cota (empate), ninguna de las dos se declara mejor; por
    # debajo (PEOR), podar evitó un ganador peor (nunca "mató"); si el argmax
    # sigue siendo el ganador real pero el oráculo prefiere a algún podado,
    # podar no cambió a quién elige la búsqueda. `frase_trial` verifica QUÉ trial
    # nombra el veredicto, no solo que contenga el texto de la rama: un
    # veredicto que nombrara al ganador real en vez del argmax también haría
    # match con los `assert` de fase si solo se mirara la frase de la rama.
    podado_gano_esperado = numero_sin_podar_esperado != ganador_real.number
    verdad_sin_podar_esperada = ganador_sin_podar_esperado[2]
    diferencia_esperada = verdad_sin_podar_esperada - mejor_verdad_txt
    # 2 * 0.0012: dos cotas de error Monte Carlo del oráculo (n_oraculo =
    # 200_000, documentada en taller_hpo.problemas.paridad.riesgo_verdadero).
    empate_oraculo = 2 * 0.0012
    frase_trial = f"el trial {numero_sin_podar_esperado} por validación"

    if podado_gano_esperado and abs(diferencia_esperada) <= empate_oraculo:
        assert "empatan dentro del error del oráculo" in veredicto, veredicto
        assert frase_trial in veredicto, veredicto
    elif podado_gano_esperado and diferencia_esperada > 0:
        assert "Sí mató un ganador tardío" in veredicto, veredicto
        assert frase_trial in veredicto, veredicto
    elif podado_gano_esperado:
        assert "Sí mató un ganador tardío" not in veredicto, veredicto
        assert "PEOR" in veredicto, veredicto
        assert frase_trial in veredicto, veredicto
    elif solo_verdad:
        assert "Sí mató un ganador tardío" not in veredicto, veredicto
        assert "igual habría elegido al mismo ganador" in veredicto, veredicto
        assert str(sorted(solo_verdad)) in veredicto, veredicto
    else:
        assert "Ningún ganador tardío" in veredicto, veredicto


@pytest.mark.gpu
def test_respuesta_3_contrafactual_recalculado_en_proceso_coincide_con_lo_impreso(
    salida_humo, estudios_persistidos
):
    """Recalcula, EN el proceso de pytest, cada re-entrenamiento que
    pasos/03_poda.py imprime para la pregunta 3: la MISMA config y semillas
    que `en_proceso` guardó en el trial PODADO (nunca otras semillas ni la
    config de otro trial) deben reproducir la validación y la verdad
    impresas, y la curva reportada hasta el punto de poda debe coincidir con
    las `intermediate_values` que Optuna ya guardó de la corrida real --
    mismos datos, mismo orden, mismo modelo, así que entrenar de nuevo hasta
    ESE punto no puede dar un número distinto.
    """
    reentrenos = re.findall(
        r"re-entrenado trial (\d+): (\d+) épocas completas \(sin podar\), "
        r"validación (\d\.\d+), verdadera (\d\.\d+)",
        salida_humo,
    )
    assert reentrenos, salida_humo
    median = estudios_persistidos["median"]
    podados_por_numero = {
        t.number: t
        for t in median.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    }
    for numero_txt, _epocas_txt, val_txt, verdad_txt in reentrenos:
        trial = podados_por_numero[int(numero_txt)]
        config = trial.user_attrs["config"]
        semillas_trial = Semillas(**trial.user_attrs["semillas"])
        curva: list[tuple[int, float]] = []

        def _registrar(epoca: int, valor: float, curva=curva) -> bool:
            curva.append((epoca, valor))
            return False

        recalculado = juguete.entrenar(config, semillas_trial, _registrar)
        assert recalculado["exactitud_val"] == pytest.approx(float(val_txt), abs=5e-4)
        assert recalculado["exactitud_verdadera"] == pytest.approx(float(verdad_txt), abs=5e-4)

        curva_hasta_la_poda = dict(curva)
        for epoca, valor in trial.intermediate_values.items():
            assert curva_hasta_la_poda[epoca] == pytest.approx(valor, abs=5e-4)


def _protocolo_pruner_esperado(pruner: str, n_total: int) -> dict:
    return {
        "paso": "03_poda",
        "pruner": pruner,
        "pruner_kwargs": PRUNER_KWARGS_ESPERADOS[pruner],
        "sampler": "TPESampler",
        "sampler_kwargs": {
            "seed_base": 20_260_302,
            "seed": "seed_base + n_previos",
            "n_startup_trials": 10,
            "multivariate": True,
            "constant_liar": False,
        },
        "politica_semillas": {
            "tipo": "CRN",
            "base": 20_260_301,
            "n": 1,
            "semillas": dataclasses.asdict(SEMILLA_BUSQUEDA_ESPERADA),
        },
        "metrica": "exactitud_val",
        "n_total": n_total,
        **juguete.PROTOCOLO_MOTOR,
    }


def _fabrica_sampler_controlada(n_previos: int) -> optuna.samplers.TPESampler:
    return optuna.samplers.TPESampler(
        seed=20_260_302 + n_previos, n_startup_trials=10, multivariate=True, constant_liar=False
    )


_CLASES_PRUNER_TEST: dict[str, type[optuna.pruners.BasePruner]] = {
    "nop": optuna.pruners.NopPruner,
    "median": optuna.pruners.MedianPruner,
    "hyperband": optuna.pruners.HyperbandPruner,
}


def _sembrar_estudio_terminado(destino: Path, *, pruner: str, n_total: int) -> None:
    """Pre-siembra `03_poda_{pruner}_humo.log` con `n_total` trials COMPLETE
    ficticios (config real dentro del espacio, valor y verdad arbitrarios):
    con `estudio.optimizar` viendo ya `n_total` trials terminados no corre
    ningún trial nuevo. Los estudios `nop`/`hyperband` no se inspeccionan en
    las pruebas que llaman a esto (solo la pregunta 3, sobre `median`), así
    que sus números no importan -- ahorra 2*n_total entrenamientos reales por
    corrida frente a dejar que el paso los corra desde cero.
    """
    estudio_previo = estudio.crear_estudio(
        f"03_poda_{pruner}",
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler_controlada,
        pruner=_CLASES_PRUNER_TEST[pruner](**PRUNER_KWARGS_ESPERADOS[pruner]),
        protocolo=_protocolo_pruner_esperado(pruner, n_total),
    )
    for _ in range(n_total):
        estudio_previo.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=0.5,
                params=CONFIG_REAL_CONTROLADA,
                distributions=DISTRIBUCIONES_ESPACIO,
                user_attrs={
                    "exactitud_verdadera": 0.5,
                    "epocas": float(juguete.DEFAULTS["epocas"]),
                },
            )
        )


def _sembrar_estudio_ganador_controlado(
    destino: Path, *, pruner: str, n_total: int, val_ganador: float, verdad_ganador: float
) -> None:
    """Pre-siembra `03_poda_{pruner}_humo.log` con UN trial COMPLETE ganador
    (validación y verdad FALSAS y controladas) y `n_total - 1` trials
    COMPLETE de relleno con menos validación (para que el ganador sea
    inequívoco): permite fijar la verdad del `best_trial` de un pruner sin
    depender de qué tan buena sea la búsqueda real (ver
    `test_respuesta_2_delta_verdad_tiene_el_signo_correcto`).
    """
    estudio_previo = estudio.crear_estudio(
        f"03_poda_{pruner}",
        destino=destino,
        direcciones=["maximize"],
        sampler=_fabrica_sampler_controlada,
        pruner=_CLASES_PRUNER_TEST[pruner](**PRUNER_KWARGS_ESPERADOS[pruner]),
        protocolo=_protocolo_pruner_esperado(pruner, n_total),
    )
    estudio_previo.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=val_ganador,
            params=CONFIG_REAL_CONTROLADA,
            distributions=DISTRIBUCIONES_ESPACIO,
            user_attrs={
                "exactitud_verdadera": verdad_ganador,
                "epocas": float(juguete.DEFAULTS["epocas"]),
            },
        )
    )
    for _ in range(n_total - 1):
        estudio_previo.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=val_ganador - 0.1,
                params=CONFIG_REAL_CONTROLADA,
                distributions=DISTRIBUCIONES_ESPACIO,
                user_attrs={
                    "exactitud_verdadera": 0.5,
                    "epocas": float(juguete.DEFAULTS["epocas"]),
                },
            )
        )


def _sembrar_estudio_median_controlado(
    estudios: Path,
    *,
    n_total: int,
    val_ganador: float,
    verdad_ganador: float,
    podados: Sequence[Semillas] | None = None,
    configs: Sequence[dict] | None = None,
    verdad_podados: float = 0.5,
) -> None:
    """Pre-siembra `estudios/03_poda_median_humo.log` con exactamente `n_total` trials
    terminados: un COMPLETE con validación y verdad falsas (el "ganador") y el resto PRUNED
    con config y semillas reales, que el paso re-entrena. Como `optimizar` ya ve `n_total`
    trials, no corre ninguno nuevo y el ganador queda fijo. También siembra los estudios
    `nop` y `hyperband` con trials ficticios, para no correrlos en cada prueba.

    - `podados`: las semillas exactas de los `n_total - 1` trials PRUNED, para forzar un par
      discordante real.
    - `configs`: la config de cada podado, para sembrar uno que diverge al re-entrenar.
    - `verdad_podados`: el `exactitud_verdadera` de cada podado, independiente de su
      validación, para distinguir si la pregunta 4 clasifica por validación o por oráculo.
    """
    if podados is None:
        podados = semillas_de(20_260_777, n_total - 1)
    assert len(podados) == n_total - 1, (
        f"se esperaban {n_total - 1} semillas podadas, no {len(podados)}"
    )
    if configs is None:
        configs = [CONFIG_REAL_CONTROLADA] * len(podados)
    assert len(configs) == len(podados), (
        f"se esperaban {len(podados)} configs (una por podado), no {len(configs)}"
    )
    estudio_previo = estudio.crear_estudio(
        "03_poda_median",
        destino=estudios / "03_poda_median_humo.log",
        direcciones=["maximize"],
        sampler=_fabrica_sampler_controlada,
        pruner=optuna.pruners.MedianPruner(**PRUNER_KWARGS_ESPERADOS["median"]),
        protocolo=_protocolo_pruner_esperado("median", n_total),
    )
    estudio_previo.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=val_ganador,
            params=CONFIG_REAL_CONTROLADA,
            distributions=DISTRIBUCIONES_ESPACIO,
            user_attrs={
                "exactitud_verdadera": verdad_ganador,
                "epocas": float(juguete.DEFAULTS["epocas"]),
            },
        )
    )
    for semillas_trial, config_trial in zip(podados, configs, strict=True):
        estudio_previo.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.PRUNED,
                params=CONFIG_REAL_CONTROLADA,
                distributions=DISTRIBUCIONES_ESPACIO,
                user_attrs={
                    "config": dict(config_trial),
                    "semillas": dataclasses.asdict(semillas_trial),
                    "exactitud_verdadera": verdad_podados,
                    "epocas": 5.0,
                },
                intermediate_values={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
            )
        )
    _sembrar_estudio_terminado(estudios / "03_poda_nop_humo.log", pruner="nop", n_total=n_total)
    _sembrar_estudio_terminado(
        estudios / "03_poda_hyperband_humo.log", pruner="hyperband", n_total=n_total
    )


@pytest.fixture(scope="module")
def podados_controlados_reentrenados() -> list[tuple[int, float, float]]:
    """`(numero_de_trial, validación, verdad)` de re-entrenar, con `entrenar`
    real, los `n_total - 1 = 7` podados por defecto de
    `_sembrar_estudio_median_controlado` (`CONFIG_REAL_CONTROLADA` con
    `semillas_de(20_260_777, 7)`, numerados 1..7 porque el COMPLETE sembrado
    ocupa el trial 0). Se calcula UNA vez y lo comparten las pruebas de abajo
    que solo cambian `val_ganador`/`verdad_ganador`, nunca los podados.
    """
    n_total = 8
    podados_semillas = semillas_de(20_260_777, n_total - 1)
    return [
        (indice + 1, resultado["exactitud_val"], resultado["exactitud_verdadera"])
        for indice, semillas_trial in enumerate(podados_semillas)
        for resultado in [juguete.entrenar(CONFIG_REAL_CONTROLADA, semillas_trial)]
    ]


@pytest.mark.parametrize(
    ("val_ganador", "verdad_ganador", "rama_esperada"),
    [
        (0.99, 0.50, "solo_verdad"),
        (0.60, 0.99, "solo_val"),
        (0.60, 0.60, "tardio_real"),
        (0.99, 0.99, "ninguno"),
    ],
)
@pytest.mark.gpu
def test_respuesta_3_las_cuatro_ramas_del_veredicto_segun_validacion_y_oraculo(
    tmp_path, val_ganador, verdad_ganador, rama_esperada, podados_controlados_reentrenados
):
    """Fuerza cada rama del veredicto sembrando un ganador de Median con
    validación/verdad FALSAS pero controladas, y trials podados con una
    config y semillas REALES cuyo re-entrenamiento cae, según el problema
    calibrado (media 0.70, rango 0.63-0.72 con DEFAULTS -- ver
    `taller_hpo.problemas.mlp_paridad`), muy por debajo de 0.99 y muy por
    encima de 0.50: eso hace que la categoría de cada combinación sea la
    misma sin importar el valor exacto que dé el entrenamiento real. El
    veredicto sale del argmax de validación (`ganador_sin_podar`), calculado
    aquí de forma independiente sobre `podados_controlados_reentrenados`
    (nunca leído de vuelta del texto impreso): con `val_ganador=0.99` ningún
    podado (val real ~0.63-0.72) le gana en validación, así que ese argmax
    sigue siendo el ganador real (rama "solo_verdad" -> "igual habría
    elegido al mismo ganador" si el oráculo además prefiere a algún podado);
    con `val_ganador=0.60` todos los podados le ganan en validación, así que
    el argmax pasa a ser un podado (rama "solo_val" -> el oráculo dice que
    ese argmax es PEOR).
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    _sembrar_estudio_median_controlado(
        estudios,
        n_total=8,
        val_ganador=val_ganador,
        verdad_ganador=verdad_ganador,
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr

    m = PATRON_RESPUESTA_3.search(salida.stdout)
    assert m, salida.stdout
    veredicto = m["veredicto"]

    candidatos = [(0, val_ganador, verdad_ganador), *podados_controlados_reentrenados]
    numero_esperado, _val_esperado, _verdad_esperado = max(sorted(candidatos), key=lambda c: c[1])
    assert int(m["num_sin_podar"]) == numero_esperado, salida.stdout
    frase_trial = f"el trial {numero_esperado} por validación"

    if rama_esperada == "tardio_real":
        assert "Sí mató un ganador tardío" in veredicto, veredicto
        assert frase_trial in veredicto, veredicto
    elif rama_esperada == "solo_verdad":
        assert "Sí mató un ganador tardío" not in veredicto, veredicto
        assert "igual habría elegido al mismo ganador" in veredicto, veredicto
    elif rama_esperada == "solo_val":
        assert "Sí mató un ganador tardío" not in veredicto, veredicto
        assert "PEOR" in veredicto, veredicto
        assert frase_trial in veredicto, veredicto
    else:
        assert "Ningún ganador tardío" in veredicto, veredicto


@pytest.mark.gpu
def test_respuesta_3_empate_dentro_del_error_del_oraculo(
    tmp_path, podados_controlados_reentrenados
):
    """Cuando el argmax de validación es un podado y su verdad está a
    <= 2 * 0.0012 (dos cotas de error Monte Carlo del oráculo,
    `taller_hpo.problemas.paridad.riesgo_verdadero`, con n_oraculo=200_000)
    de la del ganador real, ninguna de las dos puede declararse mejor ni
    peor: una diferencia de medio punto por mil no es algo que el oráculo,
    con este n_oraculo, pueda sostener.
    """
    numero_argmax, _val_argmax, verdad_argmax = max(
        sorted(podados_controlados_reentrenados), key=lambda c: c[1]
    )

    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    _sembrar_estudio_median_controlado(
        estudios,
        n_total=8,
        val_ganador=0.0,  # por debajo de TODOS los podados en validación
        verdad_ganador=verdad_argmax + 0.0005,  # dentro de la cota de empate (EMPATE_ORACULO)
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr

    m = PATRON_RESPUESTA_3.search(salida.stdout)
    assert m, salida.stdout
    assert int(m["num_sin_podar"]) == numero_argmax, salida.stdout

    veredicto = m["veredicto"]
    assert "empatan dentro del error del oráculo" in veredicto, veredicto
    assert "Sí mató un ganador tardío" not in veredicto, veredicto
    assert "PEOR" not in veredicto, veredicto


def _paciencia_real(paciencia: int, margen: float = 1e-4):
    """Reimplementación independiente de `early_stopping_por_paciencia` de
    pasos/03_poda.py: los pasos son scripts, no módulos importables sin correr
    todo lo que hacen al importarlos, así que esto verifica el COMPORTAMIENTO
    documentado (para cuando la validación no mejora en `paciencia` épocas
    seguidas) contra lo que el paso realmente imprime, en vez de confiar en
    que su propio código hace lo que su nombre promete.
    """
    estado = {"mejor": -float("inf"), "espera": 0}

    def reportar(epoca: int, valor: float) -> bool:
        if valor > estado["mejor"] + margen:
            estado["mejor"] = valor
            estado["espera"] = 0
        else:
            estado["espera"] += 1
        return estado["espera"] >= paciencia

    return reportar


@pytest.mark.parametrize(
    ("ic_bajo", "ic_alto", "esperado"),
    [
        (0.10, 0.40, "bajo"),
        (0.45, 0.55, "no_distingue"),
        (0.60, 0.90, "despegado"),
        (0.30, 0.50, "no_distingue"),  # toca 0.5 por arriba: sigue "no_distingue"
        (0.50, 0.70, "no_distingue"),  # toca 0.5 por abajo: sigue "no_distingue"
    ],
)
def test_categoria_por_ic_tres_ramas(ic_bajo, ic_alto, esperado):
    """`categoria_por_ic` (pasos/03_poda.py) clasificado con intervalos
    sintéticos, sin entrenar nada: las tres categorías con las que se
    clasifica cada corte de paciencia ('bajo el azar' / 'no se distingue del
    azar' / 'despegado') y sus dos fronteras (un intervalo que TOCA 0.5 por
    cualquier lado sigue siendo 'no_distingue', el criterio `ic_bajo <= 0.5
    <= ic_alto`).
    """
    categoria_por_ic = _cargar_funcion_pura("categoria_por_ic")
    assert categoria_por_ic(ic_bajo, ic_alto) == esperado


def test_conclusion_por_ic_valida_cortes_contra_len_ics():
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    with pytest.raises(ValueError):
        conclusion_por_ic([(0.4, 0.6)], 2)


def test_conclusion_por_ic_0_cortes():
    """Rama sin cortes (`cortes == 0`): el mismo texto que
    `pasos/03_poda.py` antepone a "no alcanzó a activarse en estas
    semillas".
    """
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    assert conclusion_por_ic([], 0) == "sin cortes no hay con qué correr una prueba pareada."


def test_conclusion_por_ic_1_corte():
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    texto = conclusion_por_ic([(0.4, 0.6)], 1)
    assert "Con un solo corte (n=1) no puedes generalizar" in texto
    assert "no se distinguen" not in texto
    assert "mezcla" not in texto


def test_conclusion_por_ic_todos_no_se_distinguen_del_azar():
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    texto = conclusion_por_ic([(0.45, 0.55), (0.40, 0.60), (0.48, 0.52)], 3)
    assert "Los 3 cortes no se distinguen del azar" in texto
    assert "cortó mientras la validación no se distinguía del azar" in texto
    # Esta frase y la de "ninguno indistinguible" nunca deben coexistir en el
    # mismo texto: si se intercambiaran, esta prueba fallaría.
    assert "Ninguno de los 3 cortes" not in texto
    assert "mezcla" not in texto


def test_conclusion_por_ic_ninguno_indistinguible_del_azar():
    """2 despegados + 1 bajo el azar: NINGUNO cae en 'no_distingue'."""
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    texto = conclusion_por_ic([(0.60, 0.90), (0.70, 0.95), (0.05, 0.20)], 3)
    assert "Ninguno de los 3 cortes se queda indistinguible del azar" in texto
    assert "2 ya se habían despegado" in texto
    assert "1 quedan por debajo del azar" in texto
    assert "no se distinguen del azar" not in texto
    assert "mezcla" not in texto


def test_conclusion_por_ic_mezcla():
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    texto = conclusion_por_ic([(0.45, 0.55), (0.60, 0.90), (0.05, 0.20)], 3)
    assert (
        "De los 3 cortes, 1 no se distinguen del azar, 1 ya se habían despegado y 1 quedan "
        "por debajo del azar: una mezcla" in texto
    )
    assert "Los 3 cortes no se distinguen" not in texto
    assert "Ninguno de los 3 cortes" not in texto


def test_conclusion_despegue_ic_completo_entero_sobre_0_5():
    """`conclusion_despegue` (pasos/03_poda.py), rama (a): el intervalo de
    confianza (t de Student) de la validación del completo, entero sobre
    0.5, basta como evidencia por sí solo -- la frase dice ANTES y cita ese
    intervalo, aunque el veredicto de la prueba pareada sea 'no concluyente'
    (que NUNCA se cita como evidencia: ver la rama de abajo y la docstring
    de la función).
    """
    conclusion_despegue = _cargar_funcion_pura("conclusion_despegue")
    texto = conclusion_despegue((0.6, 0.8), "no concluyente")
    assert "ANTES de que el modelo se despegara del azar" in texto
    assert "intervalo 95% [0.600, 0.800]" in texto
    assert "no concluyente" not in texto


def test_conclusion_despegue_ic_contiene_0_5_y_veredicto_no_concluyente():
    """Rama (b): el intervalo del completo contiene 0.5 (no alcanza por sí solo) y
    el veredicto es 'no concluyente' (tampoco alcanza): ninguna evidencia se
    cumple, así que la frase dice que la corrida no alcanza para decidir y
    nunca afirma 'ANTES'.
    """
    conclusion_despegue = _cargar_funcion_pura("conclusion_despegue")
    texto = conclusion_despegue((0.4, 0.6), "no concluyente")
    assert "no alcanza para decir" in texto
    assert "ANTES" not in texto
    assert "no concluyente" not in texto


def test_conclusion_despegue_veredicto_peor_basta_como_evidencia():
    """El intervalo del completo NO alcanza (contiene 0.5) pero el veredicto 'peor'
    de la prueba pareada (dejarlo terminar da más verdad que pararlo) sí
    basta por sí solo para afirmar 'ANTES'.
    """
    conclusion_despegue = _cargar_funcion_pura("conclusion_despegue")
    texto = conclusion_despegue((0.4, 0.6), "peor")
    assert "ANTES de que el modelo se despegara del azar" in texto
    assert "veredicto 'peor'" in texto


def test_conclusion_despegue_veredicto_equivalente_nunca_se_cita():
    """Rama (c): el veredicto 'equivalente' tampoco basta ni se cita jamás
    como evidencia, igual que 'no concluyente'.
    """
    conclusion_despegue = _cargar_funcion_pura("conclusion_despegue")
    texto = conclusion_despegue((0.4, 0.6), "equivalente")
    assert "no alcanza para decir" in texto
    assert "ANTES" not in texto
    assert "equivalente" not in texto


_PATRON_SIGLA_IC = re.compile(r"\bICs?(\d|\b)")


def test_ninguna_linea_de_pasos_usa_la_sigla_ic():
    r"""Esta prueba, sin marca gpu y sin correr ningún paso, busca la sigla
    'IC' línea por línea en el TEXTO COMPLETO de cada archivo de pasos/*.py
    (`\bICs?(\d|\b)`, como palabra suelta, en plural o seguida de un dígito):
    la política del taller pide 'intervalo de confianza' o 'intervalo',
    nunca la sigla ni su plural, en cualquier texto que el tesista lea.
    Buscar línea por línea (en vez de recorrer solo las constantes de texto
    del AST) también cubre el comentario markdown de celda
    (`# %% [markdown]`), invisible para `ast.parse`. Los identificadores del
    código (`ic_bajo`, `ic_alto`) van en minúscula, así que el patrón, en
    mayúscula, no los confunde con la sigla."""
    raiz_pasos = RAIZ / "pasos"
    archivos = sorted(raiz_pasos.glob("*.py"))
    assert archivos, raiz_pasos
    for archivo in archivos:
        fuente = archivo.read_text(encoding="utf-8")
        for n, linea in enumerate(fuente.splitlines(), 1):
            assert not _PATRON_SIGLA_IC.search(linea), (archivo.name, n, linea)


def test_ninguna_linea_de_src_plantilla_docs_ni_scripts_usa_la_sigla_ic():
    """Esta prueba, sin marca gpu y sin correr nada, aplica el mismo patrón
    de la sigla 'IC' que la de `pasos/*.py` a los archivos `.py`, `.md` y `.sh`
    de `src/`, `plantilla/`, `docs/` y `scripts/`, y al `README.md` de la
    raíz. `src/taller_hpo/honestidad.py` y `src/taller_hpo/estadistica.py`
    le hablan de tú al tesista desde sus docstrings, y `plantilla/` es el
    texto que él copia a su propio proyecto: ahí también vale 'intervalo de
    confianza' o 'intervalo', nunca la sigla."""
    archivos = [RAIZ / "README.md"]
    for carpeta in ("src", "plantilla", "docs", "scripts"):
        for patron in ("*.py", "*.md", "*.sh"):
            archivos.extend(sorted((RAIZ / carpeta).rglob(patron)))
    assert len(archivos) > 10, archivos
    for archivo in archivos:
        fuente = archivo.read_text(encoding="utf-8")
        for n, linea in enumerate(fuente.splitlines(), 1):
            assert not _PATRON_SIGLA_IC.search(linea), (str(archivo.relative_to(RAIZ)), n, linea)


_PATRON_SIGLAS_EN_INGLES = re.compile(r"\b(CI|CPUs?|GPUs?|CLI|RNGs?)\b")

# Cuando falta la tarjeta de video, torch avisa en inglés "No CUDA GPUs are available".
# Varios textos del taller citan ese aviso para explicar qué error evitan, y esa cita es
# el único lugar donde la sigla "GPUs" es aceptable. Algunas citas cambian de renglón a
# mitad de frase ("No CUDA GPUs are" / "available"), pero las cuatro que hay (en
# `paridad_torch.py`, `buscar_por_subproceso.py` y dos pruebas) traen "CUDA GPUs are"
# en una sola línea. Por eso la guarda quita de cada línea solo ese fragmento exacto,
# antes de buscar siglas: un "CUDA GPUs" suelto en una prosa nueva sí cuenta.
_CITA_DEL_ERROR_DE_TORCH = re.compile(r"CUDA GPUs are\b")

_PATRON_PROCESADOR = re.compile(r"\bprocesador(es)?\b", re.IGNORECASE)
# "cpu" en minúscula también nombra al procesador cuando va en prosa ("en la cpu").
# En el código aparece de otras dos formas, que no cuentan: como método (`.cpu()`,
# después de un punto) y como valor literal entre comillas (`"cpu"`, `'cpu'`, o entre
# comillas invertidas en la documentación).
_PATRON_CPU_EN_PROSA = re.compile(r"(?<![.`\"'])\bcpu\b(?![`\"'(])")


def _sigla_en_ingles(linea: str) -> bool:
    """Dice si `linea` usa 'CI', 'CPU', 'GPU', 'CLI' o 'RNG' (con su plural), sin
    contar la cita en inglés del error de torch."""
    return bool(_PATRON_SIGLAS_EN_INGLES.search(_CITA_DEL_ERROR_DE_TORCH.sub("", linea)))


def _nombra_al_procesador(linea: str) -> bool:
    """Dice si `linea` usa la palabra 'procesador' o 'procesadores', con o sin
    mayúscula inicial, o 'cpu' en minúscula como palabra de prosa."""
    return bool(_PATRON_PROCESADOR.search(linea) or _PATRON_CPU_EN_PROSA.search(linea))


def _archivos_de_texto_del_taller() -> list[Path]:
    """Devuelve los archivos que leen el tesista o quien mantiene el taller y que
    las guardas de redacción revisan: los `.py`, `.md` y `.sh` de `pasos/`,
    `ejercicios/`, `src/`, `plantilla/`, `docs/`, `scripts/` y `tests/`, el
    `README.md` de la raíz, los flujos de `.github/workflows/`, el `Dockerfile`
    y `compose.yaml`. Deja fuera este mismo archivo, porque tiene que nombrar
    las siglas y la palabra que busca."""
    archivos = [RAIZ / "README.md", RAIZ / "Dockerfile", RAIZ / "compose.yaml"]
    for carpeta in ("pasos", "ejercicios", "src", "plantilla", "docs", "scripts", "tests"):
        for patron in ("*.py", "*.md", "*.sh"):
            archivos.extend(sorted((RAIZ / carpeta).rglob(patron)))
    archivos.extend(sorted((RAIZ / ".github" / "workflows").glob("*.yml")))
    este_archivo = Path(__file__).resolve()
    archivos = [archivo for archivo in archivos if archivo.resolve() != este_archivo]
    assert len(archivos) > 10, archivos
    return archivos


def test_ningun_texto_del_taller_usa_las_siglas_ci_cpu_gpu_cli_ni_rng():
    """Esta prueba, sin marca gpu y sin correr nada, busca las siglas 'CI',
    'CPU', 'GPU', 'CLI' y 'RNG' como palabra suelta, en mayúscula y también en
    plural ('CPUs', 'GPUs', 'RNGs'), línea por línea en los archivos que lista
    `_archivos_de_texto_del_taller`. La política del taller pide escribir la
    palabra completa ("integración continua", "tarjeta de video", "línea de
    comandos", "generador aleatorio") en cualquier texto, nunca la sigla. La
    única excepción es la cita en inglés del error de torch
    (`_CITA_DEL_ERROR_DE_TORCH`). Los nombres en minúscula del código
    (`cli_primero`, el método `.cpu()` de torch, el parámetro `rng`) y los que
    unen la sigla con un guion bajo (`MEMORIA_GPU_WORKER_JSON`) no coinciden con
    el patrón."""
    for archivo in _archivos_de_texto_del_taller():
        fuente = archivo.read_text(encoding="utf-8")
        for n, linea in enumerate(fuente.splitlines(), 1):
            assert not _sigla_en_ingles(linea), (str(archivo.relative_to(RAIZ)), n, linea)


def test_ningun_texto_del_taller_nombra_al_procesador():
    """Esta prueba, sin marca gpu y sin correr nada, busca la palabra
    'procesador' (y 'procesadores', con o sin mayúscula) y 'cpu' en minúscula
    cuando va en prosa, línea por línea en los mismos archivos que revisa la
    guarda de las siglas. El taller entrena solo
    en la tarjeta de video, y ningún texto compara con el procesador, lo niega
    ni lo describe: si un texto necesita nombrar la memoria de la máquina, escribe
    'la memoria de tu computadora'."""
    for archivo in _archivos_de_texto_del_taller():
        fuente = archivo.read_text(encoding="utf-8")
        for n, linea in enumerate(fuente.splitlines(), 1):
            assert not _nombra_al_procesador(linea), (str(archivo.relative_to(RAIZ)), n, linea)


def test_las_guardas_de_redaccion_reconocen_el_plural_y_perdonan_solo_la_cita_de_torch():
    """Esta prueba ejercita las dos guardas de arriba con líneas escritas a mano,
    porque una guarda con el patrón roto pasa en verde sin vigilar nada: el
    plural 'GPUs'/'CPUs'/'RNGs' cuenta como sigla; la cita de torch, completa o
    partida en dos renglones, no cuenta; 'GPUs' fuera de la cita sí cuenta
    aunque comparta línea con ella, y también un 'CUDA GPUs' que no sea la cita
    ("CUDA GPUs are"); y 'procesador' cuenta en singular, en plural y con
    mayúscula."""
    assert _sigla_en_ingles("Todo el taller corre en GPUs NVIDIA, nunca en CPUs.")
    assert _sigla_en_ingles("una GPU")
    assert _sigla_en_ingles("una CPU")
    assert _sigla_en_ingles("igual en laptop, CI o Docker")
    assert _sigla_en_ingles("la CLI del taller")
    assert _sigla_en_ingles("nunca el RNG global")
    assert _sigla_en_ingles("dos RNGs distintos")
    assert not _sigla_en_ingles("RuntimeError: No CUDA GPUs are available")
    assert not _sigla_en_ingles("    CUDA GPUs are available`; un padre sin tarjeta de video")
    assert not _sigla_en_ingles('(el error en inglés de torch ("No CUDA GPUs are')
    assert _sigla_en_ingles("No CUDA GPUs are available, y tampoco hay GPUs de reserva")
    assert _sigla_en_ingles("a las CUDA GPUs se paga")
    assert not _sigla_en_ingles("cli_primero y el método .cpu() de torch")
    assert not _sigla_en_ingles("MEMORIA_GPU_WORKER_JSON")
    assert not _sigla_en_ingles("el parámetro rng y el estado RNG_ESTADO")
    assert _nombra_al_procesador("Nada de este paso corre en el procesador.")
    assert _nombra_al_procesador("Procesador al inicio de la línea")
    assert _nombra_al_procesador("varios procesadores a la vez")
    assert not _nombra_al_procesador("el procesamiento de los datos")
    assert _nombra_al_procesador("# Este paso nunca entrena en la cpu.")
    assert _nombra_al_procesador("cpu y tarjeta de video dan lo mismo")
    assert not _nombra_al_procesador("pesos = modelo.weight.detach().cpu().numpy()")
    assert not _nombra_al_procesador('torch.device("cpu")')
    assert not _nombra_al_procesador("el valor `cpu` que las pruebas rechazan")
    assert not _nombra_al_procesador("os.cpu_count() y el parámetro 'cpu'")


@pytest.mark.gpu
def test_respuesta_4_recalculado_en_proceso_coincide_con_lo_impreso(salida_humo):
    """Recalcula, EN el proceso de pytest, la medición honesta de la
    pregunta 4 (mismas semillas -- `semillas_de(20_260_304, 4, datos=...)`,
    el mismo presupuesto real) contra lo impreso: la clasificación por
    intervalo de Wilson de cada corte (valores Y extremos del intervalo, no
    solo la época), las
    medias calculadas SOLO sobre las semillas cortadas (nunca sobre las
    diez/cuatro semillas completas -- eso invertiría o borraría la
    conclusión), la pérdida en puntos PORCENTUALES con el signo correcto, la
    lista de épocas exacta (solo las que de verdad cortaron), la conclusión
    agregada (`conclusion_por_ic` real, cargada del propio paso) y el
    veredicto de la prueba pareada -- todo contra `entrenar`/`resumen`/
    `diferencia_pareada` reales, nunca contra el propio texto que imprime el
    paso.
    """
    m = PATRON_RESPUESTA_4.search(salida_humo)
    assert m, salida_humo
    n_semillas = int(m["n_semillas"])
    assert int(m["n_semillas2"]) == n_semillas
    assert int(m["n_semillas3"]) == n_semillas
    assert n_semillas == 4  # humo: N_SEMILLAS_PACIENCIA = 4 (ver pasos/03_poda.py)
    paciencia = int(m["paciencia"])
    assert paciencia == 5
    epocas_reales = int(m["epocas_reales"])
    assert epocas_reales == juguete.DEFAULTS["epocas"]
    n_val = juguete.DEFAULTS["n_val"]

    semillas_paciencia = semillas_de(20_260_304, n_semillas, datos=SEMILLA_BUSQUEDA_ESPERADA.datos)
    epocas_paro: list[float] = []
    val_parada: list[float] = []
    verdad_parada: list[float] = []
    verdad_completo: list[float] = []
    val_completo: list[float] = []
    cambios_verdad: list[float] = []
    for s in semillas_paciencia:
        completo = juguete.entrenar({}, s)
        parada = juguete.entrenar({}, s, _paciencia_real(paciencia))
        epocas_paro.append(parada["epocas"])
        val_parada.append(parada["exactitud_val"])
        verdad_parada.append(parada["exactitud_verdadera"])
        verdad_completo.append(completo["exactitud_verdadera"])
        val_completo.append(completo["exactitud_val"])
        cambios_verdad.append(parada["exactitud_verdadera"] - completo["exactitud_verdadera"])

    cortadas_esperadas = [i for i, e in enumerate(epocas_paro) if e < epocas_reales]
    cortes_esperados = len(cortadas_esperadas)
    assert int(m["cortes"]) == cortes_esperados
    # Con N_SEMILLAS_PACIENCIA['humo'] == 4 hay SIEMPRE >= 2 cortes (medido:
    # semillas_de(20_260_304, 10, datos=...) corta la 0, 3, 4, 5 y 9; las
    # primeras 4 ya incluyen dos), así que esta prueba SIEMPRE ejercita la
    # clasificación agregada y la prueba pareada -- justo lo que taller y
    # completo imprimen de verdad, nunca la rama corta de "cortes == 1". Si
    # esto deja de cumplirse (p. ej. tras cambiar la semilla base), la prueba
    # debe fallar aquí, no colarse en silencio por esa rama.
    assert cortes_esperados >= 2, (
        f"se esperaban >= 2 cortes con N_SEMILLAS_PACIENCIA=4, no {cortes_esperados}"
    )

    assert float(m["cambio_verdad"]) == pytest.approx(statistics.mean(cambios_verdad), abs=5e-4)

    epocas_totales_posibles = n_semillas * epocas_reales
    epocas_ahorradas = sum(epocas_reales - e for e in epocas_paro)
    ahorro_pct = 100 * epocas_ahorradas / epocas_totales_posibles
    m_ahorro = PATRON_AHORRO_PACIENCIA.search(m["frase_ahorro"])
    assert m_ahorro, m["frase_ahorro"]
    assert int(m_ahorro["ahorradas"]) == epocas_ahorradas
    assert int(m_ahorro["posibles"]) == epocas_totales_posibles
    assert float(m_ahorro["pct"]) == pytest.approx(ahorro_pct, abs=0.05)
    # `m["frase_ahorro"]` en realidad captura frase_ahorro + zona_median (el
    # patrón las junta en un solo grupo final): por eso el chequeo es
    # `startswith`, no una igualdad exacta. La frase impresa es SOLO lo
    # medido ("Ahorró X de Y épocas posibles (Z%)."): la idea conceptual de
    # que esto no compara entre trials vive en el markdown de la pregunta 4,
    # no en un texto fijo dentro de la respuesta.
    frase_ahorro_esperada = (
        f"Ahorró {epocas_ahorradas:.0f} de {epocas_totales_posibles:.0f} épocas posibles "
        f"({ahorro_pct:.1f}%)."
    )
    assert m["frase_ahorro"].startswith(frase_ahorro_esperada), m["frase_ahorro"]
    assert "Ahorra épocas DENTRO de un entrenamiento" not in m["frase_ahorro"], m["frase_ahorro"]

    cuerpo = m["cuerpo"]
    detalle = PATRON_DETALLE_CORTE.search(cuerpo)
    assert detalle, cuerpo

    # Ground truth SOLO sobre las semillas cortadas -- mata: promediar sobre
    # TODAS las semillas (incluida la que no cortó), invertir el signo de la
    # pérdida, o colar en la lista de épocas una semilla sin corte real.
    val_corte_prom_esperado = statistics.mean(val_parada[i] for i in cortadas_esperadas)
    verdad_corte_prom_esperado = statistics.mean(verdad_parada[i] for i in cortadas_esperadas)
    verdad_completo_corte_prom_esperado = statistics.mean(
        verdad_completo[i] for i in cortadas_esperadas
    )
    perdida_esperada_pct = 100 * (verdad_completo_corte_prom_esperado - verdad_corte_prom_esperado)

    assert float(detalle["val_corte"]) == pytest.approx(val_corte_prom_esperado, abs=5e-4)
    assert float(detalle["verdad_corte"]) == pytest.approx(verdad_corte_prom_esperado, abs=5e-4)
    assert float(detalle["verdad_completo"]) == pytest.approx(
        verdad_completo_corte_prom_esperado, abs=5e-4
    )
    # El signo importa: el paso reporta la pérdida en PUNTOS PORCENTUALES
    # (x100 sobre la fracción cruda) porque un valor como 0.189 se lee como
    # si fuera despreciable, cuando corresponde a 18.9 puntos porcentuales de
    # exactitud.
    assert float(detalle["perdida"]) == pytest.approx(perdida_esperada_pct, abs=0.05)

    coincidencias_item = list(PATRON_CORTE_ITEM.finditer(detalle["lista"]))
    items = [m.groups() for m in coincidencias_item]
    assert items, detalle["lista"]

    # La primera mención de la lista, EN EL ORDEN en que aparece, usa la
    # expansión completa "intervalo de confianza 95%"; las siguientes usan
    # la abreviada "intervalo 95%" (ver pasos/03_poda.py::lista_cortes) --
    # así la respuesta 4 no repite la frase completa una vez por cada corte.
    assert "confianza" in coincidencias_item[0].group(0), detalle["lista"]
    for coincidencia in coincidencias_item[1:]:
        assert "confianza" not in coincidencia.group(0), detalle["lista"]

    epocas_impresas = sorted(int(item[0]) for item in items)
    epocas_esperadas = sorted(int(epocas_paro[i]) for i in cortadas_esperadas)
    # Una semilla SIN corte real (epocas_paro[i] == epocas_reales) que se
    # colara en esta lista rompería esta igualdad exacta (más elementos de
    # los que de verdad cortaron).
    assert epocas_impresas == epocas_esperadas, cuerpo

    ics_esperados = [ic_proporcion(round(val_parada[i] * n_val), n_val) for i in cortadas_esperadas]
    # Comparar también los EXTREMOS de cada intervalo impreso contra los
    # esperados, no solo la época: un intervalo de Wilson mal calculado (o
    # con bajo/alto invertidos) debe fallar aquí.
    ics_impresos = sorted((float(it[2]), float(it[3])) for it in items)
    for (ic_bajo_esp, ic_alto_esp), (ic_bajo_imp, ic_alto_imp) in zip(
        sorted(ics_esperados), ics_impresos, strict=True
    ):
        assert ic_bajo_imp == pytest.approx(ic_bajo_esp, abs=5e-4)
        assert ic_alto_imp == pytest.approx(ic_alto_esp, abs=5e-4)

    categoria_por_ic = _cargar_funcion_pura("categoria_por_ic")
    conclusion_por_ic = _cargar_funcion_pura("conclusion_por_ic")
    conclusion_despegue = _cargar_funcion_pura("conclusion_despegue")
    conclusion_esperada = conclusion_por_ic(ics_esperados, cortes_esperados)

    # "ANTES de que el modelo se despegara del azar" solo se afirma cuando
    # TODOS los cortes son indistinguibles del azar Y hay evidencia MEDIDA
    # (nunca antes de medirla): la función real del paso, cargada aquí, decide
    # cuál de las dos evidencias (o ninguna) se cumplió.
    pareada_esperada = diferencia_pareada(
        {i: verdad_parada[i] for i in cortadas_esperadas},
        {i: verdad_completo[i] for i in cortadas_esperadas},
        delta=0.01,
    )
    if all(categoria_por_ic(*ic) == "no_distingue" for ic in ics_esperados):
        r_completo_corte = resumen([val_completo[i] for i in cortadas_esperadas])
        conclusion_esperada += conclusion_despegue(
            (r_completo_corte.ic_bajo, r_completo_corte.ic_alto), pareada_esperada.veredicto
        )

    texto_pareada_esperado = (
        f" Prueba pareada (parada contra completo, delta=0.01): "
        f"diferencia media {pareada_esperada.diferencia_media:+.3f} (intervalo "
        f"{100 * pareada_esperada.confianza:.0f}% [{pareada_esperada.ic_bajo:.3f}, "
        f"{pareada_esperada.ic_alto:.3f}]), veredicto: {pareada_esperada.veredicto}."
    )

    resto = detalle["resto"]
    assert resto == f" {conclusion_esperada}{texto_pareada_esperado}", resto

    # Doble verificación (además de la igualdad exacta de arriba) con el
    # patrón de la prueba pareada, por si el formato exacto de `resto` cambia
    # pero el CONTENIDO numérico de la prueba pareada no.
    m_pareada = PATRON_PAREADA_PACIENCIA.search(resto)
    assert m_pareada, resto
    assert float(m_pareada["dif"]) == pytest.approx(pareada_esperada.diferencia_media, abs=5e-4)
    assert float(m_pareada["ic_bajo"]) == pytest.approx(pareada_esperada.ic_bajo, abs=5e-4)
    assert float(m_pareada["ic_alto"]) == pytest.approx(pareada_esperada.ic_alto, abs=5e-4)
    assert m_pareada["veredicto"] == pareada_esperada.veredicto


@pytest.mark.gpu
def test_respuesta_4_zona_median_coincide_con_lo_persistido(salida_humo, estudios_persistidos):
    """Verifica que la afirmación de que Median poda 'en la misma zona donde
    la paciencia manual también corta' esté medida: recalcula, sobre el
    estudio Median REALMENTE persistido, la MISMA clasificación de intervalo
    de Wilson que `pasos/03_poda.py` aplica al último reporte de cada podado
    (`intermediate_values` en su época más alta) y compara los tres conteos
    contra lo impreso -- nunca contra el propio texto del paso.
    """
    median = estudios_persistidos["median"]
    podados = median.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    assert podados
    n_val = juguete.DEFAULTS["n_val"]

    categoria_por_ic = _cargar_funcion_pura("categoria_por_ic")
    conteo = {"bajo": 0, "no_distingue": 0, "despegado": 0}
    for trial in podados:
        epoca = max(trial.intermediate_values)
        val = trial.intermediate_values[epoca]
        ic = ic_proporcion(round(val * n_val), n_val)
        conteo[categoria_por_ic(*ic)] += 1

    m = re.search(
        r"De los (?P<total>\d+) trials que podó Median: (?P<no_distingue>\d+) con último reporte "
        r"indistinguible del azar \(la misma zona de indiferencia que clasifican estos cortes de "
        r"paciencia\), (?P<despegado>\d+) ya despegados y (?P<bajo>\d+) por debajo del azar\.",
        salida_humo,
    )
    assert m, salida_humo
    assert int(m["total"]) == len(podados)
    assert int(m["no_distingue"]) == conteo["no_distingue"]
    assert int(m["despegado"]) == conteo["despegado"]
    assert int(m["bajo"]) == conteo["bajo"]


@pytest.mark.gpu
def test_respuesta_4_zona_median_usa_validacion_no_el_oraculo(tmp_path):
    """Siembra los 7 podados de Median con `intermediate_values` fijo en 0.5
    en las cinco épocas (contiene 0.5: 'no_distingue') pero
    `exactitud_verdadera=0.8` en `user_attrs` (que clasificaría 'despegado'
    si el paso clasificara con el oráculo en vez de con la validación). Los
    PRUNED sí guardan `exactitud_verdadera` en `user_attrs` (verificado: sus
    claves son config/epocas/exactitud_verdadera/semillas), así que solo la
    validación real de `intermediate_values` distingue las dos rutas.
    """
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    _sembrar_estudio_median_controlado(
        estudios, n_total=8, val_ganador=0.9, verdad_ganador=0.9, verdad_podados=0.8
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr
    assert (
        "De los 7 trials que podó Median: 7 con último reporte indistinguible del azar "
        "(la misma zona de indiferencia que clasifican estos cortes de paciencia), "
        "0 ya despegados y 0 por debajo del azar." in salida.stdout
    ), salida.stdout


@pytest.mark.gpu
def test_respuesta_3_veredicto_sale_del_argmax_de_validacion_no_del_oraculo(tmp_path):
    """El veredicto (y `ganador_sin_podar`) deben salir de quién es el argmax
    de VALIDACIÓN entre los candidatos sin podar, nunca de quién tiene más
    verdad. Se buscan, sobre `CONFIG_REAL_CONTROLADA`, dos semillas REALES A
    y B donde A gana en validación pero B gana en verdad (la misma
    discordancia que ya documenta la pregunta 2 del paso 02: validar y el
    oráculo no siempre están de acuerdo) y se siembra el ganador de Median
    por DEBAJO de A en validación y ENTRE A y B en verdad. Si
    `ganador_sin_podar` se calculara con el oráculo en vez de con la
    validación (por ejemplo, `max(..., key=lambda c: c[2])` en vez de
    `c[1]`), señalaría a B y el veredicto diría 'Sí mató un ganador tardío';
    el correcto señala a A -- que NO supera al ganador real en verdad -- y
    debe decir que la búsqueda, sin podar, habría elegido un ganador PEOR.
    """
    candidatos = []
    for semillas_candidata in semillas_de(20_260_888, 14):
        resultado = juguete.entrenar(CONFIG_REAL_CONTROLADA, semillas_candidata)
        candidatos.append(
            (semillas_candidata, resultado["exactitud_val"], resultado["exactitud_verdadera"])
        )

    candidatos_por_val = sorted(candidatos, key=lambda c: c[1], reverse=True)
    semillas_a, val_a, verdad_a = candidatos_por_val[0]
    par_b = next((c for c in candidatos_por_val[1:] if c[2] > verdad_a), None)
    assert par_b is not None, (
        "no se encontró, en 14 semillas, una con más verdad que el argmax de validación -- "
        "ampliar la búsqueda"
    )
    semillas_b, val_b, verdad_b = par_b
    assert val_a > val_b
    assert verdad_b > verdad_a

    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    val_ganador = val_b - 0.01  # por debajo de A y de B en validación
    verdad_ganador = (verdad_a + verdad_b) / 2  # entre A y B en verdad
    otras = [c[0] for c in candidatos if c[0] not in (semillas_a, semillas_b)][:5]
    _sembrar_estudio_median_controlado(
        estudios,
        n_total=8,
        val_ganador=val_ganador,
        verdad_ganador=verdad_ganador,
        podados=[semillas_a, semillas_b, *otras],
    )

    salida = _correr("humo", resultados, estudios)
    assert salida.returncode == 0, salida.stderr

    m = PATRON_RESPUESTA_3.search(salida.stdout)
    assert m, salida.stdout
    assert int(m["num_sin_podar"]) != int(m["num_real"])
    assert float(m["val_sin_podar"]) == pytest.approx(val_a, abs=5e-4)
    assert float(m["verdad_sin_podar"]) == pytest.approx(verdad_a, abs=5e-4)

    veredicto = m["veredicto"]
    # A gana en validación pero pierde en verdad: la búsqueda sin podar habría
    # elegido a A, y el oráculo dice que A es PEOR que el ganador real -- NUNCA
    # "Sí mató un ganador tardío" (esa frase exige que el argmax de validación
    # TAMBIÉN gane en verdad).
    assert "Sí mató un ganador tardío" not in veredicto, veredicto
    assert "PEOR" in veredicto, veredicto


@pytest.mark.gpu
def test_respuesta_5_confirma_notimplementederror_sin_silenciarlo(salida_humo):
    assert "5. Trial.report en un estudio multiobjetivo levanta NotImplementedError:" in salida_humo
    # Si Optuna alguna vez dejara de lanzar este error, el propio paso revienta
    # (AssertionError -> exit != 0) antes de llegar a imprimir esto: por eso
    # "NADA (inesperado)" nunca debería aparecer en una corrida que sí salió 0.
    assert "NADA (inesperado)" not in salida_humo


@pytest.mark.gpu
def test_reabrir_humo_no_reentrena(corrida, salida_humo):
    """El nombre del archivo del estudio incluye el presupuesto
    (`03_poda_<pruner>_humo.log`): correr humo dos veces sobre el MISMO
    TALLER_ESTUDIOS reabre sin re-entrenar y da EXACTAMENTE los mismos
    números -- el presupuesto en el nombre es lo que evita que una corrida
    posterior con otro `n_total` (p. ej. taller sobre la misma carpeta) choque
    con un protocolo distinto al reabrir. Reusa la carpeta de estudios del
    fixture de módulo (`corrida`, ya corrida una vez para el resto de las
    pruebas) y corre humo UNA vez más sobre ella, en vez de correr dos
    corridas frescas desde cero.
    """
    segunda = _correr("humo", corrida["resultados"], corrida["estudios"])
    assert segunda.returncode == 0, segunda.stderr

    patron = re.compile(r"1\. Sin podar.*épocas\)\.")
    resumen_1 = patron.search(salida_humo).group(0)
    resumen_2 = patron.search(segunda.stdout).group(0)
    assert resumen_1 == resumen_2

    patron_2 = re.compile(r"2\. Mejor validación.*compararlos pareados \(el método del paso 06\)\.")
    resumen_2a = patron_2.search(salida_humo).group(0)
    resumen_2b = patron_2.search(segunda.stdout).group(0)
    assert resumen_2a == resumen_2b

    # Verificación directa sobre el archivo, no solo sobre el texto: si esta
    # segunda corrida re-entrenara sobre el mismo archivo tendría 16 trials
    # por estudio en vez de 8, y si usara un archivo sin el presupuesto en el
    # nombre, el conteo no coincidiría con lo que de verdad se corrió.
    for pruner in ("nop", "median", "hyperband"):
        destino = corrida["estudios"] / f"03_poda_{pruner}_humo.log"
        study = optuna.load_study(
            study_name=f"03_poda_{pruner}", storage=estudio.abrir_storage(destino)
        )
        assert len(study.get_trials(deepcopy=False)) == 8
