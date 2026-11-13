"""`pasos/07_conecta_tu_modelo.py` de punta a punta con PRESUPUESTO=humo, más el contenido:
las 3 respuestas numeradas y que los números medidos tengan sentido.

Cada prueba que corre el paso o importa su módulo lleva `@pytest.mark.gpu`. Cuatro corren
sin tarjeta de video: `test_paso07_presupuesto_invalido_falla_con_mensaje` y las tres que
leen el código del paso con `ast`. `salida_humo` corre el paso una sola vez y las pruebas de
contenido leen ese mismo resultado.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest

RAIZ = Path(__file__).resolve().parent.parent
PASO_07 = RAIZ / "pasos" / "07_conecta_tu_modelo.py"


def _importar(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _correr_humo(tmp_path, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    # TALLER_ESTUDIOS también fijo a tmp_path: la sección 2 de este paso crea
    # sus propios estudios de comparación en directorios temporales (ver el
    # paso), pero sin esta variable, cualquier estudio que SÍ respetara
    # TALLER_ESTUDIOS escribiría en <repo>/estudios en vez de aislarse en la
    # corrida de la prueba.
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        # cuda:0, no "cuda": este paso compara las dos rutas SIEMPRE en la
        # tarjeta por defecto (ver `dispositivo_por_subproceso` en el paso),
        # sin importar TALLER_DISPOSITIVO. Si el paso pasara `None` a
        # `_resolver_dispositivo` (que sí respeta esa variable), la config de
        # por_subproceso quedaría en "cuda:0" mientras en_proceso sigue en
        # "cuda", y el assert de la sección 3 que compara
        # `user_attrs["config"]` entre las dos rutas fallaría.
        "TALLER_DISPOSITIVO": "cuda:0",
    }
    return subprocess.run(
        [sys.executable, str(PASO_07)], env=entorno, capture_output=True, text=True, timeout=timeout
    )


@pytest.fixture(scope="module")
def corrida_humo(tmp_path_factory) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Corre el paso UNA sola vez con PRESUPUESTO=humo: las pruebas de
    contenido de abajo (PNG, respuestas 1-3, motor) reparten esta misma
    corrida en vez de relanzar el subproceso cada una -- la salida es
    completamente determinista (semillas fijas), así que compartir una sola
    corrida entre todas las pruebas de contenido no pierde cobertura y evita
    pagar el costo de arrancar el proceso una vez por prueba."""
    tmp_path = tmp_path_factory.mktemp("paso07_humo")
    return _correr_humo(tmp_path), tmp_path


@pytest.fixture(scope="module")
def salida_humo(corrida_humo) -> subprocess.CompletedProcess[str]:
    return corrida_humo[0]


# --- corre de punta a punta y contesta las 3 preguntas ---------------------------


@pytest.mark.gpu
def test_paso07_humo_sale_0_y_deja_un_png(corrida_humo):
    resultado, tmp_path = corrida_humo
    assert resultado.returncode == 0, resultado.stderr
    assert list(tmp_path.glob("*.png")), "el paso no guardó su figura"


@pytest.mark.gpu
def test_paso07_humo_imprime_las_3_respuestas_numeradas(salida_humo):
    assert salida_humo.returncode == 0, salida_humo.stderr
    salida = salida_humo.stdout
    # re.MULTILINE + ^N\. : "3." a secas es vacuo; anclar a inicio de línea
    # evita un falso positivo con cualquier otro "3." en medio de una línea.
    for patron in (
        r"^1\. check_importable aprueba train_ejemplo\.py",
        r"^2\. Misma config, misma semilla",
        r"^3\. Con 1 trial nuevo cada uno: en_proceso cuesta",
    ):
        assert re.search(patron, salida, re.MULTILINE), salida


@pytest.mark.gpu
def test_paso07_humo_imprime_el_motor_del_juguete_al_arrancar(salida_humo):
    """Convención obligatoria (docs/arquitectura.md): el paso debe imprimir
    `juguete.DESCRIPCION` al arrancar; así ves sin adivinar en qué corre."""
    assert salida_humo.returncode == 0, salida_humo.stderr
    assert "Motor resuelto para el juguete de este repo:" in salida_humo.stdout


def test_paso07_presupuesto_invalido_falla_con_mensaje(tmp_path):
    # Sin marca gpu: el paso valida PRESUPUESTO ANTES de importar
    # taller_hpo.problemas.juguete (ver el comentario junto a esa validación
    # en el paso), así que un PRESUPUESTO inválido sale con SystemExit sin
    # tocar la tarjeta de video -- esta prueba corre igual en cualquier
    # máquina.
    entorno = {**os.environ, "PRESUPUESTO": "enorme", "TALLER_RESULTADOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(PASO_07)], env=entorno, capture_output=True, text=True, timeout=30
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


# --- respuesta 1: check_importable aprueba/rechaza con el diagnóstico correcto ---


@pytest.mark.gpu
def test_paso07_humo_respuesta_1_check_importable_con_evidencia(salida_humo):
    """No solo que imprima la frase fija: que el `problema=` que reporta sea
    de verdad `None` para el bueno y mencione `parse_args` para el roto -- el
    propio paso ya lo afirma con `assert`; esta prueba verifica el efecto
    observable de esos `assert` en la salida impresa, no solo que el texto
    literal aparezca."""
    salida = salida_humo.stdout
    assert "train_ejemplo.py aprueba check_importable: True" in salida
    assert "train_roto.py aprueba check_importable: False" in salida
    assert "problema=None" in salida
    assert "parse_args" in salida


# --- respuesta 2: misma config, misma semilla -> diferencia EXACTA 0 ------------


@pytest.mark.gpu
def test_paso07_humo_respuesta_2_misma_metrica_exacta(salida_humo):
    salida = salida_humo.stdout
    diferencia = re.search(r"diferencia: ([\d.e+-]+), exactamente la misma métrica", salida)
    assert diferencia is not None, salida
    assert float(diferencia.group(1)) == 0.0


# --- respuesta 3: costos de cada ruta y la fórmula del costo relativo -----------


@pytest.mark.gpu
def test_paso07_humo_respuesta_3_costos_reanudacion_y_formula_correcta(salida_humo):
    """(a) por_subproceso cuesta más que en_proceso y las dos rutas reportan
    exactamente el 1 trial NUEVO de esta llamada (humo), no una cuenta
    heredada de una reanudación. (b) La fórmula de la respuesta 3 es costo
    relativo = sobrecosto/T (nunca sobrecosto/(T+sobrecosto): esa segunda
    fórmula mide la FRACCIÓN de un trial que es sobrecosto, no cuánto más
    lento queda un entrenamiento de T segundos, y puede dar un número que se
    contradice con la propia frase 'más lento')."""
    salida = salida_humo.stdout

    en_proceso_match = re.search(
        r"en_proceso: (\d+) trials? nuevos? en [\d.]+ s \(([\d.]+) s/trial\)", salida
    )
    assert en_proceso_match is not None, salida
    assert en_proceso_match.group(1) == "1"
    costo_en_proceso = float(en_proceso_match.group(2))
    assert costo_en_proceso > 0.0

    por_subproceso_match = re.search(
        r"por_subproceso: (\d+) trials? nuevos? en [\d.]+ s "
        r"\(([\d.]+) s/trial, ([\d.]+)x más lento por trial\)",
        salida,
    )
    assert por_subproceso_match is not None, salida
    assert por_subproceso_match.group(1) == "1"
    costo_por_subproceso = float(por_subproceso_match.group(2))
    razon = float(por_subproceso_match.group(3))
    assert costo_por_subproceso > costo_en_proceso
    assert razon > 1.0

    # humo: n_busqueda=1 -- singular ("1 trial nuevo", "la 1 config"), no el
    # "1 trials"/"1 configs" agramatical de imprimir siempre el plural.
    assert "en_proceso: 1 trial nuevo en" in salida, salida
    assert "por_subproceso: 1 trial nuevo en" in salida, salida
    # Concordancia: con 1 sola config el verbo va en singular ("da"), no el
    # "dan" agramatical de imprimir siempre el plural (ver _verbo_dar en el
    # paso, mismo patrón que _pluralizar).
    assert "la 1 config de las dos rutas da la MISMA métrica, trial a trial." in salida
    # humo: repeticiones_sobrecosto=0 (ver PRESUPUESTOS en el paso) -- medir
    # el sobrecosto fijo con una prueba pareada no cabe en el presupuesto de
    # 20 s.
    assert "En humo, este paso no mide el sobrecosto fijo con una prueba pareada" in salida, salida

    respuesta_3 = re.search(
        r"^3\. Con 1 trial nuevo cada uno: en_proceso cuesta ([\d.]+) s/trial, "
        r"por_subproceso ([\d.]+) s/trial \(([\d.]+)x más lento, sobrecosto fijo "
        r"([\d.]+) s/trial\)\. .*? El sobrecosto hace que un entrenamiento de T "
        r"segundos tarde T\+sobrecosto, es decir 100\*sobrecosto/T % más: para "
        r"T=([\d.]+) s \(este juguete\) es (-?\d+)% más; para T=60 s \(un modelo "
        r"real\) sería solo ([\d.]+)% más\.",
        salida,
        re.MULTILINE,
    )
    assert respuesta_3 is not None, salida
    (
        r3_costo_en_proceso,
        r3_costo_por_subproceso,
        r3_razon,
        r3_sobrecosto,
        r3_t,
        r3_porcentaje_juguete,
        r3_porcentaje_t60,
    ) = (float(g) for g in respuesta_3.groups())

    assert math.isclose(r3_costo_en_proceso, costo_en_proceso, abs_tol=1e-6)
    assert math.isclose(r3_costo_por_subproceso, costo_por_subproceso, abs_tol=1e-6)
    assert math.isclose(r3_razon, razon, abs_tol=0.05)
    # T y costo_en_proceso son la MISMA variable, impresas las dos con .3f:
    # no es una comparación con tolerancia por redondeo (con .2f contra .3f,
    # el tercer decimal en 5 exacto queda del lado equivocado del redondeo
    # en una fracción de las corridas -- una fuente evitable de fallas intermitentes),
    # es la MISMA cadena de dígitos dos veces, así que deben coincidir
    # EXACTO.
    assert r3_t == r3_costo_en_proceso, (r3_t, r3_costo_en_proceso)

    # sobrecosto y T con la precisión de .3f: abs_tol=2e-3 porque
    # costo_en_proceso y costo_por_subproceso ya vienen redondeados a .3f cada
    # uno por separado, así que su resta puede diferir del sobrecosto_fijo
    # (redondeado aparte, con precisión completa antes de redondear) hasta en
    # dos veces el paso de redondeo de .3f.
    sobrecosto = r3_costo_por_subproceso - r3_costo_en_proceso
    assert math.isclose(r3_sobrecosto, sobrecosto, abs_tol=2e-3), (r3_sobrecosto, sobrecosto)

    # Fórmula correcta: costo relativo = sobrecosto/T. Con T=60 (mucho más
    # grande que sobrecosto) las dos fórmulas posibles (sobrecosto/T y
    # sobrecosto/(T+sobrecosto)) casi coinciden -- ahí NO hay forma de
    # distinguirlas -- así que la fórmula equivocada se detecta contra el
    # porcentaje del JUGUETE (T comparable a sobrecosto, donde las dos
    # difieren varias veces, no unos puntos porcentuales). Tolerancia
    # RELATIVA (no absoluta): T es chico (~0.1-0.8 s) y el porcentaje
    # resultante puede ser de cientos o miles por ciento, así que hasta el
    # redondeo de .3f en T se amplifica -- una tolerancia absoluta de un par
    # de puntos porcentuales sería frágil sin que hubiera ningún bug real.
    esperado_juguete = 100 * sobrecosto / r3_costo_en_proceso
    esperado_t60 = 100 * sobrecosto / 60
    formula_alterna_juguete = 100 * sobrecosto / (r3_costo_en_proceso + sobrecosto)
    assert math.isclose(r3_porcentaje_juguete, esperado_juguete, rel_tol=0.1), (
        r3_porcentaje_juguete,
        esperado_juguete,
    )
    assert math.isclose(r3_porcentaje_t60, esperado_t60, rel_tol=0.1, abs_tol=0.2), (
        r3_porcentaje_t60,
        esperado_t60,
    )
    if sobrecosto > 0.2 * r3_costo_en_proceso:
        assert not math.isclose(r3_porcentaje_juguete, formula_alterna_juguete, rel_tol=0.2), (
            "la respuesta 3 usa sobrecosto/(T+sobrecosto) en vez de sobrecosto/T"
        )


# --- reanudación: el conteo de trials y el directorio temporal nuevo -----------


@pytest.mark.gpu
def test_paso07_dos_corridas_seguidas_dan_costos_sanos_las_dos_veces(tmp_path, salida_humo):
    """Cada corrida de la sección 3 usa un directorio TEMPORAL nuevo (nunca
    TALLER_ESTUDIOS) y, además, `_trials_de_esta_llamada` cuenta `len(
    study.trials) - antes` -- un conteo puro, sin reloj de pared: correr el
    paso dos veces SEGUIDAS debe dar costos > 0 y por_subproceso > en_proceso
    las DOS veces -- sin esa guarda, una segunda corrida que reanudara por
    accidente un estudio ya completo (un destino fijo en vez de un
    directorio nuevo) reportaría un costo dividiendo el tiempo de reloj
    entre trials que no corrieron. Reusa `salida_humo` como la PRIMERA
    corrida (regla dura: no relanzar el paso más veces de las necesarias) y
    solo lanza una SEGUNDA aquí."""
    assert salida_humo.returncode == 0, salida_humo.stderr
    segunda = _correr_humo(tmp_path)
    assert segunda.returncode == 0, segunda.stderr

    for salida in (salida_humo.stdout, segunda.stdout):
        assert "en_proceso: 1 trial nuevo en" in salida, salida
        assert "por_subproceso: 1 trial nuevo en" in salida, salida
        en_proceso_match = re.search(
            r"en_proceso: 1 trial nuevo en [\d.]+ s \(([\d.]+) s/trial\)", salida
        )
        por_subproceso_match = re.search(
            r"por_subproceso: 1 trial nuevo en [\d.]+ s \(([\d.]+) s/trial", salida
        )
        assert en_proceso_match is not None and por_subproceso_match is not None, salida
        costo_en_proceso = float(en_proceso_match.group(1))
        costo_por_subproceso = float(por_subproceso_match.group(1))
        assert costo_en_proceso > 0.0
        assert costo_por_subproceso > costo_en_proceso


# --- el módulo importado directo: CFG, dispositivo, y las funciones puras ------


@pytest.mark.gpu
def test_paso07_modulo_importado_ejercita_las_funciones_puras(tmp_path, monkeypatch):
    """Complemento de las pruebas de arriba (que solo miran la salida
    impresa de un subproceso): importa el paso directo (con PRESUPUESTO=humo,
    vía monkeypatch) para ejercitar sus funciones puras sobre el módulo YA
    IMPORTADO, sin correrlo de nuevo."""
    monkeypatch.setenv("PRESUPUESTO", "humo")
    monkeypatch.setenv("TALLER_RESULTADOS", str(tmp_path))

    modulo = _importar(PASO_07, "_paso07_bajo_prueba")

    assert modulo.DISPOSITIVO_TORCH == "cuda"
    assert modulo.CFG == modulo.PRESUPUESTOS["humo"]
    assert modulo.costo_en_proceso > 0.0
    assert modulo.costo_por_subproceso > 0.0
    assert modulo.sobrecosto_fijo_reportado == modulo.sobrecosto_fijo

    # _trials_de_esta_llamada es pura y no toca ningún reloj de pared. Con
    # `antes=0` en un estudio nuevo, TODOS sus trials (FAIL incluido) son de
    # esta llamada; con un estudio que YA tenía trials antes de la llamada,
    # `antes` debe descontarlos -- si no se descontaran, la cuenta
    # confundiría trials viejos con nuevos.
    study_prueba = optuna.create_study()
    study_prueba.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL))
    study_prueba.add_trial(optuna.trial.create_trial(value=0.5))
    assert modulo._trials_de_esta_llamada(study_prueba, 0) == 2

    antes_prueba = len(study_prueba.trials)
    study_prueba.add_trial(optuna.trial.create_trial(value=0.6))
    study_prueba.add_trial(optuna.trial.create_trial(value=0.7))
    assert modulo._trials_de_esta_llamada(study_prueba, antes_prueba) == 2
    # Sin descontar antes, la cuenta se confundiría con el total de trials,
    # no con los nuevos de esta llamada.
    total_prueba = len(study_prueba.trials)
    assert modulo._trials_de_esta_llamada(study_prueba, 0) == total_prueba != 2

    # _texto_sobrecosto_fijo: las tres direcciones (sobrecosto positivo,
    # equivalente a 0, no concluyente), con series sintéticas fáciles de
    # verificar a mano.
    valor_pos, texto_pos = modulo._texto_sobrecosto_fijo([3.0] * 5, [0.1] * 5, 5)
    assert valor_pos == pytest.approx(2.9)
    assert "~2.900 s" in texto_pos
    assert "sobrecosto fijo medido con intervalo de confianza" in texto_pos

    valor_equiv, texto_equiv = modulo._texto_sobrecosto_fijo([0.1] * 5, [0.1] * 5, 5)
    assert valor_equiv == pytest.approx(0.0)
    assert "equivalente a 0" in texto_equiv

    tiempos_cli_noconc = [0.3, 0.1, 0.3, 0.1, 0.3]
    tiempos_en_proceso_noconc = [0.1] * 5
    p_verificacion = modulo.estadistica.diferencia_pareada(
        dict(enumerate(tiempos_cli_noconc)),
        dict(enumerate(tiempos_en_proceso_noconc)),
        delta=0.10 * modulo.statistics.median(tiempos_en_proceso_noconc),
        direccion="maximizar",
    )
    assert p_verificacion.veredicto == "no concluyente", p_verificacion
    valor_noconc, texto_noconc = modulo._texto_sobrecosto_fijo(
        tiempos_cli_noconc, tiempos_en_proceso_noconc, 5
    )
    assert valor_noconc is None
    assert "no concluyente" in texto_noconc

    # _medir_sobrecosto_intercalado: intercalado por ronda (nunca todas las
    # repeticiones de una serie antes que las de otra -- ver su docstring),
    # con espías sobre las dos mediciones de una sola corrida. Con
    # repeticiones=1 (2 rondas: calentamiento + 1 medida), el par debe
    # alternar ABBA: ronda 0 (par) línea de comandos primero, ronda 1 (impar) en_proceso
    # primero -- un CFG["repeticiones_sobrecosto"] > 0 real (taller/completo)
    # corre esto mismo con más rondas.
    orden_llamadas: list[str] = []
    cli_original = modulo._medir_una_cli
    en_proceso_original = modulo._medir_una_en_proceso

    def cli_espia():
        orden_llamadas.append("cli")
        return cli_original()

    def en_proceso_espia():
        orden_llamadas.append("en_proceso")
        return en_proceso_original()

    monkeypatch.setattr(modulo, "_medir_una_cli", cli_espia)
    monkeypatch.setattr(modulo, "_medir_una_en_proceso", en_proceso_espia)
    tiempos_cli_ds, tiempos_en_proceso_ds = modulo._medir_sobrecosto_intercalado(1)
    assert orden_llamadas == ["cli", "en_proceso", "en_proceso", "cli"], orden_llamadas
    # 1 repetición + 1 calentamiento = 2 rondas; las listas devueltas
    # excluyen la ronda de calentamiento: 1 elemento cada una.
    assert len(tiempos_cli_ds) == 1 and len(tiempos_en_proceso_ds) == 1

    # _medir_una_cli: una sola corrida, con la config de la línea base de la
    # búsqueda (train_ejemplo.DEFAULTS) -- verifica que SÍ llega esa config
    # al JSON de entrada.
    entradas_escritas: list[dict] = []
    escribir_original = modulo.escribir_json_atomico

    def escribir_espia(ruta, obj):
        entradas_escritas.append(obj)
        return escribir_original(ruta, obj)

    monkeypatch.setattr(modulo, "escribir_json_atomico", escribir_espia)
    modulo._medir_una_cli()
    assert entradas_escritas, "no se escribió ninguna entrada"
    assert entradas_escritas[0]["config"] == dict(modulo.train_ejemplo.DEFAULTS)

    # _medir_una_en_proceso: una sola corrida EN PROCESO, con la misma config
    # de línea base.
    llamadas_entrenar: list[dict] = []
    entrenar_original = modulo.train_ejemplo.entrenar

    def entrenar_espia(config, semillas):
        llamadas_entrenar.append(dict(config))
        return entrenar_original(config, semillas)

    monkeypatch.setattr(modulo.train_ejemplo, "entrenar", entrenar_espia)
    t = modulo._medir_una_en_proceso()
    assert t > 0.0
    assert len(llamadas_entrenar) == 1, llamadas_entrenar
    assert llamadas_entrenar[0] == dict(modulo.train_ejemplo.DEFAULTS), llamadas_entrenar[0]


# --- _texto_sobrecosto_fijo: la llamada recibe los argumentos en orden --------


def _args_de_llamada(arbol: ast.Module, nombre_funcion: str) -> list[str]:
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and ast.unparse(nodo.func) == nombre_funcion
    ]
    assert len(llamadas) == 1, f"debe haber una única llamada a {nombre_funcion}"
    return [ast.unparse(a) for a in llamadas[0].args]


def test_ast_texto_sobrecosto_fijo_recibe_los_desgloses_en_orden(tmp_path):
    """Revisión estática (sin correr el paso): `_texto_sobrecosto_fijo` recibe
    `(tiempos_cli_sobrecosto, tiempos_en_proceso_sobrecosto,
    CFG["repeticiones_sobrecosto"])`, en ese orden -- comparado por
    `ast.unparse`, no por conteo de argumentos, para que intercambiar dos de
    esos nombres en la llamada no pase inadvertido. La segunda mitad
    demuestra, sobre una copia del código en `tmp_path` con dos argumentos
    cruzados, que esta misma comparación deja de cumplirse: si no lo hiciera,
    tampoco distinguiría un cableado correcto de uno con dos argumentos
    cruzados."""
    codigo = PASO_07.read_text(encoding="utf-8")
    arbol = ast.parse(codigo, filename=str(PASO_07))

    esperado = [
        "tiempos_cli_sobrecosto",
        "tiempos_en_proceso_sobrecosto",
        "CFG['repeticiones_sobrecosto']",
    ]
    assert _args_de_llamada(arbol, "_texto_sobrecosto_fijo") == esperado

    bloque_original = (
        "    sobrecosto_medido, texto_sobrecosto_medido = _texto_sobrecosto_fijo(\n"
        "        tiempos_cli_sobrecosto, tiempos_en_proceso_sobrecosto, "
        'CFG["repeticiones_sobrecosto"]\n'
        "    )\n"
    )
    bloque_mutado = (
        "    sobrecosto_medido, texto_sobrecosto_medido = _texto_sobrecosto_fijo(\n"
        "        tiempos_en_proceso_sobrecosto, tiempos_cli_sobrecosto, "
        'CFG["repeticiones_sobrecosto"]\n'
        "    )\n"
    )
    assert codigo.count(bloque_original) == 1, "el bloque a mutar no apareció exactamente una vez"
    codigo_mutado = codigo.replace(bloque_original, bloque_mutado)

    copia_mutada = tmp_path / "07_conecta_tu_modelo_mutado.py"
    copia_mutada.write_text(codigo_mutado, encoding="utf-8")
    arbol_mutado = ast.parse(codigo_mutado, filename=str(copia_mutada))
    assert _args_de_llamada(arbol_mutado, "_texto_sobrecosto_fijo") != esperado


def test_ast_buscar_por_subproceso_pasa_dispositivo_ya_comprobado():
    """El paso resuelve `dispositivo_por_subproceso` antes de arrancar el
    reloj de por_subproceso (ver el comentario junto a esa variable): sin
    decirle a `buscar()` que no repita la comprobación,
    `_detectar_dispositivo` -- un intérprete nuevo que importa torch --
    correría DENTRO del intervalo que mide el reloj, tal como evita
    `buscar_por_subproceso.main()` (ver
    tests/test_plantilla_busqueda.py::test_detectar_dispositivo_se_llama_una_sola_vez_en_main).
    Revisión estática (sin correr el paso, no necesita tarjeta de video):
    busca la llamada a `buscar_por_subproceso.buscar` en el árbol y comprueba
    que trae el keyword `_dispositivo_ya_comprobado=True`."""
    codigo = PASO_07.read_text(encoding="utf-8")
    arbol = ast.parse(codigo, filename=str(PASO_07))
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and ast.unparse(nodo.func) == "buscar_por_subproceso.buscar"
    ]
    assert len(llamadas) == 1, "debe haber una única llamada a buscar_por_subproceso.buscar"
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in llamadas[0].keywords}
    assert kwargs.get("_dispositivo_ya_comprobado") == "True", kwargs


def test_ast_resolver_dispositivo_pasa_cuda_explicito_y_ya_comprobado_correcto():
    """Este paso compara las dos rutas SIEMPRE en la tarjeta por defecto
    ("cuda"), sin importar TALLER_DISPOSITIVO (ver el comentario junto a
    `dispositivo_por_subproceso` en el paso): un `None` en vez de `"cuda"`
    dejaría que `_resolver_dispositivo` consultara esa variable de entorno, y
    con TALLER_DISPOSITIVO=cuda:N la config de por_subproceso ("cuda:N")
    dejaría de coincidir con la de en_proceso ("cuda"), lo que revienta el
    assert de la sección 3 que compara `user_attrs["config"]` entre las dos
    rutas. Revisión estática (sin correr el paso, no necesita tarjeta de
    video): busca la única llamada a `buscar_por_subproceso._resolver_dispositivo`
    y exige `"cuda"` y `PYTHON_COMPARACION` como sus dos argumentos
    posicionales, más `ya_comprobado=PYTHON_COMPARACION == sys.executable`
    como keyword -- un `ya_comprobado=True` fijo, en cambio, saltaría la
    comprobación de la tarjeta cuando TALLER_PYTHON apunta a otro entorno."""
    codigo = PASO_07.read_text(encoding="utf-8")
    arbol = ast.parse(codigo, filename=str(PASO_07))
    llamadas = [
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and ast.unparse(nodo.func) == "buscar_por_subproceso._resolver_dispositivo"
    ]
    assert len(llamadas) == 1, (
        "debe haber una única llamada a buscar_por_subproceso._resolver_dispositivo"
    )
    llamada = llamadas[0]
    argumentos = [ast.unparse(a) for a in llamada.args]
    assert argumentos == ["'cuda'", "PYTHON_COMPARACION"], argumentos
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in llamada.keywords}
    assert kwargs.get("ya_comprobado") == "PYTHON_COMPARACION == sys.executable", kwargs
