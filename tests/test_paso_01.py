"""Contenido de las respuestas del paso 01, no solo que corra (eso lo cubre
`tests/test_pasos.py`).

`corrida_humo` corre el paso una sola vez por módulo, con `-W error` y vía `runpy.run_path`
en un subproceso nuevo, para leer además los params del primer trial de `estudio_random` y
`estudio_qmc` de la última réplica. `salida_humo` reparte ese stdout entre las pruebas de
contenido. `salida_taller` hace lo mismo con PRESUPUESTO=taller, para la única afirmación
que necesita más réplicas (la pregunta 4 contra QMC).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import optuna
import pytest

from taller_hpo import estadistica

RAIZ = Path(__file__).resolve().parent.parent
PASO = RAIZ / "pasos" / "01_grid_vs_random.py"

# Número (entero o decimal, con exponente científico opcional): las
# respuestas del paso imprimen con formatos .2e/.2g/.1g según la magnitud
# (.4f no alcanza para regrets de orden 1e-4, que quedarían en "0.0001" sin
# ninguna cifra significativa), así que el regex de las pruebas debe aceptar
# ambas formas.
NUM = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"


def _envoltura_runpy(registro: Path) -> str:
    """`runpy.run_path(PASO, run_name='__main__')` dentro del propio subproceso,
    volcando a `registro` los params del primer trial de `estudio_random` y
    `estudio_qmc` de la última réplica -- mismo patrón que
    tests/test_paso_04.py (envolver algo y correr el paso con runpy.run_path
    dentro de un SUBPROCESO nuevo, nunca en el proceso de pytest)."""
    return (
        "import json\n"
        "import runpy\n"
        "\n"
        f"g = runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
        "salida = {\n"
        "    'random': g['estudio_random'].trials[0].params,\n"
        "    'qmc': g['estudio_qmc'].trials[0].params,\n"
        "}\n"
        f"with open({str(registro)!r}, 'w', encoding='utf-8') as _f:\n"
        "    json.dump(salida, _f)\n"
    )


@pytest.fixture(scope="module")
def corrida_humo(tmp_path_factory) -> tuple[str, dict]:
    """Corre el paso UNA sola vez con PRESUPUESTO=humo, bajo `-W error`: un
    aviso que el paso re-emita desde el bloque de QMCSampler (ver el
    comentario sobre `warnings.warn_explicit` en pasos/01_grid_vs_random.py),
    o cualquier otro que deje salir, se vuelve una excepción visible aquí, en
    vez de un renglón de stderr que ninguna prueba mira. Corre el paso con la
    envoltura de `runpy.run_path` para leer, de esta MISMA corrida, los params
    del primer trial de `estudio_random` y `estudio_qmc`: así ninguna prueba
    de contenido necesita correr el paso una segunda vez solo para comparar
    esos dos params.
    """
    resultados = tmp_path_factory.mktemp("resultados_humo")
    registro = tmp_path_factory.mktemp("registro_humo") / "params.json"
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", _envoltura_runpy(registro)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert resultado.returncode == 0, resultado.stderr
    params = json.loads(registro.read_text(encoding="utf-8"))
    return resultado.stdout, params


@pytest.fixture(scope="module")
def salida_humo(corrida_humo) -> str:
    return corrida_humo[0]


@pytest.fixture(scope="module")
def salida_taller(tmp_path_factory) -> str:
    resultados = tmp_path_factory.mktemp("resultados_taller")
    entorno = {**os.environ, "PRESUPUESTO": "taller", "TALLER_RESULTADOS": str(resultados)}
    resultado = subprocess.run(
        [sys.executable, "-W", "error", str(PASO)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert resultado.returncode == 0, resultado.stderr
    return resultado.stdout


def test_paso_01_imprime_las_cuatro_respuestas_numeradas(salida_humo):
    for inicio in ("1. Random vence", "2. Diferencia pareada", "3. En cada réplica", "4. QMC"):
        assert inicio in salida_humo, salida_humo


def test_paso_01_avisa_que_no_usa_la_tarjeta_de_video(salida_humo):
    """Único paso sin tarjeta de video del taller: el paisaje analítico no
    entrena nada, así que no hay nada que mandar a un dispositivo. La
    convención "todo entrena en la tarjeta de video" exige decir esta
    excepción en una línea del paso (aquí) y en su markdown, no darla por
    sabida.
    """
    assert "Este paso no usa la tarjeta de video" in salida_humo, salida_humo


def test_paso_01_random_vence_a_grid_en_la_mayoria_con_humo(salida_humo):
    """R=10 en humo (PRESUPUESTOS del propio paso): la afirmación de Bergstra y
    Bengio debe sostenerse ya con el presupuesto más chico, no solo con `taller`
    o `completo`. La respuesta 1 es un conteo con su propio intervalo de
    confianza de Wilson, sin ninguna frase interpretativa que invertir --
    esta mayoría es una verificación de cordura con humo, no una conclusión
    con veredicto; las conclusiones que SÍ salen de `diferencia_pareada`
    (respuestas 2 y 4) se prueban en las dos direcciones más abajo
    (test_paso_01_con_taller_qmc_se_distingue_de_random_y_la_nota_desaparece
    es la dirección "el criterio SÍ debe rechazar" para la pregunta 4).
    """
    m = re.search(
        r"Random vence a grid en (\d+) de (\d+) réplicas \(([\d.]+)%, "
        r"intervalo de confianza 95% Wilson \[([\d.]+)%, ([\d.]+)%\]\)",
        salida_humo,
    )
    assert m, salida_humo
    victorias, total, porcentaje = int(m.group(1)), int(m.group(2)), float(m.group(3))
    ic_bajo_txt, ic_alto_txt = m.group(4), m.group(5)

    assert total == 10  # PRESUPUESTOS["humo"] = 10, medido en pasos/01_grid_vs_random.py
    assert 0 <= victorias <= total
    assert victorias == round(porcentaje / 100.0 * total)  # el % impreso coincide con el conteo
    assert victorias > total // 2, salida_humo

    # El intervalo de confianza 95% Wilson impreso tiene que ser el que de
    # verdad calcula `estadistica.ic_proporcion(victorias, total)`, no un
    # intervalo con otra confianza (p. ej. 50%) rotulado "95%" -- mismo patrón que
    # tests/test_paso_05.py (esperado_bajo, esperado_alto =
    # estadistica.ic_proporcion(top_count, total_replicas); assert ic_bajo ==
    # pytest.approx(esperado_bajo, abs=0.005)).
    ic_bajo, ic_alto = estadistica.ic_proporcion(victorias, total)
    assert ic_bajo_txt == f"{100 * ic_bajo:.1f}", (ic_bajo_txt, ic_bajo)
    assert ic_alto_txt == f"{100 * ic_alto:.1f}", (ic_alto_txt, ic_alto)


def test_paso_01_diferencia_pareada_grid_menos_random_es_consistente(salida_humo):
    """La respuesta 2 imprime `diferencia_media` con su intervalo de confianza
    en unidades "grid - random" (ver el comentario en
    pasos/01_grid_vs_random.py sobre por qué el orden de argumentos de
    diferencia_pareada se invierte para lograrlo). El intervalo debe
    contener a la diferencia y, como grid pierde con humo (pregunta 1),
    debe quedar del lado positivo -- estructuralmente por qué: grid desperdicia
    24 de sus 27 evaluaciones en ejes que no importan (pregunta 3). DELTA es
    una constante fija a priori (0.01**2), no calculada desde los propios
    resultados: mezclar el criterio de "diferencia despreciable" con los
    datos que se están juzgando permitiría mover el umbral para que el
    veredicto saliera como se quiera.
    """
    patron = (
        rf"grid - random\) = ({NUM}) \[({NUM}), ({NUM})\] con delta=({NUM}) "
        r"\(regret de errar el eje importante por 1% de su rango\) -> random resulta "
        rf"(.+?) frente a grid \(p=({NUM})\)"
    )
    m = re.search(patron, salida_humo)
    assert m, salida_humo
    diferencia, ic_bajo, ic_alto, delta = (float(m.group(i)) for i in (1, 2, 3, 4))
    veredicto = m.group(5)
    p_valor = float(m.group(6))

    assert ic_bajo <= diferencia <= ic_alto
    assert math.isclose(delta, 0.01**2, rel_tol=1e-9)
    assert veredicto in ("mejor", "peor", "equivalente", "no concluyente")
    assert 0.0 <= p_valor <= 1.0
    # random gana la mayoría (pregunta 1) y su regret medio es estructuralmente
    # menor: la diferencia grid-random debe ser positiva.
    assert diferencia > 0.0
    assert ic_bajo > 0.0  # el intervalo no cruza el cero: coherente con veredicto "mejor"
    assert veredicto == "mejor"
    assert "random resulta mejor frente a grid" in salida_humo


def test_paso_01_cuenta_valores_medidos_por_eje(salida_humo):
    """La respuesta 3 imprime números MEDIDOS por réplica (trials corridos,
    valores distintos del eje importante, y cuántas réplicas grid alcanzó el
    mínimo analítico de su propia rejilla vía `regret_minimo_grid`), no
    constantes como `len(NIVELES)` o `N_TRIALS` reimpresas sin medir nada:
    una constante reimpresa seguiría pareciendo cierta aunque el paso se
    rompiera y dejara de correr esos trials de verdad. Con humo (R=10,
    sembrado) el valor medido coincide con el estructural: grid solo prueba
    los 3 niveles de su rejilla en TODAS las réplicas.
    """
    m = re.search(
        r"En cada réplica cada brazo corrió (\S+) trials \(27 pedidos\); "
        r"valores distintos del eje importante: grid (\S+), random (\S+), qmc (\S+)\. "
        r"Grid alcanzó el mínimo analítico de su propia rejilla \(regret_minimo_grid\) en "
        r"(\d+) de (\d+) réplicas",
        salida_humo,
    )
    assert m, salida_humo
    trials, grid_v, random_v, qmc_v, alcanzados, total = m.groups()
    assert trials == "27"
    assert grid_v == "3"
    assert random_v == "27"
    assert qmc_v == "27"
    assert total == "10"
    # Estructural (docstring de regret_minimo_grid): la rejilla 3x3x3 cubre
    # cada uno de sus 3 niveles en el eje importante en TODA réplica, así que
    # el mejor regret encontrado coincide con el mínimo analítico siempre.
    assert alcanzados == total
    # La razón ("agota sus 3 niveles...") solo debe aparecer cuando grid_v
    # midió exactamente 3 valores distintos: es la MISMA condición que arma
    # la frase en pasos/01_grid_vs_random.py, no una constante repetida aquí
    # sin depender de lo medido.
    assert (grid_v == "3") == ("agota sus 3 niveles del eje importante" in salida_humo)


def test_paso_01_qmc_frente_a_random_y_grid_con_numeros_medidos(salida_humo):
    """La respuesta 4 se redacta desde `.veredicto` y `.diferencia_media` de
    `diferencia_pareada`, no desde promedios crudos: comparar solo medias
    dejaría que el puro ruido de muestreo imprimiera una conclusión causal
    falsa, porque una diferencia que no es significativa igual tiene un
    signo. Con humo (R=10) el veredicto medido es el esperado: sin potencia
    para distinguir QMC de random puro ("no concluyente"), pero sí mejor que
    una rejilla 3x3x3 ("mejor").
    """
    patron = (
        rf"QMC frente a random: (.+?) \(random - qmc = ({NUM}) \[({NUM}), ({NUM})\], "
        rf"p=({NUM})\); frente a grid: (.+?) \(grid - qmc = ({NUM}) \[({NUM}), ({NUM})\], "
        rf"p=({NUM})\)"
    )
    m = re.search(patron, salida_humo)
    assert m, salida_humo
    (
        veredicto_random,
        dif_random,
        ic_bajo_random,
        ic_alto_random,
        p_random,
        veredicto_grid,
        dif_grid,
        ic_bajo_grid,
        ic_alto_grid,
        p_grid,
    ) = m.groups()

    for v in (veredicto_random, veredicto_grid):
        assert v in ("mejor", "peor", "equivalente", "no concluyente")
    assert float(ic_bajo_random) <= float(dif_random) <= float(ic_alto_random)
    assert float(ic_bajo_grid) <= float(dif_grid) <= float(ic_alto_grid)
    assert 0.0 <= float(p_random) <= 1.0
    assert 0.0 <= float(p_grid) <= 1.0

    assert veredicto_random == "no concluyente"
    assert veredicto_grid == "mejor"
    # La nota condicional (docs/arquitectura.md: nunca afirmar lo que no se
    # mide) solo sale cuando el veredicto contra random es "no concluyente",
    # como aquí.
    assert "Con estas 10 réplicas, QMC no se distingue de random." in salida_humo


def test_paso_01_con_taller_qmc_se_distingue_de_random_y_la_nota_desaparece(salida_taller):
    """Dirección contraria de la prueba anterior, con la MISMA maquinaria
    (`diferencia_pareada` sobre `qmc_vs_random`): con humo (R=10) no hay
    potencia para distinguir QMC de random puro ("no concluyente"); con
    taller (R=100) la ventaja de QMC sí alcanza a distinguirse ("mejor"), y
    la nota condicional que dice "QMC no se distingue de random" debe
    desaparecer -- si siguiera apareciendo pese al veredicto "mejor", la nota
    estaría escrita de antemano en vez de depender de
    `qmc_vs_random.veredicto`, y seguiría imprimiéndose igual aunque el brazo
    QMC fuera en realidad puro ruido con otra semilla.
    """
    patron = (
        rf"QMC frente a random: (.+?) \(random - qmc = ({NUM}) \[({NUM}), ({NUM})\], p=({NUM})\)"
    )
    m = re.search(patron, salida_taller)
    assert m, salida_taller
    veredicto_random = m.group(1)

    assert veredicto_random == "mejor", salida_taller
    assert "QMC no se distingue de random" not in salida_taller, salida_taller


def test_paso_01_avisa_una_sola_vez_que_qmcsampler_es_experimental(salida_humo):
    """QMCSampler es experimental en Optuna 5.0.0 (`@experimental_class`, ver
    optuna/samplers/_qmc.py): pasos/01_grid_vs_random.py atrapa el
    ExperimentalWarning con warnings.catch_warnings(record=True) e imprime una
    nota explícita en vez de silenciarlo sin decir nada (regla del taller). La
    nota debe salir UNA vez, no una por réplica -- si saliera R veces (R=10 en
    humo) taparía las respuestas igual que el log INFO de Optuna.
    """
    assert salida_humo.count("Nota: QMCSampler es experimental") == 1, salida_humo


def test_paso_01_re_emite_avisos_que_no_son_el_experimentalwarning_esperado(tmp_path):
    """Dentro de `warnings.catch_warnings(record=True)`, pasos/01_grid_vs_random.py
    re-emite (`warnings.warn_explicit`) cualquier aviso capturado que NO sea el
    `ExperimentalWarning` esperado de `QMCSampler`, en vez de descartarlo en
    silencio -- si no lo hiciera, un aviso real ahí adentro jamás llegaría a
    stderr, ni con `-W error` (que `corrida_humo`, más arriba, ya usa).

    Para forzar ese aviso ADICIONAL sin depender de que Optuna algún día
    empiece a emitir uno de verdad, se envuelve `QMCSampler.__init__` (patrón
    de `test_reevaluacion_usa_semillas_nuevas_y_no_optimiza_el_oraculo` en
    tests/test_paso_04.py: envolver algo y correr el paso con
    `runpy.run_path` dentro de un SUBPROCESO nuevo, nunca en el proceso de
    pytest) para que SOLO la primera llamada -- la instancia desechable de
    `pasos/01_grid_vs_random.py` dentro de `catch_warnings`, la única que
    corre antes de que empiece el bucle de réplicas -- emita además un
    `DeprecationWarning` sintético. Tiene que ser solo esa primera llamada: si
    todas las instancias de QMCSampler del bucle (3*R por corrida) lo
    emitieran, `-W error` reventaría por esa razón en cualquiera de ellas, no
    por la re-emisión que esta prueba quiere aislar.
    """
    resultados = tmp_path / "resultados"
    envoltura = (
        "import warnings\n"
        "import runpy\n"
        "\n"
        "import optuna\n"
        "\n"
        "_original = optuna.samplers.QMCSampler.__init__\n"
        "_llamadas = []\n"
        "\n"
        "\n"
        "def _envuelto(self, *args, **kwargs):\n"
        "    primera = not _llamadas\n"
        "    _llamadas.append(1)\n"
        "    if primera:\n"
        "        warnings.warn('aviso sintetico', DeprecationWarning, stacklevel=2)\n"
        "    return _original(self, *args, **kwargs)\n"
        "\n"
        "\n"
        "optuna.samplers.QMCSampler.__init__ = _envuelto\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", envoltura],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert resultado.returncode != 0, resultado.stdout
    assert "aviso sintetico" in resultado.stderr, resultado.stderr


def test_paso_01_el_texto_sigue_al_veredicto_y_a_lo_medido(tmp_path):
    """Con los datos reales, grid pierde contra random y contra QMC en todos
    los presupuestos, y grid alcanza el mínimo de su rejilla en todas las
    réplicas: con solo esas corridas, una frase fija ("random resulta mejor",
    "frente a grid: mejor", "agota sus 3 niveles...") pasaría las pruebas de
    contenido igual que la frase que sale de lo medido. Esta prueba fuerza el
    resultado contrario sin tocar el paso -- mismo patrón que
    tests/test_paso_04.py (reemplazar una función del módulo antes de
    `runpy.run_path`, dentro de un subproceso nuevo):

    - `estadistica.diferencia_pareada` recibe `a` y `b` intercambiados, así
      que la respuesta 2 y la comparación con grid de la respuesta 4 deben
      decir "peor", con la diferencia en negativo.
    - `analitico.regret_minimo_grid` devuelve un mínimo que ningún trial
      alcanza, así que la respuesta 3 debe contar 0 réplicas y callar la
      razón "agota sus 3 niveles...".
    """
    resultados = tmp_path / "resultados"
    envoltura = (
        "import runpy\n"
        "\n"
        "from taller_hpo import estadistica\n"
        "from taller_hpo.problemas import analitico\n"
        "\n"
        "_pareada = estadistica.diferencia_pareada\n"
        "_minimo = analitico.regret_minimo_grid\n"
        "\n"
        "\n"
        "def _pareada_invertida(a, b, **kwargs):\n"
        "    return _pareada(b, a, **kwargs)\n"
        "\n"
        "\n"
        "def _minimo_inalcanzable(paisaje, niveles):\n"
        "    return _minimo(paisaje, niveles) + 1.0\n"
        "\n"
        "\n"
        "estadistica.diferencia_pareada = _pareada_invertida\n"
        "analitico.regret_minimo_grid = _minimo_inalcanzable\n"
        f"runpy.run_path({str(PASO)!r}, run_name='__main__')\n"
    )
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_RESULTADOS": str(resultados)}
    resultado = subprocess.run(
        [sys.executable, "-W", "error", "-c", envoltura],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert resultado.returncode == 0, resultado.stderr
    salida = resultado.stdout

    m = re.search(rf"grid - random\) = ({NUM}) ", salida)
    assert m, salida
    assert float(m.group(1)) < 0.0, salida
    assert "random resulta peor frente a grid" in salida, salida
    assert "random resulta mejor frente a grid" not in salida, salida

    m = re.search(rf"; frente a grid: (.+?) \(grid - qmc = ({NUM}) ", salida)
    assert m, salida
    assert m.group(1) == "peor", salida
    assert float(m.group(2)) < 0.0, salida

    assert "rejilla (regret_minimo_grid) en 0 de 10 réplicas." in salida, salida
    assert "agota sus 3 niveles del eje importante" not in salida, salida


def test_paso_01_presupuesto_invalido_falla_con_mensaje(tmp_path):
    entorno = {**os.environ, "PRESUPUESTO": "enorme", "TALLER_RESULTADOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(PASO)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


def test_paso_01_qmc_sin_independent_sampler_propio_comparte_el_primer_trial_con_random():
    """`QMCSampler`, sin un `independent_sampler` explícito, usa
    `RandomSampler(seed=seed)` internamente (optuna/samplers/_qmc.py:
    `self._independent_sampler = independent_sampler or
    optuna.samplers.RandomSampler(seed=seed)`). Con el MISMO seed que el
    brazo random de la réplica, su primer trial sale idéntico al primer
    trial de random -- contaminando la comparación pareada, porque uno de
    los dos puntos emparejados deja de ser independiente del otro. Esta
    prueba fija ese acoplamiento como un hecho medido, no una suposición.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio_random = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    estudio_random.optimize(lambda t: t.suggest_float("x0", 0.0, 1.0), n_trials=1)

    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        sampler_qmc_semilla_compartida = optuna.samplers.QMCSampler(
            qmc_type="sobol", scramble=True, seed=0
        )
    estudio_qmc_semilla_compartida = optuna.create_study(
        direction="minimize", sampler=sampler_qmc_semilla_compartida
    )
    estudio_qmc_semilla_compartida.optimize(lambda t: t.suggest_float("x0", 0.0, 1.0), n_trials=1)

    assert (
        estudio_random.trials[0].params["x0"]
        == estudio_qmc_semilla_compartida.trials[0].params["x0"]
    )


def test_paso_01_qmc_con_independent_sampler_propio_no_comparte_el_primer_trial():
    """Pasar un `independent_sampler` propio (semilla
    `SEMILLA_QMC_INDEPENDIENTE + r`, desacoplada de la semilla `r` del brazo
    random) rompe el acoplamiento de la prueba anterior: el primer trial de
    QMC deja de compartir semilla con el primer trial de random. Dirección
    contraria de esa misma prueba: aquí SÍ debe diferir, en cada una de 5
    réplicas (semillas 0 a 4).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for r in range(5):
        estudio_random = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.RandomSampler(seed=r)
        )
        estudio_random.optimize(lambda t: t.suggest_float("x0", 0.0, 1.0), n_trials=1)

        with pytest.warns(optuna.exceptions.ExperimentalWarning):
            sampler_qmc = optuna.samplers.QMCSampler(
                qmc_type="sobol",
                scramble=True,
                seed=r,
                independent_sampler=optuna.samplers.RandomSampler(seed=10_000 + r),
            )
        estudio_qmc = optuna.create_study(direction="minimize", sampler=sampler_qmc)
        estudio_qmc.optimize(lambda t: t.suggest_float("x0", 0.0, 1.0), n_trials=1)

        assert estudio_random.trials[0].params["x0"] != estudio_qmc.trials[0].params["x0"]


def test_paso_01_el_paso_en_si_desacopla_qmc_de_random(corrida_humo):
    """Las dos pruebas anteriores ejercitan a `QMCSampler`/`RandomSampler`
    directamente, no al paso: si `pasos/01_grid_vs_random.py` dejara de
    pasarle `independent_sampler` a `QMCSampler` (volviendo a acoplar sus
    primeros trials con los de random), esas dos pruebas seguirían pasando
    sin cambios, porque no corren el script. `corrida_humo` (más arriba) ya
    corre el paso completo con PRESUPUESTO=humo vía `runpy.run_path` dentro de
    un subproceso nuevo y deja volcados los params del primer trial de
    `estudio_random` y `estudio_qmc` para la ÚLTIMA réplica (r=R-1, la que
    dejan apuntando esos dos nombres al salir del bucle); esta prueba solo
    los lee, sin correr el paso otra vez.
    """
    _, params = corrida_humo
    assert params["random"] != params["qmc"], params
