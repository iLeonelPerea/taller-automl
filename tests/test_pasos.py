"""Cada paso corre de punta a punta, como lo correría el tesista, con PRESUPUESTO=humo.

Cada paso corre UNA sola vez por módulo de pytest (fixture `corridas_cache`,
scope="module") y las pruebas de contenido reparten ese mismo
`CompletedProcess`: entrenar el juguete en la tarjeta de video cuesta tiempo
real, y varias pruebas de este archivo comprueban cosas distintas de la
salida del mismo paso. El taller entrena siempre en una tarjeta de video
NVIDIA con CUDA (ver `docs/arquitectura.md`): las pruebas que corren un paso
que entrena el juguete llevan la marca `gpu` (`tests/conftest.py`) y fallan,
sin tarjeta de video visible, en vez de saltarse.
"""

import ast
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as scipy_stats

RAIZ = Path(__file__).resolve().parent.parent
PASOS = sorted((RAIZ / "pasos").glob("[0-9][0-9]_*.py"))
PASO_00 = RAIZ / "pasos" / "00_linea_base.py"
PASO_05 = RAIZ / "pasos" / "05_buscar_el_modelo.py"

# Cuántos juegos de semillas entrena el paso 00 con PRESUPUESTO=humo -- tiene
# que coincidir con PRESUPUESTOS["humo"] en pasos/00_linea_base.py.
N_HUMO_PASO_00 = 3

# Pasos que entrenan el juguete vía taller_hpo.problemas.juguete e imprimen su
# DESCRIPCION al arrancar, con el prefijo EXACTO que usa cada uno -- 07 solo
# imprime la DESCRIPCION del juguete del repo como referencia (no entrena con
# él: sus secciones 2 y 3 corren plantilla/train_ejemplo.py, un modelo
# aparte, ver su cabecera) y por eso usa su propia frase en vez de
# "Motor: ", que es la de los pasos que sí entrenan el juguete.
PREFIJOS_MOTOR = {
    "00_linea_base": "Motor: ",
    "02_primer_estudio": "Motor: ",
    "03_poda": "Motor: ",
    "04_pareto": "Motor: ",
    "06_honestidad": "Motor: ",
    "07_conecta_tu_modelo": "Motor resuelto para el juguete de este repo: ",
}
# Pasos que NO pasan por taller_hpo.problemas.juguete, con el motivo. Un paso
# nuevo que no aparezca ni en PREFIJOS_MOTOR ni aquí hace fallar
# test_todos_los_pasos_estan_clasificados en vez de quedar sin revisar.
EXENTOS = {
    "01_grid_vs_random": (
        "resuelve taller_hpo.problemas.analitico (el paisaje sintético grid-vs-random); "
        "no entrena el MLP del juguete y no imprime 'Motor: '"
    ),
    "05_buscar_el_modelo": (
        "resuelve su propio dispositivo vía taller_hpo.problemas.cash_paridad "
        "(clave 'dispositivo') e imprime 'cash_paridad: dispositivo=...' "
        "en vez de 'Motor: ...' -- se comprueba aparte, ver "
        "test_paso_05_imprime_su_dispositivo_de_cash_paridad"
    ),
}
PASOS_CON_JUGUETE = [p for p in PASOS if p.stem in PREFIJOS_MOTOR]
# Los mismos, más el 05: no imprime la DESCRIPCION de `juguete` (resuelve su
# propio dispositivo vía cash_paridad, ver EXENTOS), pero sí entrena de
# verdad con torch en la tarjeta de video -- `knn` y `krr` (cash_paridad)
# corren con torch en CADA réplica del paso 05, sin importar qué algoritmo
# sorteó la búsqueda ese trial (ver pasos/05_buscar_el_modelo.py:
# `resultado_knn`/`resultado_krr` se calculan fuera del muestreo de Optuna).
# Por eso entra también a la comprobación de uso real de la tarjeta de video
# de más abajo, aunque quede fuera de PASOS_CON_JUGUETE.
PASOS_CON_GPU_REAL = [*PASOS_CON_JUGUETE, PASO_05]

_PATRON_CORRIDA = re.compile(r"semilla\s*(\d+): validación (\d+\.\d+) \| verdadera (\d+\.\d+) \|")

# Espejo exacto de la sección "Las tres respuestas" de pasos/00_linea_base.py,
# para correr en un subproceso aparte con el MISMO entorno (mismo motor) que
# ya corrió el paso. semillas_de(0, N) es determinista, así que esta segunda
# corrida reproduce los mismos valores en precisión completa -- comparar así
# evita un falso rojo por cadena de redondeos: max(verdad) - min(verdad) en
# precisión completa no siempre coincide, ya redondeado a 1 decimal, con
# restar valores que ya se leyeron redondeados a 3 decimales del stdout del
# paso -- redondear dos veces puede mover el último decimal.
#
# Límite conocido de esta plantilla con PRESUPUESTO=humo (N_HUMO_PASO_00=3):
# la respuesta 2 usa `abs(v - t)` para medir cuánto se desvía la validación de
# la verdad -- un error que cambiara ese `abs` por la resta con signo NO
# se detecta con las 3 semillas de humo, porque con pocas semillas la
# validación puede caer siempre del mismo lado de la verdad. Por eso
# `desvio_validacion` (la función pura que hace ese cálculo dentro del paso)
# tiene su propia prueba unitaria fuera del subproceso, con un caso sintético
# donde `val` cae de los dos lados de `verdad` (ver test_desvio_validacion_*,
# más abajo), extraída con `ast` con el mismo patrón que `veredictos_franjas`.
#
# Los dos veredictos pareados de la respuesta 1 (Pitman-Morgan para cuál
# franja es más ancha, y la correlación entre franjas) SÍ tienen su propia
# prueba unitaria, sin pasar por humo ni por n_pares: `veredictos_franjas`
# (ver test_veredictos_franjas_*, más abajo), extraída con `ast` del propio
# `pasos/00_linea_base.py` y ejercitada con arreglos sintéticos en las dos
# direcciones. Esta plantilla inserta esa misma función (no una copia
# manual) para que la respuesta completa siga siendo un espejo exacto del
# paso.
_PLANTILLA_RESPUESTAS_PASO_00 = """
import statistics as st

from scipy import stats as scipy_stats

from taller_hpo.estadistica import resumen
from taller_hpo.problemas import juguete
from taller_hpo.problemas.paridad import error_bayes
from taller_hpo.semillas import semillas_de

juegos = semillas_de(0, __N__)
corridas = [juguete.entrenar({}, s) for s in juegos]
val = [m["exactitud_val"] for m in corridas]
verdad = [m["exactitud_verdadera"] for m in corridas]
errores = [abs(v - t) for v, t in zip(val, verdad, strict=True)]
r = resumen(verdad)
datos_fijos = juegos[0].datos
juegos_fijos = semillas_de(0, __N__, datos=datos_fijos)
fijos = [verdad[0]] + [
    juguete.entrenar({}, s)["exactitud_verdadera"] for s in juegos_fijos[1:]
]
r_fijos = resumen(fijos)
techo = 1.0 - error_bayes(juguete.DEFAULTS["d"], juguete.DEFAULTS["a"])

__VEREDICTOS_FRANJAS__

veredicto_ancho, veredicto_junto = veredictos_franjas(total=verdad, fijos=fijos)

print(
    f"1. Con {r.n} juegos de semillas (datos, orden e init), la exactitud verdadera va de "
    f"{min(verdad):.3f} a {max(verdad):.3f} ({100 * (max(verdad) - min(verdad)):.1f} puntos, "
    f"de={100 * r.de:.1f} puntos) sin cambiar una línea de código. Con los mismos datos y "
    f"solo orden e init variando -- lo único que puede cambiar un tesista con un dataset ya "
    f"fijo -- el rango es de {100 * (max(fijos) - min(fijos)):.1f} "
    f"puntos, de={100 * r_fijos.de:.1f} puntos. ¿Cuál franja es más ancha? {veredicto_ancho}. "
    f"¿Se mueven juntas? {veredicto_junto}."
)
print(
    f"2. Con las mismas {r.n} semillas, la validación ({juguete.DEFAULTS['n_val']} datos) "
    f"se desvía de la verdad {100 * st.mean(errores):.1f} puntos en promedio "
    f"y {100 * max(errores):.1f} en el peor caso."
)
print(
    f"3. El techo (1 - error de Bayes) es {techo:.3f}. Tu línea base, con {r.n} semillas, "
    f"promedia {r.media:.3f} (intervalo de confianza {100 * r.confianza:.0f}% "
    f"[{r.ic_bajo:.3f}, {r.ic_alto:.3f}]): "
    f"le faltan entre {100 * (techo - r.ic_alto):.1f} y {100 * (techo - r.ic_bajo):.1f} puntos "
    f"(intervalo {100 * r.confianza:.0f}%; {100 * (techo - r.media):.1f} en promedio)."
)
"""


def _entorno_base():
    """Entorno del proceso de pytest tal cual, sin overrides de motor: el taller
    entrena siempre en la tarjeta de video (ver `docs/arquitectura.md`), así que la
    única corrida posible es la que vería un tesista con `uv run pasos/NN_*.py`."""
    return dict(os.environ)


def _entorno(tmp_path, *, presupuesto="humo"):
    entorno = _entorno_base()
    entorno["PRESUPUESTO"] = presupuesto
    entorno["TALLER_RESULTADOS"] = str(tmp_path)
    # TALLER_ESTUDIOS SIEMPRE con un valor propio de la prueba: sin esto, dos
    # corridas del mismo paso comparten <repo>/estudios y la segunda reabre el
    # estudio de la primera (taller_hpo.estudio.crear_estudio no repite
    # trabajo al reanudar), así que un trial de una corrida vieja se cuenta
    # como si fuera de la corrida actual.
    entorno["TALLER_ESTUDIOS"] = str(tmp_path / "estudios")
    return entorno


def test_entorno_aisla_taller_estudios_en_tmp_path(tmp_path):
    """Sin `TALLER_ESTUDIOS` en el entorno de cada corrida, dos corridas del mismo paso
    comparten `<repo>/estudios` y la segunda reabre los trials de la primera en vez de
    partir de cero (el incidente que motivó fijarlo aquí). Esta prueba ancla que `_entorno`
    de verdad lo apunta dentro de `tmp_path` -- sin ella, quitar esa línea de `_entorno`
    no lo nota ninguna otra prueba."""
    entorno = _entorno(tmp_path)
    assert Path(entorno["TALLER_ESTUDIOS"]).is_relative_to(tmp_path), entorno["TALLER_ESTUDIOS"]


def _correr(paso, entorno, *, timeout=110, medir_uso_gpu=False):
    """Corre `paso` con `entorno`. Con `medir_uso_gpu=True`, esta función no
    lanza un proceso EXTRA para medir la tarjeta de video (eso costaría
    tiempo real de entrenar dos veces): en su lugar, la MISMA corrida única
    usa un lanzador que ejecuta el paso completo con
    `runpy.run_path(..., run_name="__main__")` (equivalente a `python
    paso.py`: ninguno de los pasos lee `sys.argv` ni depende de que su
    propio directorio quede al frente de `sys.path`, lo único que cambiaría
    entre las dos formas de invocarlo) y, YA TERMINADA la corrida completa,
    imprime cuánta memoria CUDA llegó a asignar PyTorch en total
    (`torch.cuda.max_memory_allocated()`, que acumula el pico desde el
    arranque del proceso, no un valor instantáneo)."""
    comando = [sys.executable, str(paso)]
    if medir_uso_gpu:
        comando = [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "runpy.run_path(sys.argv[1], run_name='__main__')\n"
            "import torch\n"
            "print('__cuda_max_mem__', torch.cuda.max_memory_allocated())\n",
            str(paso),
        ]
    return subprocess.run(comando, env=entorno, capture_output=True, text=True, timeout=timeout)


def _descripcion_esperada(entorno):
    """La `taller_hpo.problemas.juguete.DESCRIPCION` que resolvería ESTE
    entorno, calculada en un subproceso aparte en vez de comparada contra un
    valor fijo -- así la prueba no depende de la versión de PyTorch instalada
    ni del nombre exacto de la tarjeta de video, sin necesidad de saber de
    antemano cuáles son.
    """
    resultado = subprocess.run(
        [
            sys.executable,
            "-c",
            "from taller_hpo.problemas import juguete as j; print(j.DESCRIPCION)",
        ],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode == 0, resultado.stderr
    return resultado.stdout.strip()


def _dispositivo_cash_paridad_esperado(entorno):
    """El dispositivo que resolvería `cash_paridad._resolver_dispositivo` con ESTE
    entorno: "auto" siempre resuelve a "cuda" (ver `docs/arquitectura.md`). Esta
    función ancla ese hecho para las pruebas que la usan más abajo, sin
    depender del hardware de quien corre la prueba."""
    resultado = subprocess.run(
        [
            sys.executable,
            "-c",
            "from taller_hpo.problemas import cash_paridad as c\n"
            "print(c._resolver_dispositivo(c.DEFAULTS['dispositivo']))\n",
        ],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode == 0, resultado.stderr
    return resultado.stdout.strip()


@pytest.fixture(scope="module")
def corridas_cache(tmp_path_factory):
    """Esta fixture corre cada paso UNA sola vez por módulo de pytest, con éxito o
    con timeout.

    Un `subprocess.TimeoutExpired` también se guarda en la caché (no solo el
    resultado): sin esto, cada prueba que pide este mismo paso vuelve a
    lanzar el paso completo y paga otro timeout entero, en vez de fallar una
    sola vez con un mensaje claro -- con el timeout de 110 s de este archivo,
    dos pruebas de contenido sobre el mismo paso ya duplican el costo.
    """
    cache: dict[
        str, tuple[subprocess.CompletedProcess, Path, dict] | subprocess.TimeoutExpired
    ] = {}

    def obtener(paso):
        clave = paso.stem
        if clave not in cache:
            destino = tmp_path_factory.mktemp(clave)
            entorno = _entorno(destino)
            # Medir el uso real de la tarjeta de video es la MISMA corrida (ver
            # _correr): sin costo extra, para todo paso que entra a
            # PASOS_CON_GPU_REAL -- el taller entrena siempre en la tarjeta de
            # video, así que no hay que discriminar por hardware (las pruebas
            # que llaman a `obtener` con uno de estos pasos llevan la marca
            # `gpu`, que ya falla antes de llegar aquí sin una tarjeta
            # visible). La mayoría de PASOS_CON_GPU_REAL entrena vía
            # taller_hpo.problemas.juguete (00/02/03/04/06) o cash_paridad
            # (05); 07 entra por otra vía -- su uso real de la tarjeta de
            # video viene de correr plantilla/train_ejemplo.py, en proceso y
            # por subproceso, en sus secciones 2 y 3 (la sección 1 solo
            # revisa el contrato con check_importable), no de juguete.
            medir_uso_gpu = paso in PASOS_CON_GPU_REAL
            try:
                resultado = _correr(paso, entorno, medir_uso_gpu=medir_uso_gpu)
            except subprocess.TimeoutExpired as error:
                cache[clave] = error
            else:
                cache[clave] = (resultado, destino, entorno)
        entrada = cache[clave]
        if isinstance(entrada, subprocess.TimeoutExpired):
            pytest.fail(
                f"{paso.name} excedió {entrada.timeout} s con PRESUPUESTO=humo",
                pytrace=False,
            )
        return entrada

    return obtener


@pytest.fixture(scope="module")
def descripcion_esperada():
    return _descripcion_esperada(_entorno_base())


def test_todos_los_pasos_estan_clasificados():
    stems = {p.stem for p in PASOS}
    clasificados = set(PREFIJOS_MOTOR) | set(EXENTOS)
    faltantes = stems - clasificados
    assert not faltantes, (
        f"{sorted(faltantes)} no está en PREFIJOS_MOTOR ni en EXENTOS de "
        "tests/test_pasos.py: clasifícalo antes de que la suite lo revise a medias"
    )


def test_pasos_con_juguete_no_entrenan_via_mlp_paridad_directo():
    """Estático (ast, sin ejecutar nada), sobre TODOS los pasos (no solo
    PASOS_CON_JUGUETE): un paso que importe `mlp_paridad`/`paridad_torch`
    directo podría imprimir la DESCRIPCION del motor resuelto por `juguete` y
    entrenar con el modelo de referencia de scikit-learn sin que ninguna
    prueba de contenido lo note (la salida sigue diciendo lo que
    `juguete.DESCRIPCION` dice, no con qué entrenó en realidad). Esto incluye
    pasos EXENTOS de imprimir esa DESCRIPCION (01, 05): la convención de no
    importar esos módulos directo aplica a todos, con o sin `juguete`. Sin
    ninguna excepción: ni siquiera 07, que solo importa `juguete` para
    mostrar la DESCRIPCION del juguete del repo como referencia (ver su
    cabecera) y entrena con plantilla/train_ejemplo.py, nunca con
    `mlp_paridad` ni `paridad_torch` directo.
    """
    prohibidos = {"mlp_paridad", "paridad_torch"}
    for paso in PASOS:
        arbol = ast.parse(paso.read_text(encoding="utf-8"), filename=str(paso))
        importados: set[str] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.ImportFrom) and nodo.module:
                importados.update(nodo.module.split("."))
                importados.update(alias.name for alias in nodo.names)
            elif isinstance(nodo, ast.Import):
                for alias in nodo.names:
                    importados.update(alias.name.split("."))
        encontrados = importados & prohibidos
        assert not encontrados, (
            f"{paso.name} importa {sorted(encontrados)} directo: ningún paso "
            "entrena con el modelo de referencia ni con el motor por dentro; "
            "usa taller_hpo.problemas.juguete o plantilla/train_ejemplo.py"
        )


def _parametros_por_paso(pasos):
    """`pytest.param` por paso, con la marca `gpu` para los que entran a
    PASOS_CON_GPU_REAL (entrenan el juguete o cash_paridad en la tarjeta de
    video): sin una tarjeta visible, esos items FALLAN (tests/conftest.py),
    nunca se saltan. El único paso sin la marca es 01 (el paisaje analítico,
    sin nada que entrenar -- ver EXENTOS)."""
    return [
        pytest.param(
            paso, marks=pytest.mark.gpu if paso in PASOS_CON_GPU_REAL else (), id=paso.stem
        )
        for paso in pasos
    ]


@pytest.mark.parametrize("paso", _parametros_por_paso(PASOS))
def test_paso_corre_con_presupuesto_humo(paso, corridas_cache):
    resultado, destino, _ = corridas_cache(paso)
    assert resultado.returncode == 0, resultado.stderr
    assert list(destino.glob("*.png")), "el paso no guardó su figura"
    # -W error solo aplica al proceso de pytest, no a este subproceso: un paso
    # podría emitir un aviso real (p. ej. una deprecación de una librería) sin
    # que -W error de pytest lo capture, porque vive en otro intérprete. No se
    # corre el subproceso con -W error: algunos pasos capturan a propósito un
    # ExperimentalWarning con warnings.catch_warnings(record=True) (ver
    # pasos/05_buscar_el_modelo.py) y lo convierten en una nota de una línea en
    # stdout -- bajo -W error eso se volvería una excepción en cada corrida en
    # vez de la nota que el propio paso ya imprime. En su lugar, se revisa el
    # stderr del subproceso directamente: la palabra "Warning" ahí solo puede
    # venir de un aviso real que nadie convirtió en nota.
    assert "Warning" not in resultado.stderr, resultado.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("paso", PASOS_CON_JUGUETE, ids=[p.stem for p in PASOS_CON_JUGUETE])
def test_paso_imprime_la_descripcion_del_motor_resuelto(paso, corridas_cache, descripcion_esperada):
    resultado, _, _ = corridas_cache(paso)
    linea_esperada = f"{PREFIJOS_MOTOR[paso.stem]}{descripcion_esperada}"
    assert linea_esperada in resultado.stdout, resultado.stdout


@pytest.mark.gpu
def test_motor_siempre_resuelve_a_cuda(descripcion_esperada):
    """El taller entrena siempre en la tarjeta de video (ver docs/arquitectura.md):
    la DESCRIPCION que resuelve `taller_hpo.problemas.juguete` dice "cuda" siempre.
    Esta prueba falla si la DESCRIPCION alguna vez deja de decir "cuda" -- con
    la marca `gpu`, falla en vez de pasar en silencio en una máquina sin
    tarjeta de video.
    """
    assert " en cuda " in descripcion_esperada, descripcion_esperada


@pytest.mark.gpu
@pytest.mark.parametrize("paso", PASOS_CON_GPU_REAL, ids=[p.stem for p in PASOS_CON_GPU_REAL])
def test_paso_usa_la_gpu_de_verdad(paso, corridas_cache):
    """`test_paso_imprime_la_descripcion_del_motor_resuelto` (00/02/03/04/06/07)
    y `test_paso_05_imprime_su_dispositivo_de_cash_paridad` (05) solo comparan
    TEXTO contra lo que el propio módulo dice que resolvió: un paso podría
    imprimir 'cuda' sin entrenar de verdad en la tarjeta de video, y ninguna
    de las dos lo notaría.

    Esta prueba ancla el USO real: `corridas_cache` corrió este paso con un
    lanzador que, ya terminada la corrida COMPLETA, imprime cuánta memoria
    CUDA se llegó a asignar (`torch.cuda.max_memory_allocated()`). Entrenar
    de verdad en la tarjeta de video la deja en > 0 bytes.

    Límite: esto detecta que NADA corrió en la tarjeta de video en toda la
    corrida, no que cada sección la haya usado -- en el paso 05, un algoritmo
    del CASH que fallara y cayera a otra ruta mientras knn/krr (que siempre
    entrenan con torch) siguen en la tarjeta de video dejaría
    `max_memory_allocated() > 0` igual, aportado por la parte que sí la usó.
    """
    resultado, _, _ = corridas_cache(paso)
    coincidencia = re.search(r"__cuda_max_mem__ (\d+)", resultado.stdout)
    assert coincidencia, resultado.stdout
    assert int(coincidencia.group(1)) > 0, resultado.stdout


@pytest.mark.gpu
def test_paso_05_imprime_su_dispositivo_de_cash_paridad(corridas_cache):
    resultado, _, entorno = corridas_cache(PASO_05)
    esperado = _dispositivo_cash_paridad_esperado(entorno)
    assert f"cash_paridad: dispositivo={esperado}" in resultado.stdout, resultado.stdout


def _parsear_corridas(stdout):
    return [(int(i), float(v), float(t)) for i, v, t in _PATRON_CORRIDA.findall(stdout)]


def _nodo_veredictos_franjas():
    """`ast.FunctionDef` de `veredictos_franjas`, leído de `pasos/00_linea_base.py` con
    `ast` -- mismo patrón que `test_pasos_con_juguete_no_entrenan_via_mlp_paridad_directo`
    (leer con `ast` en vez de importar: los pasos corren código de nivel de módulo, que
    entrenaría el juguete de verdad, al importarlos)."""
    arbol = ast.parse(PASO_00.read_text(encoding="utf-8"), filename=str(PASO_00))
    return next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "veredictos_franjas"
    )


def _fuente_veredictos_franjas():
    """Código fuente de `veredictos_franjas`, tal cual vive en el paso -- para insertarla
    dentro de `_PLANTILLA_RESPUESTAS_PASO_00` (reemplaza `__VEREDICTOS_FRANJAS__`) sin
    mantener una segunda copia manual que se pueda desincronizar del paso real."""
    return ast.get_source_segment(PASO_00.read_text(encoding="utf-8"), _nodo_veredictos_franjas())


@pytest.fixture(scope="module")
def veredictos_franjas():
    """La función `veredictos_franjas` real de `pasos/00_linea_base.py`, ejecutada en un
    espacio de nombres propio (sin correr el resto del paso, que entrena el juguete de
    verdad): un error en su lógica rompe esta prueba y la del paso completo por
    igual, porque las dos corren la MISMA fuente."""
    modulo = ast.Module(body=[_nodo_veredictos_franjas()], type_ignores=[])
    ast.fix_missing_locations(modulo)
    espacio_nombres = {"scipy_stats": scipy_stats}
    exec(compile(modulo, str(PASO_00), "exec"), espacio_nombres)
    return espacio_nombres["veredictos_franjas"]


def _nodo_desvio_validacion():
    """`ast.FunctionDef` de `desvio_validacion`, leído de `pasos/00_linea_base.py` con
    `ast` -- mismo patrón que `_nodo_veredictos_franjas` arriba."""
    arbol = ast.parse(PASO_00.read_text(encoding="utf-8"), filename=str(PASO_00))
    return next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "desvio_validacion"
    )


@pytest.fixture(scope="module")
def desvio_validacion():
    """La función `desvio_validacion` real de `pasos/00_linea_base.py`, ejecutada en un
    espacio de nombres propio -- mismo patrón que el fixture `veredictos_franjas` arriba:
    un error en su lógica rompe esta prueba y la del paso completo por igual, porque
    las dos corren la MISMA fuente."""
    modulo = ast.Module(body=[_nodo_desvio_validacion()], type_ignores=[])
    ast.fix_missing_locations(modulo)
    espacio_nombres = {"st": st}
    exec(compile(modulo, str(PASO_00), "exec"), espacio_nombres)
    return espacio_nombres["desvio_validacion"]


def test_desvio_validacion_no_cancela_signos_opuestos(desvio_validacion):
    """Con `val` cayendo de los dos lados de `verdad` (0.6 por debajo de 0.7, 0.8 por
    encima), un error que cambiara `abs(v - t)` por la resta con signo dejaría la
    media en 0.0 (los signos se cancelan) en vez de 0.1: la validación de humo del paso
    completo no distingue esto porque con sus 3 semillas la validación siempre cae de un
    solo lado (ver el comentario de `_PLANTILLA_RESPUESTAS_PASO_00` más arriba)."""
    media, maximo = desvio_validacion([0.6, 0.8], [0.7, 0.7])
    assert media == pytest.approx(0.1), media
    assert maximo == pytest.approx(0.1), maximo


def test_paso_00_llama_veredictos_franjas_con_los_nombres_correctos():
    """`veredictos_franjas` es keyword-only (ver su firma en pasos/00_linea_base.py):
    con dos listas del mismo tipo, invertir `total`/`fijos` en el sitio de llamada no
    lo marca ningún tipo ni ninguna prueba unitaria de la función misma (esas ejercitan
    la función, no quién la llama). Esta prueba ancla el ÚNICO sitio de llamada real
    contra los nombres de variable exactos que arman las dos franjas."""
    arbol = ast.parse(PASO_00.read_text(encoding="utf-8"), filename=str(PASO_00))
    llamada = next(
        nodo
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "veredictos_franjas"
    )
    argumentos = {
        keyword.arg: keyword.value.id
        for keyword in llamada.keywords
        if isinstance(keyword.value, ast.Name)
    }
    assert argumentos == {"total": "verdad", "fijos": "verdad_datos_fijos"}, argumentos


def _respuestas_esperadas_paso_00(entorno):
    codigo = _PLANTILLA_RESPUESTAS_PASO_00.replace("__N__", str(N_HUMO_PASO_00)).replace(
        "__VEREDICTOS_FRANJAS__", _fuente_veredictos_franjas()
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode == 0, resultado.stderr
    return resultado.stdout.strip().splitlines()


@pytest.mark.gpu
def test_paso_00_contesta_las_tres_preguntas_con_los_numeros_que_mide(corridas_cache):
    resultado, _, entorno = corridas_cache(PASO_00)
    # Sanidad barata antes de repetir el entrenamiento: el paso imprimió
    # exactamente N_HUMO_PASO_00 corridas, ni de más ni de menos.
    assert len(_parsear_corridas(resultado.stdout)) == N_HUMO_PASO_00, resultado.stdout

    for esperada in _respuestas_esperadas_paso_00(entorno):
        assert esperada in resultado.stdout, resultado.stdout


@pytest.fixture(scope="module")
def sinteticos_veredictos_franjas():
    """Arreglos sintéticos fijos (semilla 12345, sin entrenar el juguete) para ejercitar
    las ramas de `veredictos_franjas`: `total` es una franja cualquiera alrededor de 0.7
    (la escala real de una exactitud) y cada `fijos` se construye a partir de ella para
    caer, a propósito, en una rama distinta -- las simulaciones usan
    `numpy.random.default_rng` con semilla fija, mismos números en cada corrida (mismo
    patrón que tests/test_estadistica.py)."""
    rng = np.random.default_rng(12345)
    n = 10
    total = rng.normal(loc=0.7, scale=0.03, size=n)
    media = total.mean()
    return {
        "total": total.tolist(),
        # (a) casi la misma serie, con ruido chico: deben moverse juntas.
        "fijos_juntos": (total + rng.normal(scale=0.005, size=n)).tolist(),
        # (b) independiente de total: la correlación no debe distinguirse de 0.
        "fijos_independiente": rng.normal(loc=0.7, scale=0.03, size=n).tolist(),
        # (c) el triple de la desviación de total respecto a su media: fijos queda
        # más ancha (Var(fijos) ~= 9*Var(total)).
        "fijos_mas_ancha": (
            media + 3.0 * (total - media) + rng.normal(scale=0.002, size=n)
        ).tolist(),
        # (d) un tercio de la desviación: fijos queda más angosta.
        "fijos_mas_angosta": (
            media + (1.0 / 3.0) * (total - media) + rng.normal(scale=0.002, size=n)
        ).tolist(),
    }


def test_veredictos_franjas_correlacion_alta_dice_se_mueven_juntas(
    veredictos_franjas, sinteticos_veredictos_franjas
):
    s = sinteticos_veredictos_franjas
    _, veredicto_junto = veredictos_franjas(total=s["total"], fijos=s["fijos_juntos"])
    assert "se mueven juntas" in veredicto_junto, veredicto_junto


def test_veredictos_franjas_correlacion_baja_es_no_concluyente(
    veredictos_franjas, sinteticos_veredictos_franjas
):
    s = sinteticos_veredictos_franjas
    veredicto_ancho, veredicto_junto = veredictos_franjas(
        total=s["total"], fijos=s["fijos_independiente"]
    )
    assert "no concluyente" in veredicto_junto, veredicto_junto
    # También la franja ancha: sin esto, subir el umbral de Pitman-Morgan
    # (p. ej. `pm.pvalue >= 0.05` -> `pm.pvalue >= 1.1`) no lo detecta ninguna
    # prueba, porque esta es la única que pasa una franja independiente (sin
    # relación real entre las dos) a `veredictos_franjas`.
    assert veredicto_ancho.startswith("no concluyente: con 9 pares los datos no distinguen"), (
        veredicto_ancho
    )


def test_veredictos_franjas_correlacion_moderada_es_no_concluyente(veredictos_franjas):
    """Con una correlación positiva pero un intervalo que sí cruza 0, decidir por el
    estimador puntual (`correlacion.statistic > 0.0`) en vez de por el intervalo completo
    (`ic_corr.low > 0.0`) diría 'se mueven juntas' con evidencia que no lo sostiene --
    justo el antipatrón de leer un número puntual sin su margen que este paso evita en
    otras partes. `total`/`fijos` se construyen con una dependencia parcial a propósito
    (mitad de la desviación de `total` respecto a su media, más ruido independiente),
    para caer en esa zona intermedia sin necesidad de barrer semillas."""
    g = np.random.default_rng(1)
    total = g.normal(0.7, 0.03, 10)
    fijos = 0.5 * (total - total.mean()) + g.normal(0.7, 0.03, 10)
    _, veredicto_junto = veredictos_franjas(total=total.tolist(), fijos=fijos.tolist())
    assert "no concluyente" in veredicto_junto, veredicto_junto
    assert "r=0.582" in veredicto_junto, veredicto_junto


def test_veredictos_franjas_diferencias_constantes_no_llama_pitman_morgan(veredictos_franjas):
    """Dos franjas idénticas dato a dato dejan `diferencias_pm` constante (cero): sin la
    guarda explícita, `scipy.stats.pearsonr` recibe una entrada constante y devuelve NaN
    (con `ConstantInputWarning`, que -W error volvería un error real de suite) en vez de
    decidir nada -- y `nan >= 0.05` es `False`, así que el paso caería en la rama 'más
    angosta' con un p-valor sin sentido."""
    total = [0.70, 0.71, 0.69, 0.72, 0.68]
    veredicto_ancho, _ = veredictos_franjas(total=total, fijos=list(total))
    assert veredicto_ancho == (
        "no concluyente: las diferencias entre franjas son constantes, Pitman-Morgan no aplica"
    ), veredicto_ancho


def test_veredictos_franjas_fijos_mas_ancha(veredictos_franjas, sinteticos_veredictos_franjas):
    s = sinteticos_veredictos_franjas
    veredicto_ancho, _ = veredictos_franjas(total=s["total"], fijos=s["fijos_mas_ancha"])
    assert "más ancha" in veredicto_ancho, veredicto_ancho


def test_veredictos_franjas_fijos_mas_angosta(veredictos_franjas, sinteticos_veredictos_franjas):
    s = sinteticos_veredictos_franjas
    veredicto_ancho, _ = veredictos_franjas(total=s["total"], fijos=s["fijos_mas_angosta"])
    assert "más angosta" in veredicto_ancho, veredicto_ancho


def test_veredictos_franjas_pocos_pares_es_no_concluyente(
    veredictos_franjas, sinteticos_veredictos_franjas
):
    s = sinteticos_veredictos_franjas
    veredicto_ancho, veredicto_junto = veredictos_franjas(
        total=s["total"][:3], fijos=s["fijos_juntos"][:3]
    )
    assert veredicto_ancho.startswith("no concluyente: con 2 pares"), veredicto_ancho
    assert veredicto_junto.startswith("no concluyente: con 2 pares"), veredicto_junto


def test_veredictos_franjas_correlacion_negativa_dice_sentido_contrario(veredictos_franjas):
    """Con una correlación negativa cuyo intervalo no cruza el cero, la rama `elif ic_corr.high <
    0.0` debe decidir que las franjas se mueven en sentido contrario. Sin esta prueba,
    vaciar la rama (`elif False`) o invertirle el mensaje ("se mueven juntas") sobrevive:
    ninguna otra prueba de este archivo pasa una franja con correlación negativa
    significativa. Valores medidos sobre la función real: r=-0.987, intervalo
    del 95% [-0.997, -0.936]."""
    g = np.random.default_rng(7)
    x = g.normal(0, 0.03, 10)
    total = (0.7 + x).tolist()
    fijos = (0.7 - x + g.normal(0, 0.004, 10)).tolist()
    _, veredicto_junto = veredictos_franjas(total=total, fijos=fijos)
    assert veredicto_junto.startswith("se mueven en sentido contrario (r=-0.987"), veredicto_junto


def test_veredictos_franjas_pitman_morgan_cerca_del_umbral_es_no_concluyente(veredictos_franjas):
    """p=0.068, apenas sobre el umbral 0.05: el único caso no significativo que ya prueba
    `test_veredictos_franjas_correlacion_baja_es_no_concluyente` tiene p=0.568, lejos del
    umbral, así que relajarlo (p. ej. `pm.pvalue >= 0.05` -> `pm.pvalue >= 0.30`) sobrevive
    sin este caso borde. Valor medido sobre la función real: p=0.068."""
    g = np.random.default_rng(24)
    total = g.normal(0.7, 0.03, 10)
    fijos = 0.7 + 1.6 * (total - total.mean()) + g.normal(0, 0.02, 10)
    veredicto_ancho, _ = veredictos_franjas(total=total.tolist(), fijos=fijos.tolist())
    assert veredicto_ancho.startswith("no concluyente: con 9 pares los datos no distinguen"), (
        veredicto_ancho
    )
    assert "p=0.068" in veredicto_ancho, veredicto_ancho


def test_veredictos_franjas_correlacion_cerca_del_borde_es_no_concluyente(veredictos_franjas):
    """Con `confidence_level=0.95` el intervalo de la correlación cruza cero (no
    concluyente, r=0.618, intervalo [-0.079, 0.909]) pero con 0.90 no cruza (intervalo
    [0.050, 0.884], "se mueven juntas"): el único caso moderado que ya prueba
    `test_veredictos_franjas_correlacion_moderada_es_no_concluyente` (r=0.582) queda lejos
    de ese borde en las dos escalas, así que bajar el nivel de confianza de la correlación
    a 0.90 sobrevive sin este caso."""
    g = np.random.default_rng(19)
    total = g.normal(0.7, 0.03, 10)
    fijos = 0.7 * (total - total.mean()) + g.normal(0.7, 0.03, 10)
    _, veredicto_junto = veredictos_franjas(total=total.tolist(), fijos=fijos.tolist())
    assert "no concluyente (r=0.618" in veredicto_junto, veredicto_junto


@pytest.mark.gpu
def test_presupuesto_invalido_falla_con_mensaje(tmp_path):
    """PASOS[0] (00_linea_base.py) importa taller_hpo.problemas.juguete antes de
    validar PRESUPUESTO: sin una tarjeta de video, esa importación ya falla con
    su propio RuntimeError, antes de llegar al SystemExit que esta prueba
    busca -- de ahí la marca `gpu`, aunque la pregunta de esta prueba no sea
    sobre la tarjeta de video."""
    entorno = _entorno(tmp_path, presupuesto="enorme")
    resultado = _correr(PASOS[0], entorno, timeout=60)
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr
