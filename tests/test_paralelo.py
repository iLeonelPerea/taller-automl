"""Anexo paralelo: N workers de Optuna sobre el mismo estudio.

`test_lanzar_local_*` corre `pasos/anexo_paralelo.py --lanzar N` de punta a punta y
verifica además el contenido del estudio resultante. `test_docker_*` hace lo mismo contra
PostgreSQL vía `scripts/paralelo_docker.sh` y se salta solo si Docker no está disponible.

Cada prueba que corre un worker lleva `@pytest.mark.gpu`. No la llevan las de funciones
puras (`_frase_contencion`, `_contencion`, `_huella_mib`,
`_params_por_worker_tras_arranque`), las que fallan antes de tocar el juguete (argumentos
inválidos, el montaje 9p, el docker falso, el texto del `Dockerfile`) ni las que verifican
el mensaje sin tarjeta de video. Como en `tests/test_juguete.py`, el try/except de abajo
evita que la colección aborte con `pytest -m "not gpu"`.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from taller_hpo.estadistica import Pareado, diferencia_pareada
from taller_hpo.estudio import abrir_storage
from taller_hpo.semillas import semillas_de
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

RAIZ = Path(__file__).resolve().parent.parent
SCRIPT = RAIZ / "pasos" / "anexo_paralelo.py"
ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
BASE_SEMILLA_SAMPLER = 1_000_000  # ver pasos/anexo_paralelo.py
SEMILLA_BUSQUEDA = 20_260_925  # ver pasos/anexo_paralelo.py
# _SAMPLER_KWARGS de pasos/anexo_paralelo.py: misma fuente que el protocolo
# persistido (ya afirmado más abajo) Y que el TPESampler real -- se repite
# aquí para poder afirmar el segundo también (SEMILLAS_SAMPLER_JSON, que trae
# los kwargs que la fábrica de verdad le pasó a TPESampler(**kwargs)).
SAMPLER_KWARGS_ESPERADOS = {
    "multivariate": True,
    "constant_liar": True,
    "n_startup_trials": 10,
    "n_ei_candidates": 24,
    "group": False,
}
# _ESPACIO_RANGOS de pasos/anexo_paralelo.py: la misma fuente que arma tanto
# el espacio real (_espacio) como el protocolo persistido (_protocolo) -- se
# repite aquí para poder afirmar protocolo['espacio'] contra el valor real,
# no contra una suposición de qué rangos debería tener.
ESPACIO_RANGOS_ESPERADO = {
    "learning_rate_init": [1e-4, 3e-1, "log"],
    "alpha": [1e-6, 1e-1, "log"],
    "ancho": [16, 128, "step16"],
}


def _importar_anexo_paralelo():
    """Importa `pasos/anexo_paralelo.py` como módulo, sin correr ningún modo
    (guardia `if __name__ == "__main__":` del propio archivo). Patrón de
    importación por ruta copiado literalmente de `tests/test_paso_07.py`:

        def _importar(ruta: Path, nombre: str):
            spec = importlib.util.spec_from_file_location(nombre, ruta)
            modulo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modulo)
            return modulo

    Usado para probar las funciones puras del módulo (`_frase_contencion`,
    `_params_por_worker_tras_arranque`) con datos sintéticos, sin depender de
    que una corrida real produzca por azar el caso que se quiere probar."""
    spec = importlib.util.spec_from_file_location("_anexo_paralelo_bajo_prueba", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _pareado(*, veredicto: str, diferencia_media: float = 0.05) -> Pareado:
    """`Pareado` sintético con el `veredicto` pedido: los demás campos son
    valores cualesquiera (no hacen falta para `_frase_contencion`, que solo
    lee `veredicto` y `diferencia_media`/`ic_bajo`/`ic_alto`)."""
    return Pareado(
        n=5,
        diferencia_media=diferencia_media,
        de_dif=0.02,
        ic_bajo=diferencia_media - 0.01,
        ic_alto=diferencia_media + 0.01,
        p_valor=0.5,
        wilcoxon_p=None,
        rho=0.1,
        veredicto=veredicto,
        delta=0.05,
        direccion="minimizar",
        confianza=0.95,
    )


def test_frase_contencion_depende_exactamente_del_veredicto():
    """`_frase_contencion` (pasos/anexo_paralelo.py) es una función pura: los
    4 veredictos posibles de `estadistica.diferencia_pareada` se prueban aquí
    con un `Pareado` sintético cada uno -- con PRESUPUESTO=humo y pocas
    configuraciones, una corrida real casi nunca produce "mejor"/"peor" por
    azar (ver test_lanzar_comparar_aceleracion_es_reloj_1_sobre_reloj_n, que
    solo puede afirmar la frase del veredicto que de hecho salió). Un error
    que invirtiera qué frase corresponde a qué veredicto reprueba aquí, sin
    depender de qué veredicto haya salido en ninguna corrida."""
    anexo_paralelo = _importar_anexo_paralelo()

    frase_mejor = anexo_paralelo._frase_contencion(_pareado(veredicto="mejor"))
    assert "más caro bajo contención" in frase_mejor, frase_mejor
    assert "en promedio" in frase_mejor, frase_mejor
    assert "más rápido" not in frase_mejor, frase_mejor

    frase_peor = anexo_paralelo._frase_contencion(_pareado(veredicto="peor"))
    assert "más rápido que uno aislado" in frase_peor, frase_peor
    assert "más caro bajo contención" not in frase_peor, frase_peor

    frase_equivalente = anexo_paralelo._frase_contencion(_pareado(veredicto="equivalente"))
    assert "equivalente a un trial aislado" in frase_equivalente, frase_equivalente

    frase_no_concluyente = anexo_paralelo._frase_contencion(_pareado(veredicto="no concluyente"))
    assert "no concluyente con estas 5 configuraciones" in frase_no_concluyente, (
        frase_no_concluyente
    )
    # La unidad pareada es la CONFIGURACIÓN (los pares aislado/contención
    # comparten hiperparámetros), nunca la semilla: las dos corridas usan la
    # MISMA Semillas de búsqueda (CRN).
    assert "semillas" not in frase_no_concluyente, frase_no_concluyente

    # Las 4 frases son distintas entre sí: ninguna combinación de veredictos
    # comparte texto (si lo hiciera, invertir dos de ellas no se notaría).
    frases = {frase_mejor, frase_peor, frase_equivalente, frase_no_concluyente}
    assert len(frases) == 4, frases


def test_contencion_pura_detecta_las_dos_direcciones_desde_pares():
    """`_contencion` (pasos/anexo_paralelo.py) arma `segundos_solo`/
    `segundos_contencion` directo de una lista de pares sintéticos y llama a
    `estadistica.diferencia_pareada` -- se prueba aquí con pares hechos a
    mano en las dos direcciones, sin depender de que una corrida real con
    PRESUPUESTO=humo produzca "mejor"/"peor" por azar (con pocas
    configuraciones casi siempre sale "equivalente"/"no concluyente").
    Intercambiar `a`/`b` al llamar a `diferencia_pareada`, o la `direccion`
    ("minimizar" -> "maximizar"), invierte el veredicto en los dos casos de
    abajo -- los dos errores reprueban aquí."""
    anexo_paralelo = _importar_anexo_paralelo()

    # Contención claramente más CARA que un trial aislado.
    pares_caro = [
        {"trial_number": i, "segundos_solo": 1.0, "segundos_contencion": 1.5} for i in range(5)
    ]
    pareado_caro = anexo_paralelo._contencion(pares_caro)
    assert pareado_caro.veredicto == "mejor", pareado_caro
    assert pareado_caro.diferencia_media == pytest.approx(0.5), pareado_caro

    # Contención claramente más BARATA que un trial aislado (dirección opuesta).
    pares_barato = [
        {"trial_number": i, "segundos_solo": 1.5, "segundos_contencion": 1.0} for i in range(5)
    ]
    pareado_barato = anexo_paralelo._contencion(pares_barato)
    assert pareado_barato.veredicto == "peor", pareado_barato
    assert pareado_barato.diferencia_media == pytest.approx(-0.5), pareado_barato

    # Sin diferencia: ni "mejor" ni "peor" -- el intervalo colapsa dentro de la zona
    # de indiferencia (delta = 10% de la mediana aislada > 0).
    pares_iguales = [
        {"trial_number": i, "segundos_solo": 1.0, "segundos_contencion": 1.0} for i in range(5)
    ]
    pareado_igual = anexo_paralelo._contencion(pares_iguales)
    assert pareado_igual.veredicto == "equivalente", pareado_igual
    assert pareado_igual.diferencia_media == pytest.approx(0.0), pareado_igual


def test_huella_mib_resta_el_pico_menos_la_base_y_divide_entre_n():
    """`_huella_mib` (pasos/anexo_paralelo.py) es la aritmética pura detrás de
    la respuesta 6: pico-menos-base, dividido entre `n` workers. La prueba de
    punta a punta (`test_lanzar_local_memoria_gpu`, más abajo) mide sobre una
    tarjeta de video real, así que una lectura corta por ruido ambiental
    podría tapar que la resta se invirtiera; aquí, con valores sintéticos, la
    resta invertida (base menos pico, como el error real que devolvería un
    número negativo) reprueba siempre, sin depender de ninguna lectura de
    nvidia-smi."""
    anexo_paralelo = _importar_anexo_paralelo()

    huella_total, huella_por_worker = anexo_paralelo._huella_mib(
        pico_memoria_mib=2500.0, memoria_base_mib=500.0, n=4
    )
    assert huella_total == pytest.approx(2000.0), huella_total
    assert huella_por_worker == pytest.approx(500.0), huella_por_worker
    # El pico de una corrida real siempre queda por encima de la línea base:
    # una resta invertida daría un total negativo, nunca este resultado.
    assert huella_total > 0, huella_total

    # n=1 (la corrida --comparar de un solo worker aislado): la huella por
    # worker coincide con la huella total.
    huella_total_1, huella_por_worker_1 = anexo_paralelo._huella_mib(
        pico_memoria_mib=900.0, memoria_base_mib=100.0, n=1
    )
    assert huella_total_1 == pytest.approx(800.0), huella_total_1
    assert huella_por_worker_1 == pytest.approx(huella_total_1), (
        huella_por_worker_1,
        huella_total_1,
    )


def _trial_falso(number: int, worker: object, params: dict) -> SimpleNamespace:
    """Sustituto de `optuna.trial.FrozenTrial` con solo los 3 atributos que
    `_params_por_worker_tras_arranque` lee (`number`, `user_attrs`, `params`):
    más barato que crear un estudio real de Optuna para cada combinación."""
    return SimpleNamespace(number=number, user_attrs={"worker": worker}, params=params)


def test_params_por_worker_tras_arranque_agrupa_por_worker_y_filtra_arranque():
    """Datos sintéticos (number, worker, params) fabricados a mano, no un
    estudio real: la recomputación de `coinciden_params`/`comparados` en
    `_lanzar` (pasos/anexo_paralelo.py) parte de `params_originales`/
    `params_repetidos`, así que un error de agrupación o de filtrado en
    `_params_por_worker_tras_arranque` no dejaría rastro si esta función se
    probara solo indirectamente, a través de una corrida completa. Aquí se
    prueba sola, contra el resultado esperado escrito a mano."""
    anexo_paralelo = _importar_anexo_paralelo()
    n_arranque = 2

    trials = [
        _trial_falso(0, worker=0, params={"x": "arranque_w0"}),  # <= n_arranque: fuera
        _trial_falso(1, worker=1, params={"x": "arranque_w1"}),  # <= n_arranque: fuera
        _trial_falso(3, worker=0, params={"x": "a"}),
        _trial_falso(4, worker=1, params={"x": "b"}),
        _trial_falso(5, worker=0, params={"x": "c"}),
    ]

    agrupado = anexo_paralelo._params_por_worker_tras_arranque(trials, n_arranque)

    # Sin el filtro `t.number > n_arranque`, los 2 trials de arranque (number
    # 0 y 1) entrarían también -- esto falla si el filtro se pierde.
    assert agrupado == {
        0: [{"x": "a"}, {"x": "c"}],
        1: [{"x": "b"}],
    }, agrupado
    # Agrupar todo bajo una sola clave (p. ej. por número global de trial en
    # vez de por worker) dejaría UNA sola clave con los 3 params en vez de 2
    # claves -- esto lo detecta.
    assert set(agrupado) == {0, 1}, agrupado
    assert len(agrupado[0]) == 2 and len(agrupado[1]) == 1, agrupado


def _razon_no_docker() -> str | None:
    """`None` si Docker está listo para usarse; si no, por qué (para un
    `pytest.skip` que distinga "no hay binario" de "el daemon no responde" --
    dentro del contenedor de `docker compose run --rm taller` lo primero es
    lo normal, ahí no hay daemon).

    La prueba que llama a esta función lleva además `@pytest.mark.gpu`: sin
    tarjeta de video, `tests/conftest.py` ya la hace fallar en el setup,
    antes de que este código corra -- por eso ninguna rama de aquí necesita
    (ni debería) saltar la prueba por falta de tarjeta de video."""
    if shutil.which("docker") is None:
        return "no hay binario docker (dentro del contenedor no hay daemon)"
    resultado = subprocess.run(["docker", "info"], capture_output=True, timeout=30)
    if resultado.returncode != 0:
        return "'docker info' falló (el daemon no está corriendo o no es alcanzable)"
    return None


# --- modo local: --lanzar 3 sobre un Journal en tmp, corrido UNA vez --------


@pytest.fixture(scope="module")
def corrida_lanzar_3(tmp_path_factory):
    """Corre `pasos/anexo_paralelo.py --lanzar 3` UNA sola vez con
    PRESUPUESTO=humo (convención del taller para pruebas que corren un paso
    por subproceso: un fixture scope='module' lo corre UNA vez y reparte la
    salida -- mismo patrón que `corrida_humo`/`salida_humo` de
    tests/test_paso_07.py). Las pruebas de conteo, semillas de sampler y
    secuencia de abajo leen todas de esta misma corrida en vez de relanzar el
    subproceso cada una: cada relanzamiento son 3 subprocesos con su propio
    arranque de Python e import de torch/optuna, bastante más caro que el
    entrenamiento en sí con PRESUPUESTO=humo.

    TALLER_N_TOTAL=15 en vez de los 6 de PRESUPUESTOS["humo"]: con 6, un
    worker que arranca tarde a veces encuentra el estudio ya completo por los
    otros dos y no aporta ningún trial (con este juguete un trial es barato
    frente al arranque del proceso, así que 6 se agotan casi de inmediato);
    15 les da a los 3 workers margen de sobra para arrancar antes de que el
    estudio se cierre.
    """
    tmp_path = tmp_path_factory.mktemp("lanzar_3_humo")
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",  # PRESUPUESTOS["humo"] existe pero N_TOTAL se fija abajo
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_compartido",
        "TALLER_N_TOTAL": "15",
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "3"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return resultado, tmp_path


@pytest.mark.gpu
def test_lanzar_local_conteo_por_worker_y_sin_running(corrida_lanzar_3):
    resultado, tmp_path = corrida_lanzar_3
    assert resultado.returncode == 0, resultado.stderr

    # Las respuestas numeradas que pide docs/arquitectura.md para un paso.
    for inicio in (
        "1. Trials totales",
        "2. Trials por worker",
        "3. Verificación",
        "4. Tiempo de reloj",
        "5. Secuencia de workers",
        "6. Memoria de la tarjeta de video",
    ):
        assert inicio in resultado.stdout, resultado.stdout

    # PRESUPUESTO=humo deja --comparar inactivo por defecto: sin esto se
    # correrían dos lanzamientos extra (comparación + repetición de orden),
    # incumpliendo el techo de tiempo del presupuesto humo.
    assert "no es una aceleración" in resultado.stdout, resultado.stdout
    assert "no se repitió (sin --comparar, PRESUPUESTO=humo)" in resultado.stdout, resultado.stdout

    # Convención de paso: el proceso padre de --lanzar imprime el motor leído
    # del log del worker 0, nunca importando juguete él mismo (docstring de
    # pasos/anexo_paralelo.py). Primera línea del stdout del padre (no solo
    # "aparece en algún lado"): si _correr_worker dejara de vaciar su búfer
    # al imprimir "Motor: " (flush=True), o si _spawn_n_workers no la leyera
    # del log a tiempo, la línea seguiría apareciendo pero tarde, después de
    # las respuestas numeradas -- esta afirmación posicional lo nota.
    primera_linea_stdout = resultado.stdout.splitlines()[0]
    assert primera_linea_stdout == f"Motor: {juguete.DESCRIPCION}", resultado.stdout

    figura = tmp_path / "resultados" / "anexo_paralelo.png"
    assert figura.exists(), "--lanzar no guardó su figura"

    # Contenido real del estudio, no solo lo que el script dice de sí mismo.
    destino = tmp_path / "estudios" / "anexo_paralelo.log"
    storage = abrir_storage(destino)
    study = optuna.load_study(study_name="paralelo_compartido", storage=storage)
    terminados = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    corriendo = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))

    n_total = 15  # TALLER_N_TOTAL fijado en corrida_lanzar_3 (ver su docstring)
    n_workers = 3
    assert n_total <= len(terminados) <= n_total + n_workers - 1
    assert corriendo == []

    por_worker: dict[int, int] = {}
    for t in terminados:
        w = t.user_attrs["worker"]
        assert isinstance(w, int)
        por_worker[w] = por_worker.get(w, 0) + 1
    # Los 3 workers (0, 1, 2: --lanzar los numera explícito) aportaron al menos un trial.
    assert set(por_worker) == {0, 1, 2}
    assert all(n >= 1 for n in por_worker.values())
    assert sum(por_worker.values()) == len(terminados)

    # Protocolo idéntico para los 3: crear_estudio lo hubiera rechazado si no.
    protocolo = study.user_attrs["protocolo"]
    assert protocolo["sampler"] == "tpe"
    # La semilla del sampler NO entra al protocolo (docs/arquitectura.md, Anexo).
    assert "seed" not in protocolo
    # juguete.PROTOCOLO_MOTOR (motor y dispositivo) SÍ entra: el worker entrena
    # con taller_hpo.problemas.juguete, nunca con mlp_paridad/paridad_torch
    # directo (docs/arquitectura.md).
    assert protocolo["motor"] == juguete.PROTOCOLO_MOTOR["motor"]
    assert protocolo["dispositivo"] == juguete.PROTOCOLO_MOTOR["dispositivo"]
    # _SAMPLER_KWARGS es la ÚNICA fuente tanto del protocolo persistido como
    # del TPESampler real (pasos/anexo_paralelo.py), así que los dos no
    # deberían desviarse entre sí -- pero esto NO verifica al sampler real: una
    # fábrica podría capturar y reportar estos valores sin de verdad
    # pasárselos a TPESampler(**kwargs). Esa verificación (SAMPLER_REAL_JSON,
    # leída de study.sampler ya construido) la hace
    # test_workers_usan_semillas_de_sampler_distintas, más abajo.
    assert protocolo["constant_liar"] is True
    assert protocolo["multivariate"] is True
    assert protocolo["n_startup_trials"] == 10
    assert protocolo["n_ei_candidates"] == 24
    assert protocolo["group"] is False

    # Convención de paso: cada worker imprime juguete.DESCRIPCION al arrancar
    # (así se ve, sin adivinar, con qué versión de PyTorch y con qué tarjeta
    # de video corrió). --lanzar manda el stdout de cada worker a su propio
    # archivo de log, no al stdout del proceso padre (arriba ya verificado) --
    # se lee de ahí.
    log_worker_0 = (tmp_path / "estudios" / "anexo_paralelo_worker_0.log.txt").read_text(
        encoding="utf-8"
    )
    assert f"Motor: {juguete.DESCRIPCION}" in log_worker_0

    # Protocolo completo: politica_semillas trae los datos CRN reales de la
    # búsqueda (docstring de _protocolo en pasos/anexo_paralelo.py), espacio
    # es la misma fuente que _espacio usó para muestrear, y semilla_sampler
    # documenta la fórmula real de _fabrica_sampler -- las tres claves deben
    # coincidir con lo que el script de verdad usó, no solo con "sampler" y
    # los _SAMPLER_KWARGS ya afirmados arriba.
    assert protocolo["politica_semillas"] == {
        "tipo": "CRN",
        "busqueda": dataclasses.asdict(semillas_de(SEMILLA_BUSQUEDA, 1)[0]),
    }, protocolo
    assert protocolo["espacio"] == ESPACIO_RANGOS_ESPERADO, protocolo
    assert protocolo["semilla_sampler"] == {
        "base": BASE_SEMILLA_SAMPLER,
        "formula": "base + 1000*worker + n_previos",
    }, protocolo


@pytest.mark.gpu
def test_lanzar_local_memoria_gpu(corrida_lanzar_3):
    """Memoria de la tarjeta de video de la respuesta 6, separada de
    `test_lanzar_local_conteo_por_worker_y_sin_running` para que un fallo aquí no apague
    las afirmaciones del protocolo.

    La huella total (pico menos base de `memory.used`, medida por el proceso padre) es una
    lectura de toda la tarjeta y otros procesos pueden moverla, pero debe quedar por encima
    de lo que los 3 workers reservaron con el asignador de torch, que es por proceso. En
    las mediciones de referencia el margen fue de cientos de MiB, muy por encima del ruido
    ambiental. Con la resta invertida, la huella sale negativa y la cota falla siempre."""
    resultado, tmp_path = corrida_lanzar_3
    assert resultado.returncode == 0, resultado.stderr

    n_workers = 3
    # Memoria de la tarjeta de video: MEMORIA_JSON trae la huella medida
    # por el proceso PADRE (nvidia-smi, pico durante la corrida menos la línea
    # base antes de lanzar) más el desglose por worker (asignador de torch).
    linea_memoria = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("MEMORIA_JSON ")
    )
    memoria = json.loads(linea_memoria.removeprefix("MEMORIA_JSON "))
    por_worker = memoria["por_worker"]
    assert por_worker, "cada worker debe reportar la memoria de la tarjeta de video que usó"
    for worker, datos in por_worker.items():
        assert datos["asignador_mib"] > 0, (worker, datos)
        # --lanzar arranca los n workers juntos: el nvidia-smi antes/después
        # del primer tensor de cada uno (contexto_mib, ver _correr_worker)
        # atrapa también los contextos de los demás y queda inflado -- la
        # respuesta 6 solo agrega por worker el asignador de torch, que sí
        # es por proceso incluso bajo concurrencia (ver
        # pasos/anexo_paralelo.py); contexto_mib nunca entra aquí.
        assert "contexto_mib" not in datos, (worker, datos)
        assert set(datos) == {"asignador_mib"}, (worker, datos)
    assert memoria["huella_total_mib"] is not None, memoria
    suma_asignadores = sum(datos["asignador_mib"] for datos in por_worker.values())
    # Cota de magnitud sobre la huella global (ver el docstring, arriba, para
    # el margen medido): una resta invertida en `_huella_mib` da un número
    # negativo, siempre por debajo de esta suma positiva.
    assert memoria["huella_total_mib"] >= suma_asignadores, (memoria, suma_asignadores)
    huella_por_worker = memoria["huella_por_worker_mib"]
    # huella_por_worker_mib es SIEMPRE huella_total_mib / n_workers (nunca
    # el total sin dividir): comprobado con exactitud, no solo con una
    # cota, para detectar una división que se pierda. Esta división es
    # aritmética pura sobre lo que ya midió el padre -- determinista sin
    # importar cuánto ruido ambiental haya en la lectura de nvidia-smi.
    assert huella_por_worker == pytest.approx(memoria["huella_total_mib"] / n_workers), memoria


@pytest.mark.gpu
def test_workers_usan_semillas_de_sampler_distintas(corrida_lanzar_3):
    """Si `1000*worker` dejara de sumarse a la semilla del sampler, los 3
    workers propondrían la MISMA primera configuración (mismos datos por CRN,
    mismo sampler, misma semilla). Esta prueba verifica sobre el estudio real
    que produjo una corrida de verdad, no sobre la fábrica en aislamiento, y,
    además, sobre la semilla real que cada worker
    registró (`SEMILLAS_SAMPLER_JSON`): la fórmula completa es
    `BASE_SEMILLA_SAMPLER + 1000*worker + n_previos`, así que solo mirar los
    params no distinguiría "se rompió el *1000" de "con otro `n_previos`, por
    azar, salió una config distinta de todos modos".
    """
    resultado, tmp_path = corrida_lanzar_3
    assert resultado.returncode == 0, resultado.stderr

    storage = abrir_storage(tmp_path / "estudios" / "anexo_paralelo.log")
    study = optuna.load_study(study_name="paralelo_compartido", storage=storage)
    terminados = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)

    primer_trial_por_worker: dict[int, dict] = {}
    for t in sorted(terminados, key=lambda t: t.number):
        primer_trial_por_worker.setdefault(t.user_attrs["worker"], t.params)

    assert len(primer_trial_por_worker) == 3
    claves = {tuple(sorted(params.items())) for params in primer_trial_por_worker.values()}
    assert len(claves) > 1, (
        f"el primer trial de los 3 workers es idéntico: {primer_trial_por_worker} -- "
        "revisar que BASE_SEMILLA_SAMPLER + 1000*worker se siga aplicando"
    )

    linea_semillas = next(
        linea
        for linea in resultado.stdout.splitlines()
        if linea.startswith("SEMILLAS_SAMPLER_JSON ")
    )
    semillas = json.loads(linea_semillas.removeprefix("SEMILLAS_SAMPLER_JSON "))
    assert len(semillas) == 3, semillas
    valores: set[int] = set()
    for worker_str, datos in semillas.items():
        worker = int(worker_str)
        esperado = BASE_SEMILLA_SAMPLER + 1000 * worker + datos["n_previos"]
        assert datos["semilla"] == esperado, (worker, datos)
        valores.add(datos["semilla"])

        # `kwargs` es el dict COMPLETO que la fábrica dice haberle pasado a
        # TPESampler(**kwargs), capturado ANTES de esa llamada: prueba qué se
        # LE PASÓ a la construcción, no qué construyó Optuna con eso (una
        # fábrica que capturara un valor y le pasara otro a TPESampler
        # engañaría esta parte sola). La parte de abajo (SAMPLER_REAL_JSON)
        # cierra ese hueco leyendo los atributos del sampler REAL ya
        # construido.
        kwargs = datos["kwargs"]
        assert kwargs is not None, datos
        assert kwargs["seed"] == esperado, datos
        for clave, valor in SAMPLER_KWARGS_ESPERADOS.items():
            assert kwargs[clave] == valor, (clave, datos)
    assert len(valores) == 3, semillas

    # Verificación independiente sobre el sampler REAL ya construido (nunca
    # sobre el dict `kwargs` de arriba, que solo prueba qué se LE PASÓ a
    # TPESampler): si la fábrica dejara de pasar `**_SAMPLER_KWARGS` (p.ej.
    # `TPESampler(seed=semilla)` a secas) pero por error capturara y
    # reportara el dict completo de todos modos, la prueba de arriba seguiría
    # en verde -- SAMPLER_REAL_JSON, en cambio, sale de `getattr(study.
    # sampler, '_' + clave)` (pasos/anexo_paralelo.py:_correr_worker), así
    # que refleja lo que Optuna de verdad construyó.
    for worker in range(3):
        log_worker = (tmp_path / "estudios" / f"anexo_paralelo_worker_{worker}.log.txt").read_text(
            encoding="utf-8"
        )
        linea_sampler_real = next(
            linea for linea in log_worker.splitlines() if linea.startswith("SAMPLER_REAL_JSON ")
        )
        sampler_real = json.loads(linea_sampler_real.removeprefix("SAMPLER_REAL_JSON "))
        assert sampler_real["worker"] == worker, sampler_real
        for clave, valor in SAMPLER_KWARGS_ESPERADOS.items():
            assert sampler_real[clave] == valor, (worker, clave, sampler_real)


@pytest.mark.gpu
def test_secuencia_impresa_coincide_con_el_estudio(corrida_lanzar_3):
    """La respuesta 5 imprime la secuencia entera (no solo quién ganó el
    trial 0): se compara contra la que de verdad quedó en el estudio."""
    resultado, tmp_path = corrida_lanzar_3
    assert resultado.returncode == 0, resultado.stderr

    linea = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("SECUENCIA_JSON ")
    )
    secuencia_impresa = json.loads(linea.removeprefix("SECUENCIA_JSON "))

    storage = abrir_storage(tmp_path / "estudios" / "anexo_paralelo.log")
    study = optuna.load_study(study_name="paralelo_compartido", storage=storage)
    terminados = study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    secuencia_real = [
        t.user_attrs.get("worker") for t in sorted(terminados, key=lambda t: t.number)
    ]

    assert secuencia_impresa == secuencia_real


_TIPOS_FS_9P = frozenset({"9p", "v9fs", "drvfs"})


def _tipo_fs_de(ruta: Path) -> str | None:
    """Tipo de sistema de archivos del punto de montaje que contiene `ruta`,
    o `None` si `/proc/self/mountinfo` no existe (cualquier SO que no sea
    Linux). Mismo criterio (y misma firma) que `_tipo_fs_de` de
    `pasos/anexo_paralelo.py`, releído aquí en vez de importado -- aunque ese
    módulo ya es importable (guardia `if __name__ == "__main__":`, ver
    `_importar_anexo_paralelo` más abajo), duplicar esta función chica evita
    pagar sus imports pesados (optuna/matplotlib/sqlalchemy) solo para
    afirmar un mensaje de error."""
    try:
        texto = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError:
        return None
    resuelta = str(ruta.resolve())
    mejor_punto = ""
    mejor_tipo: str | None = None
    for linea in texto.splitlines():
        izquierda, separador, derecha = linea.partition(" - ")
        if not separador:
            continue
        campos_izq = izquierda.split()
        campos_der = derecha.split()
        if len(campos_izq) < 5 or not campos_der:
            continue
        punto_montaje = campos_izq[4]
        coincide = resuelta == punto_montaje or resuelta.startswith(
            punto_montaje if punto_montaje.endswith("/") else punto_montaje + "/"
        )
        if coincide and len(punto_montaje) >= len(mejor_punto):
            mejor_punto = punto_montaje
            mejor_tipo = campos_der[0]
    return mejor_tipo


def test_lanzar_en_montaje_9p_falla_con_mensaje_claro(tmp_path):
    """El Journal compartido por N procesos falla en montajes 9p/DrvFs de WSL
    (medido: 3 de 6 repeticiones con `--lanzar 3` en `/mnt/c` terminaron con
    `OSError: [Errno 61] No data available`, 0 de 6 en `/tmp` ext4).
    `--lanzar` debe salir ANTES con un mensaje claro, nunca dejar que
    ese `OSError` opaco llegue al tesista.

    Esta prueba usa el propio punto de montaje de RAIZ (el repo), nunca
    otro montaje 9p elegido al azar entre los que haya en la máquina (p. ej.
    /mnt/c cuando el repo vive en /mnt/d): escribir en la raíz de un montaje
    que no es el del repo no tiene nada que ver con lo que este test
    verifica, y arriesga tocar un directorio ajeno. Si RAIZ no vive en 9p,
    la prueba se salta -- no busca otro montaje donde sí se reproduzca."""
    tipo = _tipo_fs_de(RAIZ)
    if tipo not in _TIPOS_FS_9P:
        pytest.skip(f"RAIZ ({RAIZ}) no vive en un montaje 9p/v9fs/drvfs (tipo detectado: {tipo!r})")

    estudios_9p = RAIZ / "estudios" / f"_prueba_9p_{os.getpid()}"
    estudios_9p.mkdir(parents=True, exist_ok=False)
    try:
        entorno = {
            **os.environ,
            "PRESUPUESTO": "humo",
            "TALLER_RESULTADOS": str(tmp_path / "resultados"),
            "TALLER_ESTUDIOS": str(estudios_9p),
        }
        resultado = subprocess.run(
            [sys.executable, str(SCRIPT), "--lanzar", "2"],
            env=entorno,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resultado.returncode != 0
        assert "montajes 9p/DrvFs de WSL" in resultado.stderr, resultado.stderr
    finally:
        # Sin ignore_errors: si la limpieza de verdad falla, que se vea en vez
        # de tragárselo en silencio (regla de credenciales/errores del taller).
        shutil.rmtree(estudios_9p)


@pytest.mark.gpu
def test_lanzar_local_reanuda_sin_repetir_trabajo(tmp_path):
    """Volver a llamar --lanzar sobre el MISMO Journal no repite los primeros
    trials: es el mismo contrato de reanudar que ya prueba test_estudio.py para
    `optimizar`, aquí en el caso real de N workers concurrentes.
    """
    entorno_base = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_reanudar",
    }

    primero = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2"],
        env=entorno_base,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert primero.returncode == 0, primero.stderr

    destino = tmp_path / "estudios" / "anexo_paralelo.log"
    storage = abrir_storage(destino)
    study = optuna.load_study(study_name="paralelo_reanudar", storage=storage)
    numeros_sesion_1 = {
        t.number for t in study.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    }
    assert len(numeros_sesion_1) >= 6

    entorno_reanudar = {**entorno_base, "TALLER_N_TOTAL": "10"}
    segundo = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2"],
        env=entorno_reanudar,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert segundo.returncode == 0, segundo.stderr
    # La segunda corrida entrenó ALGO nuevo (10 > 6): no es el caso de
    # reanudación completa (ver test_lanzar_comparar_dos_veces_...), así que
    # sí debe reportar trials nuevos, no "REANUDACIÓN" con 0 nuevos.
    assert "REANUDACIÓN" in segundo.stdout, segundo.stdout
    assert "0 nuevos de esta sesión" not in segundo.stdout, segundo.stdout

    # Si la fábrica dejara de sumar `n_previos` a la semilla (quedándose solo
    # con BASE_SEMILLA_SAMPLER + 1000*worker), esta segunda sesión propondría
    # la MISMA primera configuración que la primera sesión ya propuso --
    # ningún assert de arriba lo nota porque ambas sesiones siguen reportando
    # trials nuevos, solo el sampler quedaría desincronizado.
    linea_semillas_2 = next(
        linea for linea in segundo.stdout.splitlines() if linea.startswith("SEMILLAS_SAMPLER_JSON ")
    )
    semillas_2 = json.loads(linea_semillas_2.removeprefix("SEMILLAS_SAMPLER_JSON "))
    assert len(semillas_2) == 2, semillas_2
    for worker_str, datos in semillas_2.items():
        worker = int(worker_str)
        # numeros_sesion_1 ya tiene >= 6 trials (assert de arriba) y los dos
        # workers de la sesión 2 abren el estudio ANTES de producir su propio
        # primer trial (crear_estudio cuenta `_trials_existentes` en ese
        # instante) -- los dos deben ver al menos esos 6 como `n_previos`.
        assert datos["n_previos"] >= 6, (worker, datos)
        assert datos["semilla"] == BASE_SEMILLA_SAMPLER + 1000 * worker + datos["n_previos"], (
            worker,
            datos,
        )

    study_reabierto = optuna.load_study(study_name="paralelo_reanudar", storage=storage)
    terminados_2 = study_reabierto.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    numeros_sesion_2 = {t.number for t in terminados_2}
    # Los trials de la primera sesión siguen ahí tal cual (no se repitieron).
    assert numeros_sesion_1 <= numeros_sesion_2
    assert len(terminados_2) >= 10


def test_lanzar_menor_a_uno_falla_con_mensaje(tmp_path):
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "0"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "--lanzar debe ser >= 1" in resultado.stderr


def test_lanzar_sin_tarjeta_de_video_falla_con_mensaje_claro(tmp_path):
    """`--lanzar` es el modo que usa el tesista: sin tarjeta de video, debe
    fallar con un mensaje claro que llegue a esta pantalla (el `stderr` del
    proceso PADRE), nunca dejar que el mensaje real quede enterrado solo
    dentro del log de un worker que nadie abre a simple vista (ver `_lanzar`
    en pasos/anexo_paralelo.py). Esta prueba oculta la tarjeta de video con
    `CUDA_VISIBLE_DEVICES=""`, sin la marca `gpu`: necesita precisamente
    correr sin tarjeta de video visible, la tenga o no la máquina que corre
    la suite."""
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    # No solo una subcadena: un traceback que TERMINE con esta frase (p. ej.
    # copiada del log de un worker fallido, ver _ultima_linea_no_vacia)
    # también la contendría sin que el proceso PADRE hubiera fallado limpio
    # ANTES de lanzar ningún worker -- las tres afirmaciones de abajo juntas
    # exigen la salida limpia de una sola línea que `_exigir_tarjeta_de_video`
    # produce, no un traceback que solo mencione la misma frase de pasada.
    assert "Traceback" not in resultado.stderr, resultado.stderr
    assert resultado.stderr.strip() == MENSAJE_SIN_TARJETA, resultado.stderr
    assert not (tmp_path / "estudios" / "anexo_paralelo_worker_0.log.txt").exists()


def test_correr_worker_sin_tarjeta_de_video_falla_con_mensaje_claro(tmp_path):
    """El modo worker (default, sin `--lanzar` ni `--preparar`) es el que
    corre cada contenedor `worker` de `compose.yaml` y cada
    proceso que `--lanzar` arranca por su cuenta: debe fallar con la misma
    línea limpia que `--lanzar`, nunca con el traceback completo que da
    importar `taller_hpo.problemas.juguete` sin tarjeta de video (`_correr_worker`
    llama a `_exigir_tarjeta_de_video` ANTES de ese import perezoso, ver
    pasos/anexo_paralelo.py). Regresión: sin esta llamada, este modo vuelve a
    fallar con el traceback de `juguete.py` en vez de esta línea. Esta prueba
    oculta la tarjeta de video con `CUDA_VISIBLE_DEVICES=""`, sin la marca
    `gpu`: necesita precisamente correr sin tarjeta de video visible."""
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "Traceback" not in resultado.stderr, resultado.stderr
    assert resultado.stderr.strip() == MENSAJE_SIN_TARJETA, resultado.stderr


def test_preparar_sin_tarjeta_de_video_falla_con_mensaje_claro(tmp_path):
    """El modo `--preparar` (el servicio `preparar` de `compose.yaml`, perfil
    `paralelo`) debe fallar con la misma línea limpia, antes de tocar el
    estudio -- mismo motivo y misma regresión que la prueba de arriba para el
    modo worker, pero para `_preparar_estudio`."""
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--preparar"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "Traceback" not in resultado.stderr, resultado.stderr
    assert resultado.stderr.strip() == MENSAJE_SIN_TARJETA, resultado.stderr
    assert not (tmp_path / "estudios").exists()


def test_presupuesto_invalido_falla_con_mensaje(tmp_path):
    entorno = {
        **os.environ,
        "PRESUPUESTO": "enorme",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "1"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


def test_comparar_sin_lanzar_falla_con_mensaje(tmp_path):
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--comparar"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "--comparar solo aplica junto con --lanzar" in resultado.stderr


def test_preparar_y_lanzar_juntos_falla_con_mensaje(tmp_path):
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--preparar", "--lanzar", "1"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0
    assert "--preparar y --lanzar son incompatibles" in resultado.stderr


# --- garantías centrales del anexo: aceleración real, orden, protocolo -----


@pytest.fixture(scope="module")
def corrida_comparar(tmp_path_factory):
    """Corre `pasos/anexo_paralelo.py --lanzar 2 --comparar` UNA sola vez
    (mismo patrón que `corrida_lanzar_3`, más arriba): las pruebas de
    aceleración/contención, ORDEN_JSON/semillas y memoria de abajo leen
    todas de esta misma corrida en vez de relanzar el subproceso cada una.
    Un fallo real en la prueba de memoria (más abajo) no debe apagar las
    afirmaciones de aceleración/contención u ORDEN_JSON, que no dependen de
    la memoria de la tarjeta de video -- por eso viven en pruebas separadas
    sobre la MISMA corrida, no repetidas dentro de un único test con un `if`
    de por medio."""
    tmp_path = tmp_path_factory.mktemp("comparar_humo")
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_comparar",
    }
    resultado = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2", "--comparar"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return resultado, tmp_path


@pytest.mark.gpu
def test_lanzar_comparar_aceleracion_es_reloj_1_sobre_reloj_n(corrida_comparar):
    """La aceleración sale de dos relojes independientes (una corrida con 1 worker y otra
    con n), nunca de sum(segundos)/reloj, que la contención infla. El estudio de comparación
    entrena exactamente los trials que produjo la sesión con n workers, no n_total, para
    que una reanudación parcial no compare cargas distintas.

    El script borra ese estudio auxiliar, así que la prueba lee el log del worker aislado:
    `reloj_1`, medido por el padre, nunca puede ser menor que el tiempo de `optimizar()` ni
    que la vida de proceso que reportó el worker (`VIDA_PROCESO_JSON`).
    `TRIALS_PROPIOS_JSON` da una fuente independiente de `params`/`segundos_solo` por par,
    que atrapa un desalineo de `segundos_solo` en `pares_contencion`.
    """
    resultado, tmp_path = corrida_comparar
    assert resultado.returncode == 0, resultado.stderr

    linea = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("ACELERACION_JSON ")
    )
    datos = json.loads(linea.removeprefix("ACELERACION_JSON "))
    assert datos["reloj_1"] > 0
    assert datos["reloj_n"] > 0
    assert datos["aceleracion"] == pytest.approx(datos["reloj_1"] / datos["reloj_n"], rel=1e-9)
    assert datos["n_trials_comparacion"] >= 1, datos

    # CONTENCION_JSON: el costo por trial bajo contención frente a uno
    # aislado sale de estadistica.diferencia_pareada (nunca de un cociente
    # entre sum(segundos) bajo contención y reloj_1, que no miden lo mismo).
    # (1) El texto impreso debe coincidir EXACTAMENTE con `veredicto` -- ver
    # test_frase_contencion_depende_exactamente_del_veredicto, que prueba los
    # 4 veredictos con datos sintéticos (una corrida real casi nunca produce
    # "mejor"/"peor" con PRESUPUESTO=humo, así que solo esta prueba no basta
    # para cubrir los 4). (2) params_solo (los params que de verdad entrenó
    # el estudio de 1 worker, leídos ANTES de borrarlo) deben ser IDÉNTICOS,
    # en el MISMO orden, a los params de los trials nuevos del estudio
    # PRINCIPAL -- la verificación de que el worker solo entrenó las MISMAS
    # configuraciones (encoladas con enqueue_trial), nunca "otras" que un
    # TPESampler recién sembrado hubiera propuesto por su cuenta.
    linea_contencion = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("CONTENCION_JSON ")
    )
    contencion = json.loads(linea_contencion.removeprefix("CONTENCION_JSON "))
    linea_frase_contencion = next(
        linea
        for linea in resultado.stdout.splitlines()
        if linea.startswith("   Costo por trial bajo contención frente a uno aislado: ")
    )
    frases_esperadas_por_veredicto = {
        "mejor": "más caro bajo contención",
        "peor": "más rápido que uno aislado",
        "equivalente": "equivalente a un trial aislado dentro del 10%",
        "no concluyente": "no concluyente con estas",
    }
    assert contencion["veredicto"] in frases_esperadas_por_veredicto, contencion
    assert frases_esperadas_por_veredicto[contencion["veredicto"]] in linea_frase_contencion, (
        contencion,
        linea_frase_contencion,
    )

    # Toda frase que interpreta un resultado (aquí, el veredicto de
    # contención) tiene que salir de una prueba medida en la MISMA corrida,
    # no solo de que el texto impreso corresponda al JSON que el propio
    # script ya produjo (eso solo prueba `_frase_contencion`, ver
    # test_frase_contencion_depende_exactamente_del_veredicto): esta prueba
    # recalcula `diferencia_pareada` desde cero, a partir de
    # `contencion["pares"]` (ya verificado más abajo contra el estudio
    # principal), con la MISMA `delta`/`direccion` que usa `_contencion`
    # (pasos/anexo_paralelo.py): si el veredicto o la diferencia media
    # persistida no coincidieran con este recálculo independiente, el código
    # real se habría desviado de lo que _frase_contencion interpreta.
    solo_recalculo = {p["trial_number"]: p["segundos_solo"] for p in contencion["pares"]}
    cont_recalculo = {p["trial_number"]: p["segundos_contencion"] for p in contencion["pares"]}
    pareado_recalculado = diferencia_pareada(
        solo_recalculo,
        cont_recalculo,
        delta=0.1 * statistics.median(solo_recalculo.values()),
        direccion="minimizar",
    )
    assert contencion["veredicto"] == pareado_recalculado.veredicto, (
        contencion,
        pareado_recalculado,
    )
    assert contencion["diferencia_media"] == pytest.approx(pareado_recalculado.diferencia_media), (
        contencion,
        pareado_recalculado,
    )
    assert contencion["ic_bajo"] == pytest.approx(pareado_recalculado.ic_bajo), (
        contencion,
        pareado_recalculado,
    )
    assert contencion["ic_alto"] == pytest.approx(pareado_recalculado.ic_alto), (
        contencion,
        pareado_recalculado,
    )

    storage_principal = abrir_storage(tmp_path / "estudios" / "anexo_paralelo.log")
    study_principal = optuna.load_study(study_name="paralelo_comparar", storage=storage_principal)
    # Estudio fresco (TALLER_ESTUDIO nuevo, sin sesión previa): TODOS los
    # trials del estudio principal son "nuevos" de esta sesión.
    params_principal = [
        t.params
        for t in sorted(
            study_principal.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS),
            key=lambda t: t.number,
        )
    ]
    assert contencion["params_solo"] == params_principal, (
        contencion["params_solo"],
        params_principal,
    )

    # Cada par de CONTENCION_JSON["pares"] se cruza contra el trial del MISMO
    # NÚMERO en el estudio PRINCIPAL (nunca contra una posición de lista, que
    # un emparejamiento incorrecto -- p. ej. con `reversed` -- podría
    # desalinear) -- params y segundos_contencion deben coincidir exactamente
    # con lo que ese trial real tiene.
    principal_por_numero = {
        t.number: t for t in study_principal.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    }
    assert len(contencion["pares"]) == datos["n_trials_comparacion"], contencion
    for par in contencion["pares"]:
        trial_principal = principal_por_numero[par["trial_number"]]
        assert par["params"] == trial_principal.params, (par, trial_principal.params)
        assert par["segundos_contencion"] == trial_principal.user_attrs["segundos"], (
            par,
            trial_principal.user_attrs,
        )
        assert par["segundos_solo"] > 0, par
    # params y segundos_solo de cada par salen del MISMO t_solo (ver
    # pares_contencion en pasos/anexo_paralelo.py): ya verificado arriba que
    # params_solo == params_principal, así que una reversión de terminados_solo
    # (que movería params Y segundos_solo juntos) ya habría reprobado esa
    # aserción -- aquí solo hace falta la marca de agua de que las dos claves
    # existen y son coherentes entre sí par a par.
    assert [p["params"] for p in contencion["pares"]] == contencion["params_solo"], contencion

    # El script ya borró el estudio auxiliar tras medir: esta prueba lee su
    # log en vez de reabrirlo con optuna.load_study.
    log_solo = (tmp_path / "estudios" / "anexo_paralelo_solo_worker_0.log.txt").read_text(
        encoding="utf-8"
    )

    # TRIALS_PROPIOS_JSON: fuente independiente de `params`/`segundos_solo`
    # por par, leída DENTRO del worker aislado directo de su propio estudio
    # (ver `_correr_worker` en pasos/anexo_paralelo.py) -- nunca reconstruida
    # por el `zip(nuevos_ordenados, terminados_solo)` que arma
    # `pares_contencion` en `_lanzar`. Un error que desalineara solo la
    # extracción de `segundos_solo` en ese zip (dejando `params` intacto) no
    # la notaría la verificación de `params_solo` de arriba, pero sí este
    # cruce, porque viene de un proceso y un cómputo distintos.
    linea_trials_propios = next(
        linea for linea in log_solo.splitlines() if linea.startswith("TRIALS_PROPIOS_JSON ")
    )
    trials_propios_solo = json.loads(linea_trials_propios.removeprefix("TRIALS_PROPIOS_JSON "))[
        "trials"
    ]
    assert len(trials_propios_solo) == len(contencion["pares"]), (
        trials_propios_solo,
        contencion["pares"],
    )
    for par, propio in zip(contencion["pares"], trials_propios_solo, strict=True):
        assert par["params"] == propio["params"], (par, propio)
        assert par["segundos_solo"] == propio["segundos"], (par, propio)

    linea_worker_solo = next(
        linea for linea in log_solo.splitlines() if linea.startswith("worker 0: aportó ")
    )
    coincidencia = re.search(
        r"aportó (\d+) trial\(s\) propio\(s\); estudio con (\d+) terminados "
        r"\(meta n_total=(\d+)\) en ([\d.]+) s",
        linea_worker_solo,
    )
    assert coincidencia is not None, linea_worker_solo
    propios_solo, terminados_solo, n_total_solo, segundos_optimizar_solo = coincidencia.groups()
    assert int(propios_solo) == datos["n_trials_comparacion"], (propios_solo, datos)
    assert int(terminados_solo) == datos["n_trials_comparacion"], (terminados_solo, datos)
    assert int(n_total_solo) == datos["n_trials_comparacion"], (n_total_solo, datos)
    assert datos["reloj_1"] >= float(segundos_optimizar_solo), (
        datos["reloj_1"],
        segundos_optimizar_solo,
    )

    # Cota floja: la vida medida por VIDA_PROCESO_JSON de este worker solo
    # (el import de torch, el contexto CUDA y el entrenamiento -- NUNCA el
    # arranque del intérprete ni el import de optuna/matplotlib/sqlalchemy,
    # que _INICIO_PROCESO en pasos/anexo_paralelo.py deja fuera a propósito),
    # nunca solo lo que optimizar() midió por dentro -- si `reloj_1` viniera
    # de la suma de 'segundos' bajo contención del run con N workers (en vez
    # de medir de verdad ESTE proceso aislado con Popen.wait()), no habría
    # por qué respetar esta cota tampoco.
    linea_vida_solo = next(
        linea for linea in log_solo.splitlines() if linea.startswith("VIDA_PROCESO_JSON ")
    )
    vida_solo = json.loads(linea_vida_solo.removeprefix("VIDA_PROCESO_JSON "))
    vida_propia_solo_s = vida_solo["fin"] - vida_solo["inicio"]
    assert datos["reloj_1"] >= vida_propia_solo_s, (datos["reloj_1"], vida_propia_solo_s)

    # Cota DIRECTA y ajustada (no floja): `inicio_reloj_solo` y
    # `fin_reloj_solo` (time.monotonic(), tomados por el PADRE justo antes y
    # justo después de _spawn_n_workers(1, ...)) encierran exactamente la
    # misma llamada que `reloj_1` mide con perf_counter() -- en Linux, los dos
    # leen CLOCK_MONOTONIC --, así que las dos duraciones deben coincidir salvo
    # el código sin E/S que corre entre las dos lecturas (la ruta del script ya
    # viene resuelta desde que se importó el paso: resolverla ahí, sobre un
    # montaje 9p de WSL, tocaría el disco dentro de la ventana), y una
    # tolerancia de 50 ms sí es un detector confiable. Con time.time() no lo
    # sería: el reloj de pared salta cuando el sistema sincroniza la hora (en
    # WSL2 pasa durante una misma corrida) y un salto dentro de la ventana
    # movería una duración y no la otra. Además, vida_solo (medida DENTRO del worker con el mismo
    # CLOCK_MONOTONIC, que en Linux es uno solo para todos los procesos) debe
    # caer adentro de la ventana [inicio_reloj_solo, fin_reloj_solo] del
    # padre: el worker arranca después del Popen() y termina antes de que
    # Popen.wait() retorne. La cadena completa también exige que la vida sea
    # coherente por dentro (inicio <= fin, y al menos lo que optimizar() midió
    # por dentro): un worker que mezclara relojes o tomara su inicio al final
    # dejaría las cotas de arriba vacías sin que ninguna fallara.
    assert "inicio_reloj_solo" in datos and "fin_reloj_solo" in datos, datos
    duracion_reloj_solo = datos["fin_reloj_solo"] - datos["inicio_reloj_solo"]
    assert abs(datos["reloj_1"] - duracion_reloj_solo) < 0.05, (datos, duracion_reloj_solo)
    assert (
        datos["inicio_reloj_solo"]
        <= vida_solo["inicio"]
        <= vida_solo["fin"]
        <= datos["fin_reloj_solo"]
    ), (datos, vida_solo)
    assert float(segundos_optimizar_solo) <= vida_propia_solo_s <= datos["reloj_1"], (
        segundos_optimizar_solo,
        vida_propia_solo_s,
        datos["reloj_1"],
    )


@pytest.mark.gpu
def test_lanzar_comparar_memoria_gpu_worker_aislado(corrida_comparar):
    """Memoria del worker aislado (--comparar), en una prueba separada de la
    aceleración/contención (arriba) y del ORDEN_JSON (abajo): ninguna de esas
    otras dos depende de la memoria, y las tres corren sobre la misma corrida
    vía `corrida_comparar`.

    `contexto_mib` (nvidia-smi antes/después del primer tensor de este
    worker, ver `_medir_contexto_mib` en pasos/anexo_paralelo.py) es UNA sola
    lectura de UN worker real: con la tarjeta de video compartida entre otros
    procesos de la máquina, esa lectura sola puede salir corta o incluso
    negativa sin que sea un fallo de este worker (el "antes" nunca puede
    pesar más que el "después" por cuenta propia de este worker, pero otro
    proceso ajeno sí puede mover `memory.used` en esa misma ventana) -- por
    eso esta prueba no le exige un piso de magnitud a esta única lectura,
    solo que exista y sea un número real. La cota sobre la magnitud (mediana
    > 200 MiB) vive en test_medir_contexto_mib_da_una_mediana_alta_bajo_carga
    (sobre la función suelta) y en
    test_correr_worker_reporta_contexto_mib_con_mediana_alta_bajo_carga (sobre
    el modo worker de verdad): las dos repiten la medición K veces en
    procesos nuevos en vez de confiar en esta lectura sola."""
    resultado, tmp_path = corrida_comparar
    assert resultado.returncode == 0, resultado.stderr

    log_solo = (tmp_path / "estudios" / "anexo_paralelo_solo_worker_0.log.txt").read_text(
        encoding="utf-8"
    )
    linea_memoria_solo = next(
        linea for linea in log_solo.splitlines() if linea.startswith("MEMORIA_GPU_WORKER_JSON ")
    )
    datos_mem_solo = json.loads(linea_memoria_solo.removeprefix("MEMORIA_GPU_WORKER_JSON "))
    assert set(datos_mem_solo) == {
        "worker",
        "contexto_mib",
        "asignador_mib",
        "total_mib",
    }, datos_mem_solo
    assert math.isfinite(datos_mem_solo["contexto_mib"]), datos_mem_solo

    # La respuesta 6 (agregada, sobre los n workers concurrentes) nunca
    # imprime contexto_mib por worker -- ver test_lanzar_local_memoria_gpu,
    # que lo exige sobre la corrida principal de --lanzar sin --comparar.
    linea_memoria = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("MEMORIA_JSON ")
    )
    memoria = json.loads(linea_memoria.removeprefix("MEMORIA_JSON "))
    for worker, datos_worker in memoria["por_worker"].items():
        assert set(datos_worker) == {"asignador_mib"}, (worker, datos_worker)


@pytest.mark.gpu
def test_medir_contexto_mib_da_una_mediana_alta_bajo_carga():
    """`_medir_contexto_mib` es la única fuente de `contexto_mib`, y una sola lectura puede
    salir corta o negativa bajo carga ajena. Por eso la prueba la corre 7 veces, cada una
    en un proceso nuevo, y exige la mediana.

    `PISO_CONTEXTO_MIB=200` deja margen: con la tarjeta compartida y carga adicional, las
    medianas medidas quedaron alrededor de 450-540 MiB, aunque lecturas sueltas cayeran bajo
    0. Invertir la resta da una mediana cercana a -450 MiB, y medir el "antes" después del
    primer tensor da una cercana a 0: los dos errores quedan bajo el piso."""
    codigo = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(SCRIPT)!r})\n"
        "modulo = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(modulo)\n"
        "print(modulo._medir_contexto_mib('cuda'))\n"
    )
    valores = []
    for _ in range(7):
        resultado = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, timeout=60
        )
        assert resultado.returncode == 0, resultado.stderr
        valores.append(float(resultado.stdout.strip()))
    mediana = statistics.median(valores)
    assert mediana > 200, valores


@pytest.mark.gpu
def test_correr_worker_reporta_contexto_mib_con_mediana_alta_bajo_carga(tmp_path_factory):
    """`test_medir_contexto_mib_da_una_mediana_alta_bajo_carga` (arriba)
    protege `_medir_contexto_mib` aislada, pero el camino real -- el que de
    verdad llega al tesista en MEMORIA_GPU_WORKER_JSON -- es `_correr_worker`,
    que la llama una sola vez ANTES de crear cualquier otro tensor en el
    dispositivo (ver el comentario de `_correr_worker` en
    pasos/anexo_paralelo.py). Si un tensor de calentamiento se colara antes de
    esa medición, el contexto CUDA ya existiría al medir y `contexto_mib`
    saldría cerca de 0 en vez de reflejar el contexto real -- esta prueba
    corre el modo worker de verdad (nunca la función suelta) K=7 veces, cada
    vez en un proceso nuevo con un solo trial (PRESUPUESTO=humo,
    TALLER_N_TOTAL=1), y exige la MEDIANA sobre el `contexto_mib` que cada uno
    reportó.

    Con la tarjeta compartida, tres corridas reales del modo worker dieron un
    `contexto_mib` de entre 450 y 900 MiB; adelantar la asignación con un tensor de
    calentamiento lo deja cerca de 0, muy por debajo del piso de 200 MiB."""
    valores = []
    for intento in range(7):
        tmp_path = tmp_path_factory.mktemp(f"contexto_worker_{intento}")
        entorno = {
            **os.environ,
            "PRESUPUESTO": "humo",
            "TALLER_RESULTADOS": str(tmp_path / "resultados"),
            "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
            "TALLER_ESTUDIO": "contexto_worker_solo",
            "TALLER_N_TOTAL": "1",
            "TALLER_WORKER": "0",
        }
        resultado = subprocess.run(
            [sys.executable, str(SCRIPT)],
            env=entorno,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert resultado.returncode == 0, resultado.stderr
        linea_memoria = next(
            linea
            for linea in resultado.stdout.splitlines()
            if linea.startswith("MEMORIA_GPU_WORKER_JSON ")
        )
        datos = json.loads(linea_memoria.removeprefix("MEMORIA_GPU_WORKER_JSON "))
        valores.append(datos["contexto_mib"])
    mediana = statistics.median(valores)
    assert mediana > 200, valores


@pytest.mark.gpu
def test_lanzar_comparar_orden_y_semillas(corrida_comparar):
    """ORDEN_JSON: el veredicto impreso debe ser exactamente el resultado de
    comparar las dos secuencias (o, con longitudes distintas, decir que no
    son comparables), nunca una frase fija, y los conteos de configuraciones
    que coinciden se recalculan aquí de forma independiente a partir de los
    params y las semillas que ORDEN_JSON trae de las dos corridas -- además,
    `params_originales` se recalcula también de cero a partir del estudio
    real (nunca solo repitiendo lo que ORDEN_JSON ya afirma sobre sí mismo):
    si `_lanzar` dejara de filtrar por `n_startup_trials` al construir ese
    campo, esta prueba lo nota aunque el resto del recálculo lea el mismo
    JSON. Separada de la aceleración/contención y de la memoria (arriba) por
    la misma razón que ellas: un fallo real en una no debe apagar las
    afirmaciones de las otras dos."""
    resultado, tmp_path = corrida_comparar
    assert resultado.returncode == 0, resultado.stderr

    linea_orden = next(
        linea for linea in resultado.stdout.splitlines() if linea.startswith("ORDEN_JSON ")
    )
    orden = json.loads(linea_orden.removeprefix("ORDEN_JSON "))

    # El veredicto de "coincide" solo compara secuencias de la MISMA longitud;
    # con longitudes distintas debe declararse "no comparables", nunca
    # comparar un prefijo arbitrario (secuencia[:len(rep)] == rep puede dar
    # "coincidió" sin que las dos corridas se parezcan en nada cuando las
    # longitudes vienen de corridas distintas).
    if len(orden["secuencia_original"]) == len(orden["secuencia_repetida"]):
        esperado = orden["secuencia_original"] == orden["secuencia_repetida"]
        assert orden["coincide"] == esperado, orden
    else:
        assert orden["coincide"] is None, orden

    linea_secuencia_repetida = next(
        linea
        for linea in resultado.stdout.splitlines()
        if linea.startswith("   Secuencia repetida: ")
    )
    if orden["coincide"] is None:
        assert "no comparables" in linea_secuencia_repetida, linea_secuencia_repetida
    elif orden["coincide"]:
        assert "coincidió con la corrida de arriba" in linea_secuencia_repetida, (
            linea_secuencia_repetida
        )
    else:
        assert "cambió respecto a la corrida de arriba" in linea_secuencia_repetida, (
            linea_secuencia_repetida
        )

    # conteo_por_worker_repetido se computa en el código con una expresión
    # INDEPENDIENTE de secuencia_repetida (mismo query, otro Counter): si
    # alguien reemplazara secuencia_repetida por una copia de
    # secuencia_original (en vez de derivarla de verdad de la repetición),
    # los dos ya no coincidirían.
    conteo_esperado = {str(k): v for k, v in Counter(orden["secuencia_repetida"]).items()}
    assert orden["conteo_por_worker_repetido"] == conteo_esperado, orden

    # params_originales se recalcula aquí DIRECTO del estudio principal
    # (nunca solo repitiendo lo que ORDEN_JSON ya trae): agrupado por
    # user_attrs["worker"], en orden de number, quedándose solo con los
    # trials con number > n_startup_trials -- si `_lanzar` dejara de pasar
    # ese filtro a `_params_por_worker_tras_arranque` (p. ej. pasando 0 en
    # vez de n_arranque), el código seguiría imprimiendo el mismo
    # "n_startup_trials=10" en el texto, pero params_originales incluiría
    # también los trials de arranque, y esta recomputación independiente
    # -- que nunca pasa por esa función mutada -- lo nota.
    storage_principal = abrir_storage(tmp_path / "estudios" / "anexo_paralelo.log")
    study_principal = optuna.load_study(study_name="paralelo_comparar", storage=storage_principal)
    terminados_principal = study_principal.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    n_arranque_esperado = SAMPLER_KWARGS_ESPERADOS["n_startup_trials"]
    params_originales_esperado: dict[str, list[dict]] = {}
    for t in sorted(terminados_principal, key=lambda t: t.number):
        if t.number > n_arranque_esperado:
            params_originales_esperado.setdefault(str(t.user_attrs.get("worker")), []).append(
                t.params
            )
    assert orden["params_originales"] == params_originales_esperado, (
        orden["params_originales"],
        params_originales_esperado,
    )

    # coinciden_params/comparados se recalculan aquí de cero a partir de los
    # params y las semillas efectivas que ORDEN_JSON trae de las dos
    # corridas -- si el código dejara de filtrar por semilla coincidente
    # (contando siempre, sin el `if`) o contara mal, esta recomputación
    # independiente lo nota.
    comparados_esperado = 0
    coinciden_esperado = 0
    for worker_str, propios in orden["params_originales"].items():
        semilla_orig = orden["semillas_originales"].get(worker_str)
        semilla_rep = orden["semillas_repetidas"].get(worker_str)
        if semilla_orig is None or semilla_rep is None or semilla_orig != semilla_rep:
            continue
        repetidos = orden["params_repetidos"].get(worker_str, [])
        for a, b in zip(propios, repetidos, strict=False):
            comparados_esperado += 1
            if a == b:
                coinciden_esperado += 1
    assert orden["comparados"] == comparados_esperado, orden
    assert orden["coinciden_params"] == coinciden_esperado, orden

    # La frase de "semillas efectivas iguales en K de N" debe coincidir con
    # las semillas REALES que la repetición escribió en sus propios logs
    # (SEMILLA_SAMPLER_JSON), no con lo que el código de _lanzar afirma sobre
    # sí mismo sin verificación externa.
    linea_semillas_iguales = next(
        linea
        for linea in resultado.stdout.splitlines()
        if linea.startswith("   Semillas efectivas de sampler iguales en ")
    )
    semillas_rep_logs: dict[str, int] = {}
    for worker_str in orden["semillas_originales"]:
        log_rep = (
            tmp_path / "estudios" / f"anexo_paralelo_orden_worker_{worker_str}.log.txt"
        ).read_text(encoding="utf-8")
        linea_json_rep = next(
            linea for linea in log_rep.splitlines() if linea.startswith("SEMILLA_SAMPLER_JSON ")
        )
        semillas_rep_logs[worker_str] = json.loads(
            linea_json_rep.removeprefix("SEMILLA_SAMPLER_JSON ")
        )["semilla"]
    coincidencias_esperadas = sum(
        1
        for worker_str, semilla in orden["semillas_originales"].items()
        if semillas_rep_logs.get(worker_str) == semilla
    )
    assert f"iguales en {coincidencias_esperadas} de 2 worker(s)" in linea_semillas_iguales, (
        linea_semillas_iguales,
        orden["semillas_originales"],
        semillas_rep_logs,
    )
    # La razón impresa junto a "iguales en K de N" depende EXACTAMENTE de si
    # coincidieron todas las semillas efectivas (`_razon_semillas` en
    # pasos/anexo_paralelo.py): con K < N, n_previos depende de la carrera de
    # apertura del estudio; con K == N, las dos corridas vieron el mismo
    # n_previos por worker. Invertir esa condición dejaría pasar el texto
    # equivocado sin que el assert de arriba (que solo mira "iguales en K de
    # N") lo note.
    if coincidencias_esperadas < 2:
        assert "n_previos depende de quién abrió el estudio primero" in linea_semillas_iguales, (
            linea_semillas_iguales
        )
    else:
        assert "las dos corridas vieron el mismo n_previos" in linea_semillas_iguales, (
            linea_semillas_iguales
        )


@pytest.mark.gpu
def test_lanzar_comparar_reanudacion_parcial_compara_solo_lo_nuevo(tmp_path):
    """Con una reanudación PARCIAL (0 < restante < n_total), la corrida de
    comparación de 1 worker debe entrenar EXACTAMENTE los mismos trials que
    produjo la sesión con n workers (`len(nuevos)`), nunca `n_total` a secas:
    comparar 1 worker entrenando `n_total` DESDE CERO contra n workers que
    solo entrenaron lo que faltaba sería una comparación desigual (1 worker
    haría más trabajo del que le tocaba comparar)."""
    entorno_base = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_parcial",
    }
    primero = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2"],
        env=entorno_base,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert primero.returncode == 0, primero.stderr

    entorno_parcial = {**entorno_base, "TALLER_N_TOTAL": "12"}
    segundo = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2", "--comparar"],
        env=entorno_parcial,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert segundo.returncode == 0, segundo.stderr
    assert "REANUDACIÓN" in segundo.stdout, segundo.stdout

    # Reanudación PARCIAL: n_previos (y por lo tanto la semilla efectiva de
    # cada worker) depende de quién ganó la carrera de abrir el estudio
    # primero -- "repetir con las mismas semillas" sería falso aquí (ver
    # _fabrica_sampler en pasos/anexo_paralelo.py), así que la repetición de
    # orden/reproducibilidad debe omitirse, nunca afirmar "las mismas
    # semillas".
    assert "Repetición con las mismas semillas" not in segundo.stdout, segundo.stdout
    assert "Repetición de orden/reproducibilidad omitida: REANUDACIÓN parcial" in segundo.stdout, (
        segundo.stdout
    )

    linea_secuencia = next(
        linea for linea in segundo.stdout.splitlines() if linea.startswith("SECUENCIA_JSON ")
    )
    nuevos_segundo = json.loads(linea_secuencia.removeprefix("SECUENCIA_JSON "))
    # Reanudación PARCIAL de verdad: ni 0 (eso sería reanudación completa,
    # otro test) ni 12 (eso sería una sesión fresca, sin nada previo).
    assert 0 < len(nuevos_segundo) < 12, nuevos_segundo

    linea = next(
        linea for linea in segundo.stdout.splitlines() if linea.startswith("ACELERACION_JSON ")
    )
    datos = json.loads(linea.removeprefix("ACELERACION_JSON "))
    assert datos["n_trials_comparacion"] == len(nuevos_segundo), (datos, nuevos_segundo)

    # El estudio auxiliar de comparación ya se borró (optuna.delete_study):
    # se verifica lo que de verdad entrenó a través de su log, que sigue en
    # disco.
    log_solo = (tmp_path / "estudios" / "anexo_paralelo_solo_worker_0.log.txt").read_text(
        encoding="utf-8"
    )
    linea_worker_solo = next(
        linea for linea in log_solo.splitlines() if linea.startswith("worker 0: aportó ")
    )
    coincidencia = re.search(
        r"aportó (\d+) trial\(s\) propio\(s\); estudio con (\d+) terminados "
        r"\(meta n_total=(\d+)\) en ([\d.]+) s",
        linea_worker_solo,
    )
    assert coincidencia is not None, linea_worker_solo
    _propios, terminados_solo, n_total_solo, _segundos = coincidencia.groups()
    assert int(terminados_solo) == len(nuevos_segundo), (terminados_solo, nuevos_segundo)
    assert int(n_total_solo) == len(nuevos_segundo), (n_total_solo, nuevos_segundo)
    assert int(n_total_solo) != 12, "la comparación no debe entrenar n_total completo desde cero"


@pytest.mark.gpu
def test_lanzar_comparar_dos_veces_no_reporta_datos_de_la_corrida_anterior(tmp_path):
    """Con una reanudación COMPLETA (el estudio principal ya alcanzó
    n_total), las respuestas 1, 2, 4 y 5 no deben calcularse sobre todo el
    estudio (que incluiría los trials de la corrida anterior) -- eso
    reportaría una 'aceleración' y una secuencia que en realidad pertenecen a
    otra sesión. Aquí se corre dos veces y se exige que la SEGUNDA reconozca
    la reanudación, no imprima ACELERACION_JSON (no hay nada nuevo que
    comparar), reporte una secuencia vacía (0 trials nuevos) y omita la
    repetición de orden/reproducibilidad (tampoco hay nada nuevo con qué
    repetirla -- nunca un veredicto sobre una comparación vacía)."""
    entorno = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_comparar_dos_veces",
        "TALLER_N_TOTAL": "6",
    }
    primero = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2", "--comparar"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert primero.returncode == 0, primero.stderr
    assert "REANUDACIÓN" not in primero.stdout, primero.stdout
    assert any(linea.startswith("ACELERACION_JSON ") for linea in primero.stdout.splitlines()), (
        primero.stdout
    )

    segundo = subprocess.run(
        [sys.executable, str(SCRIPT), "--lanzar", "2", "--comparar"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert segundo.returncode == 0, segundo.stderr

    assert "REANUDACIÓN" in segundo.stdout, segundo.stdout
    assert "0 nuevos de esta sesión" in segundo.stdout, segundo.stdout
    assert "El paso no mide una aceleración aquí" in segundo.stdout, segundo.stdout
    assert not any(
        linea.startswith("ACELERACION_JSON ") for linea in segundo.stdout.splitlines()
    ), segundo.stdout

    # La repetición de orden/reproducibilidad tampoco debe correr: sin trials
    # nuevos, "cambió"/"coincidió con la corrida de arriba" sería un veredicto
    # sobre una comparación vacía (secuencia == [] de los dos lados).
    assert "Repetición de orden/reproducibilidad omitida" in segundo.stdout, segundo.stdout
    assert "cambió respecto a la corrida de arriba" not in segundo.stdout, segundo.stdout
    assert "coincidió con la corrida de arriba" not in segundo.stdout, segundo.stdout

    linea_secuencia = next(
        linea for linea in segundo.stdout.splitlines() if linea.startswith("SECUENCIA_JSON ")
    )
    assert json.loads(linea_secuencia.removeprefix("SECUENCIA_JSON ")) == []


@pytest.mark.gpu
def test_preparar_registra_protocolo_y_worker_lo_reabre_sin_crearlo(tmp_path):
    """El modo --preparar (servicio 'preparar' de compose.yaml) crea el
    estudio con 0 trials; un worker que reabre después no lo vuelve a crear ni
    lo rechaza -- exactamente lo que evita la carrera de dos conexiones contra
    un PostgreSQL vacío creando el esquema a la vez (ver el docstring de
    pasos/anexo_paralelo.py)."""
    entorno_base = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_preparado",
        "TALLER_N_TOTAL": "3",
    }

    preparar = subprocess.run(
        [sys.executable, str(SCRIPT), "--preparar"],
        env=entorno_base,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert preparar.returncode == 0, preparar.stderr
    assert "preparado" in preparar.stdout, preparar.stdout
    # La contraseña nunca debería aparecer aquí (no hay una: destino es un
    # Journal local, no Postgres), pero el print pasa por _destino_redactado
    # de todos modos -- ver test_docker_paralelo_dos_workers_contra_postgres
    # para la verificación real con una URL de Postgres.

    destino = tmp_path / "estudios" / "anexo_paralelo.log"
    storage = abrir_storage(destino)
    study = optuna.load_study(study_name="paralelo_preparado", storage=storage)
    assert study.get_trials(deepcopy=False) == []
    assert study.user_attrs["protocolo"]["motor"] == juguete.PROTOCOLO_MOTOR["motor"]

    worker = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**entorno_base, "TALLER_WORKER": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert worker.returncode == 0, worker.stderr

    study_reabierto = optuna.load_study(study_name="paralelo_preparado", storage=storage)
    terminados = study_reabierto.get_trials(deepcopy=False, states=ESTADOS_TERMINADOS)
    assert len(terminados) >= 3


@pytest.mark.gpu
def test_reabrir_con_protocolo_distinto_falla_con_mensaje(tmp_path):
    """`crear_estudio` (`taller_hpo.estudio`) rechaza reabrir un estudio con un
    protocolo distinto al que ya tiene guardado -- el docstring de
    `pasos/anexo_paralelo.py` lo dice explícito: el protocolo guardado
    registra que PyTorch entrenó en la tarjeta de video, y si reabres un
    Journal con un protocolo distinto, `crear_estudio` lo rechaza, igual que
    con otro sampler o espacio. Esto protege a un tesista que reabre un
    Journal viejo, guardado con un protocolo distinto al de la corrida
    actual: el worker debe rechazarlo con un mensaje claro, nunca mezclar en
    silencio trials de dos búsquedas distintas.

    Esta prueba altera a mano `dispositivo` en el protocolo ya persistido:
    prueba el rechazo de `crear_estudio` ante un protocolo distinto, no
    cuántos valores puede tomar `dispositivo`."""
    entorno_base = {
        **os.environ,
        "PRESUPUESTO": "humo",
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_ESTUDIO": "paralelo_protocolo_distinto",
        "TALLER_N_TOTAL": "3",
    }

    preparar = subprocess.run(
        [sys.executable, str(SCRIPT), "--preparar"],
        env=entorno_base,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert preparar.returncode == 0, preparar.stderr

    destino = tmp_path / "estudios" / "anexo_paralelo.log"
    storage = abrir_storage(destino)
    study = optuna.load_study(study_name="paralelo_protocolo_distinto", storage=storage)
    # Esta prueba simula un Journal escrito por otra corrida, con un
    # protocolo que ya no coincide con el que este worker va a pedir al
    # reabrir -- el propio anexo nunca lo produce así, porque siempre pide
    # dispositivo "cuda".
    protocolo_alterado = {**study.user_attrs["protocolo"], "dispositivo": "cuda_de_otra_corrida"}
    study.set_user_attr("protocolo", protocolo_alterado)

    worker = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env={**entorno_base, "TALLER_WORKER": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert worker.returncode != 0, worker.stdout
    assert "protocolo distinto" in worker.stderr, worker.stderr


def test_paralelo_docker_sh_no_filtra_password_generada_sin_daemon(tmp_path):
    """Regresión de credenciales para la ruta por defecto de `scripts/paralelo_docker.sh`:
    sin `TALLER_PG_PASSWORD`, el script genera una con `secrets.token_hex(16)` y la exporta,
    pero nunca debe imprimirla. `test_docker_paralelo_dos_workers_contra_postgres` siempre
    fija la variable, así que no cubre esta rama.

    Sin Docker real: un ejecutable `docker` falso, primero en PATH, captura
    `$TALLER_PG_PASSWORD` del entorno y responde lo mínimo para que el script complete su
    camino con N=1. El control positivo (la contraseña capturada tiene el formato de
    `token_hex(16)`) evita que la ausencia en stdout/stderr sea trivial."""
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    captura = tmp_path / "password_capturada.txt"
    docker_falso = fake_bin / "docker"
    docker_falso.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [ "$1" != "compose" ]; then
    echo "fake docker: solo se soporta 'compose', no: $*" >&2
    exit 1
fi
shift
# argv ahora: --profile paralelo -p PROYECTO <subcomando> [resto...]
# (mismo prefijo constante que arma COMPOSE en scripts/paralelo_docker.sh).
sub="$5"

case "$sub" in
    up)
        # Captura lo que el script YA exportó a su propio entorno -- nunca
        # se interpola compose.yaml aquí, este fake no monta nada real.
        printf '%s' "${TALLER_PG_PASSWORD:-}" > "$FAKE_DOCKER_CAPTURA"
        exit 0
        ;;
    ps)
        resto="$*"
        if [[ "$resto" == *"--format json"* ]]; then
            if [[ "$resto" == *"--status running"* ]]; then
                exit 0  # ninguna línea: 0 réplicas "corriendo" (N=1, ya listo)
            fi
            echo '{"ExitCode":0}'
        fi
        exit 0
        ;;
    logs)
        exit 0
        ;;
    run)
        echo 'RESUMEN_JSON {"trials_totales": 1, "por_worker": {"0": 1}, "running": 0}'
        exit 0
        ;;
    down)
        exit 0
        ;;
    *)
        echo "fake docker: subcomando compose no soportado: $sub ($*)" >&2
        exit 1
        ;;
esac
""",
        encoding="utf-8",
    )
    modo = docker_falso.stat().st_mode
    docker_falso.chmod(modo | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    script = RAIZ / "scripts" / "paralelo_docker.sh"
    entorno = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
        "FAKE_DOCKER_CAPTURA": str(captura),
        "TALLER_COMPOSE_PROJECT": f"taller-hpo-paralelo-fake-{os.getpid()}",
    }
    entorno.pop("TALLER_PG_PASSWORD", None)  # la ruta por defecto: SIN definir

    resultado = subprocess.run(
        ["bash", str(script), "1"],
        cwd=RAIZ,
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, f"stdout:\n{resultado.stdout}\n\nstderr:\n{resultado.stderr}"
    assert "TALLER_PG_PASSWORD generada solo para esta corrida" in resultado.stdout, (
        resultado.stdout
    )

    assert captura.exists(), "el docker falso nunca vio TALLER_PG_PASSWORD (revisar el fake)"
    password_capturada = captura.read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{32}", password_capturada), password_capturada

    assert password_capturada not in resultado.stdout, "la contraseña generada salió en stdout"
    assert password_capturada not in resultado.stderr, "la contraseña generada salió en stderr"


# La línea `# syntax=docker/dockerfile:1` le pide a BuildKit bajar de docker.io la imagen
# del analizador de Dockerfile en cada build. Docker acepta espacios y mayúsculas en esa
# directiva ("#syntax = ..."), así que el patrón también.
_DIRECTIVA_SYNTAX = re.compile(r"^\s*#\s*syntax\s*=", re.IGNORECASE)


def _lineas_con_directiva_syntax(dockerfile: str) -> list[int]:
    """Los números de línea de `dockerfile` que son una directiva `# syntax=`."""
    lineas = dockerfile.splitlines()
    return [n for n, linea in enumerate(lineas, 1) if _DIRECTIVA_SYNTAX.match(linea)]


def test_dockerfile_no_pide_el_analizador_de_dockerfile_a_docker_io():
    """El `Dockerfile` no lleva la directiva `# syntax=`: con ella, cada build
    consulta docker.io por la imagen `docker/dockerfile`, y un tropiezo de red en
    ese registro rompe `test_docker_paralelo_dos_workers_contra_postgres` antes de
    construir nada. Sin Docker, sin tarjeta de video y sin red: la prueba solo lee el
    texto del archivo.

    Prueba el patrón en las dos direcciones (un patrón roto dejaría pasar la línea
    en verde): marca la directiva con espacios y en mayúscula, y deja pasar un
    comentario que solo la nombra."""
    assert _lineas_con_directiva_syntax((RAIZ / "Dockerfile").read_text(encoding="utf-8")) == []

    assert _lineas_con_directiva_syntax("# syntax=docker/dockerfile:1\nFROM base\n") == [1]
    assert _lineas_con_directiva_syntax("#syntax = docker/dockerfile:1.7\nFROM base\n") == [1]
    assert _lineas_con_directiva_syntax("# SYNTAX=docker/dockerfile:1\nFROM base\n") == [1]
    comentario = "# Sin la línea `# syntax=docker/dockerfile:1`, el build no baja nada.\n"
    assert _lineas_con_directiva_syntax(comentario) == []


# Una imagen de un registro (lleva ":" o "/", a diferencia del nombre de una etapa como
# "base") que aparece en `FROM` o en `COPY --from=` sin digest hace que BuildKit consulte
# ese registro en cada build, aunque tenga la imagen en su caché.
_REFERENCIA_DE_IMAGEN = re.compile(r"^\s*(?:FROM\s+(?:--platform=\S+\s+)?|COPY\s+--from=)(\S+)")
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _imagenes_de_registro_sin_digest(dockerfile: str) -> list[int]:
    """Los números de línea de `dockerfile` que usan una imagen de registro sin digest."""
    sin_digest = []
    for n, linea in enumerate(dockerfile.splitlines(), 1):
        coincidencia = _REFERENCIA_DE_IMAGEN.match(linea)
        if coincidencia is None:
            continue
        referencia = coincidencia.group(1)
        es_de_registro = ":" in referencia or "/" in referencia
        if es_de_registro and not _DIGEST.search(referencia):
            sin_digest.append(n)
    return sin_digest


def test_dockerfile_fija_las_imagenes_base_por_digest():
    """Las dos imágenes base del `Dockerfile` (python y uv) van fijadas por digest: con
    solo la etiqueta, cada build le pregunta al registro por sus metadatos, y un corte
    de red rompe `test_docker_paralelo_dos_workers_contra_postgres` antes de construir
    nada. Sin Docker, sin tarjeta de video y sin red: la prueba solo lee el texto del
    archivo.

    Prueba el patrón en las dos direcciones: marca una imagen con solo etiqueta en
    `FROM` y en `COPY --from=`, y deja pasar las etapas internas (`FROM base AS worker`)
    y las imágenes con digest."""
    assert _imagenes_de_registro_sin_digest((RAIZ / "Dockerfile").read_text(encoding="utf-8")) == []

    digest = "@sha256:" + "a" * 64
    assert _imagenes_de_registro_sin_digest("FROM python:3.12-slim AS base\n") == [1]
    assert _imagenes_de_registro_sin_digest(
        "COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/\n"
    ) == [1]
    assert _imagenes_de_registro_sin_digest("FROM --platform=linux/amd64 python:3.12\n") == [1]
    assert _imagenes_de_registro_sin_digest(f"FROM python:3.12-slim{digest} AS base\n") == []
    assert (
        _imagenes_de_registro_sin_digest(
            f"COPY --from=ghcr.io/astral-sh/uv:0.12.19{digest} /uv /\n"
        )
        == []
    )
    assert (
        _imagenes_de_registro_sin_digest("FROM base AS worker\nCOPY --from=worker /app /app\n")
        == []
    )


# --- Docker: postgres real + N réplicas de worker (scripts/paralelo_docker.sh) ---


@pytest.mark.gpu
@pytest.mark.docker
@pytest.mark.timeout(1600)
def test_docker_paralelo_dos_workers_contra_postgres():
    razon = _razon_no_docker()
    if razon:
        pytest.skip(razon)

    script = RAIZ / "scripts" / "paralelo_docker.sh"
    # El ciclo repite la corrida 2 veces, cada una con un proyecto de compose
    # (y por lo tanto una base de datos) nuevos: la carrera de CREATE TYPE
    # studydirection que corrige el servicio "preparar" (dos conexiones
    # creando el esquema de PostgreSQL a la vez) solo se reproduce contra una
    # base VACÍA, así que una sola corrida en verde no basta como regresión.
    marcador_password = f"marcador-de-prueba-no-secreto-{os.getpid()}"
    for intento in range(2):
        entorno = {
            **os.environ,
            "PRESUPUESTO": "humo",
            "TALLER_N_TOTAL": "6",
            "TALLER_ESTUDIO": f"paralelo_docker_test_{intento}",
            # Contraseña propia y claramente marcada (nunca la real generada
            # por el script): así se puede afirmar que NUNCA sale en stdout ni
            # en stderr, ni siquiera en el log de "preparar" (que el script
            # imprime siempre) -- ver _destino_redactado en
            # pasos/anexo_paralelo.py.
            "TALLER_PG_PASSWORD": marcador_password,
            # Sufijo con el pid: dos corridas de esta prueba en paralelo (dos
            # sesiones de pytest, o un re-run mientras otra sigue viva) no
            # comparten proyecto de compose ni se pisan el "down -v" una a la
            # otra.
            "TALLER_COMPOSE_PROJECT": f"taller-hpo-paralelo-pytest-{os.getpid()}-{intento}",
        }
        resultado = subprocess.run(
            ["bash", str(script), "2"],
            cwd=RAIZ,
            env=entorno,
            capture_output=True,
            text=True,
            timeout=700,
        )
        assert resultado.returncode == 0, (
            f"intento {intento}: stdout:\n{resultado.stdout}\n\nstderr:\n{resultado.stderr}"
        )

        assert marcador_password not in resultado.stdout, (
            f"intento {intento}: la contraseña de PostgreSQL salió en stdout"
        )
        assert marcador_password not in resultado.stderr, (
            f"intento {intento}: la contraseña de PostgreSQL salió en stderr"
        )

        # scripts/paralelo_docker.sh imprime "docker compose logs worker" antes
        # del resumen: cada worker real (dentro del contenedor, con la tarjeta
        # de video) debe haber impreso juguete.DESCRIPCION al arrancar, la
        # misma convención que ya verifica el modo local. El taller entrena
        # siempre en la tarjeta de video (compose.yaml reserva `driver:
        # nvidia` en "worker" y en "preparar"): el entrenamiento corre
        # siempre con torch en cuda -- ver taller_hpo.problemas.juguete.
        assert "Motor: PyTorch" in resultado.stdout, f"intento {intento}: {resultado.stdout}"
        assert "en cuda" in resultado.stdout, f"intento {intento}: {resultado.stdout}"

        # Docker no tiene un proceso PADRE que muestree nvidia-smi como
        # --lanzar (cada réplica de "worker" es un contenedor independiente,
        # sin un Popen que las junte): el desglose por worker
        # (MEMORIA_GPU_WORKER_JSON, en los logs que "docker compose logs
        # worker" ya volcó a stdout) es la única medición de memoria
        # disponible aquí. `asignador_mib` (torch.cuda.max_memory_reserved,
        # propio del proceso) sí debe ser > 0 siempre. `contexto_mib`
        # (nvidia-smi antes/después del primer tensor) puede salir negativo
        # con la tarjeta de video compartida entre otros procesos de la máquina que
        # corre Docker: si alguno de
        # ellos libera memoria en esa misma ventana, el delta medido puede
        # superar lo que este worker de verdad reservó -- así que solo se
        # exige que sea un número real (nunca NaN, que es lo que da cuando
        # nvidia-smi falla del todo). "in linea", no
        # "startswith": "docker compose logs worker" antepone
        # "<servicio>-<réplica> | " a cada línea (Docker Compose v5.5.1), y anclar al
        # principio de línea, como hace RESUMEN_JSON más abajo (que viene de "docker
        # compose run", sin ese prefijo), aquí perdería las dos líneas sin avisar.
        lineas_memoria_docker = [
            linea for linea in resultado.stdout.splitlines() if "MEMORIA_GPU_WORKER_JSON " in linea
        ]
        assert len(lineas_memoria_docker) >= 2, f"intento {intento}: {resultado.stdout}"
        for linea_mem in lineas_memoria_docker:
            _, _, cola = linea_mem.partition("MEMORIA_GPU_WORKER_JSON ")
            datos_mem = json.loads(cola)
            assert datos_mem["asignador_mib"] > 0, (intento, datos_mem)
            assert datos_mem["contexto_mib"] == datos_mem["contexto_mib"], (intento, datos_mem)

        # scripts/paralelo_docker.sh imprime, además de las líneas legibles, un
        # "RESUMEN_JSON {...}": se parsea ESE (json.loads), no las líneas de
        # texto pensadas para una persona.
        linea_json = next(
            linea for linea in resultado.stdout.splitlines() if linea.startswith("RESUMEN_JSON ")
        )
        resumen = json.loads(linea_json.removeprefix("RESUMEN_JSON "))

        assert resumen["running"] == 0
        assert resumen["trials_totales"] in (6, 7)  # n_total=6, N=2 -> [6, 6+2-1]
        por_worker = resumen["por_worker"]
        assert len(por_worker) == 2, (
            f"intento {intento}: se esperaban 2 workers, salió {por_worker}"
        )
        assert sum(por_worker.values()) == resumen["trials_totales"]
        assert all(n >= 1 for n in por_worker.values())


@pytest.mark.docker
def test_docker_n_menor_a_uno_falla_sin_tocar_docker():
    script = RAIZ / "scripts" / "paralelo_docker.sh"
    resultado = subprocess.run(
        ["bash", str(script), "0"], cwd=RAIZ, capture_output=True, text=True, timeout=30
    )
    assert resultado.returncode != 0
    assert "N debe ser" in resultado.stderr
