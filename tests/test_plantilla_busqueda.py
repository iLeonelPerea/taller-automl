"""`plantilla/buscar_en_proceso.py` y `plantilla/buscar_por_subproceso.py`: las dos rutas
completas (línea base sembrada, `TPESampler` y `optimizar`) contra `train_ejemplo.py`.

Las pruebas que entrenan llevan `@pytest.mark.gpu`. `_resolver_dispositivo` siempre lanza
un intérprete a comprobar la tarjeta, con cualquier valor, así que casi todas sus pruebas
también la llevan. No la llevan las que fallan antes de tocar torch: `PRESUPUESTO`
inválido, `train.py` sin la clave `'dispositivo'` o un `dispositivo` con forma inválida.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import optuna
import pytest
import torch

from taller_hpo.objetivo import en_proceso
from taller_hpo.semillas import Semillas
from taller_hpo.subproceso import por_subproceso
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

optuna.logging.set_verbosity(optuna.logging.WARNING)

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLA = RAIZ / "plantilla"
BUSCAR_EN_PROCESO_PY = PLANTILLA / "buscar_en_proceso.py"
BUSCAR_POR_SUBPROCESO_PY = PLANTILLA / "buscar_por_subproceso.py"


def _importar(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


buscar_en_proceso = _importar(BUSCAR_EN_PROCESO_PY, "_buscar_en_proceso_bajo_prueba")
buscar_por_subproceso = _importar(BUSCAR_POR_SUBPROCESO_PY, "_buscar_por_subproceso_bajo_prueba")


# --- buscar(): la ruta completa produce un estudio real -------------------------


@pytest.mark.gpu
def test_buscar_en_proceso_produce_un_estudio_con_n_total_trials(tmp_path):
    study = buscar_en_proceso.buscar(n_total=3, destino=tmp_path / "estudio.log")
    completados = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completados) == 3
    assert all(0.0 <= t.value <= 1.0 for t in completados)


@pytest.mark.gpu
def test_buscar_por_subproceso_produce_un_estudio_con_n_total_trials(tmp_path):
    study = buscar_por_subproceso.buscar(
        n_total=3,
        destino=tmp_path / "estudio.log",
        directorio_trials=tmp_path / "trials",
        timeout_s=30.0,
    )
    completados = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completados) == 3
    assert all(0.0 <= t.value <= 1.0 for t in completados)


@pytest.mark.gpu
def test_buscar_en_proceso_siembra_la_linea_base_primero(tmp_path):
    # Con n_total=1, el ÚNICO trial que corre es el encolado por
    # sembrar_linea_base: Optuna siempre vacía la cola antes de muestrear.
    study = buscar_en_proceso.buscar(n_total=1, destino=tmp_path / "estudio.log")
    (trial,) = study.trials
    assert trial.user_attrs.get("rol") == "linea_base"
    modulo_train = buscar_en_proceso._importar_train_ejemplo()
    esperado = {clave: modulo_train.DEFAULTS[clave] for clave in buscar_en_proceso.DISTRIBUCIONES}
    assert trial.params == esperado


@pytest.mark.gpu
def test_buscar_por_subproceso_siembra_la_misma_linea_base(tmp_path):
    study = buscar_por_subproceso.buscar(
        n_total=1,
        destino=tmp_path / "estudio.log",
        directorio_trials=tmp_path / "trials",
        timeout_s=30.0,
    )
    (trial,) = study.trials
    assert trial.user_attrs.get("rol") == "linea_base"
    assert trial.params == buscar_por_subproceso._LINEA_BASE


def test_buscar_por_subproceso_copia_defaults_no_ha_divergido_de_train_ejemplo():
    """Esta prueba no necesita una tarjeta de video: solo importa
    `train_ejemplo.py` para leer sus `DEFAULTS`, sin llamar a `entrenar` ni
    resolver ningún dispositivo. `buscar_por_subproceso.py` NO importa
    `train_ejemplo.py` (por_subproceso
    lo corre como proceso completo, no como módulo -- ver el docstring), así
    que `_FIJOS_PROBLEMA`/`_LINEA_BASE` son una copia manual de sus DEFAULTS.
    Esta prueba es la guarda contra que esa copia divergiera en silencio si
    alguien cambia `train_ejemplo.py` sin actualizarla. `_FIJOS_PROBLEMA` debe
    traer TODAS las claves de DEFAULTS que no son del espacio de búsqueda
    (incluidas n_ruido/n_capas/activacion/batch_size, no solo
    n_train/n_val/ruido/epocas) salvo 'dispositivo', que NO se copia estático
    aquí: se resuelve en tiempo de ejecución (ver `_resolver_dispositivo`)."""
    modulo_train = buscar_en_proceso._importar_train_ejemplo()
    esperado_fijos = {
        clave: valor
        for clave, valor in modulo_train.DEFAULTS.items()
        if clave not in buscar_por_subproceso.DISTRIBUCIONES and clave != "dispositivo"
    }
    esperado_linea_base = {
        clave: modulo_train.DEFAULTS[clave] for clave in buscar_por_subproceso.DISTRIBUCIONES
    }
    assert buscar_por_subproceso._FIJOS_PROBLEMA == esperado_fijos
    assert buscar_por_subproceso._LINEA_BASE == esperado_linea_base


# --- buscar_en_proceso._resolver_dispositivo_auto: valida CUALQUIER valor ---


def test_resolver_dispositivo_auto_sin_tarjeta_de_video_falla_con_mensaje_claro(monkeypatch):
    """Esta prueba fuerza `torch.cuda.is_available` a `False` para simular
    esa ausencia sin necesitar una tarjeta de video real, así corre igual en
    cualquier máquina. `'cpu'` es un valor inválido como cualquier otro
    fuera de `auto`/`cuda(:N)`. Esta prueba llama a la función con valores
    de formato válido pero distintos de `'auto'` (`'cuda'` y `'cuda:1'`) a
    propósito: la comprobación de la tarjeta de video corre SIEMPRE, con
    cualquier valor que pase el chequeo de formato -- nunca solo cuando vale
    `'auto'` -- así que un `train.py` que declare `DEFAULTS['dispositivo'] =
    'cuda'` sin que haya tarjeta visible debe fallar igual que uno con
    `'auto'`. Esta prueba compara la cadena EXACTA del mensaje, no solo que
    contenga "tarjeta de video": una implementación que fallara con otro
    mensaje que también mencionara esa frase pasaría desapercibida si esta
    prueba solo buscara la subcadena."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="dispositivo"):
        buscar_en_proceso._resolver_dispositivo_auto("cpu")
    for valor in ("auto", "cuda", "cuda:1"):
        with pytest.raises(RuntimeError) as excinfo:
            buscar_en_proceso._resolver_dispositivo_auto(valor)
        assert str(excinfo.value) == MENSAJE_SIN_TARJETA, (valor, excinfo.value)


def test_resolver_dispositivo_auto_conserva_el_indice_de_tarjeta_de_video(monkeypatch):
    """Con una tarjeta de video disponible, `_resolver_dispositivo_auto` solo
    debe resolver `'auto'` a `'cuda'`: un valor ya concreto como `'cuda:1'`
    debe llegar INTACTO, con su índice, para que un `train.py` que pida esa
    tarjeta en particular la reciba de verdad. Esta prueba fuerza
    `torch.cuda.is_available` a `True` sin necesitar una tarjeta de video
    real."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert buscar_en_proceso._resolver_dispositivo_auto("cuda:1") == "cuda:1"
    assert buscar_en_proceso._resolver_dispositivo_auto("auto") == "cuda"


_TRAIN_CON_DISPOSITIVO_CUDA_FIJO = '''"""train.py de prueba: DEFAULTS declara \'dispositivo\':
\'cuda\' -- ya concreto, nunca \'auto\' -- para comprobar que
buscar_en_proceso.buscar() valida la tarjeta de video con CUALQUIER valor,
no solo con \'auto\'."""

DEFAULTS = {
    "n_train": 300,
    "n_val": 100,
    "ruido": 0.25,
    "n_ruido": 65,
    "n_capas": 1,
    "ancho": 16,
    "activacion": "relu",
    "learning_rate_init": 1e-2,
    "alpha": 1e-4,
    "batch_size": 32,
    "epocas": 15,
    "dispositivo": "cuda",
}


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5, "epocas": 1.0}
'''


def test_buscar_en_proceso_dispositivo_cuda_fijo_sin_tarjeta_falla_antes_de_crear_journal(
    tmp_path, monkeypatch
):
    """Un `train.py` que declare `DEFAULTS['dispositivo'] = 'cuda'` (un valor
    ya concreto, nunca `'auto'`) debe pasar por la MISMA comprobación de
    tarjeta de video que uno con `'auto'`: la clave obligatoria deja que
    `buscar()` compruebe la tarjeta de video antes de crear el estudio, para
    cualquier `train.py` que fijara el dispositivo de antemano. Esta prueba
    fuerza `torch.cuda.is_available` a `False` para simular esa ausencia sin
    necesitar una tarjeta de video real. `buscar()` debe fallar con el
    mensaje EXACTO de `MENSAJE_SIN_TARJETA`, sin dejar el Journal creado."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train_path = tmp_path / "train_dispositivo_cuda_fijo.py"
    train_path.write_text(_TRAIN_CON_DISPOSITIVO_CUDA_FIJO, encoding="utf-8")
    destino = tmp_path / "estudio.log"
    with pytest.raises(RuntimeError) as excinfo:
        buscar_en_proceso.buscar(n_total=1, destino=destino, train_path=train_path)
    assert str(excinfo.value) == MENSAJE_SIN_TARJETA
    assert not destino.exists()


# --- protocolo: sampler, pruner, semillas y dispositivo quedan en la huella ------


@pytest.mark.gpu
def test_buscar_en_proceso_protocolo_registra_sampler_pruner_semillas_y_fijos(tmp_path):
    study = buscar_en_proceso.buscar(n_total=1, destino=tmp_path / "estudio.log")
    protocolo = study.user_attrs["protocolo"]
    assert protocolo["sampler"] == "TPESampler"
    assert protocolo["sampler_kwargs"] == {
        "seed_base": buscar_en_proceso.SEMILLA_BASE,
        "n_startup_trials": 2,
    }
    assert protocolo["pruner"] == "NopPruner"
    assert protocolo["politica_semillas"]["semillas"] == dataclasses.asdict(
        buscar_en_proceso.SEMILLAS_BUSQUEDA
    )
    assert protocolo["metrica"] == "exactitud_val"
    # 'dispositivo' queda resuelto a 'cuda', nunca 'auto': ver el docstring
    # de buscar() y taller_hpo.problemas.juguete.PROTOCOLO_MOTOR.
    assert protocolo["fijos"]["dispositivo"] == "cuda"
    assert set(protocolo["fijos"]) == {
        "n_train",
        "n_val",
        "ruido",
        "n_ruido",
        "n_capas",
        "activacion",
        "batch_size",
        "epocas",
        "dispositivo",
    }


@pytest.mark.gpu
def test_buscar_por_subproceso_protocolo_registra_sampler_pruner_semillas_y_fijos(tmp_path):
    study = buscar_por_subproceso.buscar(
        n_total=1,
        destino=tmp_path / "estudio.log",
        directorio_trials=tmp_path / "trials",
        timeout_s=30.0,
    )
    protocolo = study.user_attrs["protocolo"]
    assert protocolo["sampler"] == "TPESampler"
    assert protocolo["sampler_kwargs"] == {
        "seed_base": buscar_por_subproceso.SEMILLA_BASE,
        "n_startup_trials": 2,
    }
    assert protocolo["pruner"] == "NopPruner"
    assert protocolo["politica_semillas"]["semillas"] == dataclasses.asdict(
        buscar_por_subproceso.SEMILLAS_BUSQUEDA
    )
    assert protocolo["metrica"] == "exactitud_val"
    assert protocolo["fijos"]["dispositivo"] == "cuda"


@pytest.mark.gpu
def test_en_proceso_y_por_subproceso_protocolos_son_iguales_salvo_fijos(tmp_path):
    """Las dos plantillas dicen ser espejo una de la otra (ver los docstrings
    de los dos módulos): su protocolo debe coincidir en todo salvo 'fijos'
    (que en_proceso arma leyendo DEFAULTS en vivo y por_subproceso con una
    copia manual -- test_buscar_por_subproceso_copia_defaults_no_ha_divergido
    ya prueba que esa copia no diverge)."""
    study_ep = buscar_en_proceso.buscar(n_total=1, destino=tmp_path / "ep.log")
    study_ps = buscar_por_subproceso.buscar(
        n_total=1,
        destino=tmp_path / "ps.log",
        directorio_trials=tmp_path / "trials",
        timeout_s=30.0,
    )
    protocolo_ep = dict(study_ep.user_attrs["protocolo"])
    protocolo_ps = dict(study_ps.user_attrs["protocolo"])
    del protocolo_ep["fijos"]
    del protocolo_ps["fijos"]
    assert protocolo_ep == protocolo_ps


@pytest.mark.gpu
def test_buscar_en_proceso_reabrir_con_otras_semillas_de_busqueda_falla(tmp_path, monkeypatch):
    """Un tesista que cambie SEMILLAS_BUSQUEDA (o SEMILLA_BASE) y reanude el
    MISMO estudio no debe poder contaminarlo en silencio: crear_estudio
    guarda 'politica_semillas' en el protocolo y rechaza reabrir con uno
    distinto (ver taller_hpo.estudio.crear_estudio)."""
    destino = tmp_path / "estudio.log"
    buscar_en_proceso.buscar(n_total=1, destino=destino)
    monkeypatch.setattr(
        buscar_en_proceso, "SEMILLAS_BUSQUEDA", Semillas(datos=99, orden=100, init=101)
    )
    with pytest.raises(ValueError, match="protocolo distinto"):
        buscar_en_proceso.buscar(n_total=1, destino=destino)


@pytest.mark.gpu
def test_buscar_por_subproceso_reabrir_con_otras_semillas_de_busqueda_falla(tmp_path, monkeypatch):
    """Misma guarda que la prueba de arriba, para buscar_por_subproceso.py
    (requiere torch: a diferencia de la prueba de arriba, la PRIMERA llamada
    sí corre un trial real por subproceso contra train_ejemplo.py)."""
    destino = tmp_path / "estudio.log"
    buscar_por_subproceso.buscar(
        n_total=1, destino=destino, directorio_trials=tmp_path / "trials", timeout_s=30.0
    )
    monkeypatch.setattr(
        buscar_por_subproceso, "SEMILLAS_BUSQUEDA", Semillas(datos=99, orden=100, init=101)
    )
    with pytest.raises(ValueError, match="protocolo distinto"):
        buscar_por_subproceso.buscar(
            n_total=1, destino=destino, directorio_trials=tmp_path / "trials", timeout_s=30.0
        )


# --- atrapar: MetricaNoFinita marca el trial FAIL y la búsqueda sigue -----------

_TRAIN_FALLA_UNA_VEZ = '''"""train.py de prueba: falla (NaN) en la PRIMERA llamada a entrenar, y
funciona en las siguientes. Sirve para probar que buscar_en_proceso.buscar()
atrapa MetricaNoFinita (ver taller_hpo.objetivo.en_proceso) y sigue la
búsqueda en vez de propagar la excepción -- no necesita torch."""

DEFAULTS = {
    "n_train": 300,
    "n_val": 100,
    "ruido": 0.25,
    "n_ruido": 65,
    "n_capas": 1,
    "ancho": 16,
    "activacion": "relu",
    "learning_rate_init": 1e-2,
    "alpha": 1e-4,
    "batch_size": 32,
    "epocas": 15,
    "dispositivo": "cuda",
}

_llamadas = []


def entrenar(config, semillas, reportar=None):
    _llamadas.append(1)
    if len(_llamadas) == 1:
        return {"exactitud_val": float("nan"), "epocas": 1.0}
    return {"exactitud_val": 0.5, "epocas": 1.0}
'''


def test_buscar_en_proceso_atrapa_metrica_no_finita_y_sigue(tmp_path, monkeypatch):
    """`buscar_en_proceso.buscar()` pasa `atrapar=(MetricaNoFinita,)` a
    `optimizar()`: esta prueba comprueba que de verdad funciona (un NaN marca
    el trial FAIL y la búsqueda sigue), no solo que el argumento esté escrito.
    Con n_total=1, la línea base (primera
    llamada a entrenar) falla con NaN y queda FAIL; optimizar() reintenta con
    un trial nuevo (TPE), que sí completa -- el estudio termina con 1 FAIL +
    1 COMPLETE, sin propagar MetricaNoFinita. El `train.py` de prueba declara
    `DEFAULTS['dispositivo'] = 'cuda'` (obligatorio, ver
    `_dispositivo_resuelto_de_modulo`) pero su `entrenar` no toca torch para
    nada: esta prueba no necesita una tarjeta de video real, así que fuerza
    `torch.cuda.is_available` a `True` para pasar esa comprobación sin
    depender de la máquina que corre la suite."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    train_path = tmp_path / "train_falla_una_vez.py"
    train_path.write_text(_TRAIN_FALLA_UNA_VEZ, encoding="utf-8")
    study = buscar_en_proceso.buscar(
        n_total=1, destino=tmp_path / "estudio.log", train_path=train_path
    )
    fallidos = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    completados = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(fallidos) == 1, study.trials
    assert len(completados) == 1, study.trials


# --- main(): "ya tenía N trials" mide lo que había, no lo que se pidió ----------


@pytest.mark.gpu
def test_en_proceso_main_ya_tenia_dice_lo_que_habia_no_lo_que_se_pidio(tmp_path):
    """`main()` debe reportar cuántos trials tenía de verdad el estudio (lo
    que devuelve `_trials_terminados` antes de llamar a `buscar()`), no el
    presupuesto PEDIDO en la llamada que no corrió nada -- son números
    distintos en cuanto el estudio ya tiene más trials que los que se piden
    en esta llamada. Se siembra el estudio con 3 trials DIRECTO (buscar(), en
    proceso,
    barato) y se reabre `main()` por subprocess con PRESUPUESTO=humo
    (n_total=2 < 3): el mensaje debe decir "ya tenía 3", no "ya tenía 2"."""
    estudios = tmp_path / "estudios"
    buscar_en_proceso.buscar(n_total=3, destino=estudios / "buscar_en_proceso.log")

    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_ESTUDIOS": str(estudios)}
    resultado = subprocess.run(
        [sys.executable, str(BUSCAR_EN_PROCESO_PY)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "ya tenía 3 trials terminados" in resultado.stdout, resultado.stdout
    assert "pediste 2" in resultado.stdout, resultado.stdout
    assert "trials nuevos en" not in resultado.stdout, resultado.stdout


@pytest.mark.gpu
def test_por_subproceso_main_ya_tenia_dice_lo_que_habia_no_lo_que_se_pidio(tmp_path):
    """Misma prueba que la de arriba, para buscar_por_subproceso.py."""
    estudios = tmp_path / "estudios"
    buscar_por_subproceso.buscar(
        n_total=3,
        destino=estudios / "buscar_por_subproceso.log",
        directorio_trials=estudios / "buscar_por_subproceso_trials",
        timeout_s=30.0,
    )

    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_ESTUDIOS": str(estudios)}
    resultado = subprocess.run(
        [sys.executable, str(BUSCAR_POR_SUBPROCESO_PY)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "ya tenía 3 trials terminados" in resultado.stdout, resultado.stdout
    assert "pediste 2" in resultado.stdout, resultado.stdout
    assert "trials nuevos en" not in resultado.stdout, resultado.stdout


# --- misma config, misma métrica por las dos rutas -------------------------------


@pytest.mark.gpu
def test_en_proceso_y_por_subproceso_dan_la_misma_metrica_con_la_misma_config(tmp_path):
    """Las dos plantillas comparten `espacio` (misma secuencia de `suggest_*`)
    y `SEMILLAS_BUSQUEDA`: con el MISMO sampler sembrado, deben sugerir la
    MISMA config y, con `train_ejemplo.entrenar` determinista, dar el MISMO
    valor -- la comparación real que hace `pasos/07_conecta_tu_modelo.py`,
    aquí aislada por plantilla en vez de por script completo."""
    modulo_train = buscar_en_proceso._importar_train_ejemplo()
    fijos = {
        clave: modulo_train.DEFAULTS[clave] for clave in ("n_train", "n_val", "ruido", "epocas")
    }
    assert buscar_en_proceso.SEMILLAS_BUSQUEDA == buscar_por_subproceso.SEMILLAS_BUSQUEDA

    objetivo_en_proceso = en_proceso(
        modulo_train.entrenar,
        buscar_en_proceso.espacio,
        metricas=["exactitud_val"],
        semillas=buscar_en_proceso.SEMILLAS_BUSQUEDA,
        fijos=fijos,
    )
    study_en_proceso = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    study_en_proceso.optimize(objetivo_en_proceso, n_trials=1)

    objetivo_por_subproceso = por_subproceso(
        [sys.executable, str(buscar_por_subproceso.TRAIN_EJEMPLO)],
        buscar_por_subproceso.espacio,
        metricas=["exactitud_val"],
        semillas=buscar_por_subproceso.SEMILLAS_BUSQUEDA,
        timeout_s=30.0,
        directorio=tmp_path,
        fijos=fijos,
    )
    study_por_subproceso = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11))
    study_por_subproceso.optimize(objetivo_por_subproceso, n_trials=1)

    assert study_en_proceso.trials[0].params == study_por_subproceso.trials[0].params
    assert study_en_proceso.trials[0].value == study_por_subproceso.trials[0].value


# --- TALLER_PYTHON: las dos direcciones -----------------------------------------


@pytest.mark.gpu
def test_buscar_por_subproceso_sin_taller_python_usa_sys_executable(tmp_path, monkeypatch):
    monkeypatch.delenv("TALLER_PYTHON", raising=False)
    # No debe reventar: el default (sys.executable, el python de este repo)
    # corre train_ejemplo.py sin problema. Cubre la misma ruta que ya ejercitan
    # las pruebas de "produce un estudio" de arriba; se deja explícita para que
    # el par con la prueba de abajo (TALLER_PYTHON apuntando a un binario que
    # no existe) sea evidente.
    study = buscar_por_subproceso.buscar(
        n_total=1, destino=tmp_path / "e.log", directorio_trials=tmp_path / "t", timeout_s=30.0
    )
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


def test_buscar_por_subproceso_taller_python_bogus_falla_al_resolver_el_dispositivo(
    tmp_path, monkeypatch
):
    """TALLER_PYTHON SÍ se está leyendo: con un binario que no existe,
    `_resolver_dispositivo` -- que SIEMPRE lanza el intérprete que va a
    entrenar para comprobar que ve la tarjeta de video, incluso con
    `dispositivo="cuda"` ya concreto -- revienta primero, con un `ValueError`
    claro sobre TALLER_PYTHON, en vez de dejar pasar el valor concreto sin
    comprobar nada: eso dejaría que el error de verdad (`FileNotFoundError`
    de `Popen`) solo apareciera más tarde, al lanzar el primer trial -- un
    diagnóstico peor para el mismo problema."""
    monkeypatch.setenv("TALLER_PYTHON", str(tmp_path / "no_existe_python_aqui"))
    with pytest.raises(ValueError, match="el taller no pudo lanzar el intérprete"):
        buscar_por_subproceso.buscar(
            n_total=1,
            destino=tmp_path / "e.log",
            directorio_trials=tmp_path / "t",
            timeout_s=10.0,
            dispositivo="cuda",
        )
    assert not (tmp_path / "e.log").exists()


# --- _resolver_dispositivo: no pierde el índice de tarjeta de video, y valida su formato -----


@pytest.mark.gpu
def test_resolver_dispositivo_conserva_indice_pero_el_protocolo_lo_normaliza():
    """`_resolver_dispositivo` (con `dispositivo` explícito, ni `None` ni
    `'auto'`) conserva el índice de tarjeta de video: normalizarlo aquí haría
    que `buscar(dispositivo='cuda:7')` entrenara en `cuda:0` sin decirlo. La
    normalización vive aparte, en `_normalizar_dispositivo_protocolo`, y solo
    aplica al protocolo del estudio (ver
    test_buscar_por_subproceso_cuda_con_indice_invalido_falla_en_el_hijo_no_en_cuda_0
    para la prueba de integración con un índice real inválido). Esta prueba
    necesita una tarjeta de video real: aunque el valor ya sea concreto,
    `_resolver_dispositivo` SIEMPRE lanza un intérprete a comprobar que la
    ve (ver `_detectar_dispositivo`), nunca salta esa comprobación."""
    assert buscar_por_subproceso._resolver_dispositivo("cuda:7", sys.executable) == "cuda:7"
    assert buscar_por_subproceso._normalizar_dispositivo_protocolo("cuda:7") == "cuda"
    assert buscar_por_subproceso._normalizar_dispositivo_protocolo("cuda") == "cuda"


def test_resolver_dispositivo_formato_invalido_falla_antes_de_lanzar_ningun_interprete():
    """`'cudaa'` y `'cpu'` fallan por FORMATO, en el chequeo del patrón --
    antes de que `_resolver_dispositivo` llegue a lanzar ningún intérprete a
    comprobar la tarjeta de video, así que esta prueba no necesita una
    tarjeta de video real. `'cpu'` es un valor inválido como cualquier otro
    nombre fuera de `auto`/`cuda(:N)`."""
    with pytest.raises(ValueError, match="dispositivo"):
        buscar_por_subproceso._resolver_dispositivo("cudaa", sys.executable)
    with pytest.raises(ValueError, match="dispositivo"):
        buscar_por_subproceso._resolver_dispositivo("cpu", sys.executable)


@pytest.mark.gpu
def test_resolver_dispositivo_taller_dispositivo_valor_concreto_se_respeta_pero_se_comprueba(
    monkeypatch,
):
    """TALLER_DISPOSITIVO con un valor concreto (aquí `'cuda:1'`) elige QUÉ
    tarjeta usar, con su índice intacto -- pero no salta la comprobación de
    que el intérprete la ve de verdad (ver `_detectar_dispositivo`): esta
    prueba SÍ necesita una tarjeta de video real, porque
    `_resolver_dispositivo` la lanza siempre, aunque el valor ya sea
    concreto. Sin esta comprobación, `buscar_por_subproceso.py` aceptaría un
    valor concreto sin comprobar nada, crearía el estudio y lanzaría 11
    trials fallidos con TALLER_DISPOSITIVO=cuda y sin tarjeta de video
    visible."""
    monkeypatch.setenv("TALLER_DISPOSITIVO", "cuda:1")
    assert buscar_por_subproceso._resolver_dispositivo("auto", sys.executable) == "cuda:1"


@pytest.mark.gpu
def test_resolver_dispositivo_taller_dispositivo_auto_no_llega_intacto(monkeypatch):
    """`TALLER_DISPOSITIVO=auto` no es un valor concreto: no debe llegar tal
    cual a `fijos` ni al protocolo del estudio (contra "nunca 'auto' en la
    huella" -- ver `buscar_en_proceso.py`), así que `_resolver_dispositivo`
    debe seguir detectando con el intérprete (necesita una tarjeta de video
    real ahí, de ahí la marca `gpu`) y devolver `'cuda'` resuelto, nunca el
    string `'auto'` literal."""
    monkeypatch.setenv("TALLER_DISPOSITIVO", "auto")
    resuelto = buscar_por_subproceso._resolver_dispositivo("auto", sys.executable)
    assert resuelto == "cuda", resuelto


_BLOQUEADOR_TORCH_INTERPRETE = f"""#!{sys.executable}
import sys


class _BloqueaTorch:
    def find_spec(self, nombre, ruta=None, destino=None):
        if nombre == "torch" or nombre.startswith("torch."):
            raise ModuleNotFoundError("torch bloqueado para la prueba", name=nombre)
        return None


sys.meta_path.insert(0, _BloqueaTorch())
if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    exec(sys.argv[2])
else:
    raise SystemExit("uso: python_sin_torch -c <codigo>")
"""


def _escribir_interprete_sin_torch(tmp_path: Path) -> Path:
    """Un "intérprete" que es el mismo `sys.executable` (vía shebang) con
    torch bloqueado por un `MetaPathFinder` (nunca
    `sys.modules["torch"] = None`, que rompería otra cosa distinta). Sirve
    para pasarlo como `python` a
    `_resolver_dispositivo`/`_detectar_dispositivo` y simular "el intérprete
    que va a entrenar no tiene torch" sin depender de
    que la máquina que corre la suite tenga un python de sistema sin torch."""
    ruta = tmp_path / "python_sin_torch"
    ruta.write_text(_BLOQUEADOR_TORCH_INTERPRETE, encoding="utf-8")
    ruta.chmod(0o755)
    return ruta


@pytest.mark.skipif(sys.platform == "win32", reason="el intérprete simulado usa un shebang POSIX")
def test_resolver_dispositivo_sin_torch_en_el_hijo_falla_con_o_sin_taller_dispositivo(
    tmp_path, monkeypatch
):
    """`_resolver_dispositivo(None, python)` detecta la tarjeta de video con
    el intérprete que de verdad va a entrenar (`_detectar_dispositivo`),
    NUNCA con el proceso padre que orquesta Optuna: consultar al padre daría
    la tarjeta de video del padre, no la del hijo, y con `TALLER_PYTHON`
    apuntando a otro entorno el dispositivo saldría mal en las dos
    direcciones (un padre con tarjeta de video y un hijo sin ella entrenaría
    con `'cuda'` y el trial fallaría de verdad al llegar ahí; un padre sin
    tarjeta de video y un hijo que sí la tiene se detendría aquí mismo, con
    un mensaje claro, en vez de dejar que el hijo entrene sin que nadie lo
    note). Con el intérprete simulado sin torch, la comprobación revienta con
    un `ValueError` claro tanto SIN `TALLER_DISPOSITIVO` (dispositivo='auto',
    se detecta) como CON ella definida a un valor YA concreto ('cuda:3'): la
    comprobación de que el intérprete hijo ve la tarjeta de video corre
    SIEMPRE, nunca se salta por tener ya un valor concreto: sin esa
    comprobación, el segundo caso devolvería 'cuda:3' sin comprobar nada."""
    python_sin_torch = str(_escribir_interprete_sin_torch(tmp_path))
    monkeypatch.delenv("TALLER_DISPOSITIVO", raising=False)
    with pytest.raises(ValueError, match="torch"):
        buscar_por_subproceso._resolver_dispositivo(None, python_sin_torch)

    monkeypatch.setenv("TALLER_DISPOSITIVO", "cuda:3")
    with pytest.raises(ValueError, match="torch"):
        buscar_por_subproceso._resolver_dispositivo(None, python_sin_torch)


def _escribir_wrapper_shell(tmp_path: Path, nombre: str, cuerpo: str) -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(cuerpo, encoding="utf-8")
    ruta.chmod(0o755)
    return ruta


@pytest.mark.skipif(sys.platform == "win32", reason="el wrapper usa un shebang #!/bin/sh")
def test_buscar_dispositivo_auto_falla_claro_si_el_interprete_hijo_no_ve_la_tarjeta(
    tmp_path, monkeypatch
):
    """`TALLER_PYTHON` apunta a un wrapper que oculta la tarjeta de video SOLO
    para el hijo (`CUDA_VISIBLE_DEVICES=''`), sin importar lo que vea el padre
    (este proceso de pytest) -- por eso esta prueba NO necesita una tarjeta de
    video real, corre igual en cualquier máquina. Consultar torch en el PADRE
    en vez de en el hijo fijaría `'dispositivo': 'cuda'` para un hijo que no
    ve la tarjeta de video, y los trials fallarían tarde, con `RuntimeError:
    No CUDA GPUs are available` en vez de un mensaje claro sobre POR QUÉ.
    Detectando con el intérprete del hijo (ver `_detectar_dispositivo`), la
    resolución falla aquí mismo, con un `RuntimeError` explícito, ANTES de
    crear ningún trial. Esta prueba compara la cadena EXACTA de `MENSAJE_SIN_TARJETA`
    (con `.startswith`, porque `_detectar_dispositivo` le agrega detalle del
    intérprete): un `RuntimeError` de otra parte del taller (por ejemplo, de
    `optimizar` agotando `max_fallos`) también tiene ese tipo, y un `match`
    laxo sobre "tarjeta de video" no distinguiría los dos."""
    wrapper = _escribir_wrapper_shell(
        tmp_path,
        "wrapper_sin_cuda.sh",
        f"#!/bin/sh\nCUDA_VISIBLE_DEVICES='' exec {shlex.quote(sys.executable)} \"$@\"\n",
    )
    monkeypatch.setenv("TALLER_PYTHON", str(wrapper))
    monkeypatch.delenv("TALLER_DISPOSITIVO", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        buscar_por_subproceso.buscar(
            n_total=1, destino=tmp_path / "e.log", directorio_trials=tmp_path / "t", timeout_s=30.0
        )
    assert str(excinfo.value).startswith(MENSAJE_SIN_TARJETA), excinfo.value
    assert not (tmp_path / "e.log").exists()


def test_taller_dispositivo_cuda_sin_tarjeta_falla_de_entrada_sin_dejar_journal(tmp_path):
    """TALLER_DISPOSITIVO=cuda (un valor YA CONCRETO) solo elige qué tarjeta
    usar -- nunca salta la comprobación de que el intérprete hijo la ve de
    verdad (ver `_resolver_dispositivo`). Esta prueba corre el script
    COMPLETO por subproceso (no solo `buscar()` en proceso) con
    `CUDA_VISIBLE_DEVICES=''` para forzar la ausencia de tarjeta de video de
    forma portable, sin wrapper de shell -- por eso no necesita una tarjeta
    de video real ni se salta en Windows. Sin esta comprobación, este caso
    saltaría la validación, crearía el estudio y lanzaría varios trials que
    fallarían uno por uno hasta agotar `max_fallos`; el script debe detenerse
    de entrada, con el mensaje claro, sin llegar a crear el estudio ni su
    Journal."""
    estudios = tmp_path / "estudios"
    entorno = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "TALLER_DISPOSITIVO": "cuda",
        "PRESUPUESTO": "humo",
        "TALLER_ESTUDIOS": str(estudios),
    }
    resultado = subprocess.run(
        [sys.executable, str(BUSCAR_POR_SUBPROCESO_PY)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0
    assert MENSAJE_SIN_TARJETA in (resultado.stdout + resultado.stderr)
    assert not (estudios / "buscar_por_subproceso.log").exists()


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform == "win32", reason="el wrapper usa un shebang #!/bin/sh")
def test_buscar_dispositivo_auto_usa_el_interprete_hijo_no_el_padre_con_cuda(tmp_path):
    """Caso inverso al de arriba: el padre que orquesta Optuna no ve la tarjeta de video
    (`CUDA_VISIBLE_DEVICES=''`), pero `TALLER_PYTHON` apunta a un wrapper que la restaura
    para el hijo. Consultar al padre en vez de al hijo detendría la búsqueda aunque el hijo
    sí tenga tarjeta.

    `buscar()` corre en un intérprete nuevo, lanzado sin tarjeta visible desde el inicio:
    con `monkeypatch.setenv` sobre pytest no bastaría, porque la marca `gpu` ya inicializó
    CUDA en este proceso. Así la prueba distingue una implementación correcta de una que
    consultara al padre."""
    wrapper = _escribir_wrapper_shell(
        tmp_path,
        "wrapper_con_cuda.sh",
        f'#!/bin/sh\nexec env -u CUDA_VISIBLE_DEVICES {shlex.quote(sys.executable)} "$@"\n',
    )
    destino = tmp_path / "e.log"
    directorio_trials = tmp_path / "t"
    script = f"""
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_buscar_por_subproceso_subproc", {str(BUSCAR_POR_SUBPROCESO_PY)!r}
)
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
study = modulo.buscar(
    n_total=1,
    destino={str(destino)!r},
    directorio_trials={str(directorio_trials)!r},
    timeout_s=30.0,
)
print("DISPOSITIVO=" + study.user_attrs["protocolo"]["fijos"]["dispositivo"])
print("ESTADO=" + study.trials[0].state.name)
"""
    entorno = {**os.environ, "TALLER_PYTHON": str(wrapper), "CUDA_VISIBLE_DEVICES": ""}
    entorno.pop("TALLER_DISPOSITIVO", None)
    resultado = subprocess.run(
        [sys.executable, "-c", script], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "DISPOSITIVO=cuda" in resultado.stdout, resultado.stdout
    assert "ESTADO=COMPLETE" in resultado.stdout, resultado.stdout


@pytest.mark.gpu
def test_buscar_por_subproceso_cuda_con_indice_invalido_falla_en_el_hijo_no_en_cuda_0(tmp_path):
    """Un `'cuda:N'` con N fuera de rango debe fallar de VERDAD en el hijo
    (torch revienta con un índice de tarjeta de video inválido), nunca entrenar en
    silencio en `cuda:0` -- lo que pasaría si `_resolver_dispositivo`
    recortara el índice antes de que llegue a `fijos` (ver
    `_normalizar_dispositivo_protocolo`, que sí lo recorta pero solo para el
    protocolo). `max_fallos=0`: el dispositivo fijo hace que CUALQUIER
    reintento falle igual, así que un solo trial ya prueba el punto -- no
    hace falta pagar los 11 arranques de intérprete del `max_fallos=10` por
    defecto. `pytest.raises(RuntimeError)` solo no bastaría: pasaría igual
    si el hijo fallara por cualquier OTRA causa (un timeout, un JSON
    corrupto); esta prueba lee `trial_0_stderr.txt` y afirma el mensaje real
    de torch. El protocolo, en cambio, sí debe decir 'cuda' (normalizado, sin
    índice)."""
    indice_invalido = f"cuda:{torch.cuda.device_count() + 5}"
    destino = tmp_path / "estudio.log"
    directorio_trials = tmp_path / "trials"
    with pytest.raises(RuntimeError):
        buscar_por_subproceso.buscar(
            n_total=1,
            destino=destino,
            directorio_trials=directorio_trials,
            timeout_s=30.0,
            dispositivo=indice_invalido,
            max_fallos=0,
        )
    stderr_trial_0 = (directorio_trials / "trial_0_stderr.txt").read_text(encoding="utf-8")
    assert "invalid device ordinal" in stderr_trial_0, stderr_trial_0
    reabierto = optuna.load_study(
        study_name="buscar_por_subproceso", storage=buscar_por_subproceso.abrir_storage(destino)
    )
    assert reabierto.user_attrs["protocolo"]["fijos"]["dispositivo"] == "cuda"


# --- buscar_en_proceso: un train.py sin la clave 'dispositivo' obligatoria -----

_TRAIN_SIN_DISPOSITIVO = '''"""train.py sin la clave 'dispositivo' en DEFAULTS: no cumple el
contrato de plantilla/README.md, que la exige siempre -- el taller entrena
siempre en la tarjeta de video, así que ningún train.py (ni uno de numpy
puro, como este) puede quedar fuera de esa comprobación."""

DEFAULTS = {"ancho": 16, "learning_rate_init": 1e-2, "alpha": 1e-4}


def get_config(defaults, **overrides):
    desconocidas = sorted(set(overrides) - set(defaults))
    if desconocidas:
        raise KeyError(f"claves desconocidas: {desconocidas}")
    return {**defaults, **overrides}


def entrenar(config, semillas, reportar=None):
    c = get_config(DEFAULTS, **config)
    return {"exactitud_val": 0.5 + 0.0 * c["ancho"], "epocas": 1.0}


def main(argv=None):
    pass
'''


def test_buscar_en_proceso_train_py_sin_clave_dispositivo_falla_de_entrada(tmp_path):
    """El contrato de `plantilla/README.md` exige `'dispositivo'` en
    `DEFAULTS` para cualquier `train.py`: la clave obligatoria deja que
    `buscar_en_proceso.buscar()` compruebe la tarjeta de video antes de
    crear el estudio; entrenar en ella es trabajo de tu `entrenar`. Por eso
    el script se detiene de entrada, ANTES de crear el estudio, si tu
    `train.py` no declara la clave. Esta prueba no necesita una tarjeta de
    video real: el `train.py` de prueba ni siquiera importa torch, y el
    chequeo de la clave corre antes de tocar cualquier dispositivo."""
    train_path = tmp_path / "train_sin_dispositivo.py"
    train_path.write_text(_TRAIN_SIN_DISPOSITIVO, encoding="utf-8")
    destino = tmp_path / "estudio.log"
    with pytest.raises(ValueError, match="dispositivo"):
        buscar_en_proceso.buscar(n_total=1, destino=destino, train_path=train_path)
    assert not destino.exists()


def test_buscar_en_proceso_main_sin_tarjeta_falla_de_entrada_sin_dejar_journal(tmp_path):
    """`main()` importa `train_ejemplo.py` y resuelve su dispositivo ANTES de
    llamar a `_trials_terminados` (que abre el Journal, aunque el estudio no
    exista todavía): sin ese orden, un `train.py` sin tarjeta de video
    dejaría un Journal VACÍO a medio camino en vez de detenerse de entrada.
    Esta prueba corre el script COMPLETO por subproceso (no solo `buscar()`
    en proceso) con `CUDA_VISIBLE_DEVICES=''` para forzar la ausencia de
    tarjeta de video de forma portable, sin wrapper de shell -- por eso no
    necesita una tarjeta de video real ni se salta en Windows. `main()` debe
    detenerse de entrada, con el mensaje claro, sin llegar a crear el
    Journal."""
    estudios = tmp_path / "estudios"
    entorno = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PRESUPUESTO": "humo",
        "TALLER_ESTUDIOS": str(estudios),
    }
    resultado = subprocess.run(
        [sys.executable, str(BUSCAR_EN_PROCESO_PY)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0
    assert MENSAJE_SIN_TARJETA in (resultado.stdout + resultado.stderr)
    assert not (estudios / "buscar_en_proceso.log").exists()


@pytest.mark.gpu
def test_detectar_dispositivo_se_llama_una_sola_vez_en_main(tmp_path, monkeypatch):
    """Sin el argumento `_dispositivo_ya_comprobado`, `_resolver_dispositivo`
    lanzaría `_detectar_dispositivo` -- un intérprete nuevo que importa
    torch -- dos veces por cada llamada a `main()`: una explícita, ANTES de
    arrancar el reloj (ver su comentario), y otra DENTRO de `buscar()`, ya
    adentro del intervalo que reporta 's/trial'. Esta prueba llama a
    `main()` en ESTE mismo proceso (nunca por subprocess: así el espía puede
    contar las llamadas) con PRESUPUESTO=humo, y cuenta las invocaciones
    reales de `_detectar_dispositivo` con un espía sobre el atributo del
    módulo: debe haber UNA sola, la que corre en `main()` antes de
    `_trials_terminados` y de arrancar el reloj -- `buscar()`, avisada con
    `_dispositivo_ya_comprobado=True`, no vuelve a llamarla."""
    monkeypatch.setenv("PRESUPUESTO", "humo")
    monkeypatch.setenv("TALLER_ESTUDIOS", str(tmp_path))
    monkeypatch.delenv("TALLER_PYTHON", raising=False)
    monkeypatch.delenv("TALLER_DISPOSITIVO", raising=False)

    llamadas: list[str] = []
    original = buscar_por_subproceso._detectar_dispositivo

    def espia(python):
        llamadas.append(python)
        return original(python)

    monkeypatch.setattr(buscar_por_subproceso, "_detectar_dispositivo", espia)
    buscar_por_subproceso.main()
    assert len(llamadas) == 1, llamadas


# --- línea de comandos: PRESUPUESTO -----------------------------------------------


@pytest.mark.parametrize(
    "script", [BUSCAR_EN_PROCESO_PY, BUSCAR_POR_SUBPROCESO_PY], ids=["en_proceso", "por_subproceso"]
)
def test_presupuesto_invalido_falla_con_mensaje(script, tmp_path):
    entorno = {**os.environ, "PRESUPUESTO": "enorme", "TALLER_ESTUDIOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(script)], env=entorno, capture_output=True, text=True, timeout=30
    )
    assert resultado.returncode != 0
    assert "PRESUPUESTO debe ser uno de" in resultado.stderr


@pytest.mark.gpu
@pytest.mark.parametrize(
    "script", [BUSCAR_EN_PROCESO_PY, BUSCAR_POR_SUBPROCESO_PY], ids=["en_proceso", "por_subproceso"]
)
def test_main_corre_con_humo_e_imprime_mejor_config_y_advertencia(script, tmp_path):
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_ESTUDIOS": str(tmp_path)}
    resultado = subprocess.run(
        [sys.executable, str(script)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "trials nuevos en" in resultado.stdout
    assert "mejor exactitud_val" in resultado.stdout
    assert "advertencia" in resultado.stdout
    assert "(maldición del ganador)" in resultado.stdout


@pytest.mark.gpu
@pytest.mark.parametrize(
    "script", [BUSCAR_EN_PROCESO_PY, BUSCAR_POR_SUBPROCESO_PY], ids=["en_proceso", "por_subproceso"]
)
def test_main_segunda_corrida_no_repite_trabajo_y_lo_dice(script, tmp_path):
    """`optimizar()` reanuda un estudio existente sin repetir trabajo (ver
    `taller_hpo.estudio`): con el MISMO TALLER_ESTUDIOS, una segunda corrida
    de `main()` no debe fingir que corrió `n_total` trials nuevos -- dividir
    el tiempo de reloj de una llamada que no entrenó nada entre `n_total`
    reportaría un costo por trial que no midió nada."""
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_ESTUDIOS": str(tmp_path)}
    primera = subprocess.run(
        [sys.executable, str(script)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert primera.returncode == 0, primera.stderr
    assert "2 trials nuevos en" in primera.stdout, primera.stdout

    segunda = subprocess.run(
        [sys.executable, str(script)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert segunda.returncode == 0, segunda.stderr
    assert "el estudio ya tenía 2 trials" in segunda.stdout, segunda.stdout
    assert "nada que correr" in segunda.stdout, segunda.stdout
    assert "trials nuevos en" not in segunda.stdout, segunda.stdout


@pytest.mark.gpu
@pytest.mark.parametrize(
    "modulo,script",
    [(buscar_en_proceso, BUSCAR_EN_PROCESO_PY), (buscar_por_subproceso, BUSCAR_POR_SUBPROCESO_PY)],
    ids=["en_proceso", "por_subproceso"],
)
def test_main_con_un_solo_trial_nuevo_usa_singular(modulo, script, tmp_path):
    """`main()` usa el mismo `_pluralizar` que `pasos/07_conecta_tu_modelo.py`
    (ver su docstring): con exactamente 1 trial nuevo la salida debe decir
    'trial nuevo', nunca el '1 trials nuevos' agramatical. Se siembra el
    estudio con 1 trial de menos que el `n_total` de humo (2 en las dos
    plantillas), así la corrida por subproceso de más abajo completa
    exactamente 1."""
    estudios = tmp_path / "estudios"
    if modulo is buscar_en_proceso:
        modulo.buscar(n_total=1, destino=estudios / "buscar_en_proceso.log")
    else:
        modulo.buscar(
            n_total=1,
            destino=estudios / "buscar_por_subproceso.log",
            directorio_trials=estudios / "buscar_por_subproceso_trials",
            timeout_s=30.0,
        )
    entorno = {**os.environ, "PRESUPUESTO": "humo", "TALLER_ESTUDIOS": str(estudios)}
    resultado = subprocess.run(
        [sys.executable, str(script)], env=entorno, capture_output=True, text=True, timeout=60
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "1 trial nuevo en" in resultado.stdout, resultado.stdout
    assert "1 trials nuevos" not in resultado.stdout, resultado.stdout
