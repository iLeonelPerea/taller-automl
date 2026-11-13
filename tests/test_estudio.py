"""taller_hpo.estudio -- guardas sobre Study/Trial reales de Optuna.

`destino` esta parametrizado sobre JournalStorage (un archivo en tmp_path) y
sqlite (RDBStorage vía URL): las pruebas del contrato de `crear_estudio` corren
sobre los dos backends, porque el codigo no deberia depender de cual es.
"""

from __future__ import annotations

import errno
import os
import subprocess
import sys

import numpy as np
import optuna
import pytest

from taller_hpo import estudio

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sampler(seed: int = 0) -> optuna.samplers.BaseSampler:
    return optuna.samplers.RandomSampler(seed=seed)


def _pruner() -> optuna.pruners.BasePruner:
    return optuna.pruners.NopPruner()


def _estudio_simple(tmp_path, nombre: str = "e") -> optuna.Study:
    return estudio.crear_estudio(
        nombre,
        destino=tmp_path / f"{nombre}.log",
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )


@pytest.fixture(params=["journal", "sqlite"])
def destino(tmp_path, request):
    if request.param == "journal":
        return tmp_path / "estudio.log"
    return f"sqlite:///{tmp_path}/estudio.db"


# --- abrir_storage ---


def test_abrir_storage_reconoce_urls_rdb_tal_cual():
    assert estudio.abrir_storage("sqlite:///x.db") == "sqlite:///x.db"
    assert estudio.abrir_storage("postgresql://u@h/db") == "postgresql://u@h/db"
    assert estudio.abrir_storage("postgresql+psycopg://u@h/db") == "postgresql+psycopg://u@h/db"


def test_abrir_storage_ruta_de_archivo_crea_carpeta_padre_y_journalstorage(tmp_path):
    ruta = tmp_path / "sub" / "dir" / "estudio.log"
    assert not ruta.parent.exists()
    storage = estudio.abrir_storage(ruta)
    assert isinstance(storage, optuna.storages.JournalStorage)
    assert ruta.parent.exists()
    assert ruta.exists()


def test_abrir_storage_usa_openlock_y_no_el_symlinklock_por_defecto(tmp_path, monkeypatch):
    """JournalFileBackend sin lock_obj usa JournalFileSymlinkLock, cuyo
    acquire() llama a os.symlink en cada intento. En Windows eso falla con
    PermissionError ("A required privilege is not held by the client",
    documentado en el propio docstring de JournalStorage). Si abrir_storage no
    pasara lock_obj=JournalFileOpenLock (que abre un archivo, no crea un
    symlink), este test fallaria con ese mismo error aunque corra en Linux: por
    eso se simula la falla de os.symlink en vez de depender del SO real.
    """

    def _symlink_falla(*args, **kwargs):
        raise OSError(errno.EPERM, "A required privilege is not held by the client")

    monkeypatch.setattr(os, "symlink", _symlink_falla)

    study = estudio.crear_estudio(
        "e",
        destino=tmp_path / "e.log",
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objetivo, n_trials=2)
    estados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    assert len(study.get_trials(deepcopy=False, states=estados)) == 2


# --- crear_estudio: contrato de sampler/pruner sin default, y de reapertura ---


def test_crear_estudio_sin_sampler_lanza_typeerror(destino):
    with pytest.raises(TypeError):
        estudio.crear_estudio(
            "e", destino=destino, direcciones=["maximize"], pruner=_pruner(), protocolo={}
        )


def test_crear_estudio_sin_pruner_lanza_typeerror(destino):
    with pytest.raises(TypeError):
        estudio.crear_estudio(
            "e", destino=destino, direcciones=["maximize"], sampler=_sampler(), protocolo={}
        )


def test_reabrir_con_mismo_protocolo_y_direcciones_no_falla(destino):
    protocolo = {"metrica": "exactitud_val", "sampler": "random"}
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    reabierto = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    assert reabierto.user_attrs["protocolo"] == protocolo
    assert reabierto.directions == [optuna.study.StudyDirection.MAXIMIZE]


def test_reabrir_con_protocolo_distinto_lanza_valueerror_con_detalle(destino):
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={"metrica": "exactitud_val"},
    )
    with pytest.raises(ValueError, match="protocolo") as excinfo:
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={"metrica": "otra"},
        )
    assert "exactitud_val" in str(excinfo.value)
    assert "otra" in str(excinfo.value)
    assert "pediste {'metrica': 'otra'}" in str(excinfo.value)


def test_crear_estudio_direcciones_como_str_suelto_lanza_typeerror(destino):
    # "maximize" es un Sequence[str]: sin la guarda, list("maximize") lo parte
    # en las letras 'm', 'a', 'x', ... y Optuna falla con un mensaje confuso.
    with pytest.raises(TypeError, match="direcciones"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones="maximize",
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={},
        )


def test_direccion_not_set_lanza_valueerror(destino):
    # StudyDirection tiene un miembro NOT_SET que no es una direccion valida:
    # hasattr(StudyDirection, "NOT_SET") es True, asi que un chequeo por hasattr
    # lo dejaria pasar.
    with pytest.raises(ValueError, match="dirección inválida"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["not_set"],
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={},
        )


def test_reabrir_con_el_mismo_protocolo_con_tupla_no_falla(destino):
    # El storage persistente (journal/sqlite) serializa user_attrs a JSON: al
    # releer, una tupla guardada vuelve como lista. Sin normalizar TAMBIEN el
    # protocolo recien pedido antes de comparar, el mismo protocolo (literal, la
    # misma tupla) pasado dos veces se veia como "distinto" al reabrir, porque
    # el guardado (ya normalizado por el storage) dejaba de calzar con dict()
    # crudo del segundo pedido (que conserva la tupla).
    protocolo = {"sampler": "tpe", "rango_lr": (0.0001, 0.1)}
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    reabierto = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    assert reabierto.user_attrs["protocolo"] == {"sampler": "tpe", "rango_lr": [0.0001, 0.1]}


def test_reabrir_con_el_mismo_protocolo_con_clave_int_no_falla(destino):
    protocolo = {1: "a"}
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    reabierto = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    assert reabierto.user_attrs["protocolo"] == {"1": "a"}


def test_reabrir_con_protocolo_de_claves_mixtas_no_falla(tmp_path):
    """json.dumps(sort_keys=True) ordena las claves ANTES de convertirlas a str:

    con una clave int y otra str en el mismo dict, ese ordenamiento levanta
    TypeError ("'<' not supported between instances of 'str' and 'int'") aunque
    el protocolo sea perfectamente serializable a JSON. La comparacion al
    reabrir es por dict (!=), que ignora el orden, asi que sort_keys no hacia
    falta para nada.
    """
    destino = tmp_path / "e.log"
    protocolo = {1: "a", "b": 2}
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    reabierto = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    assert reabierto.user_attrs["protocolo"] == {"1": "a", "b": 2}


def test_crear_estudio_protocolo_no_serializable_lanza_typeerror_sin_crear_estudio(tmp_path):
    destino = tmp_path / "e.log"
    with pytest.raises(TypeError, match="serializable"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={"sampler": optuna.samplers.RandomSampler()},
        )
    # El estudio no debe quedar creado a medias: la normalizacion de protocolo
    # pasa ANTES de abrir el storage, asi que ni siquiera se toca el archivo.
    assert not destino.exists()


def test_crear_estudio_protocolo_con_nan_lanza_valueerror_sin_crear_estudio(tmp_path):
    destino = tmp_path / "e.log"
    with pytest.raises(ValueError, match="serializable"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={"epsilon": float("nan")},
        )
    assert not destino.exists()


def test_reabrir_con_direcciones_distintas_lanza_valueerror(destino):
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    with pytest.raises(ValueError, match="direcciones") as excinfo:
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["minimize"],
            sampler=_sampler(),
            pruner=_pruner(),
            protocolo={},
        )
    assert "maximize" in str(excinfo.value)
    assert "minimize" in str(excinfo.value)
    assert "tiene ['maximize'], pediste ['minimize']" in str(excinfo.value)


def test_crear_estudio_guarda_versiones_de_optuna_y_python(tmp_path):
    study = _estudio_simple(tmp_path)
    assert study.user_attrs["version_optuna"] == optuna.__version__
    assert study.user_attrs["version_python"]


# --- crear_estudio: sampler como instancia o como fabrica ---


def test_crear_estudio_con_fabrica_de_sampler_no_repite_params_al_reanudar(destino):
    """Reabrir con una FABRICA (n_previos) -> BaseSampler debe pedir, en la
    segunda sesion, un sampler sembrado con un seed DISTINTO del de la primera
    (aqui, base + n_previos). Esta prueba cierra esa condicion: reabrir con la
    MISMA instancia sembrada (RandomSampler(seed=0)) repetiria exactamente los
    primeros x de la sesion anterior -- el caso de uso real de "volver a
    correr el script" (paso 02) no avanzaria la busqueda.
    """
    base = 1000

    def fabrica(n_previos: int) -> optuna.samplers.BaseSampler:
        return optuna.samplers.RandomSampler(seed=base + n_previos)

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=fabrica,
        pruner=_pruner(),
        protocolo={},
    )
    estudio.optimizar(study, objetivo, n_total=3)
    x_sesion_1 = {t.params["x"] for t in study.trials}

    reabierto = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=fabrica,
        pruner=_pruner(),
        protocolo={},
    )
    estudio.optimizar(reabierto, objetivo, n_total=6)
    x_sesion_2 = {t.params["x"] for t in reabierto.trials if t.params["x"] not in x_sesion_1}

    assert len(x_sesion_1) == 3
    assert len(x_sesion_2) == 3
    assert x_sesion_1.isdisjoint(x_sesion_2)
    # el protocolo no lo toca la fabrica: sigue igual entre sesiones.
    assert reabierto.user_attrs["protocolo"] == {}


def test_crear_estudio_con_instancia_de_sampler_y_trials_existentes_avisa(destino):
    """Reabrir con una INSTANCIA (no una fabrica) cuando el estudio YA tiene
    trials debe avisar: si la instancia trae semilla fija, la sesion nueva
    reproduce la misma secuencia de numeros aleatorios que la anterior (por
    ejemplo, un RandomSampler(seed=5) reabierto repite el x de sus primeros
    trials).
    """
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(seed=5),
        pruner=_pruner(),
        protocolo={},
    )
    estudio.optimizar(study, lambda t: t.suggest_float("x", 0.0, 1.0), n_total=2)

    with pytest.warns(UserWarning, match="fábrica"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=_sampler(seed=5),
            pruner=_pruner(),
            protocolo={},
        )


def test_crear_estudio_con_instancia_de_sampler_sin_trials_no_avisa(destino, recwarn):
    # La primera creacion (sin trials todavia) no tiene nada que repetir: no
    # deberia avisar nunca, ni siquiera con una instancia sembrada.
    estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    avisos = [w for w in recwarn.list if issubclass(w.category, UserWarning)]
    assert avisos == []


def test_crear_estudio_sampler_no_es_basesampler_ni_fabrica_lanza_typeerror(destino):
    with pytest.raises(TypeError, match="sampler"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler="no_es_sampler",
            pruner=_pruner(),
            protocolo={},
        )


def test_crear_estudio_fabrica_que_no_devuelve_basesampler_lanza_typeerror(destino):
    with pytest.raises(TypeError, match="fábrica"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=lambda n: "no_es_sampler",
            pruner=_pruner(),
            protocolo={},
        )


# --- crear_estudio: protocolo ausente al reabrir (carrera de N workers) ---


def test_estudio_existente_sin_protocolo_y_sin_trials_registra_protocolo_al_reabrir(tmp_path):
    """Simula perder la carrera de N workers del anexo paralelo: alguien ya paso
    por optuna.create_study DIRECTO (sin pasar por crear_estudio) pero el
    proceso murio antes de registrar protocolo -- o, en la carrera real, otro
    worker gano create_study y este todavia no llego a su propio
    set_user_attr("protocolo", ...). Como el estudio no tiene trials todavia,
    crear_estudio debe completar el registro con el MISMO protocolo pedido, no
    fallar: eso es lo que hace idempotente la carrera.
    """
    destino = tmp_path / "e.log"
    storage = estudio.abrir_storage(destino)
    optuna.create_study(
        study_name="e", storage=storage, directions=[optuna.study.StudyDirection.MAXIMIZE]
    )

    protocolo = {"metrica": "exactitud_val"}
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo=protocolo,
    )
    assert study.user_attrs["protocolo"] == protocolo
    assert study.user_attrs["version_optuna"] == optuna.__version__
    assert study.user_attrs["version_python"]


def test_estudio_existente_sin_protocolo_pero_con_trials_lanza_valueerror(tmp_path):
    """Un estudio creado FUERA de crear_estudio que YA tiene trials, sin
    protocolo, no es una carrera: es un estudio contaminado. Registrar el
    protocolo en silencio le atribuiria un metodo de busqueda a trials que
    corrieron sin el.
    """
    destino = tmp_path / "e.log"
    storage = estudio.abrir_storage(destino)
    study_directo = optuna.create_study(
        study_name="e",
        storage=storage,
        directions=[optuna.study.StudyDirection.MAXIMIZE],
        sampler=_sampler(),
    )
    study_directo.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)

    with pytest.raises(ValueError, match="sin protocolo registrado"):
        estudio.crear_estudio(
            "e",
            destino=destino,
            direcciones=["maximize"],
            sampler=lambda n: optuna.samplers.RandomSampler(seed=n),
            pruner=_pruner(),
            protocolo={},
        )


# --- recuperar_huerfanos ---


def test_recuperar_huerfanos_sin_running_no_hace_nada(tmp_path):
    study = _estudio_simple(tmp_path)
    assert estudio.recuperar_huerfanos(study) == []


_SCRIPT_CORTE = """
import os
import sys

import optuna

from taller_hpo import estudio

destino = sys.argv[1]
nombre = sys.argv[2]
storage = estudio.abrir_storage(destino)
study = optuna.load_study(study_name=nombre, storage=storage)
trial = study.ask()
trial.suggest_float("x", 0.0, 1.0)
os._exit(1)
"""


def _correr_y_cortar(tmp_path, destino, nombre: str, nombre_script: str) -> None:
    """Corre `_SCRIPT_CORTE` en un subproceso que muere via os._exit a mitad de
    un trial, sin llegar nunca a study.tell(). Simula un corte real (kill -9,
    corte de luz): ni JournalStorage ni un RDBStorage sin heartbeat configurado
    notan la diferencia entre esto y un proceso que sigue vivo, asi que sin
    recuperar_huerfanos ese trial se queda RUNNING para siempre.
    """
    script = tmp_path / nombre_script
    script.write_text(_SCRIPT_CORTE, encoding="utf-8")
    resultado = subprocess.run(
        [sys.executable, str(script), str(destino), nombre],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode != 0, resultado.stderr


def test_recuperar_huerfanos_convierte_un_running_real_cortado_en_fail(tmp_path, destino):
    estudio.crear_estudio(
        "corte",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    _correr_y_cortar(tmp_path, destino, "corte", "corte_subproceso.py")

    # Al reanudar, fábrica de sampler (el camino recomendado): no repite ni avisa.
    study = estudio.crear_estudio(
        "corte",
        destino=destino,
        direcciones=["maximize"],
        sampler=lambda n: optuna.samplers.RandomSampler(seed=n),
        pruner=_pruner(),
        protocolo={},
    )
    antes = study.get_trials(deepcopy=False)
    assert len(antes) == 1
    assert antes[0].state == optuna.trial.TrialState.RUNNING

    recuperados = estudio.recuperar_huerfanos(study)
    assert recuperados == [0]

    despues = study.get_trials(deepcopy=False)
    assert despues[0].state == optuna.trial.TrialState.FAIL
    assert despues[1].state == optuna.trial.TrialState.WAITING
    assert despues[1].user_attrs["reintento_de"] == 0
    # El reencolado debe llevar los params REALES del huerfano, no un dict
    # vacio: un enqueue_trial({}, ...) tambien dejaria reintento_de correcto y
    # pasaria las demas afirmaciones de este test.
    assert despues[0].params  # el huerfano de verdad alcanzo a sugerir "x"
    assert despues[1].system_attrs["fixed_params"] == despues[0].params


def test_recuperar_huerfanos_preserva_rol_linea_base(tmp_path, destino):
    """Si el huerfano era la linea base, el reencolado debe conservar
    rol=linea_base ademas de reintento_de: sin eso, sembrar_linea_base no ve el
    trial reencolado como una linea base activa y encola una segunda copia,
    gastando un trial del presupuesto en un duplicado.
    """
    study = estudio.crear_estudio(
        "corte_lb",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True
    _correr_y_cortar(tmp_path, destino, "corte_lb", "corte_lb_subproceso.py")

    study = estudio.crear_estudio(
        "corte_lb",
        destino=destino,
        direcciones=["maximize"],
        sampler=lambda n: optuna.samplers.RandomSampler(seed=n),
        pruner=_pruner(),
        protocolo={},
    )
    recuperados = estudio.recuperar_huerfanos(study)
    assert recuperados == [0]

    trials = study.get_trials(deepcopy=False)
    assert len(trials) == 2
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[0].user_attrs["rol"] == "linea_base"
    assert trials[1].state == optuna.trial.TrialState.WAITING
    assert trials[1].user_attrs == {"rol": "linea_base", "reintento_de": 0}

    # El reintento debe reencolar los params REALES de la linea base, no solo lo
    # que el huerfano alcanzo a sugerir antes del corte ('x' unicamente, ver
    # _SCRIPT_CORTE): 'n' y 'cat' solo viven en system_attrs["fixed_params"] del
    # huerfano. Sin leer de ahi, sembrar_linea_base(...) sigue dando False (la
    # etiqueta rol=linea_base se conserva) pero el trial que de verdad corre
    # tiene params distintos de _PARAMS_OK -- la linea base real nunca corre.
    assert trials[1].system_attrs["fixed_params"] == _PARAMS_OK

    def objetivo(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        n = trial.suggest_int("n", 1, 5)
        trial.suggest_categorical("cat", ["a", "b", "c"])
        return x + n

    study.optimize(objetivo, n_trials=1)
    reintento = study.trials[1]
    assert reintento.state == optuna.trial.TrialState.COMPLETE
    assert reintento.params == _PARAMS_OK

    # sembrar_linea_base ve el WAITING reencolado como linea base activa: no
    # encola una segunda copia ni gasta un trial extra del presupuesto.
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is False
    assert len(study.get_trials(deepcopy=False)) == 2


# --- sembrar_linea_base ---

_DIST_SIMPLE = {
    "x": optuna.distributions.FloatDistribution(0.0, 1.0),
    "n": optuna.distributions.IntDistribution(1, 5),
    "cat": optuna.distributions.CategoricalDistribution(["a", "b", "c"]),
}
_PARAMS_OK = {"x": 0.5, "n": 3, "cat": "b"}


def test_sembrar_linea_base_encola_y_devuelve_true(tmp_path):
    study = _estudio_simple(tmp_path)
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True
    esperando = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    assert len(esperando) == 1


def test_sembrar_linea_base_clave_de_mas_lanza_valueerror(tmp_path):
    study = _estudio_simple(tmp_path)
    with pytest.raises(ValueError, match="sobran"):
        estudio.sembrar_linea_base(study, {**_PARAMS_OK, "extra": 1}, _DIST_SIMPLE)


def test_sembrar_linea_base_clave_de_menos_lanza_valueerror(tmp_path):
    study = _estudio_simple(tmp_path)
    incompleto = dict(_PARAMS_OK)
    del incompleto["n"]
    with pytest.raises(ValueError, match="faltan"):
        estudio.sembrar_linea_base(study, incompleto, _DIST_SIMPLE)


def test_sembrar_linea_base_valor_fuera_de_rango_lanza_valueerror(tmp_path):
    study = _estudio_simple(tmp_path)
    # match ajustado a la clave Y al valor: "x" solo (sin anclar) casaria con
    # casi cualquier mensaje, incluido uno que reporte la clave equivocada.
    with pytest.raises(ValueError, match=r"'x'=5\.0 fuera de rango"):
        estudio.sembrar_linea_base(study, {**_PARAMS_OK, "x": 5.0}, _DIST_SIMPLE)


def test_sembrar_linea_base_categorica_fuera_de_choices_lanza_valueerror(tmp_path):
    study = _estudio_simple(tmp_path)
    with pytest.raises(ValueError, match=r"'cat': 'z' not in"):
        estudio.sembrar_linea_base(study, {**_PARAMS_OK, "cat": "z"}, _DIST_SIMPLE)


@pytest.mark.parametrize(
    ("distribuciones", "params", "debe_fallar"),
    [
        # IntDistribution con step=2: 4 no esta alineado (1, 3, 5, 7, 9 si son
        # los unicos valores validos) -> fuera de rango, no un TypeError.
        ({"n": optuna.distributions.IntDistribution(1, 9, step=2)}, {"n": 4}, True),
        # 2.5 tampoco alinea con el step, aunque sea float-castable.
        ({"n": optuna.distributions.IntDistribution(1, 9, step=2)}, {"n": 2.5}, True),
        ({"n": optuna.distributions.IntDistribution(1, 9, step=2)}, {"n": 3}, False),
        # FloatDistribution log=True rechaza 0.0 (log(0) no esta definido).
        ({"lr": optuna.distributions.FloatDistribution(1e-4, 1.0, log=True)}, {"lr": 0.0}, True),
        ({"lr": optuna.distributions.FloatDistribution(1e-4, 1.0, log=True)}, {"lr": 0.1}, False),
    ],
)
def test_sembrar_linea_base_valida_step_desalineado_y_log_con_cero(
    tmp_path, distribuciones, params, debe_fallar
):
    study = _estudio_simple(tmp_path)
    if debe_fallar:
        with pytest.raises(ValueError):
            estudio.sembrar_linea_base(study, params, distribuciones)
    else:
        assert estudio.sembrar_linea_base(study, params, distribuciones) is True


@pytest.mark.parametrize(
    ("distribuciones", "params"),
    [
        ({"x": optuna.distributions.FloatDistribution(0.0, 1.0)}, {"x": "0.5"}),
        ({"x": optuna.distributions.FloatDistribution(0.0, 1.0)}, {"x": True}),
        ({"n": optuna.distributions.IntDistribution(1, 5)}, {"n": "3"}),
        ({"n": optuna.distributions.IntDistribution(1, 5)}, {"n": False}),
    ],
)
def test_sembrar_linea_base_valor_no_numerico_lanza_valueerror(tmp_path, distribuciones, params):
    # to_internal_repr solo hace float(v): un str float-castable ('0.5') o un
    # bool (subclase de int) pasarian esa conversion en silencio.
    study = _estudio_simple(tmp_path)
    with pytest.raises(ValueError, match="no es un número"):
        estudio.sembrar_linea_base(study, params, distribuciones)


@pytest.mark.parametrize(
    ("distribuciones", "params"),
    [
        ({"x": optuna.distributions.FloatDistribution(0.0, 1.0)}, {"x": 0.5}),
        ({"n": optuna.distributions.IntDistribution(1, 5)}, {"n": 3}),
    ],
)
def test_sembrar_linea_base_valor_numerico_real_no_falla(tmp_path, distribuciones, params):
    study = _estudio_simple(tmp_path)
    assert estudio.sembrar_linea_base(study, params, distribuciones) is True


def test_sembrar_linea_base_params_validos_no_falla(tmp_path):
    study = _estudio_simple(tmp_path)
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True


def test_sembrar_linea_base_convierte_escalares_numpy_a_tipos_nativos_en_storage_persistente(
    destino,
):
    """np.float32 es numbers.Real (pasa la validacion de tipo), pero
    enqueue_trial lo guarda tal cual en fixed_params: en un storage persistente
    (journal o sqlite, a diferencia del backend en memoria que usan la mayoria
    de las pruebas) eso revienta con TypeError al serializar a JSON: "Object
    of type float32 is not JSON serializable". Mismo defecto que test_objetivo.
    test_semillas_con_enteros_de_numpy_se_guardan_en_storage_persistente ya
    cubre para las semillas.
    """
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    params_numpy = {"x": np.float32(0.5), "n": np.int64(3), "cat": "b"}
    assert estudio.sembrar_linea_base(study, params_numpy, _DIST_SIMPLE) is True

    esperando = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    guardado = esperando[0].system_attrs["fixed_params"]
    assert guardado == {"x": 0.5, "n": 3, "cat": "b"}
    assert type(guardado["x"]) is float
    assert type(guardado["n"]) is int


def test_reejecutar_no_duplica_la_linea_base(tmp_path):
    study = _estudio_simple(tmp_path)
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is False
    assert len(study.get_trials(deepcopy=False)) == 1


def test_reejecutar_no_duplica_la_linea_base_ya_completa(destino):
    # Caso real de "re-ejecutar el paso": la linea base ya termino COMPLETE (no
    # solo WAITING) y se vuelve a llamar sembrar_linea_base. Sin RUNNING/COMPLETE
    # en _ESTADOS_LINEA_BASE_ACTIVA, este caso encolaria una segunda copia.
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True

    def objetivo(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        n = trial.suggest_int("n", 1, 5)
        trial.suggest_categorical("cat", ["a", "b", "c"])
        return x + n

    study.optimize(objetivo, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is False
    assert len(study.get_trials(deepcopy=False)) == 1


def test_reejecutar_no_duplica_la_linea_base_en_curso(destino):
    # La linea base paso de WAITING a RUNNING (study.ask() la sirvio) pero
    # todavia no termino: tambien cuenta como activa, no solo WAITING/COMPLETE.
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True
    trial = study.ask()
    assert trial.user_attrs["rol"] == "linea_base"

    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is False
    assert len(study.get_trials(deepcopy=False)) == 1


def test_linea_base_fallida_se_reencola(destino):
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )
    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True

    def objetivo_falla(trial):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        study.optimize(objetivo_falla, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL

    assert estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE) is True
    esperando = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,))
    assert len(esperando) == 1


def test_trial_0_corre_con_esos_params_y_rol_linea_base(tmp_path):
    study = _estudio_simple(tmp_path)
    estudio.sembrar_linea_base(study, _PARAMS_OK, _DIST_SIMPLE)

    def objetivo(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        n = trial.suggest_int("n", 1, 5)
        trial.suggest_categorical("cat", ["a", "b", "c"])
        return x + n

    study.optimize(objetivo, n_trials=1)
    trial0 = study.trials[0]
    assert trial0.number == 0
    assert trial0.params == _PARAMS_OK
    assert trial0.user_attrs["rol"] == "linea_base"


# --- optimizar ---


def test_optimizar_llega_exactamente_a_n_total(tmp_path):
    study = _estudio_simple(tmp_path)

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    estudio.optimizar(study, objetivo, n_total=5)
    estados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    assert len(study.get_trials(deepcopy=False, states=estados)) == 5


def test_optimizar_al_reanudar_no_repite_trabajo(destino):
    study = estudio.crear_estudio(
        "e",
        destino=destino,
        direcciones=["maximize"],
        sampler=_sampler(),
        pruner=_pruner(),
        protocolo={},
    )

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    estados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    estudio.optimizar(study, objetivo, n_total=3)
    assert len(study.get_trials(deepcopy=False, states=estados)) == 3

    estudio.optimizar(study, objetivo, n_total=3)
    assert len(study.get_trials(deepcopy=False, states=estados)) == 3

    estudio.optimizar(study, objetivo, n_total=5)
    assert len(study.get_trials(deepcopy=False, states=estados)) == 5


def test_optimizar_cuenta_pruned_como_terminado(tmp_path):
    """COMPLETE + PRUNED, no solo COMPLETE: un estudio con mucha poda no debe
    correr de mas, y al reanudar no debe repetir trabajo ya podado.
    """
    study = _estudio_simple(tmp_path)

    def objetivo(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number % 2 == 1:
            raise optuna.TrialPruned()
        return x

    estudio.optimizar(study, objetivo, n_total=4)
    trials = study.get_trials(deepcopy=False)
    assert len(trials) == 4
    estados = [t.state for t in trials]
    assert estados.count(optuna.trial.TrialState.COMPLETE) == 2
    assert estados.count(optuna.trial.TrialState.PRUNED) == 2

    # Reanudar no repite trabajo: los PRUNED ya cuentan como terminados.
    estudio.optimizar(study, objetivo, n_total=4)
    assert len(study.get_trials(deepcopy=False)) == 4


def test_optimizar_excepcion_en_atrapar_marca_fail_y_la_busqueda_sigue(tmp_path):
    study = _estudio_simple(tmp_path)
    llamadas = []

    def objetivo(trial):
        llamadas.append(trial.number)
        if trial.number < 2:
            raise KeyError("falla controlada")
        return trial.suggest_float("x", 0.0, 1.0)

    estudio.optimizar(study, objetivo, n_total=1, atrapar=(KeyError,))
    assert llamadas == [0, 1, 2]
    estados = [t.state for t in study.trials]
    assert estados == [
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.FAIL,
        optuna.trial.TrialState.COMPLETE,
    ]


def test_optimizar_mas_de_max_fallos_lanza_runtimeerror(tmp_path):
    study = _estudio_simple(tmp_path)

    def objetivo_falla(trial):
        raise KeyError("boom")

    with pytest.raises(RuntimeError, match="max_fallos"):
        estudio.optimizar(study, objetivo_falla, n_total=1, atrapar=(KeyError,), max_fallos=2)
    fallos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,))
    assert len(fallos) == 3


def test_optimizar_fallos_previos_no_cuentan_para_max_fallos(tmp_path):
    """`max_fallos` cuenta los FAIL de ESTA llamada, no los del estudio entero:

    un estudio reanudado que ya traia fallos de corridas anteriores no debe
    abortar antes de tiempo por fallos que no ocurrieron ahora.
    """
    study = _estudio_simple(tmp_path)
    for _ in range(5):
        study.tell(study.ask(), state=optuna.trial.TrialState.FAIL)

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    estudio.optimizar(study, objetivo, n_total=2, max_fallos=2)
    completos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    assert len(completos) == 2


def test_optimizar_fallos_previos_no_perdonan_fallos_nuevos(tmp_path):
    study = _estudio_simple(tmp_path)
    for _ in range(5):
        study.tell(study.ask(), state=optuna.trial.TrialState.FAIL)

    def objetivo_falla(trial):
        raise KeyError("boom")

    with pytest.raises(RuntimeError, match="max_fallos"):
        estudio.optimizar(study, objetivo_falla, n_total=1, atrapar=(KeyError,), max_fallos=2)
    # 5 fallos previos + exactamente 3 nuevos (el tercero supera max_fallos=2).
    fallos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.FAIL,))
    assert len(fallos) == 8


def test_optimizar_hasta_max_fallos_exacto_no_lanza(tmp_path):
    study = _estudio_simple(tmp_path)
    intentos = {"n": 0}

    def objetivo(trial):
        intentos["n"] += 1
        if intentos["n"] <= 2:
            raise KeyError("falla controlada")
        return trial.suggest_float("x", 0.0, 1.0)

    estudio.optimizar(study, objetivo, n_total=1, atrapar=(KeyError,), max_fallos=2)
    completos = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    assert len(completos) == 1


def test_optimizar_excepcion_no_atrapada_se_propaga(tmp_path):
    study = _estudio_simple(tmp_path)

    def objetivo(trial):
        raise ValueError("no deberia atraparse")

    with pytest.raises(ValueError, match="no deberia atraparse"):
        estudio.optimizar(study, objetivo, n_total=1, atrapar=(KeyError,))
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


# --- importancia_con_piso ---


def test_importancia_con_piso_favorece_el_hiperparametro_real_sobre_el_placebo(tmp_path):
    """Esta prueba usa un objetivo sintético, sin entrenar nada: `learning_rate_init`
    decide el valor devuelto y `placebo` no entra en la fórmula, así que su importancia
    real es cero. El evaluador es `PedAnovaImportanceEvaluator`, el mismo que usa
    `pasos/02_primer_estudio.py`.
    """

    def objetivo(trial):
        lr = trial.suggest_float("learning_rate_init", 1e-4, 3e-1, log=True)
        trial.suggest_float("placebo", -10.0, 10.0)
        return -((lr - 0.05) ** 2)

    study = estudio.crear_estudio(
        "importancia",
        destino=tmp_path / "importancia.log",
        direcciones=["maximize"],
        sampler=_sampler(seed=3),
        pruner=_pruner(),
        protocolo={},
    )
    estudio.optimizar(study, objetivo, n_total=30)

    evaluador = optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.1)
    resultado = estudio.importancia_con_piso(study, evaluador=evaluador, placebo="placebo")

    assert set(resultado) == {"learning_rate_init", "placebo"}
    assert resultado["placebo"]["sobre_el_piso"] is False
    assert resultado["learning_rate_init"]["sobre_el_piso"] is True
    assert resultado["learning_rate_init"]["importancia"] > resultado["placebo"]["importancia"]


def test_importancia_con_piso_placebo_ausente_lanza_keyerror_con_disponibles(tmp_path):
    study = _estudio_simple(tmp_path)

    def objetivo(trial):
        return trial.suggest_float("x", 0.0, 1.0)

    estudio.optimizar(study, objetivo, n_total=3)
    evaluador = optuna.importance.PedAnovaImportanceEvaluator(target_quantile=0.1)
    with pytest.raises(KeyError) as excinfo:
        estudio.importancia_con_piso(study, evaluador=evaluador, placebo="no_existe")
    # match="x" a secas no sirve: "no_existe" tambien contiene una "x", asi que
    # un KeyError(placebo) SIN la lista de disponibles pasaría esta prueba
    # igual. Por eso la prueba afirma la lista completa.
    assert "disponibles ['x']" in str(excinfo.value)
