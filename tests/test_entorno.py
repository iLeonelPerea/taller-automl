"""El entorno instala, Optuna 5 persiste un estudio y el dashboard lo lee.

No prueba nada de ML. Solo que las piezas que usan los pasos existen con la
API que esperamos.
"""

import json
import sys
from wsgiref.util import setup_testing_defaults

import optuna
import optuna_dashboard
import pytest
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage


def _journal(ruta):
    # OpenLock y no el SymlinkLock por defecto: crear symlinks falla en Windows.
    return JournalStorage(JournalFileBackend(str(ruta), lock_obj=JournalFileOpenLock(str(ruta))))


def _cuadrado(trial):
    return trial.suggest_float("x", -1.0, 1.0) ** 2


def test_versiones():
    assert sys.version_info[:2] >= (3, 12)
    assert optuna.__version__.split(".")[0] == "5"


@pytest.mark.parametrize("tipo", ["journal", "sqlite"])
def test_estudio_se_reabre_y_continua(tmp_path, tipo):
    # Cubre cerrar y reabrir, no un corte a mitad de trial: el Journal no tiene
    # heartbeat y ese trial quedaría RUNNING (para eso está `recuperar_huerfanos`).
    if tipo == "journal":
        storage, reabrir = (
            _journal(tmp_path / "estudio.log"),
            lambda: _journal(tmp_path / "estudio.log"),
        )
    else:
        url = f"sqlite:///{tmp_path / 'estudio.db'}"
        storage, reabrir = url, lambda: url

    estudio = optuna.create_study(
        study_name="humo",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=0),
    )
    estudio.optimize(_cuadrado, n_trials=3)
    antes = [(t.params, t.value) for t in estudio.trials]

    # Reabrir desde otro objeto storage: lo que quedó en disco, no en memoria.
    reanudado = optuna.load_study(
        study_name="humo",
        storage=reabrir(),
        sampler=optuna.samplers.RandomSampler(seed=1),
    )
    reanudado.optimize(_cuadrado, n_trials=2)

    estados = [t.state for t in reanudado.trials]
    assert estados == [optuna.trial.TrialState.COMPLETE] * 5
    assert [(t.params, t.value) for t in reanudado.trials[:3]] == antes


def test_dashboard_lee_el_journal(tmp_path):
    ruta = tmp_path / "estudio.log"
    optuna.create_study(
        study_name="visible", storage=_journal(ruta), sampler=optuna.samplers.RandomSampler(seed=0)
    ).optimize(_cuadrado, n_trials=1)

    # La ruta y no el objeto storage: así lo abre `optuna-dashboard estudios/x.log`,
    # y obliga a leer del disco (con el mismo objeto el test pasaba con el archivo vacío).
    app = optuna_dashboard.wsgi(str(ruta))
    environ = {}
    setup_testing_defaults(environ)
    environ["PATH_INFO"] = "/api/studies"
    respuesta = {}

    def start_response(status, headers, exc_info=None):
        respuesta["status"] = status

    cuerpo = b"".join(app(environ, start_response))

    assert respuesta["status"].startswith("200")
    nombres = [s["study_name"] for s in json.loads(cuerpo)["study_summaries"]]
    assert nombres == ["visible"]
