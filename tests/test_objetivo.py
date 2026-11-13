"""en_proceso envuelve entrenar(config, semillas, reportar) como objective de Optuna.

Los primeros tests usan un `entrenar` de mentira para aislar cada regla del
contrato (config, semillas, metricas, atributos, poda) sin pagar el costo de
entrenar una red real. El ultimo test integra con `taller_hpo.problemas.juguete`
(que entrena en la tarjeta de video) para confirmar que la poda de verdad
ocurre fuera del modelo.

Importar `taller_hpo.problemas.juguete` sin tarjeta de video visible ya falla
con un `RuntimeError` (ver ese modulo): sin el try/except de abajo, esa falla
ocurriria al IMPORTAR este archivo de pruebas, y una corrida completa de la
suite con `pytest -m "not gpu"` -- la que usa la integracion continua, sin
tarjeta de video -- abortaria la COLECCION entera en vez de solo excluir la
prueba `gpu` de este archivo (mismo patron que ya usa `tests/test_juguete.py`).
"""

from __future__ import annotations

import numpy as np
import optuna
import pytest

from taller_hpo import estudio
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.semillas import Semillas, semillas_de

try:
    from taller_hpo.problemas import juguete
except RuntimeError as _error:
    if "tarjeta de video" not in str(_error):
        raise
    juguete = None

optuna.logging.set_verbosity(optuna.logging.WARNING)

S = Semillas(datos=1, orden=2, init=3)


class _PrunerSiempre(optuna.pruners.BasePruner):
    """Poda cualquier trial sin mirar lo reportado: determinista por diseno."""

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return True


class _PrunerEnPrimeraEpoca(optuna.pruners.BasePruner):
    """Poda solo cuando la ultima epoca reportada es la 0: determinista por diseno.

    Sirve para distinguir una poda pegajosa de una que se puede "despodar": un
    entrenar terco que sigue llamando a `reportar` despues del primer True vería
    should_prune() volver a False en la epoca 1 si `podado` no fuera monotono.
    """

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        return trial.last_step == 0


def _espacio(trial: optuna.Trial) -> dict:
    return {"x": trial.suggest_float("x", 0.0, 1.0)}


def _entrenar_fake(metricas: dict, *, epocas: int = 1, registro: list | None = None):
    """entrenar() de mentira: llama reportar `epocas` veces y devuelve `metricas`.

    `registro`, si se da, acumula la epoca de cada llamada a `reportar` (antes de
    mirar si dijo que parara): sirve para contar cuantas veces entrenar() de
    verdad siguio corriendo, no solo si al final se levanto TrialPruned.
    """

    def entrenar(config, semillas, reportar=None):
        if reportar is not None:
            for epoca in range(epocas):
                if registro is not None:
                    registro.append(epoca)
                if reportar(epoca, 0.5):
                    break
        return dict(metricas)

    return entrenar


def _study(direction: str = "maximize", pruner=None) -> optuna.Study:
    return optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=pruner or optuna.pruners.NopPruner(),
    )


def test_config_es_fijos_mas_espacio_y_queda_en_user_attrs():
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S, fijos={"otro": 7}
    )
    study = _study()
    study.optimize(objetivo, n_trials=1)
    config = study.trials[0].user_attrs["config"]
    assert config["otro"] == 7
    assert set(config) == {"otro", "x"}


def test_espacio_sin_fijos_no_agrega_claves_de_mas():
    objetivo = en_proceso(_entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S)
    study = _study()
    study.optimize(objetivo, n_trials=1)
    assert set(study.trials[0].user_attrs["config"]) == {"x"}


def test_semillas_fija_se_guarda_como_dict_en_user_attrs():
    objetivo = en_proceso(_entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S)
    study = _study()
    study.optimize(objetivo, n_trials=1)
    assert study.trials[0].user_attrs["semillas"] == {"datos": 1, "orden": 2, "init": 3}


def test_semillas_con_enteros_de_numpy_se_guardan_en_storage_persistente(tmp_path):
    """asdict() no convierte np.int64 a int de Python: si `semillas` se armo a
    mano con numpy (en vez de via `semillas_de`, que ya castea), guardar esos
    valores directo en un storage persistente (a diferencia del backend en
    memoria que usan la mayoria de las pruebas) revienta con TypeError.
    """
    semillas_numpy = Semillas(datos=np.int64(1), orden=np.int64(2), init=np.int64(3))
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=semillas_numpy
    )

    study = estudio.crear_estudio(
        "e",
        destino=tmp_path / "e.log",
        direcciones=["maximize"],
        sampler=optuna.samplers.RandomSampler(seed=0),
        pruner=optuna.pruners.NopPruner(),
        protocolo={},
    )
    study.optimize(objetivo, n_trials=1)
    guardado = study.trials[0].user_attrs["semillas"]
    assert guardado == {"datos": 1, "orden": 2, "init": 3}
    assert all(type(v) is int for v in guardado.values())


def test_semillas_callable_recibe_el_trial_y_varia_por_trial():
    juegos = semillas_de(1, 3)
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0}),
        _espacio,
        metricas=["m"],
        semillas=lambda trial: juegos[trial.number],
    )
    study = _study()
    study.optimize(objetivo, n_trials=3)
    for trial, esperado in zip(study.trials, juegos, strict=True):
        assert trial.user_attrs["semillas"] == {
            "datos": esperado.datos,
            "orden": esperado.orden,
            "init": esperado.init,
        }


def test_metrica_ausente_lanza_keyerror():
    objetivo = en_proceso(_entrenar_fake({"otra": 1.0}), _espacio, metricas=["m"], semillas=S)
    with pytest.raises(KeyError, match="m"):
        objetivo(_study().ask())


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), float("-inf")])
def test_metrica_no_finita_lanza_metricanofinita(valor):
    objetivo = en_proceso(_entrenar_fake({"m": valor}), _espacio, metricas=["m"], semillas=S)
    with pytest.raises(MetricaNoFinita, match=r"métrica 'm' = .* no es finita"):
        objetivo(_study().ask())


def test_metrica_finita_no_lanza_nada_y_se_devuelve_tal_cual():
    objetivo = en_proceso(_entrenar_fake({"m": 0.42}), _espacio, metricas=["m"], semillas=S)
    assert objetivo(_study().ask()) == 0.42


def test_atributos_se_guardan_como_user_attrs_y_nunca_se_devuelven():
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0, "oraculo": 99.0}),
        _espacio,
        metricas=["m"],
        semillas=S,
        atributos=["oraculo"],
    )
    study = _study()
    study.optimize(objetivo, n_trials=1)
    trial = study.trials[0]
    assert trial.user_attrs["oraculo"] == 99.0
    assert trial.value == 1.0


def test_atributos_se_guardan_incluso_en_trial_podado():
    """Un trial PRUNED tambien corrio entrenar(): su resultado (p.ej. el oraculo)
    sirve para medir cuanto computo ahorro la poda, asi que debe guardarse igual
    que en un trial COMPLETE.
    """
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0, "oraculo": 99.0}),
        _espacio,
        metricas=["m"],
        semillas=S,
        atributos=["oraculo"],
    )
    study = _study(pruner=_PrunerSiempre())
    trial = study.ask()
    with pytest.raises(optuna.TrialPruned):
        objetivo(trial)
    assert trial.user_attrs["oraculo"] == 99.0


def test_atributo_ausente_lanza_keyerror():
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S, atributos=["no_existe"]
    )
    with pytest.raises(KeyError, match="no_existe"):
        objetivo(_study().ask())


def test_metricas_vacio_lanza_valueerror():
    with pytest.raises(ValueError, match="metricas no puede estar vacío"):
        en_proceso(_entrenar_fake({}), _espacio, metricas=[], semillas=S)


def test_metricas_str_suelto_lanza_typeerror():
    # "acc" tambien es un Sequence[str]: sin la guarda, se trataria como
    # multiobjetivo sobre las letras 'a', 'c', 'c' (len==3), reportar=None
    # apagaria la poda en silencio y entrenar() recibiria un reportar distinto
    # del que el contrato promete para una sola metrica.
    with pytest.raises(TypeError, match="lista"):
        en_proceso(_entrenar_fake({"acc": 1.0}), _espacio, metricas="acc", semillas=S)


def test_atributos_str_suelto_lanza_typeerror():
    with pytest.raises(TypeError, match="lista"):
        en_proceso(
            _entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S, atributos="oraculo"
        )


def test_metricas_y_atributos_con_clave_compartida_lanza_valueerror():
    # Sin esta guarda, poner el oraculo en las dos listas a la vez lo
    # optimizaria en silencio -- justo lo que el contrato prohibe ("el oraculo
    # se registra, nunca se optimiza").
    with pytest.raises(ValueError, match=r"compartir claves: \['oraculo'\]"):
        en_proceso(
            _entrenar_fake({"oraculo": 1.0}),
            _espacio,
            metricas=["oraculo"],
            semillas=S,
            atributos=["oraculo"],
        )


def test_metricas_y_atributos_disjuntas_no_lanza_nada():
    objetivo = en_proceso(
        _entrenar_fake({"m": 1.0, "oraculo": 2.0}),
        _espacio,
        metricas=["m"],
        semillas=S,
        atributos=["oraculo"],
    )
    assert objetivo(_study().ask()) == 1.0


def test_mono_objetivo_devuelve_float():
    objetivo = en_proceso(_entrenar_fake({"m": 1.0}), _espacio, metricas=["m"], semillas=S)
    valor = objetivo(_study().ask())
    assert isinstance(valor, float)
    assert valor == 1.0


def test_multiobjetivo_devuelve_tupla_en_el_orden_pedido():
    objetivo = en_proceso(
        _entrenar_fake({"a": 1.0, "b": 2.0}), _espacio, metricas=["b", "a"], semillas=S
    )
    study = optuna.create_study(
        directions=["minimize", "maximize"], sampler=optuna.samplers.RandomSampler(seed=0)
    )
    valores = objetivo(study.ask())
    assert isinstance(valores, tuple)
    assert valores == (2.0, 1.0)


def test_multiobjetivo_no_le_pasa_reportar_a_entrenar():
    visto = {}

    def entrenar(config, semillas, reportar=None):
        visto["reportar"] = reportar
        return {"a": 1.0, "b": 2.0}

    objetivo = en_proceso(entrenar, _espacio, metricas=["a", "b"], semillas=S)
    study = optuna.create_study(
        directions=["minimize", "maximize"], sampler=optuna.samplers.RandomSampler(seed=0)
    )
    objetivo(study.ask())
    assert visto["reportar"] is None


def test_podado_levanta_trialpruned_despues_de_que_entrenar_regresa():
    """No basta con que al final se levante TrialPruned: entrenar() debe haber
    parado en la primera epoca podada, no seguir corriendo las demas. Sin esto,
    un `reportar` que siempre devuelve False (ignorando should_prune()) pasaria
    la prueba igual, porque `podado` (la variable interna) se sigue calculando
    bien aunque su valor de retorno hacia entrenar este roto.
    """
    llamadas: list[int] = []
    entrenar = _entrenar_fake({"m": 1.0}, epocas=3, registro=llamadas)
    objetivo = en_proceso(entrenar, _espacio, metricas=["m"], semillas=S)
    study = _study(pruner=_PrunerSiempre())
    with pytest.raises(optuna.TrialPruned):
        objetivo(study.ask())
    assert llamadas == [0]


def test_no_podado_no_levanta_trialpruned_y_devuelve_el_valor():
    llamadas: list[int] = []
    entrenar = _entrenar_fake({"m": 1.0}, epocas=3, registro=llamadas)
    objetivo = en_proceso(entrenar, _espacio, metricas=["m"], semillas=S)
    study = _study(pruner=optuna.pruners.NopPruner())
    assert objetivo(study.ask()) == 1.0
    assert llamadas == [0, 1, 2]


def test_podado_es_pegajoso_aunque_entrenar_siga_reportando():
    """Un entrenar terco que ignora el True de reportar y sigue llamandolo debe
    seguir podado: should_prune() puede volver a decir False en una epoca
    posterior (aqui, cualquiera despues de la 0), y eso NO debe "despodar" el
    trial. Sin la guarda de pegajosidad, `podado` terminaria en False y el trial
    quedaria COMPLETE en vez de PRUNED, violando el contrato.
    """
    decisiones = []

    def entrenar_terco(config, semillas, reportar=None):
        for epoca in range(3):
            decisiones.append(reportar(epoca, 0.5))
        return {"m": 1.0}

    objetivo = en_proceso(entrenar_terco, _espacio, metricas=["m"], semillas=S)
    study = _study(pruner=_PrunerEnPrimeraEpoca())
    with pytest.raises(optuna.TrialPruned):
        objetivo(study.ask())
    assert decisiones == [True, True, True]


# --- integracion real con taller_hpo.problemas.juguete (entrena en la tarjeta de video) ---

_RAPIDO_JUGUETE = {"n_train": 300, "n_val": 150, "n_oraculo": 5_000, "ancho": 8, "epocas": 15}


def _espacio_juguete(trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
    return {**_RAPIDO_JUGUETE, "learning_rate_init": lr}


@pytest.mark.gpu
def test_en_proceso_con_juguete_poda_de_verdad_y_registra_el_oraculo():
    juegos = semillas_de(20260924, 20)

    def semillas_por_trial(trial: optuna.Trial) -> Semillas:
        return juegos[trial.number % len(juegos)]

    objetivo = en_proceso(
        juguete.entrenar,
        _espacio_juguete,
        metricas=["exactitud_val"],
        semillas=semillas_por_trial,
        atributos=["exactitud_verdadera"],
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=7),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2, n_min_trials=1),
    )
    study.optimize(objetivo, n_trials=20)

    estados = {t.state for t in study.trials}
    assert optuna.trial.TrialState.PRUNED in estados, "se esperaba al menos un trial podado"
    assert optuna.trial.TrialState.COMPLETE in estados

    podados = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    for trial in podados:
        # La poda de verdad corta el entrenamiento: un trial PRUNED reporto menos
        # epocas que el total pedido. Si `entrenar` ignorara la poda y siguiera
        # corriendo, intermediate_values tendria las 15 epocas completas.
        assert len(trial.intermediate_values) < _RAPIDO_JUGUETE["epocas"]

    completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for trial in completos:
        assert {"learning_rate_init", "epocas"} <= set(trial.user_attrs["config"])
        assert set(trial.user_attrs["semillas"]) == {"datos", "orden", "init"}
        assert "exactitud_verdadera" in trial.user_attrs
        # el oraculo se registra, nunca es el valor que se optimizo
        assert trial.value != trial.user_attrs["exactitud_verdadera"]
