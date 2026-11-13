"""taller_hpo.honestidad -- medir la maldición del ganador contra el oráculo.

`_estudio` arma un estudio de mentira (métrica "m" = "x", un solo hiperparámetro)
con `enqueue_trial` para controlar el valor exacto de cada trial: así el orden de
"los mejores" es conocido de antemano y las pruebas no dependen del sampler.
"""

from __future__ import annotations

import statistics as st

import optuna
import pytest
from optuna.distributions import FloatDistribution

from taller_hpo import estadistica, estudio
from taller_hpo.honestidad import (
    Inflacion,
    PlanSemillas,
    PruebaSellada,
    congelar,
    inflacion_ganador,
    mejores_de,
    plan_semillas,
    reevaluar,
)
from taller_hpo.objetivo import en_proceso
from taller_hpo.semillas import Semillas, semillas_de

optuna.logging.set_verbosity(optuna.logging.WARNING)

S = Semillas(datos=1, orden=2, init=3)


def _espacio(trial: optuna.Trial) -> dict:
    return {"x": trial.suggest_float("x", 0.0, 10.0)}


def _entrenar_identidad(config, semillas, reportar=None):
    return {"m": config["x"]}


def _estudio(
    valores: list[float],
    *,
    direction: str = "maximize",
    linea_base: float | None = None,
    fijos: dict | None = None,
) -> optuna.Study:
    """Un trial COMPLETE por cada valor de `valores` (métrica "m" == "x").

    Si `linea_base` no es None, se siembra primero vía
    `estudio.sembrar_linea_base` (así queda con `user_attrs["rol"] ==
    "linea_base"`, el mismo marcador que usa `mejores_de`) y corre como el
    trial 0; los `valores` se encolan después, en orden.
    """
    objetivo = en_proceso(_entrenar_identidad, _espacio, metricas=["m"], semillas=S, fijos=fijos)
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=0))
    total = 0
    if linea_base is not None:
        estudio.sembrar_linea_base(study, {"x": linea_base}, {"x": FloatDistribution(0.0, 10.0)})
        total += 1
    for v in valores:
        study.enqueue_trial({"x": v})
        total += 1
    study.optimize(objetivo, n_trials=total)
    return study


# --- plan_semillas ---


def test_plan_semillas_evaluacion_comparte_datos_con_busqueda():
    plan = plan_semillas(20260925, 5)
    assert isinstance(plan, PlanSemillas)
    assert len(plan.evaluacion) == 5
    for e in plan.evaluacion:
        assert e.datos == plan.busqueda.datos


def test_plan_semillas_evaluacion_disjunta_de_busqueda_en_orden_e_init():
    plan = plan_semillas(20260925, 8)
    usados_por_busqueda = {plan.busqueda.orden, plan.busqueda.init}
    for e in plan.evaluacion:
        assert e.orden not in usados_por_busqueda
        assert e.init not in usados_por_busqueda
    # también disjuntas entre sí
    pares = [(e.orden, e.init) for e in plan.evaluacion]
    assert len(set(pares)) == len(pares)


def test_plan_semillas_es_determinista():
    assert plan_semillas(42, 3) == plan_semillas(42, 3)


def test_plan_semillas_bases_distintas_dan_planes_distintos():
    assert plan_semillas(1, 3) != plan_semillas(2, 3)


@pytest.mark.parametrize("n_evaluacion", [0, -1, True])
def test_plan_semillas_n_evaluacion_invalido_lanza_valueerror(n_evaluacion):
    with pytest.raises(ValueError, match="n_evaluacion"):
        plan_semillas(1, n_evaluacion)


def test_plan_semillas_evaluacion_orden_e_init_son_todos_distintos():
    """El contrato dice "orden e init nuevos" para cada elemento de `evaluacion`:
    no basta con que los PARES (orden, init) sean únicos (ya lo cubre
    test_plan_semillas_evaluacion_disjunta_de_busqueda_en_orden_e_init) si un
    error reusara el mismo `orden` en todos los elementos y solo variara
    `init` -- los pares seguirían siendo únicos, pero "orden nuevo" en el
    contrato dejaría de cumplirse.
    """
    plan = plan_semillas(20260925, 8)
    ordenes = [e.orden for e in plan.evaluacion]
    inits = [e.init for e in plan.evaluacion]
    assert len(set(ordenes)) == len(ordenes), "cada evaluación necesita su propio orden"
    assert len(set(inits)) == len(inits), "cada evaluación necesita su propio init"


# --- congelar ---


def test_congelar_estable_ante_orden_de_claves():
    assert congelar({"a": 1, "b": 2}) == congelar({"b": 2, "a": 1})


def test_congelar_distinto_ante_cambio_de_valor():
    assert congelar({"a": 1}) != congelar({"a": 2})


def test_congelar_devuelve_hexdigest_sha256():
    h = congelar({"a": 1})
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# --- mejores_de ---


def test_mejores_de_respeta_direccion_maximizar():
    study = _estudio([0.1, 0.9, 0.5, 0.3], linea_base=0.2)
    seleccion = mejores_de(study, 2)
    assert seleccion["top_1"]["valor"] == pytest.approx(0.9)
    assert seleccion["top_2"]["valor"] == pytest.approx(0.5)
    assert seleccion["linea_base"]["valor"] == pytest.approx(0.2)


def test_mejores_de_respeta_direccion_minimizar():
    study = _estudio([0.1, 0.9, 0.5, 0.3], direction="minimize", linea_base=0.2)
    seleccion = mejores_de(study, 2)
    assert seleccion["top_1"]["valor"] == pytest.approx(0.1)
    assert seleccion["top_2"]["valor"] == pytest.approx(0.3)


def test_mejores_de_excluye_la_linea_base_de_los_top():
    # la línea base (0.99) sería el "top_1" si no se excluyera de la selección.
    study = _estudio([0.1, 0.5], linea_base=0.99)
    seleccion = mejores_de(study, 2)
    valores = sorted([seleccion["top_1"]["valor"], seleccion["top_2"]["valor"]])
    assert valores == pytest.approx([0.1, 0.5])
    assert seleccion["linea_base"]["valor"] == pytest.approx(0.99)


def test_mejores_de_sin_incluir_linea_base_no_trae_esa_clave():
    study = _estudio([0.1, 0.5], linea_base=0.2)
    seleccion = mejores_de(study, 1, incluir_linea_base=False)
    assert "linea_base" not in seleccion
    assert set(seleccion) == {"top_1"}


def test_mejores_de_incluir_linea_base_sin_linea_base_en_el_estudio_lanza_valueerror():
    study = _estudio([0.1, 0.5])  # sin linea_base=...
    with pytest.raises(ValueError, match="linea_base"):
        mejores_de(study, 1, incluir_linea_base=True)


def test_mejores_de_k_mayor_que_trials_disponibles_lanza_valueerror():
    study = _estudio([0.1, 0.5], linea_base=0.2)
    with pytest.raises(ValueError, match="se pidieron 3 mejores"):
        mejores_de(study, 3, incluir_linea_base=False)


@pytest.mark.parametrize("k", [0, -1, True])
def test_mejores_de_k_invalido_lanza_valueerror(k):
    study = _estudio([0.1, 0.5], linea_base=0.2)
    with pytest.raises(ValueError, match="k debe ser"):
        mejores_de(study, k)


def test_mejores_de_params_es_la_config_completa_no_solo_lo_sugerido():
    study = _estudio([0.5, 0.7], linea_base=0.3, fijos={"otro": 42})
    seleccion = mejores_de(study, 1)
    assert seleccion["top_1"]["params"] == {"x": pytest.approx(0.7), "otro": 42}
    assert seleccion["linea_base"]["params"] == {"x": pytest.approx(0.3), "otro": 42}


def test_mejores_de_trial_es_el_numero_correcto():
    study = _estudio([0.5, 0.7], linea_base=0.3)
    seleccion = mejores_de(study, 2)
    # trial 0 es la linea base (se siembra primero en _estudio); 0.7 fue el
    # segundo `enqueue_trial`, así que es el trial número 2.
    assert seleccion["top_1"]["trial"] == 2
    assert seleccion["linea_base"]["trial"] == 0


def test_mejores_de_excluye_trials_podados_aunque_su_ultimo_valor_reportado_sea_el_mejor():
    """`mejores_de` filtra por `TrialState.COMPLETE`: un trial PRUNED conserva
    como `trial.value` el último valor reportado antes de podarse (verificado
    abajo contra Optuna 5.0.0, no asumido), así que sin ese filtro un error
    que también incluyera PRUNED podría devolver como "top_1" un trial que la
    búsqueda decidió cortar, y no uno que de verdad terminó.
    """

    def _espacio(trial: optuna.Trial) -> dict:
        return {"x": trial.suggest_float("x", 0.0, 10.0)}

    def objetivo(trial: optuna.Trial) -> float:
        config = _espacio(trial)
        trial.set_user_attr("config", dict(config))
        if config["x"] > 5.0:
            # Simula el trial que la busqueda decide podar tras reportar un
            # valor alto: sigue teniendo trial.value == 0.99 al terminar.
            trial.report(0.99, 0)
            raise optuna.TrialPruned()
        return config["x"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    study.enqueue_trial({"x": 9.0})  # se poda, value == 0.99 (el más alto de los tres)
    study.enqueue_trial({"x": 0.1})
    study.enqueue_trial({"x": 0.5})
    study.optimize(objetivo, n_trials=3, catch=())

    podados = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))
    assert len(podados) == 1
    assert podados[0].value == pytest.approx(0.99), (
        "verificación de la premisa: un trial PRUNED sí conserva el último valor reportado"
    )

    seleccion = mejores_de(study, 1, incluir_linea_base=False)
    assert seleccion["top_1"]["valor"] == pytest.approx(0.5)


def test_mejores_de_con_dos_lineas_base_complete_usa_la_de_mayor_numero_de_trial():
    """`estudio.sembrar_linea_base` nunca deja llegar aquí (no reencola mientras la
    línea base activa siga COMPLETE/RUNNING/WAITING), pero `mejores_de` no exige que
    `study` se haya sembrado por ahí -- son funciones delgadas sobre un `Study` real,
    y nada impide encolar dos "líneas base" a mano con `study.enqueue_trial`. Si eso
    pasa, el desempate documentado (la de MAYOR número de trial, la más reciente) debe
    ganarle a la más vieja aunque su valor sea peor: mutar `max` por `min` en
    `mejores_de` no lo detecta ninguna otra prueba de este archivo.
    """
    objetivo = en_proceso(_entrenar_identidad, _espacio, metricas=["m"], semillas=S)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0))
    study.enqueue_trial({"x": 0.11}, user_attrs={"rol": "linea_base"})  # trial 0: la vieja
    study.enqueue_trial({"x": 0.3})  # trial 1: candidato normal, no es línea base
    study.enqueue_trial({"x": 0.5}, user_attrs={"rol": "linea_base"})  # trial 2: la nueva
    study.optimize(objetivo, n_trials=3)

    seleccion = mejores_de(study, 1, incluir_linea_base=True)
    assert seleccion["top_1"]["trial"] == 1
    assert seleccion["linea_base"]["trial"] == 2
    assert seleccion["linea_base"]["valor"] == pytest.approx(0.5)


# --- reevaluar ---


def test_reevaluar_usa_las_mismas_semillas_para_todas_las_configs_y_en_orden():
    vistas: dict[str, list[Semillas]] = {"a": [], "b": []}

    def entrenar(config, semillas, reportar=None):
        assert reportar is None
        vistas[config["etq"]].append(semillas)
        return {"m": config["etq"] == "a"}

    juegos = semillas_de(0, 3)
    resultado = reevaluar({"a": {"etq": "a"}, "b": {"etq": "b"}}, entrenar, juegos)

    assert list(resultado) == ["a", "b"]
    assert len(resultado["a"]) == 3
    assert len(resultado["b"]) == 3
    assert vistas["a"] == juegos
    assert vistas["b"] == juegos


def test_reevaluar_config_gana_sobre_fijos_si_se_repite_la_clave():
    vistos = []

    def entrenar(config, semillas, reportar=None):
        vistos.append(dict(config))
        return {"m": 0.0}

    reevaluar({"a": {"x": 99}}, entrenar, [S], fijos={"x": 1, "y": 2})
    assert vistos == [{"x": 99, "y": 2}]


def test_reevaluar_devuelve_las_metricas_completas_de_entrenar():
    def entrenar(config, semillas, reportar=None):
        return {"exactitud_val": 0.5, "exactitud_verdadera": 0.6}

    resultado = reevaluar({"a": {}}, entrenar, [S, S])
    assert resultado["a"] == [
        {"exactitud_val": 0.5, "exactitud_verdadera": 0.6},
        {"exactitud_val": 0.5, "exactitud_verdadera": 0.6},
    ]


def test_reevaluar_semillas_vacia_lanza_valueerror():
    with pytest.raises(ValueError, match="semillas"):
        reevaluar({"a": {}}, _entrenar_identidad, [])


# --- inflacion_ganador ---

# Asimétrico a propósito (media 0.792 != mediana 0.81): así una inflacion
# calculada con la mediana en vez de la media no coincide con los valores
# exactos de las pruebas de fórmula.
_REEVALUADOS = [0.70, 0.80, 0.81, 0.82, 0.83]


def test_inflacion_ganador_maximizar_positiva_si_la_busqueda_vio_mas_que_la_media():
    resultado = inflacion_ganador(0.90, _REEVALUADOS, direccion="maximizar")
    assert isinstance(resultado, Inflacion)
    assert resultado.valor_busqueda == 0.90
    assert resultado.media_reevaluada == pytest.approx(st.mean(_REEVALUADOS))
    assert resultado.inflacion > 0
    assert resultado.ic_bajo <= resultado.inflacion <= resultado.ic_alto


def test_inflacion_ganador_maximizar_negativa_si_la_busqueda_vio_menos_que_la_media():
    resultado = inflacion_ganador(0.65, _REEVALUADOS, direccion="maximizar")
    assert resultado.inflacion < 0
    assert resultado.ic_bajo <= resultado.inflacion <= resultado.ic_alto


def test_inflacion_ganador_minimizar_invierte_el_signo_de_maximizar():
    r_max = inflacion_ganador(0.90, _REEVALUADOS, direccion="maximizar")
    r_min = inflacion_ganador(0.90, _REEVALUADOS, direccion="minimizar")
    assert r_min.inflacion == pytest.approx(-r_max.inflacion)


def test_inflacion_ganador_direccion_invalida_lanza_valueerror():
    with pytest.raises(ValueError, match="direccion"):
        inflacion_ganador(0.9, _REEVALUADOS, direccion="arriba")


def test_inflacion_ganador_formula_exacta_maximizar():
    """`inflacion` y su intervalo salen de `estadistica.resumen` (la t de
    Student, no z=1.96) sobre `_REEVALUADOS`: comparar contra los valores
    exactos de `Resumen` (no solo el signo o la contención) atrapa una
    `resumen` con la normal en vez de la t (intervalo más angosto que el
    real), un intervalo casi de ancho cero, o una `inflacion` calculada con la
    mediana en vez de la media.
    """
    r = estadistica.resumen(_REEVALUADOS)
    resultado = inflacion_ganador(0.90, _REEVALUADOS, direccion="maximizar")
    assert resultado.inflacion == pytest.approx(0.90 - r.media)
    assert resultado.ic_bajo == pytest.approx(0.90 - r.ic_alto)
    assert resultado.ic_alto == pytest.approx(0.90 - r.ic_bajo)


def test_inflacion_ganador_formula_exacta_minimizar():
    """Mismo chequeo que la versión "maximizar", con el signo y los extremos
    del intervalo invertidos (`minimizar` resta al revés:
    `media - valor_busqueda`).
    """
    r = estadistica.resumen(_REEVALUADOS)
    resultado = inflacion_ganador(0.90, _REEVALUADOS, direccion="minimizar")
    assert resultado.inflacion == pytest.approx(r.media - 0.90)
    assert resultado.ic_bajo == pytest.approx(r.ic_bajo - 0.90)
    assert resultado.ic_alto == pytest.approx(r.ic_alto - 0.90)


@pytest.mark.parametrize("valor", [float("nan"), float("inf"), float("-inf")])
def test_inflacion_ganador_valor_busqueda_no_finito_lanza_valueerror(valor):
    with pytest.raises(ValueError, match="finito"):
        inflacion_ganador(valor, _REEVALUADOS, direccion="maximizar")


# --- PruebaSellada ---


def test_pruebasellada_abrir_sin_sellar_lanza_runtimeerror():
    prueba = PruebaSellada(lambda params: 1.0)
    with pytest.raises(RuntimeError, match=r"primero llama a sellar\(params\)"):
        prueba.abrir({"x": 1})


def test_pruebasellada_abrir_con_params_distintos_lanza_valueerror():
    prueba = PruebaSellada(lambda params: 1.0)
    prueba.sellar({"x": 1})
    with pytest.raises(ValueError, match="no coincide"):
        prueba.abrir({"x": 2})


def test_pruebasellada_abrir_dos_veces_lanza_runtimeerror():
    prueba = PruebaSellada(lambda params: 1.0)
    prueba.sellar({"x": 1})
    prueba.abrir({"x": 1})
    with pytest.raises(RuntimeError, match="ya abriste la prueba"):
        prueba.abrir({"x": 1})


def test_pruebasellada_abrir_con_los_mismos_params_no_lanza_y_llama_evaluar_una_vez():
    llamadas = []

    def evaluar(params):
        llamadas.append(dict(params))
        return 0.42

    prueba = PruebaSellada(evaluar)
    prueba.sellar({"x": 1, "y": 2})
    assert prueba.abrir({"y": 2, "x": 1}) == 0.42  # orden de claves distinto, misma config
    assert llamadas == [{"y": 2, "x": 1}]


def test_pruebasellada_resellar_antes_de_abrir_reemplaza_la_config_congelada():
    """Cambiar de opinión ANTES de la primera apertura sí está permitido: todavía
    no se miró ninguna prueba con ninguna config, así que no hay maldición del
    ganador que evitar. `abrir` con la config VIEJA debe rechazarse por hash
    (ya no es la sellada) y con la NUEVA debe funcionar.
    """
    llamadas = []

    def evaluar(params):
        llamadas.append(dict(params))
        return params["x"]

    prueba = PruebaSellada(evaluar)
    prueba.sellar({"x": 1})
    prueba.sellar({"x": 2})
    with pytest.raises(ValueError, match="no coincide"):
        prueba.abrir({"x": 1})
    assert prueba.abrir({"x": 2}) == 2
    assert llamadas == [{"x": 2}]


def test_pruebasellada_abrir_marca_abierta_incluso_si_evaluar_lanza():
    """Decisión documentada en el docstring de `abrir`: la apertura se marca como
    consumida ANTES de llamar a `evaluar`, así que si `evaluar` revienta (una
    config inválida, una divergencia del entrenamiento) esa apertura ya cuenta
    como la única permitida -- reintentar tras un error dejaría "probar hasta
    que no reviente" como una forma de espiar la prueba varias veces.
    """

    def evaluar_que_revienta(params):
        raise RuntimeError("el entrenamiento diverge")

    prueba = PruebaSellada(evaluar_que_revienta)
    prueba.sellar({"x": 1})
    with pytest.raises(RuntimeError, match="diverge"):
        prueba.abrir({"x": 1})
    # la apertura ya quedó consumida, aunque evaluar nunca devolvió un número.
    with pytest.raises(RuntimeError, match="ya abriste la prueba"):
        prueba.abrir({"x": 1})


def test_pruebasellada_abrir_que_no_revienta_tambien_consume_la_apertura():
    """Contraparte de la prueba anterior: cuando `evaluar` SÍ devuelve un
    número, la apertura también queda consumida de inmediato -- el mismo
    `_abierta = True` cubre las dos rutas (con y sin excepción), y ninguna
    permite una segunda llamada.
    """
    llamadas = []

    def evaluar(params):
        llamadas.append(dict(params))
        return 1.0

    prueba = PruebaSellada(evaluar)
    prueba.sellar({"x": 1})
    assert prueba.abrir({"x": 1}) == 1.0
    with pytest.raises(RuntimeError, match="ya abriste la prueba"):
        prueba.abrir({"x": 1})
    assert llamadas == [{"x": 1}]


def test_pruebasellada_sellar_despues_de_abrir_lanza_runtimeerror():
    """La maldición del ganador un nivel más arriba: sin esta guarda,
    `sellar(A); abrir(A); sellar(B); abrir(B)` dejaría "se abre una vez" como
    una frase del docstring que el objeto no hace cumplir.
    """
    prueba = PruebaSellada(lambda params: 1.0)
    prueba.sellar({"x": 1})
    prueba.abrir({"x": 1})
    with pytest.raises(RuntimeError, match="ya abriste la prueba"):
        prueba.sellar({"x": 2})
