"""Paso 01: el paisaje analítico reproduce Bergstra y Bengio (2012).

Todo se calcula con una fórmula cerrada (nada de entrenar un modelo), así que
la comparación grid vs random se puede repetir con 100 réplicas (semilla por
réplica), sin que la suite se vuelva lenta ni inestable.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pytest

from taller_hpo.problemas.analitico import Paisaje, crear_paisaje, regret_minimo_grid

NIVELES = (1 / 6, 1 / 2, 5 / 6)


def _uniforme_grueso(
    muestra: Sequence[float],
    *,
    low: float = 0.05,
    high: float = 0.95,
    bins: int = 9,
    tolerancia: float = 0.30,
) -> bool:
    """Heurística gruesa de uniformidad: determinista, barata y fácil de razonar.

    Divide `[low, high]` en `bins` cajas iguales y compara cada frecuencia
    observada contra la esperada bajo una uniforme, con una tolerancia relativa
    `tolerancia`. No sustituye una prueba de hipótesis formal (para eso está
    `scipy.stats.kstest`); basta para distinguir, en las dos direcciones, una
    muestra uniforme de verdad de un rango encogido o de una muestra discreta.
    """
    muestra = np.asarray(muestra, dtype=float)
    esperada = len(muestra) / bins
    hist, _ = np.histogram(muestra, bins=bins, range=(low, high))
    return bool(np.all(np.abs(hist - esperada) <= tolerancia * esperada))


def test_regret_solo_depende_del_eje_importante():
    paisaje = Paisaje(n_ejes=3, eje_importante=1, optimo=0.3)
    assert paisaje.regret({"x0": 0.0, "x1": 0.3, "x2": 0.0}) == pytest.approx(0.0)
    for x0, x2 in [(0.0, 0.0), (1.0, 1.0), (0.5, 0.9)]:
        variante = {"x0": x0, "x1": 0.3, "x2": x2}
        assert paisaje.regret(variante) == pytest.approx(0.0), variante

    # Solo mover el eje importante cambia el regret.
    otro = {"x0": 0.0, "x1": 0.8, "x2": 0.0}
    assert paisaje.regret(otro) == pytest.approx((0.8 - 0.3) ** 2)


def test_espacio_es_el_cubo_unitario_en_cada_eje():
    paisaje = Paisaje(n_ejes=4, eje_importante=2, optimo=0.5)
    assert paisaje.espacio() == {
        "x0": (0.0, 1.0),
        "x1": (0.0, 1.0),
        "x2": (0.0, 1.0),
        "x3": (0.0, 1.0),
    }


def test_regret_rechaza_clave_faltante_o_sobrante():
    paisaje = Paisaje(n_ejes=2, eje_importante=0, optimo=0.5)
    completo = {"x0": 0.5, "x1": 0.0}
    assert paisaje.regret(completo) == pytest.approx(0.0)  # el completo sí funciona

    with pytest.raises(ValueError, match="x1"):
        paisaje.regret({"x0": 0.5})
    with pytest.raises(ValueError, match="x2"):
        paisaje.regret({**completo, "x2": 0.1})


def test_regret_rechaza_valor_fuera_de_rango():
    paisaje = Paisaje(n_ejes=2, eje_importante=0, optimo=0.5)
    assert paisaje.regret({"x0": 0.0, "x1": 0.5}) == pytest.approx(0.25)  # borde 0.0 válido
    assert paisaje.regret({"x0": 1.0, "x1": 0.5}) == pytest.approx(0.25)  # borde 1.0 válido

    with pytest.raises(ValueError, match="x0"):
        paisaje.regret({"x0": -0.01, "x1": 0.5})
    with pytest.raises(ValueError, match="x1"):
        paisaje.regret({"x0": 0.5, "x1": 1.01})


def test_paisaje_rechaza_n_ejes_invalido():
    Paisaje(n_ejes=1, eje_importante=0, optimo=0.5)  # el mínimo válido sí construye

    for n_ejes_invalido in (0, -1, 1.5):
        with pytest.raises(ValueError, match="n_ejes"):
            Paisaje(n_ejes=n_ejes_invalido, eje_importante=0, optimo=0.5)


def test_paisaje_rechaza_eje_importante_fuera_de_rango():
    assert Paisaje(n_ejes=3, eje_importante=0, optimo=0.5).eje_importante == 0  # borde válido
    assert Paisaje(n_ejes=3, eje_importante=2, optimo=0.5).eje_importante == 2  # borde válido

    for eje_invalido in (-1, 3, 2.0):
        with pytest.raises(ValueError, match="eje_importante"):
            Paisaje(n_ejes=3, eje_importante=eje_invalido, optimo=0.5)


def test_paisaje_rechaza_optimo_fuera_de_rango():
    assert Paisaje(n_ejes=2, eje_importante=0, optimo=0.0).optimo == 0.0  # borde 0.0 válido
    assert Paisaje(n_ejes=2, eje_importante=0, optimo=1.0).optimo == 1.0  # borde 1.0 válido

    for optimo_invalido in (-0.01, 1.01):
        with pytest.raises(ValueError, match="optimo"):
            Paisaje(n_ejes=2, eje_importante=0, optimo=optimo_invalido)


def test_crear_paisaje_reproducible_por_semilla():
    assert crear_paisaje(7) == crear_paisaje(7)
    assert crear_paisaje(7, n_ejes=5) == crear_paisaje(7, n_ejes=5)


def test_crear_paisaje_varia_entre_semillas():
    paisajes = [crear_paisaje(s) for s in range(30)]
    ejes = {p.eje_importante for p in paisajes}
    optimos = [p.optimo for p in paisajes]

    assert ejes == set(range(3))  # con 30 semillas deben salir los 3 ejes posibles
    assert all(0.05 <= o <= 0.95 for o in optimos)
    # óptimos que cubren casi todo el rango [0.05, 0.95], no un intervalo encogido
    assert min(optimos) < 0.2 and max(optimos) > 0.8


def test_crear_paisaje_es_uniforme_en_todos_los_ejes_y_en_el_optimo():
    """Con muchas semillas deben aparecer TODOS los ejes 0..n_ejes-1 con
    frecuencias razonables (no solo "aparecer" un par de ellos), y el óptimo
    debe cubrir casi todo [0.05, 0.95] con una distribución aproximadamente
    uniforme -- no solo un rango amplio con extremos densos y el medio vacío.

    Esta prueba mira las dos direcciones (docs/arquitectura.md): el código real
    pasa el criterio de `_uniforme_grueso`, y una muestra sembrada con el
    MISMO criterio falla cuando el rango está encogido o cuando la muestra es
    discreta -- justo la clase de defecto (rango [0.10, 0.90], [0.15, 0.85],
    óptimo sesgado o discreto) que un criterio más débil dejaría pasar. Números
    verificados contra el código real (1000 semillas): n_ejes=3 da cuentas
    [327, 319, 354] (esperado 333 +/- 83); n_ejes=5 da [185, 212, 183, 209,
    211] (esperado 200 +/- 50); el óptimo da min=0.0503, max=0.9498 y un
    histograma de 9 cajas dentro de 111 +/- 33.
    """
    n_semillas = 1000

    for n_ejes in (3, 5):
        paisajes = [crear_paisaje(s, n_ejes=n_ejes) for s in range(n_semillas)]
        cuentas = np.bincount([p.eje_importante for p in paisajes], minlength=n_ejes)
        esperada = n_semillas / n_ejes
        assert len(cuentas) == n_ejes, cuentas  # ningún eje fuera de [0, n_ejes)
        assert np.all(np.abs(cuentas - esperada) <= 0.25 * esperada), cuentas

    optimos = [crear_paisaje(s).optimo for s in range(n_semillas)]
    assert min(optimos) < 0.06 and max(optimos) > 0.94
    assert _uniforme_grueso(optimos)

    # Dirección contraria: el MISMO criterio debe fallar con un rango encogido
    # (por ejemplo uniform(0.15, 0.85), el defecto que un criterio más débil no ve)...
    rng = np.random.default_rng(0)
    encogido = rng.uniform(0.15, 0.85, size=n_semillas)
    assert not _uniforme_grueso(encogido)

    # ...y con una muestra discreta de pocos valores.
    discreta = rng.choice([0.1, 0.5, 0.9], size=n_semillas)
    assert not _uniforme_grueso(discreta)


def test_crear_paisaje_rechaza_n_ejes_invalido():
    assert crear_paisaje(1, n_ejes=1).n_ejes == 1  # el mínimo válido sí funciona

    for n_ejes_invalido in (0, -1, 1.5):
        with pytest.raises(ValueError, match="n_ejes"):
            crear_paisaje(1, n_ejes=n_ejes_invalido)


@pytest.mark.parametrize(
    ("semilla", "n_ejes", "niveles"),
    [
        (11, 3, (0.2, 0.4, 0.6, 0.9)),
        (0, 2, (0.1, 0.5, 0.75)),
        (5, 4, (0.05, 0.3, 0.55, 0.8, 0.95)),
    ],
)
def test_regret_minimo_grid_coincide_con_fuerza_bruta(semilla, n_ejes, niveles):
    paisaje = crear_paisaje(semilla, n_ejes=n_ejes)

    fuerza_bruta = min(
        paisaje.regret({f"x{i}": v for i, v in enumerate(combinacion)})
        for combinacion in itertools.product(niveles, repeat=paisaje.n_ejes)
    )
    assert regret_minimo_grid(paisaje, niveles) == pytest.approx(fuerza_bruta)


def test_regret_minimo_grid_es_cero_solo_si_el_optimo_cae_en_la_rejilla():
    # Dirección 1: el óptimo cae EXACTAMENTE en un nivel de la rejilla -> regret 0.
    en_la_rejilla = Paisaje(n_ejes=3, eje_importante=0, optimo=0.5)
    assert regret_minimo_grid(en_la_rejilla, NIVELES) == pytest.approx(0.0, abs=1e-12)

    # Dirección 2 (contraria): si NO cae en la rejilla, el regret no es cero.
    fuera_de_la_rejilla = Paisaje(n_ejes=3, eje_importante=0, optimo=0.3)
    assert regret_minimo_grid(fuera_de_la_rejilla, NIVELES) > 1e-4


def test_regret_minimo_grid_rechaza_niveles_vacios():
    paisaje = crear_paisaje(1, n_ejes=2)
    assert regret_minimo_grid(paisaje, (0.5,)) >= 0.0  # un nivel sí funciona

    with pytest.raises(ValueError, match="vac"):
        regret_minimo_grid(paisaje, ())


def test_regret_minimo_grid_rechaza_nivel_fuera_de_rango():
    paisaje = crear_paisaje(1, n_ejes=2)
    assert regret_minimo_grid(paisaje, (0.0, 0.5, 1.0)) >= 0.0  # bordes válidos sí funcionan

    with pytest.raises(ValueError, match="1.5"):
        regret_minimo_grid(paisaje, (0.2, 1.5))
    with pytest.raises(ValueError, match="-0.1"):
        regret_minimo_grid(paisaje, (-0.1, 0.5))


def test_random_supera_a_grid_con_el_mismo_presupuesto():
    """La afirmación de Bergstra y Bengio (2012), medida como distribución.

    Con 27 evaluaciones y 3 ejes de los que solo 1 importa: una rejilla 3x3x3
    prueba nada más 3 valores distintos del eje que importa (los otros 8x3
    puntos repiten esos mismos 3 valores variando ejes que no hacen nada);
    27 puntos al azar prueban 27 valores distintos de ese eje. Se mide sobre
    100 réplicas (semilla por réplica, paisaje distinto en cada una) en vez de
    una corrida suelta, para comparar una distribución de resultados.
    """
    n_replicas = 100
    n_ejes = 3
    n_grid = len(NIVELES) ** n_ejes  # 27: mismo presupuesto para los dos lados

    diferencias = []
    for replica in range(n_replicas):
        paisaje = crear_paisaje(replica, n_ejes=n_ejes)
        regret_grid = regret_minimo_grid(paisaje, NIVELES)

        rng = np.random.default_rng(10_000 + replica)
        puntos = rng.uniform(0.0, 1.0, size=(n_grid, n_ejes))
        regret_random = min(
            paisaje.regret({f"x{i}": float(v) for i, v in enumerate(punto)}) for punto in puntos
        )
        diferencias.append(regret_grid - regret_random)

    diferencias = np.asarray(diferencias)
    victorias_random = int(np.sum(diferencias > 0.0))

    # Dirección 1: random gana la mayoría de las réplicas...
    assert victorias_random > n_replicas // 2
    # ...y la diferencia pareada media favorece a random (grid - random > 0 en promedio).
    assert float(np.mean(diferencias)) > 0.0

    # Dirección contraria: si el presupuesto de random se reduce a un solo punto, el
    # MISMO criterio (mayoría + diferencia pareada media) ya no debe sostenerse: un
    # solo tiro no repite sistemáticamente la ventaja de explorar 27 valores distintos.
    diferencias_un_punto = []
    victorias_con_un_punto = 0
    for replica in range(n_replicas):
        paisaje = crear_paisaje(replica, n_ejes=n_ejes)
        regret_grid = regret_minimo_grid(paisaje, NIVELES)
        rng = np.random.default_rng(20_000 + replica)
        punto = rng.uniform(0.0, 1.0, size=n_ejes)
        regret_un_punto = paisaje.regret({f"x{i}": float(v) for i, v in enumerate(punto)})
        diferencias_un_punto.append(regret_grid - regret_un_punto)
        if regret_grid > regret_un_punto:
            victorias_con_un_punto += 1
    diferencias_un_punto = np.asarray(diferencias_un_punto)

    assert victorias_con_un_punto <= n_replicas // 2
    assert float(np.mean(diferencias_un_punto)) < 0.0


def test_ganancia_de_random_desaparece_sin_ejes_placebo():
    """El MISMO criterio de la prueba anterior (mayoría + diferencia pareada media)
    deja de favorecer a random cuando no hay ejes placebo que desperdiciar: con un
    único eje (que por fuerza es el importante), la rejilla ya no reparte puntos en
    ejes inútiles, así que pierde la desventaja estructural frente a random.
    """
    n_replicas = 100
    n_ejes = 1
    n_grid = len(NIVELES) ** n_ejes  # 3: mismo presupuesto que niveles, sin placebo

    diferencias = []
    for replica in range(n_replicas):
        paisaje = crear_paisaje(replica, n_ejes=n_ejes)
        regret_grid = regret_minimo_grid(paisaje, NIVELES)
        rng = np.random.default_rng(10_000 + replica)
        puntos = rng.uniform(0.0, 1.0, size=(n_grid, n_ejes))
        regret_random = min(
            paisaje.regret({f"x{i}": float(v) for i, v in enumerate(punto)}) for punto in puntos
        )
        diferencias.append(regret_grid - regret_random)

    diferencias = np.asarray(diferencias)
    victorias_random = int(np.sum(diferencias > 0.0))

    # Ni la mayoría...
    assert victorias_random <= n_replicas // 2
    # ...ni la diferencia pareada media favorecen aquí a random.
    assert float(np.mean(diferencias)) < 0.0


def test_ganancia_de_random_se_revierte_si_el_optimo_cae_en_la_rejilla():
    """Otra forma de romper el MISMO criterio: si el óptimo cae exactamente en un
    nivel de la rejilla, `regret_minimo_grid` da 0 (el mejor resultado posible) y
    ningún punto aleatorio continuo lo iguala en la práctica, así que random deja
    de ganar -- pierde, sistemáticamente, en las 100 réplicas.

    `victorias_random == 0` aquí es una consecuencia algebraica (regret_grid es
    el mínimo posible, 0, y regret_random >= 0 siempre), no un resultado
    empírico que dependa de las semillas. La prueba de este archivo con valor
    estadístico real -- la que sí puede fallar según las semillas -- es
    `test_ganancia_de_random_desaparece_sin_ejes_placebo` (n_ejes=1).
    """
    n_replicas = 100
    n_ejes = 3
    paisaje = Paisaje(n_ejes=n_ejes, eje_importante=0, optimo=0.5)  # 0.5 == NIVELES[1]
    regret_grid = regret_minimo_grid(paisaje, NIVELES)
    assert regret_grid == pytest.approx(0.0, abs=1e-12)
    n_grid = len(NIVELES) ** n_ejes

    diferencias = []
    for replica in range(n_replicas):
        rng = np.random.default_rng(30_000 + replica)
        puntos = rng.uniform(0.0, 1.0, size=(n_grid, n_ejes))
        regret_random = min(
            paisaje.regret({f"x{i}": float(v) for i, v in enumerate(punto)}) for punto in puntos
        )
        diferencias.append(regret_grid - regret_random)

    diferencias = np.asarray(diferencias)
    victorias_random = int(np.sum(diferencias > 0.0))

    assert victorias_random == 0
    assert float(np.mean(diferencias)) < 0.0
