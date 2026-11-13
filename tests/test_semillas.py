"""Los juegos de semillas son reproducibles, independientes y separables por fuente."""

import pytest

from taller_hpo.semillas import Semillas, semillas_de


def test_misma_base_mismas_semillas():
    assert semillas_de(42, 5) == semillas_de(42, 5)
    assert semillas_de(42, 5) != semillas_de(43, 5)


def test_juegos_y_fuentes_distintos_entre_si():
    juegos = semillas_de(42, 20)
    valores = [v for s in juegos for v in (s.datos, s.orden, s.init)]
    assert len(set(valores)) == len(valores)


def test_datos_fijos_y_el_resto_varia():
    juegos = semillas_de(42, 4, datos=7)
    assert {s.datos for s in juegos} == {7}
    assert len({(s.orden, s.init) for s in juegos}) == 4


def test_semillas_es_inmutable():
    s = Semillas(datos=1, orden=2, init=3)
    with pytest.raises(AttributeError):
        s.datos = 5


def test_n_invalido():
    with pytest.raises(ValueError):
        semillas_de(1, 0)
