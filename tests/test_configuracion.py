"""La configuración valida claves y tipos en vez de aceptar errores en silencio."""

import numpy as np
import pytest

from taller_hpo.configuracion import get_config

DEFAULTS = {"learning_rate_init": 1e-3, "epocas": 30, "rotar": False, "activacion": "relu"}


def test_sin_overrides_devuelve_una_copia():
    config = get_config(DEFAULTS)
    assert config == DEFAULTS
    config["epocas"] = 1
    assert DEFAULTS["epocas"] == 30


def test_override_se_aplica_y_convierte_al_tipo_del_default():
    config = get_config(DEFAULTS, learning_rate_init=1, epocas=10.0, activacion="tanh")
    assert config["learning_rate_init"] == 1.0 and isinstance(config["learning_rate_init"], float)
    assert config["epocas"] == 10 and isinstance(config["epocas"], int)
    assert get_config(DEFAULTS, epocas=np.int64(5))["epocas"] == 5


def test_clave_desconocida_falla_con_sugerencia():
    with pytest.raises(KeyError, match="quisiste decir 'learning_rate_init'"):
        get_config(DEFAULTS, learning_rate=0.1)


@pytest.mark.parametrize(
    "override",
    [
        {"epocas": 2.5},
        {"epocas": True},
        {"rotar": 1},
        {"learning_rate_init": "0.1"},
        {"activacion": 3},
    ],
)
def test_tipo_incompatible_falla(override):
    with pytest.raises(TypeError):
        get_config(DEFAULTS, **override)
