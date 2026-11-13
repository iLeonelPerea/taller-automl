"""check_importable detecta código a nivel de módulo, y train_ejemplo.py
cumple el contrato que plantilla/README.md le pide a un `train.py`.

Las pruebas de `check_importable` cubren las dos direcciones: los tres
`train.py` mal formados (parse_args al importar, trabajo pesado al importar,
sin `entrenar`) deben fallar, y dos bien formados (`train_ejemplo.py`, y uno
que usa argparse pero SOLO dentro de `main`) deben aprobar. Sin el segundo,
una implementación que rechazara cualquier script con la palabra `argparse`
pasaría igual las pruebas -- ninguna de estas pruebas necesita una tarjeta de
video (no entrenan nada, solo revisan la forma del script).

`train_ejemplo.py` entrena siempre en la tarjeta de video (torch es una
dependencia fija del repo, ver `pyproject.toml`, y `"dispositivo": "auto"`
resuelve siempre a `"cuda"`, ver su docstring): toda prueba de este archivo
que de verdad LLAME a `entrenar` lleva `@pytest.mark.gpu`, que
`tests/conftest.py` hace fallar con un mensaje claro sin una tarjeta de video
NVIDIA con CUDA visible, en vez de saltarla.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch

from taller_hpo.objetivo import en_proceso
from taller_hpo.semillas import Semillas
from taller_hpo.subproceso import por_subproceso
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
PLANTILLA = RAIZ / "plantilla"
TRAIN_EJEMPLO = PLANTILLA / "train_ejemplo.py"
CHECK_IMPORTABLE = PLANTILLA / "check_importable.py"

CONFIG_RAPIDO = {"n_train": 150, "n_val": 150, "epocas": 8, "ancho": 8}
# Forma en la que llegan las semillas por el JSON de por_subproceso (dict). Para
# llamar a train_ejemplo.entrenar directo hace falta el objeto con atributos:
# train_ejemplo.Semillas(**SEMILLAS).
SEMILLAS = {"datos": 1, "orden": 2, "init": 3}
# Config encontrada por una búsqueda TPE real de 15 trials sobre el mismo
# espacio que usa plantilla/buscar_en_proceso.py (ancho/learning_rate_init/
# alpha), con Semillas(datos=0, orden=1, init=2) -- la línea base de
# plantilla/buscar_*.py (SEMILLA_BASE=0) -- y n_ruido=65 (el DEFAULTS de
# train_ejemplo.py). Con ESA semilla exacta da exactitud_val=0.79 contra 0.68
# de DEFAULTS, pero esa es UNA sola corrida: es el mismo par que eligió la
# búsqueda, así que comparar solo ahí es la maldición del ganador (ver el
# docstring de train_ejemplo.py).
# `test_train_ejemplo_defaults_queda_por_debajo_de_una_buena_config_en_media`,
# más abajo, mide la brecha en media sobre semillas nuevas de orden/init (con
# datos=0: cifra optimista, ver su docstring).
CONFIG_BUENA = {
    "ancho": 56,
    "learning_rate_init": 0.0029131813222817086,
    "alpha": 0.07221575421635598,
}


# torch es una dependencia fija del repo (ver pyproject.toml): el taller
# entrena siempre en la tarjeta de video. Las pruebas que de verdad entrenan
# llevan `@pytest.mark.gpu` (ver el docstring del módulo), que
# `tests/conftest.py` hace fallar con un mensaje claro sin una tarjeta de
# video visible.


CON_PARSE_ARGS_EN_MODULO = """
import argparse

_parser = argparse.ArgumentParser()
_args = _parser.parse_args()


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

CON_TRABAJO_PESADO_EN_MODULO = """
import time

time.sleep(5)


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

SIN_ENTRENAR = """
def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

CON_PARSE_KNOWN_ARGS_EN_MODULO = """
import argparse

_parser = argparse.ArgumentParser()
_args, _resto = _parser.parse_known_args()


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

CON_POSICIONAL_A_NIVEL_DE_MODULO = """
import argparse

_p = argparse.ArgumentParser()
_p.add_argument("datos")
ARGS = _p.parse_args()


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

BIEN_FORMADO_CON_ARGPARSE_EN_MAIN = """
import argparse

DEFAULTS = {"dispositivo": "auto"}


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--salida", required=True)
    parser.parse_args(argv)


if __name__ == "__main__":
    main()
"""

SIN_DISPOSITIVO_EN_DEFAULTS = """
DEFAULTS = {"ancho": 16}


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""

CON_DISPOSITIVO_CPU_EN_DEFAULTS = """
DEFAULTS = {"ancho": 16, "dispositivo": "cpu"}


def entrenar(config, semillas, reportar=None):
    return {"exactitud_val": 0.5}


def main(argv=None):
    pass


if __name__ == "__main__":
    main()
"""


def _importar(ruta: Path, nombre: str):
    """Importa un .py fuera de sys.path (mismo truco que usa check_importable.py
    para revisar un train.py sin ensuciar el proceso de pruebas)."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


check_importable = _importar(CHECK_IMPORTABLE, "_check_importable_bajo_prueba")


# --- check_importable: los tres defectos que debe detectar ---------------------


def test_detecta_parse_args_a_nivel_de_modulo(tmp_path):
    script = tmp_path / "con_parse_args.py"
    script.write_text(CON_PARSE_ARGS_EN_MODULO, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "parse_args" in problema


def test_detecta_trabajo_pesado_a_nivel_de_modulo(tmp_path):
    script = tmp_path / "con_sleep.py"
    script.write_text(CON_TRABAJO_PESADO_EN_MODULO, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=0.5)
    assert problema is not None
    assert "trabajo pesado" in problema


def test_detecta_falta_de_entrenar(tmp_path):
    script = tmp_path / "sin_entrenar.py"
    script.write_text(SIN_ENTRENAR, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "entrenar" in problema


def test_detecta_parse_known_args_a_nivel_de_modulo(tmp_path):
    # parse_known_args() es tolerante a argumentos desconocidos: con un
    # fragmento que en vez de "--help" pasara un argumento inventado
    # (sys.argv = [ruta, "--argumento-inexistente"]), parse_known_args() lo
    # ignoraría en silencio y revisar() aprobaría el script igual (devolvería
    # None). "--help" lo intercepta igual que a parse_args(), sea estricto o
    # tolerante.
    script = tmp_path / "con_parse_known_args.py"
    script.write_text(CON_PARSE_KNOWN_ARGS_EN_MODULO, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "parse_args" in problema


def test_detecta_parse_args_con_posicional_a_nivel_de_modulo(tmp_path):
    # Cubre el override de sys.argv dentro del fragmento (`sys.argv = [ruta,
    # "--help"]`): sin él, `parse_args()` vería solo [ruta] (el fragmento corre
    # como `python -c FRAGMENTO ruta`) y el posicional "datos" lo consumiría sin
    # quejarse de nada, así que un script con un parser a nivel de módulo se
    # aprobaría igual pese a tener el defecto que esta revisión busca.
    script = tmp_path / "con_posicional.py"
    script.write_text(CON_POSICIONAL_A_NIVEL_DE_MODULO, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "parse_args" in problema


# --- check_importable: los dos bien formados que debe aprobar ------------------


def test_aprueba_train_ejemplo():
    assert check_importable.revisar(str(TRAIN_EJEMPLO), timeout_s=20.0) is None


def test_aprueba_argparse_dentro_de_main(tmp_path):
    script = tmp_path / "bien_formado.py"
    script.write_text(BIEN_FORMADO_CON_ARGPARSE_EN_MAIN, encoding="utf-8")
    assert check_importable.revisar(str(script), timeout_s=5.0) is None


def test_aprueba_train_py_que_importa_un_modulo_hermano(tmp_path):
    # Falso positivo real: un train.py que hace `from datos import cargar`
    # funciona con `python train.py` (la carpeta del script entra a sys.path),
    # pero fallaba en check_importable porque `python -c` no la agrega.
    (tmp_path / "datos.py").write_text("def cargar():\n    return [1, 2, 3]\n", encoding="utf-8")
    script = tmp_path / "usa_modulo_hermano.py"
    script.write_text(
        "from datos import cargar\n"
        "\n"
        "DEFAULTS = {'dispositivo': 'auto'}\n"
        "\n"
        "\n"
        "def entrenar(config, semillas, reportar=None):\n"
        "    return {'exactitud_val': sum(cargar())}\n"
        "\n"
        "\n"
        "def main(argv=None):\n"
        "    pass\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )
    assert check_importable.revisar(str(script), timeout_s=5.0) is None


# --- check_importable: DEFAULTS['dispositivo'] es parte del contrato ------------


def test_detecta_falta_de_dispositivo_en_defaults(tmp_path):
    """El taller entrena siempre en la tarjeta de video (ver
    `plantilla/README.md`): un `train.py` que define `entrenar`/`main` pero no
    declara `'dispositivo'` en `DEFAULTS` no cumple el contrato. Esta prueba
    comprueba que `check_importable.revisar` detecta esa falta y menciona
    `'dispositivo'` en el mensaje."""
    script = tmp_path / "sin_dispositivo.py"
    script.write_text(SIN_DISPOSITIVO_EN_DEFAULTS, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "dispositivo" in problema


def test_detecta_dispositivo_cpu_en_defaults(tmp_path):
    """La revisión no se detiene en que la clave EXISTA: también valida su
    FORMATO. Un `train.py` con `DEFAULTS['dispositivo'] = 'cpu'` declara la
    clave, pero ese valor no tiene la forma `'auto'` o `'cuda'` (con `':N'`
    opcional) que el taller acepta, así que tampoco cumple el contrato."""
    script = tmp_path / "dispositivo_cpu.py"
    script.write_text(CON_DISPOSITIVO_CPU_EN_DEFAULTS, encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "dispositivo" in problema


@pytest.mark.parametrize("valor", ["cuda:0x", "autocpu"])
def test_detecta_formato_de_dispositivo_invalido_en_defaults(tmp_path, valor):
    """La validación de FORMATO exige que `'dispositivo'` sea exactamente
    `'auto'` o `'cuda'` (con `':N'` opcional) de punta a punta, vía
    `re.fullmatch` en `check_importable.py`: sin esta prueba, una
    implementación que usara `re.match` (que solo exige que el patrón
    coincida con el PRINCIPIO de la cadena, no con toda ella) aprobaría
    `'cuda:0x'` (empieza como `'cuda:0'`) y `'autocpu'` (empieza como
    `'auto'`) igual, con código 0."""
    script = tmp_path / f"dispositivo_formato_invalido_{valor.replace(':', '_')}.py"
    script.write_text(
        "DEFAULTS = {'ancho': 16, 'dispositivo': " + repr(valor) + "}\n"
        "\n"
        "\n"
        "def entrenar(config, semillas, reportar=None):\n"
        "    return {'exactitud_val': 0.5}\n"
        "\n"
        "\n"
        "def main(argv=None):\n"
        "    pass\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n",
        encoding="utf-8",
    )
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "dispositivo" in problema


# --- check_importable: mensajes y casos borde -----------------------------------


def test_falta_la_guarda_main_da_codigo_2_con_mensaje_sobre_la_guarda(tmp_path):
    # `main()` llamado a nivel de módulo (sin el `if __name__ == "__main__":`)
    # dispara el mismo código 2 que un parse_args() suelto: el mensaje debe
    # cubrir las dos causas, no solo "hay un parse_args() a nivel de módulo"
    # (que aquí sería engañoso: parse_args SÍ está dentro de main()).
    script = tmp_path / "sin_guarda.py"
    script.write_text(
        "import argparse\n"
        "\n"
        "\n"
        "def entrenar(config, semillas, reportar=None):\n"
        "    return {'exactitud_val': 0.5}\n"
        "\n"
        "\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config', required=True)\n"
        "    parser.add_argument('--salida', required=True)\n"
        "    parser.parse_args(argv)\n"
        "\n"
        "\n"
        "main()\n",  # falta la guarda: se ejecuta al importar
        encoding="utf-8",
    )
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "guarda" in problema


def test_rechaza_entrenar_y_main_no_invocables(tmp_path):
    # hasattr(modulo, "entrenar") es True incluso si entrenar = None: el
    # contrato exige poder LLAMARLOS, no solo que el nombre exista.
    script = tmp_path / "no_invocables.py"
    script.write_text("entrenar = None\nmain = None\n", encoding="utf-8")
    problema = check_importable.revisar(str(script), timeout_s=5.0)
    assert problema is not None
    assert "entrenar" in problema
    assert "main" in problema


# --- check_importable: línea de comandos (código de salida + mensaje) -----------


def test_cli_sale_0_y_confirma_si_aprueba():
    resultado = subprocess.run(
        [sys.executable, str(CHECK_IMPORTABLE), str(TRAIN_EJEMPLO)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "entrenar" in resultado.stdout


def test_cli_sale_1_y_explica_si_falla(tmp_path):
    script = tmp_path / "sin_entrenar.py"
    script.write_text(SIN_ENTRENAR, encoding="utf-8")
    resultado = subprocess.run(
        [sys.executable, str(CHECK_IMPORTABLE), str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert resultado.returncode == 1
    assert "entrenar" in resultado.stderr


# --- train_ejemplo.py: el contrato del tesista ----------------------------------


@pytest.mark.gpu
def test_train_ejemplo_main_produce_el_json_esperado(tmp_path):
    entrada = tmp_path / "entrada.json"
    salida = tmp_path / "salida.json"
    entrada.write_text(
        json.dumps({"config": CONFIG_RAPIDO, "semillas": SEMILLAS}), encoding="utf-8"
    )

    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_main")
    train_ejemplo.main(["--config", str(entrada), "--salida", str(salida)])

    metricas = json.loads(salida.read_text(encoding="utf-8"))
    assert set(metricas) == {"exactitud_val", "epocas"}
    assert 0.0 <= metricas["exactitud_val"] <= 1.0
    assert metricas["epocas"] == CONFIG_RAPIDO["epocas"]


def test_train_ejemplo_clave_desconocida_falla():
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_clave")
    with pytest.raises(KeyError, match="tasa_de_aprendizaje"):
        train_ejemplo.entrenar({"tasa_de_aprendizaje": 0.1}, train_ejemplo.Semillas(**SEMILLAS))


@pytest.mark.parametrize("dispositivo", ["tpu", "cpu"])
def test_train_ejemplo_dispositivo_invalido_falla(dispositivo):
    # "cpu" es un valor inválido como cualquier otro fuera de auto/cuda(:N)
    # (ver el docstring de `_resolver_dispositivo`): "tpu" falla igual. Las
    # dos ramas fallan por el mismo ValueError de forma, ANTES de tocar
    # `torch.cuda.is_available()`, así que esta prueba no necesita una
    # tarjeta de video real.
    train_ejemplo = _importar(
        TRAIN_EJEMPLO, f"_train_ejemplo_bajo_prueba_dispositivo_{dispositivo}"
    )
    with pytest.raises(ValueError, match="dispositivo"):
        train_ejemplo.entrenar({"dispositivo": dispositivo}, train_ejemplo.Semillas(**SEMILLAS))


def test_train_ejemplo_activacion_invalida_falla():
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_activacion")
    with pytest.raises(ValueError, match="activacion"):
        train_ejemplo.entrenar({"activacion": "gelu"}, train_ejemplo.Semillas(**SEMILLAS))


@pytest.mark.gpu
def test_train_ejemplo_misma_semilla_mismo_resultado():
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_repro")
    s = train_ejemplo.Semillas(**SEMILLAS)
    config = {**CONFIG_RAPIDO, "dispositivo": "cuda"}
    assert train_ejemplo.entrenar(config, s) == train_ejemplo.entrenar(config, s)


@pytest.mark.gpu
def test_train_ejemplo_cada_fuente_de_azar_cambia_el_resultado():
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_azar")
    base = train_ejemplo.entrenar(CONFIG_RAPIDO, train_ejemplo.Semillas(**SEMILLAS))
    for otra in (
        {"datos": 9, "orden": 2, "init": 3},
        {"datos": 1, "orden": 9, "init": 3},
        {"datos": 1, "orden": 2, "init": 9},
    ):
        assert train_ejemplo.entrenar(CONFIG_RAPIDO, train_ejemplo.Semillas(**otra)) != base, otra


# Tres semillas independientes para
# `test_train_ejemplo_cada_hiperparametro_cambia_el_resultado`: con una sola
# semilla, alpha=0.1 y batch_size=17 mueven exactitud_val apenas 1 acierto de
# 150, margen que un simple cambio de orden de suma en BLAS entre máquinas
# podría borrar por accidente. Verificado que, con las 3 semillas, ninguna de
# las 9 combinaciones de abajo empata en las 3 a la vez.
_SEMS_HIPERPARAM = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("clave", "valor"),
    [
        ("ancho", 24),
        ("learning_rate_init", 1e-3),
        ("alpha", 1e-1),
        ("n_capas", 2),
        ("activacion", "tanh"),
        ("batch_size", 17),
        ("ruido", 0.4),
        ("n_ruido", 5),
        ("n_val", 50),
    ],
)
def test_train_ejemplo_cada_hiperparametro_cambia_el_resultado(clave, valor):
    """Cada clave de `DEFAULTS` que es hiperparámetro de búsqueda llega a `entrenar` y
    cambia su salida: una clave que `get_config` acepta pero que se ignora en silencio
    haría que una búsqueda explorara sin que nada cambie. Quedan fuera `n_train` (fijo en
    `CONFIG_RAPIDO`), `epocas` y `dispositivo`, que cubren otras pruebas; `n_val` entra
    porque ninguna otra lo varía.

    Compara sobre tres semillas (`_SEMS_HIPERPARAM`). Cada valor alterno cambia el resultado
    en al menos dos de las tres, así que un empate por redondeo no da un falso rojo."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, f"_train_ejemplo_bajo_prueba_hiperparam_{clave}")
    semillas = [train_ejemplo.Semillas(*trio) for trio in _SEMS_HIPERPARAM]
    base = [train_ejemplo.entrenar(CONFIG_RAPIDO, s) for s in semillas]
    mutado = [train_ejemplo.entrenar({**CONFIG_RAPIDO, clave: valor}, s) for s in semillas]
    assert mutado != base, (clave, valor, base, mutado)


@pytest.mark.gpu
def test_train_ejemplo_reportar_recibe_cada_epoca_y_puede_parar():
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_reportar")
    config = {**CONFIG_RAPIDO, "epocas": 10}
    llamadas = []

    def reportar(epoca, valor):
        llamadas.append(epoca)
        return epoca == 2

    metricas = train_ejemplo.entrenar(config, train_ejemplo.Semillas(**SEMILLAS), reportar)
    assert llamadas == [0, 1, 2]
    assert metricas["epocas"] == 3


@pytest.mark.gpu
def test_train_ejemplo_no_toca_el_rng_global_de_torch():
    """`entrenar` no cambia el estado del generador aleatorio global de torch (ni el del
    anfitrión ni el de la tarjeta de video), con una o varias llamadas y cualquier semilla.

    El estado "antes" se fija con un centinela (`torch.manual_seed(20260925)`) y la prueba
    usa una semilla de `init` (424242) que no usa ninguna otra: sin eso, una implementación
    con `torch.manual_seed(semillas.init)` podría dejar "antes" y "después" iguales por
    accidente. `test_comparacion_de_estado_de_rng_es_sensible_a_manual_seed` prueba que
    esta comparación sí detecta un cambio real.
    """
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_rng_global")
    estado_anfitrion_original = torch.get_rng_state().clone()
    estado_cuda_original = torch.cuda.get_rng_state().clone()
    try:
        torch.manual_seed(20260925)
        estado_anfitrion_antes = torch.get_rng_state().clone()
        estado_cuda_antes = torch.cuda.get_rng_state().clone()

        s = train_ejemplo.Semillas(datos=1, orden=2, init=424242)
        train_ejemplo.entrenar(CONFIG_RAPIDO, s)
        train_ejemplo.entrenar({**CONFIG_RAPIDO, "dispositivo": "cuda"}, s)

        assert torch.equal(torch.get_rng_state(), estado_anfitrion_antes)
        assert torch.equal(torch.cuda.get_rng_state(), estado_cuda_antes)
    finally:
        torch.set_rng_state(estado_anfitrion_original)
        torch.cuda.set_rng_state(estado_cuda_original)


def test_comparacion_de_estado_de_rng_es_sensible_a_manual_seed():
    """La otra dirección de la prueba de arriba: si algo SÍ llama a
    `torch.manual_seed` (el generador global), la misma comparación de estado
    antes/después debe detectarlo. Sin esta prueba, la de arriba podría estar
    comparando dos objetos que siempre resultan "iguales" por accidente (por
    ejemplo, si `torch.get_rng_state()` devolviera una copia nueva y vacía cada
    vez) y aprobar aunque `entrenar` sí tocara el generador aleatorio global."""
    estado_antes = torch.get_rng_state().clone()
    try:
        torch.manual_seed(999999)
        assert not torch.equal(torch.get_rng_state(), estado_antes)
    finally:
        torch.set_rng_state(estado_antes)


@pytest.mark.gpu
def test_importar_train_ejemplo_no_inicializa_cuda():
    """Importar `train_ejemplo.py` no debe tocar CUDA: todo el entrenamiento
    debe correr en la tarjeta de video, pero eso es responsabilidad de
    `entrenar` (vía `_resolver_dispositivo`), no del módulo al cargarse --
    inicializar CUDA al importar pagaría crear el contexto CUDA (parte del
    costo de CUDA por trial, ver `plantilla/README.md`) incluso para quien
    solo quiere `check_importable.py` o importar el módulo por sus
    constantes, sin entrenar nada.

    Esta prueba corre en un SUBPROCESO limpio (nunca en el proceso de la
    suite: aquí ya se importó `torch` y probablemente ya se inicializó CUDA
    por otras pruebas de este archivo, así que la comparación tendría que ser
    un antes/después frágil en vez de una lectura directa de
    `torch.cuda.is_initialized()` justo después de importar). La prueba
    necesita una tarjeta de video real (marca `gpu`): sin ella,
    `torch.cuda.is_initialized()` daría `False` de todos modos porque no hay
    nada que inicializar, así que la prueba no distinguiría un import
    correcto de uno que sí inicializa CUDA."""
    fragmento = (
        "import importlib.util\n"
        "import torch\n"
        f"spec = importlib.util.spec_from_file_location('t', {str(TRAIN_EJEMPLO)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(torch.cuda.is_initialized())\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", fragmento], capture_output=True, text=True, timeout=30
    )
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stdout.strip() == "False", resultado.stdout


@pytest.mark.gpu
def test_train_ejemplo_funciona_con_en_proceso_y_coincide_con_por_subproceso(tmp_path):
    """train_ejemplo.entrenar acepta el `Semillas` real de `taller_hpo` (acceso
    por atributo) sin importarlo: `en_proceso` se lo pasa tal cual porque llama
    a `entrenar` directo, y `por_subproceso` manda el mismo `dataclasses.asdict`
    por JSON al mismo script. Las dos rutas deben dar el mismo valor para la
    misma config y las mismas semillas (esto exige que `entrenar` acceda a
    `semillas` por ATRIBUTO, no por clave: `en_proceso` pasa un objeto
    `Semillas`, no un dict)."""
    config = {**CONFIG_RAPIDO, "dispositivo": "cuda"}
    semillas_core = Semillas(datos=1, orden=2, init=3)

    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_en_proceso")
    objetivo_en_proceso = en_proceso(
        train_ejemplo.entrenar,
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=semillas_core,
        fijos=config,
    )
    study_directo = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study_directo.optimize(objetivo_en_proceso, n_trials=1)
    (t_directo,) = study_directo.trials
    assert t_directo.state == optuna.trial.TrialState.COMPLETE

    objetivo_subproceso = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=semillas_core,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=config,
    )
    study_subproceso = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study_subproceso.optimize(objetivo_subproceso, n_trials=1)
    (t_subproceso,) = study_subproceso.trials
    assert t_subproceso.state == optuna.trial.TrialState.COMPLETE

    assert t_directo.value == t_subproceso.value


# 8 semillas de orden/init NUEVAS (no las de línea base que usó la búsqueda
# que encontró CONFIG_BUENA), con datos=0 fijo -- el mismo dataset que usan
# plantilla/buscar_*.py. Compartidas por las dos pruebas de abajo (positiva y
# negativa) para que midan sobre la MISMA muestra de semillas.
_SEMILLAS_GAP = [(100 + i, 200 + i) for i in range(8)]


@pytest.mark.gpu
def test_train_ejemplo_defaults_queda_por_debajo_de_una_buena_config_en_media():
    """DEFAULTS queda por debajo de CONFIG_BUENA en media sobre 8 semillas nuevas de
    orden/init (`_SEMILLAS_GAP`), nunca con una sola: comparar contra el ganador con la
    misma semilla con la que se eligió es la maldición del ganador. `datos=0` queda fijo:
    es el dataset de línea base de `plantilla/buscar_*.py`.

    En la medición de referencia, CONFIG_BUENA ganó en las 8 semillas por unos 3 puntos
    (cifra optimista: `datos=0` es la partición con la que se eligió) y, fuera de selección,
    en 17 de 19 datasets. Los umbrales (>=0.015 de diferencia y >=6/8 victorias) dejan
    margen para diferencias de punto flotante. La dirección contraria la prueba
    `test_train_ejemplo_sin_ruido_defaults_no_queda_por_debajo`."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_gap_medio")

    defaults = [
        train_ejemplo.entrenar({}, train_ejemplo.Semillas(datos=0, orden=o, init=i))[
            "exactitud_val"
        ]
        for o, i in _SEMILLAS_GAP
    ]
    buena = [
        train_ejemplo.entrenar(CONFIG_BUENA, train_ejemplo.Semillas(datos=0, orden=o, init=i))[
            "exactitud_val"
        ]
        for o, i in _SEMILLAS_GAP
    ]
    media_defaults = sum(defaults) / len(defaults)
    media_buena = sum(buena) / len(buena)
    gana_buena = sum(1 for d, b in zip(defaults, buena, strict=True) if b > d)

    assert media_defaults <= 0.78, (media_defaults, defaults)
    assert media_buena - media_defaults >= 0.015, (media_defaults, media_buena)
    assert gana_buena >= 6, (defaults, buena)


@pytest.mark.gpu
def test_train_ejemplo_sin_ruido_defaults_no_queda_por_debajo():
    """Dirección de sensibilidad de la prueba de arriba (para que no sea un
    cheque vacío): con `n_ruido=0` el problema pierde las columnas de ruido
    que le dan dificultad (ver el docstring del módulo) y `DEFAULTS` ya queda
    cerca del techo del espacio de búsqueda -- ahí CONFIG_BUENA (más ancho,
    más `alpha`) no ayuda, y de hecho pierde: medido con las mismas 8
    semillas de `_SEMILLAS_GAP`, media DEFAULTS=0.8812 contra media
    CONFIG_BUENA=0.8412 (diferencia -0.0400). Si esta versión del problema se
    volviera trivial otra vez (por ejemplo, si alguien bajara `n_ruido` en
    DEFAULTS), la prueba de arriba fallaría -- y si alguien "arreglara" esa
    prueba solo aflojando los umbrales sin que la brecha real exista, ESTA
    prueba seguiría fallando en su propia dirección."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_gap_n_ruido_0")

    defaults = [
        train_ejemplo.entrenar({"n_ruido": 0}, train_ejemplo.Semillas(datos=0, orden=o, init=i))[
            "exactitud_val"
        ]
        for o, i in _SEMILLAS_GAP
    ]
    buena = [
        train_ejemplo.entrenar(
            {**CONFIG_BUENA, "n_ruido": 0}, train_ejemplo.Semillas(datos=0, orden=o, init=i)
        )["exactitud_val"]
        for o, i in _SEMILLAS_GAP
    ]
    media_defaults = sum(defaults) / len(defaults)
    media_buena = sum(buena) / len(buena)
    gana_buena = sum(1 for d, b in zip(defaults, buena, strict=True) if b > d)

    assert media_defaults >= 0.85, (media_defaults, defaults)
    assert media_buena - media_defaults <= 0.01, (media_defaults, media_buena)
    # La brecha no solo "no es positiva": se invierte de verdad. Sin estos dos
    # asserts, un error que dejara la brecha en 0 (o en un +0.009 que aún
    # cumple <= 0.01 de arriba) pasaría igual, aunque el docstring de este
    # módulo y el de arriba afirmen una inversión que no estarían probando.
    assert media_buena < media_defaults, (media_defaults, media_buena)
    assert gana_buena <= 2, (defaults, buena)


@pytest.mark.gpu
def test_train_ejemplo_dispositivo_auto_siempre_resuelve_a_cuda():
    """Todo el entrenamiento corre siempre en la tarjeta de video (ver el
    docstring del módulo): `_resolver_dispositivo("auto")` debe devolver
    siempre "cuda", sin importar ninguna otra condición de la máquina."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_dispositivo_auto")
    assert train_ejemplo.DEFAULTS["dispositivo"] == "auto"
    assert train_ejemplo._resolver_dispositivo("auto") == "cuda"


def test_train_ejemplo_sin_tarjeta_de_video_falla_con_mensaje_claro(monkeypatch):
    """Política dura del taller: sin una tarjeta de video NVIDIA con CUDA
    visible, `entrenar` se detiene con un `RuntimeError` explícito. Esta
    prueba fuerza `torch.cuda.is_available` a `False` para simular esa
    ausencia sin necesitar una tarjeta de video real, así corre igual en
    cualquier máquina."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_sin_cuda")
    monkeypatch.setattr(train_ejemplo.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="tarjeta de video"):
        train_ejemplo._resolver_dispositivo("auto")
    with pytest.raises(RuntimeError, match="tarjeta de video"):
        train_ejemplo._resolver_dispositivo("cuda")


def test_train_ejemplo_mensaje_sin_tarjeta_coincide_con_la_fuente_unica(monkeypatch):
    """`train_ejemplo.py` es autónomo (un tesista lo copia sin `taller_hpo`,
    ver su docstring): no importa `taller_hpo.tarjeta_de_video`, así que
    conserva su PROPIA copia literal de `MENSAJE_SIN_TARJETA`. Esta prueba
    guarda que esa copia no diverja en silencio de la fuente única -- sin
    ella, alguien podría cambiar una sola palabra de un lado y dejar al
    tesista viendo dos mensajes distintos para el mismo aviso. La prueba no
    necesita una tarjeta de video real: fuerza su ausencia para leer el texto
    completo del `RuntimeError`."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_mensaje_sin_tarjeta")
    monkeypatch.setattr(train_ejemplo.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        train_ejemplo._resolver_dispositivo("auto")
    assert str(excinfo.value) == MENSAJE_SIN_TARJETA


@pytest.mark.gpu
def test_train_ejemplo_datos_residen_en_el_dispositivo_sin_copias_por_lote():
    """Detecta copiar cada minibatch del anfitrión al dispositivo DENTRO del
    bucle, en vez de subir los datos residentes una sola vez: cuenta las
    llamadas a `torch.Tensor.to` durante un `entrenar()` completo.

    Con los datos residentes de antemano, el número de llamadas es FIJO
    (los 4 tensores de datos + lo que `modulo.to(dispositivo)` mueve por
    parámetro + un vector de orden por época), nunca proporcional al número
    de lotes. Medido con CONFIG_RAPIDO (epocas=8, batch_size default=32,
    5 lotes/época con n_train=150): 16 llamadas -- muy por debajo
    de las ~50+ que darían copias por lote (8 épocas x 5 lotes x al menos 1
    `.to()` por lote, más las 16 de la versión correcta)."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_residencia")
    contador = {"n": 0}
    original_to = torch.Tensor.to

    def to_contado(self, *args, **kwargs):
        contador["n"] += 1
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = to_contado
    try:
        train_ejemplo.entrenar(CONFIG_RAPIDO, train_ejemplo.Semillas(**SEMILLAS))
    finally:
        torch.Tensor.to = original_to

    assert contador["n"] <= CONFIG_RAPIDO["epocas"] + 12, contador["n"]


@pytest.mark.gpu
@pytest.mark.parametrize("dispositivo", ["auto", "cuda"])
def test_train_ejemplo_las_capas_corren_en_el_dispositivo_pedido(monkeypatch, dispositivo):
    """Que `_resolver_dispositivo` devuelva el dispositivo correcto no basta:
    esta prueba verifica que `entrenar` de verdad LO USE. Detecta dos
    defectos que las pruebas de arriba no notarían: ignorar el dispositivo ya
    resuelto (la prueba de `_resolver_dispositivo` de arriba solo mira el
    resolvedor, no si `entrenar` respeta su resultado) y mover solo el MODELO
    al dispositivo equivocado sin tocar los tensores de datos (la prueba de
    residencia de arriba solo cuenta llamadas a `Tensor.to`, no en qué
    dispositivo terminan).

    Espía `torch.nn.functional.linear` (lo que llama `_MLP.forward` en cada
    capa, vía `nn.functional.linear`) y registra en qué dispositivo están `x`
    y el peso en CADA llamada -- deben ser todas el mismo dispositivo
    esperado, nunca una mezcla ni el dispositivo equivocado."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, f"_train_ejemplo_bajo_prueba_disp_{dispositivo}")
    vistos = set()
    original_linear = torch.nn.functional.linear

    def linear_espia(x, peso, sesgo=None):
        vistos.add((x.device.type, peso.device.type))
        return original_linear(x, peso, sesgo)

    monkeypatch.setattr(torch.nn.functional, "linear", linear_espia)
    train_ejemplo.entrenar(
        {**CONFIG_RAPIDO, "dispositivo": dispositivo}, train_ejemplo.Semillas(**SEMILLAS)
    )
    assert vistos == {("cuda", "cuda")}, vistos


@pytest.mark.gpu
def test_train_ejemplo_sin_copias_anfitrion_a_tarjeta_por_lote():
    """Detecta copiar cada minibatch del anfitrión al dispositivo DENTRO del
    bucle con `torch.tensor(x_tr[sel], device=dispositivo)`, en vez de subir
    los datos residentes una sola vez: la prueba de arriba
    (`test_train_ejemplo_datos_residen_en_el_dispositivo_sin_copias_por_lote`)
    cuenta llamadas a `Tensor.to`, pero `torch.tensor(..., device=...)` NO
    pasa por `Tensor.to` y ese defecto la esquivaría sin que ninguna otra
    prueba lo note. Perfila los eventos CUDA de un `entrenar()` completo y
    cuenta cuántos son transferencias host-a-device (`Memcpy HtoD`): con los datos
    residentes el número es FIJO (tensores de datos + parámetros + un vector
    de orden por época), nunca proporcional al número de lotes."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_htod")
    # Calienta: la primera llamada en cuda paga alocación de memoria/kernels
    # que no son transferencias de datos, pero conviene no mezclarlas con la
    # medición.
    train_ejemplo.entrenar(
        {**CONFIG_RAPIDO, "dispositivo": "cuda"}, train_ejemplo.Semillas(**SEMILLAS)
    )
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as perfil:
        train_ejemplo.entrenar(
            {**CONFIG_RAPIDO, "dispositivo": "cuda"}, train_ejemplo.Semillas(**SEMILLAS)
        )
    htod = [e for e in perfil.events() if "HtoD" in e.name]
    assert len(htod) <= CONFIG_RAPIDO["epocas"] + 12, len(htod)


@pytest.mark.gpu
def test_train_ejemplo_permuta_el_orden_en_cada_epoca():
    """Detecta recalcular
    `np.random.default_rng(semillas.orden).permutation(n)` DENTRO del bucle,
    con un generador NUEVO en cada época, en vez de llamar `.permutation()`
    una vez por época sobre el MISMO generador ya avanzado: un generador
    nuevo sembrado igual siempre da la MISMA primera permutación, así que con
    ese defecto todas las épocas verían el mismo orden -- ninguna otra prueba
    de este archivo lo notaría porque ninguna mira más de una época a la vez.

    Espía `torch.from_numpy`, llamado una vez por época con el array de la
    permutación (`orden_t = torch.from_numpy(orden).to(dispositivo)`),
    distinguible de los cuatro tensores de datos (llamados antes del bucle,
    una sola vez) por su dtype: `.permutation()` devuelve `int64`, los datos
    se suben como `float32`."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_permuta")
    n_train = CONFIG_RAPIDO["n_train"]
    ordenes = []
    original_from_numpy = torch.from_numpy

    def from_numpy_espia(arr):
        if arr.ndim == 1 and arr.dtype == np.int64 and arr.shape[0] == n_train:
            ordenes.append(arr.copy())
        return original_from_numpy(arr)

    torch.from_numpy = from_numpy_espia
    try:
        train_ejemplo.entrenar(CONFIG_RAPIDO, train_ejemplo.Semillas(**SEMILLAS))
    finally:
        torch.from_numpy = original_from_numpy

    assert len(ordenes) == CONFIG_RAPIDO["epocas"], len(ordenes)
    for orden in ordenes:
        assert sorted(orden.tolist()) == list(range(n_train))
    assert not np.array_equal(ordenes[0], ordenes[1]), "misma permutación en dos épocas distintas"


@pytest.mark.gpu
def test_train_ejemplo_acepta_semillas_con_enteros_de_numpy():
    """`objetivo.en_proceso` pasa el `Semillas` que recibe DIRECTO a
    `entrenar`, sin convertir sus campos a int nativo (el `int()` que sí
    tiene el `objective` que devuelve `objetivo.en_proceso` es solo para los
    `user_attrs` serializables) -- ver el comentario junto a
    `torch.Generator().manual_seed(int(semillas.init))` en train_ejemplo.py.
    `torch.Generator.manual_seed` revienta con
    `TypeError: an integer is required` ante un entero de numpy (np.int64),
    verificado ejecutándolo.
    `tests/test_subproceso.py::test_semillas_con_enteros_numpy_quedan_como_int_nativo`
    ya cubre la ruta por_subproceso (que sí serializa a JSON, y ahí los enteros de numpy
    llegan como int nativo del otro lado); esta prueba cubre la ruta
    en_proceso, donde `Semillas` llega tal cual."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_bajo_prueba_semillas_numpy")
    s_nativo = train_ejemplo.Semillas(datos=1, orden=2, init=3)
    s_numpy = train_ejemplo.Semillas(datos=np.int64(1), orden=np.uint32(2), init=np.int64(3))

    resultado_nativo = train_ejemplo.entrenar(CONFIG_RAPIDO, s_nativo)
    resultado_numpy = train_ejemplo.entrenar(CONFIG_RAPIDO, s_numpy)

    assert resultado_nativo == resultado_numpy
