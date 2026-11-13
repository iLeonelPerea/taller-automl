"""Lo que comparten los verificadores de los ejercicios, y nada más.

Cada ejercicio vive en su carpeta (`ejercicios/NN_nombre/`) con cuatro archivos:
`README.md`, `ejercicio.py` (el que tú completas), `solucion.py` (el mismo archivo ya
resuelto) y `comprobar.py` (las revisiones). `verificar.py` carga `comprobar.py`, le
entrega tu archivo y te muestra qué está bien y qué te falta. Este módulo guarda las
cuatro piezas que todos los ejercicios usan:

- `Revision`: el resultado de una revisión (lo que revisa, si pasó y qué hacer si no).
- `revisar`: arma una `Revision` a partir de una prueba y traduce a un mensaje amable los
  dos tropiezos típicos de un ejercicio a medias: un hueco sin completar y un error de
  tu código.
- `cargar_modulo`: carga un `.py` por su ruta, para revisar `ejercicio.py` o `solucion.py`
  sin copiarlos a ningún lado.
- `CONFIG_RAPIDA`: los ajustes que acortan lo que sí puedes acortar del modelo de práctica.

## Cómo escribes el `comprobar.py` de un ejercicio

Defines `comprobar(modulo) -> list[Revision]`, donde `modulo` es tu `ejercicio.py` (o
`solucion.py`) ya cargado. Cada revisión llama a UNA función del tesista con entradas que
tú eliges y compara con lo que calculas por tu cuenta:

    from _comun import Revision, revisar

    def comprobar(modulo) -> list[Revision]:
        def devuelve_el_doble() -> str | None:
            obtenido = modulo.doble(21)
            if obtenido != 42:
                return f"Con 21 esperaba 42 y obtuve {obtenido!r}."
            return None

        return [revisar("`doble` devuelve el doble de su argumento", devuelve_el_doble,
                        "Multiplica el argumento por 2 y devuélvelo.")]

La prueba devuelve `None` si todo está bien, o una frase que cuenta qué vio mal. `revisar`
junta esa frase con la sugerencia (`que_hacer`) y arma el mensaje completo. Cuatro reglas:

1. Una revisión por cada cosa que puede estar mal, escrita como un hecho en presente
   ("`doble` devuelve el doble"): `verificar.py` la imprime tras "Bien:" o tras "Te falta:".
2. Revisiones discriminantes: una función que devuelve una constante, que ignora un
   argumento o que usa otra semilla no debe pasar. Para eso, llama a la función con al
   menos dos entradas distintas y compara con un cálculo independiente del tuyo.
3. Ninguna revisión debe pasar con el `ejercicio.py` vacío: toda revisión llama a algo
   que el tesista completa. `tests/test_ejercicios.py` lo exige.
4. Rápido y determinista: semillas explícitas, pocos entrenamientos, nada de reloj.

Para demostrar la regla 2, `comprobar.py` puede declarar `MUTANTES: list[tuple[str, str, str]]`,
una lista de `(descripción, texto original, texto nuevo)`. `tests/test_ejercicios.py` aplica
cada cambio a una copia de `solucion.py` (el texto original debe aparecer una sola vez) y
exige que alguna revisión falle con ella. Cada cambio imita un error típico: una función que
ignora un argumento, que usa otra semilla o que olvida un término de la cuenta. Además, esa
prueba cambia por sí sola cada función con hueco por una que devuelve una constante.
`ejercicios/01_primer_entrenamiento/comprobar.py` es el ejemplo de referencia.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# El modelo de práctica ya entrena en una fracción de segundo con sus ajustes de fábrica,
# así que lo único que conviene acortar es el oráculo: los datos frescos con los que mide
# la exactitud verdadera. No acortes `n_train` ni `epocas`: con menos datos de
# entrenamiento o menos épocas, la red se queda cerca del azar y a los ejercicios no les
# queda nada que observar.
CONFIG_RAPIDA: dict[str, int] = {"n_oraculo": 20_000}

# Los errores que comete un código a medias: una cuenta mal hecha, un nombre mal escrito,
# un argumento de más. `revisar` los convierte en un mensaje con el nombre del error y su
# texto. Cualquier otro (una interrupción del teclado, un fallo de la propia comprobación)
# sube sin tocarse, porque ese no es un tropiezo del tesista.
ERRORES_DEL_TESISTA: tuple[type[Exception], ...] = (
    ArithmeticError,
    AttributeError,
    IndexError,
    KeyError,
    NameError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class Revision:
    """El resultado de una revisión.

    `que_se_reviso`: lo que debe cumplirse, escrito como un hecho en presente.
    `paso`: `True` si tu código lo cumple.
    `que_hacer`: si no pasó, qué ver y qué cambiar, en una voz amable y de tú.
    `sin_completar`: `True` si la revisión no pasó porque el hueco sigue sin completar.
    """

    que_se_reviso: str
    paso: bool
    que_hacer: str = ""
    sin_completar: bool = False


def revisar(que_se_reviso: str, prueba: Callable[[], str | None], que_hacer: str) -> Revision:
    """Corre `prueba` y arma la `Revision`.

    `prueba` devuelve `None` si todo está bien, o una frase que cuenta qué vio mal.
    Si tu función lanza `NotImplementedError` (un hueco sin completar), la revisión dice
    cuál falta. Si lanza uno de `ERRORES_DEL_TESISTA`, la revisión cuenta qué error fue.
    """
    try:
        detalle = prueba()
    except NotImplementedError as hueco:
        return Revision(que_se_reviso, False, str(hueco), sin_completar=True)
    except ERRORES_DEL_TESISTA as error:
        return Revision(
            que_se_reviso,
            False,
            f"Tu código lanzó {type(error).__name__}: {error}. {que_hacer}",
        )
    if detalle is None:
        return Revision(que_se_reviso, True)
    return Revision(que_se_reviso, False, f"{detalle} {que_hacer}")


def cargar_modulo(ruta: Path, nombre: str) -> ModuleType:
    """Carga el archivo `ruta` como un módulo llamado `nombre`, sin correr su `main`.

    Cada archivo de un ejercicio protege su programa con `if __name__ == "__main__":`,
    así que cargarlo solo define sus funciones. El módulo queda registrado en
    `sys.modules` porque Python lo necesita para resolver clases definidas dentro.
    """
    especificacion = importlib.util.spec_from_file_location(nombre, ruta)
    if especificacion is None or especificacion.loader is None:
        raise ValueError(f"no puedo cargar {ruta} como un módulo de Python")
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[nombre] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo
