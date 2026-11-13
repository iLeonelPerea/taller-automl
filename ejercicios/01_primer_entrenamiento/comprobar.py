"""Las revisiones del ejercicio 01.

`verificar.py` carga este archivo, le entrega tu `ejercicio.py` (o `solucion.py`) y
muestra el resultado de cada revisión. Cada revisión llama a una de tus funciones con
entradas elegidas aquí y compara con un cálculo independiente: una función que
devuelve siempre lo mismo, que ignora un argumento o que usa otras semillas no pasa.

`MUTANTES` lista cambios que rompen la solución a propósito. `tests/test_ejercicios.py`
los aplica a una copia de `solucion.py` y exige que alguna revisión falle con cada uno.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from _comun import CONFIG_RAPIDA, Revision, revisar

from taller_hpo.problemas import juguete
from taller_hpo.semillas import Semillas

SEMILLAS_A = Semillas(datos=11, orden=12, init=13)
SEMILLAS_B = Semillas(datos=21, orden=22, init=23)

# Dos configuraciones que difieren en varias perillas y en el tamaño de la validación.
CONFIG_A = {**CONFIG_RAPIDA, "epocas": 6, "ancho": 12}
CONFIG_B = {**CONFIG_RAPIDA, "epocas": 4, "ancho": 10, "n_capas": 2, "n_val": 250}

CLAVES = ("exactitud_val", "exactitud_verdadera", "n_parametros", "epocas", "segundos")
# `segundos` cambia en cada corrida aunque todo lo demás sea igual, así que no se compara.
CAMPOS_COMPARABLES = ("exactitud_val", "exactitud_verdadera", "n_parametros", "epocas")

# Casos de `aciertos_en_validacion`: (exactitud_val, n_val, aciertos esperados).
# Los primeros multiplican sin ruido de decimales; los últimos caen un pelo debajo del
# entero (0.29 * 100 da 28.999999999999996), y ahí `int` pierde un acierto.
ACIERTOS_EXACTOS = ((0.5, 200, 100), (0.625, 400, 250), (0.25, 40, 10), (0.75, 12, 9))
ACIERTOS_CON_RUIDO = ((0.29, 100, 29), (0.57, 100, 57), (0.58, 100, 58))

# Casos de `contar_parametros`: (d, k_ruido, ancho, n_capas). La cuenta correcta es la que
# reporta el propio entrenamiento, así que se compara contra él.
FORMAS_DE_UNA_CAPA = ((3, 10, 64, 1), (2, 3, 12, 1), (4, 0, 7, 1))
FORMAS_DE_VARIAS_CAPAS = ((3, 10, 10, 2), (2, 3, 7, 3), (3, 10, 5, 4))

MUTANTES: list[tuple[str, str, str]] = [
    (
        "entrenar_una_vez ignora las semillas que recibe",
        "return juguete.entrenar(config, semillas)",
        "return juguete.entrenar(config, Semillas(datos=1, orden=2, init=3))",
    ),
    (
        "entrenar_una_vez ignora la config que recibe",
        "return juguete.entrenar(config, semillas)",
        "return juguete.entrenar({}, semillas)",
    ),
    (
        "entrenar_una_vez devuelve solo la exactitud de validación",
        "return juguete.entrenar(config, semillas)",
        'return {"exactitud_val": juguete.entrenar(config, semillas)["exactitud_val"]}',
    ),
    (
        "aciertos_en_validacion corta con int en vez de redondear",
        "return round(exactitud_val * n_val)",
        "return int(exactitud_val * n_val)",
    ),
    (
        "aciertos_en_validacion devuelve un decimal",
        "return round(exactitud_val * n_val)",
        "return exactitud_val * n_val",
    ),
    (
        "aciertos_en_validacion ignora n_val",
        "return round(exactitud_val * n_val)",
        "return round(exactitud_val * 300)",
    ),
    (
        "contar_parametros olvida los sesgos",
        "total += entradas * ancho + ancho",
        "total += entradas * ancho",
    ),
    (
        "contar_parametros olvida la capa final",
        "total += entradas * 1 + 1",
        "total += 0",
    ),
    (
        "contar_parametros cuenta una sola capa oculta",
        "for _ in range(n_capas):",
        "for _ in range(1):",
    ),
    (
        "contar_parametros ignora el número de entradas",
        "entradas = n_entradas",
        "entradas = 13",
    ),
]


@functools.cache
def _referencia(config: tuple[tuple[str, Any], ...], semillas: Semillas) -> dict[str, float]:
    """El entrenamiento de referencia para esa config y esas semillas. Con las mismas
    semillas el resultado no cambia, así que se guarda: varias revisiones lo piden."""
    return juguete.entrenar(dict(config), semillas)


def _entrenar_de_referencia(config: Mapping[str, Any], semillas: Semillas) -> dict[str, float]:
    return _referencia(tuple(sorted(config.items())), semillas)


def _etiqueta(config: Mapping[str, Any]) -> str:
    """Las perillas que cambian en `config`, sin los ajustes rápidos que no cambian nada."""
    return ", ".join(f"{k}={v}" for k, v in config.items() if k not in CONFIG_RAPIDA)


def _primera_diferencia(
    esperado: Mapping[str, float], obtenido: Mapping[str, float], contexto: str
) -> str | None:
    """Cuenta la primera métrica comparable en que `obtenido` se aparta de `esperado`."""
    for campo in CAMPOS_COMPARABLES:
        if campo not in obtenido:
            return f"{contexto} a tu resultado le falta la métrica {campo!r}."
        if not math.isclose(obtenido[campo], esperado[campo], rel_tol=0.0, abs_tol=1e-9):
            return (
                f"{contexto} esperaba {campo}={esperado[campo]:.6g} "
                f"y obtuve {campo}={obtenido[campo]:.6g}."
            )
    return None


def _cuenta_de_pesos(forma: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Devuelve (n_entradas, ancho, n_capas, pesos) con los pesos que reporta el entrenamiento."""
    d, k_ruido, ancho, n_capas = forma
    config = {"d": d, "k_ruido": k_ruido, "ancho": ancho, "n_capas": n_capas}
    config |= {"epocas": 1, "n_oraculo": 1000}
    reportados = round(_entrenar_de_referencia(config, SEMILLAS_A)["n_parametros"])
    return d + k_ruido, ancho, n_capas, reportados


def comprobar(modulo: ModuleType) -> list[Revision]:
    def devuelve_las_cinco_metricas() -> str | None:
        obtenido = modulo.entrenar_una_vez(CONFIG_A, SEMILLAS_A)
        if not isinstance(obtenido, Mapping):
            return f"Esperaba un diccionario con las cinco métricas y obtuve {obtenido!r}."
        if set(obtenido) != set(CLAVES):
            return f"Esperaba las claves {sorted(CLAVES)} y obtuve {sorted(obtenido)}."
        return None

    def usa_la_config_que_recibe() -> str | None:
        for config in (CONFIG_A, CONFIG_B):
            esperado = _entrenar_de_referencia(config, SEMILLAS_A)
            obtenido = modulo.entrenar_una_vez(config, SEMILLAS_A)
            diferencia = _primera_diferencia(esperado, obtenido, f"Con {_etiqueta(config)},")
            if diferencia is not None:
                return diferencia
        return None

    def usa_las_semillas_que_recibe() -> str | None:
        for semillas in (SEMILLAS_A, SEMILLAS_B):
            esperado = _entrenar_de_referencia(CONFIG_A, semillas)
            obtenido = modulo.entrenar_una_vez(CONFIG_A, semillas)
            contexto = f"Con semillas datos={semillas.datos}, orden={semillas.orden}, "
            contexto += f"init={semillas.init},"
            diferencia = _primera_diferencia(esperado, obtenido, contexto)
            if diferencia is not None:
                return diferencia
        return None

    def multiplica_exactitud_por_total() -> str | None:
        for exactitud, n_val, esperado in ACIERTOS_EXACTOS:
            obtenido = modulo.aciertos_en_validacion(exactitud, n_val)
            if obtenido != esperado:
                return (
                    f"Con exactitud_val={exactitud} y n_val={n_val} esperaba {esperado} "
                    f"aciertos y obtuve {obtenido!r}."
                )
        return None

    def devuelve_un_entero_exacto() -> str | None:
        for exactitud, n_val, esperado in ACIERTOS_CON_RUIDO:
            obtenido = modulo.aciertos_en_validacion(exactitud, n_val)
            if obtenido != esperado or not isinstance(obtenido, int):
                return (
                    f"Con exactitud_val={exactitud} y n_val={n_val} esperaba el entero "
                    f"{esperado} y obtuve {obtenido!r}."
                )
        return None

    def cuenta_los_pesos(formas: tuple[tuple[int, int, int, int], ...]) -> str | None:
        for forma in formas:
            n_entradas, ancho, n_capas, reportados = _cuenta_de_pesos(forma)
            obtenido = modulo.contar_parametros(n_entradas, ancho, n_capas)
            if obtenido != reportados:
                return (
                    f"Con n_entradas={n_entradas}, ancho={ancho} y n_capas={n_capas}, el "
                    f"entrenamiento reporta {reportados} pesos y tu cuenta da {obtenido!r}."
                )
        return None

    return [
        revisar(
            "`entrenar_una_vez` devuelve las cinco métricas del entrenamiento",
            devuelve_las_cinco_metricas,
            "Llama a `juguete.entrenar(config, semillas)` y devuelve lo que te regresa, "
            "sin cambiarlo.",
        ),
        revisar(
            "`entrenar_una_vez` entrena con la config que recibe",
            usa_la_config_que_recibe,
            "Pasa la `config` que recibes a `juguete.entrenar`, sin cambiarla por otra.",
        ),
        revisar(
            "`entrenar_una_vez` entrena con las semillas que recibe",
            usa_las_semillas_que_recibe,
            "Pasa las `semillas` que recibes a `juguete.entrenar`, sin cambiarlas por otras.",
        ),
        revisar(
            "`aciertos_en_validacion` multiplica la exactitud por el total de datos",
            multiplica_exactitud_por_total,
            "Multiplica `exactitud_val` por `n_val` y devuelve el resultado.",
        ),
        revisar(
            "`aciertos_en_validacion` devuelve un entero exacto, sin perder aciertos",
            devuelve_un_entero_exacto,
            "Devuelve un `int`: `round(...)` redondea al entero más cercano, mientras que "
            "`int(...)` corta los decimales y puede perder un acierto (la computadora da "
            "28.999999999999996 al multiplicar 0.29 por 100).",
        ),
        revisar(
            "`contar_parametros` coincide con los pesos de una red de una capa oculta",
            lambda: cuenta_los_pesos(FORMAS_DE_UNA_CAPA),
            "Cada capa aprende `entradas × salidas` pesos más un sesgo por cada salida. "
            "Suma la capa oculta y la capa final, que produce un solo número.",
        ),
        revisar(
            "`contar_parametros` coincide con los pesos de una red de varias capas ocultas",
            lambda: cuenta_los_pesos(FORMAS_DE_VARIAS_CAPAS),
            "Desde la segunda capa oculta, las entradas de cada capa son el ancho de la "
            "capa anterior. Recorre las capas ocultas una por una y suma cada una.",
        ),
    ]
