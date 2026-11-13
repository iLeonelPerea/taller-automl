"""Ejercicio 01: entrena el modelo de práctica una vez y aprende a leer lo que devuelve.

Tú completas las tres funciones marcadas con `# TU TURNO`, corres este archivo para ver lo
que produce y corres el verificador para saber qué está bien y qué te falta:

    uv run python ejercicios/01_primer_entrenamiento/ejercicio.py
    uv run python ejercicios/verificar.py 01

Mientras un hueco siga sin completar, este archivo se detiene con un mensaje que te dice
cuál es. El README de esta carpeta explica cada paso y trae una pista por paso.
"""

from __future__ import annotations

import textwrap
from typing import Any

from taller_hpo.problemas import juguete
from taller_hpo.semillas import Semillas

# Las perillas (hiperparámetros) del modelo: números que tú eliges antes de entrenar. Las
# que no escribes aquí quedan en su valor de fábrica, que vive en `juguete.DEFAULTS`.
CONFIG = {"n_capas": 1, "ancho": 64, "epocas": 30}

# Una semilla fija una fuente de azar: con las mismas semillas repites el mismo entrenamiento.
# `datos` decide qué ejemplos genera el problema, `orden` en qué orden los ve la red y
# `init` con qué números empieza a aprender.
SEMILLAS = Semillas(datos=1, orden=2, init=3)

# La configuración completa: los valores de fábrica, con tus perillas encima.
CONFIG_COMPLETA = {**juguete.DEFAULTS, **CONFIG}

# Lo que mide cada una de las cinco métricas que devuelve el entrenamiento.
SIGNIFICADO = {
    "exactitud_val": (
        "La fracción de datos de validación que el modelo acierta. Son datos que el modelo "
        "no usó para aprender, y 0.70 significa 70 aciertos de cada 100."
    ),
    "exactitud_verdadera": (
        "La misma fracción, medida con una cantidad enorme de datos nuevos. Se parece mucho "
        "a la calidad real del modelo, y solo el problema de práctica puede medirla."
    ),
    "n_parametros": (
        "Cuántos números (pesos) aprende la red. Entre más tiene, más grande es el modelo."
    ),
    "epocas": "Cuántas veces vio el modelo todos los datos de entrenamiento.",
    "segundos": (
        "Lo que tardó el entrenamiento. Depende de tu máquina y no dice nada sobre la "
        "calidad del modelo."
    ),
}


def entrenar_una_vez(config: dict[str, Any], semillas: Semillas) -> dict[str, float]:
    """Entrena el modelo de práctica una vez y devuelve sus cinco métricas.

    `juguete.entrenar(config, semillas)` hace el entrenamiento completo y devuelve un
    diccionario con las métricas. Tu función le pasa los dos argumentos que recibe y
    devuelve lo que ella entrega.
    """
    # TU TURNO: llama a juguete.entrenar con la config y las semillas, y devuelve su resultado.
    raise NotImplementedError("Completa entrenar_una_vez (ver el paso 1 del README)")


def aciertos_en_validacion(exactitud_val: float, n_val: int) -> int:
    """Convierte la exactitud de validación en cuántos datos acertó el modelo.

    La exactitud es una fracción: aciertos entre total. Con una exactitud de 0.5 y 200 datos
    de validación, el modelo acertó 100. Devuelve un número entero.
    """
    # TU TURNO: multiplica la exactitud por n_val y devuelve el resultado como un entero exacto.
    raise NotImplementedError("Completa aciertos_en_validacion (ver el paso 2 del README)")


def contar_parametros(n_entradas: int, ancho: int, n_capas: int) -> int:
    """Cuenta cuántos pesos aprende la red del modelo de práctica.

    La red tiene `n_capas` capas ocultas del mismo `ancho` y una capa final que produce un
    solo número. Cada capa aprende un peso por cada conexión entre sus entradas y sus
    salidas, más un número extra (el sesgo) por cada salida. La primera capa recibe
    `n_entradas` números.
    """
    # TU TURNO: suma los pesos y los sesgos de cada capa, desde la primera hasta la final.
    raise NotImplementedError("Completa contar_parametros (ver el paso 3 del README)")


def main() -> None:
    metricas = entrenar_una_vez(CONFIG, SEMILLAS)
    print("Entrenaste el modelo de práctica una vez. Esto devolvió:\n")
    for nombre, significado in SIGNIFICADO.items():
        print(f"{nombre} = {metricas[nombre]:.4g}")
        print(textwrap.fill(significado, width=96, initial_indent="    ", subsequent_indent="    "))

    n_val = CONFIG_COMPLETA["n_val"]
    aciertos = aciertos_en_validacion(metricas["exactitud_val"], n_val)
    print(f"\nCon {n_val} datos de validación, el modelo acertó {aciertos}.")

    n_entradas = CONFIG_COMPLETA["d"] + CONFIG_COMPLETA["k_ruido"]
    contados = contar_parametros(n_entradas, CONFIG_COMPLETA["ancho"], CONFIG_COMPLETA["n_capas"])
    reportados = round(metricas["n_parametros"])
    print(f"Contaste {contados} pesos a mano y el entrenamiento reportó {reportados}.")
    print("Tu cuenta coincide." if contados == reportados else "Tu cuenta no coincide: revísala.")


if __name__ == "__main__":
    # Mientras un hueco siga sin completar, el programa se detiene con un mensaje amable en
    # vez de un error largo. Esta parte no es un hueco: déjala como está.
    try:
        main()
    except NotImplementedError as hueco:
        raise SystemExit(f"Todavía te falta un hueco: {hueco}") from None
