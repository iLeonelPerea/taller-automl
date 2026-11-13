"""Revisa tu ejercicio y te dice qué está bien y qué te falta.

Corre este programa desde la raíz del repo, con el número del ejercicio:

    uv run python ejercicios/verificar.py 04             revisa tu ejercicio.py
    uv run python ejercicios/verificar.py 04 --solucion  revisa solucion.py, para comprobar
                                                         que las revisiones aprueban una
                                                         respuesta correcta

Imprime cada revisión como "Bien: ..." o "Te falta: ..." (y "Te falta completar: ..." si el
hueco sigue sin completar), seguida de lo que puedes hacer, y cierra con un resumen. El
programa termina con código 0 si todas las revisiones pasan y con código 1 si alguna falla.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from _comun import Revision, cargar_modulo

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

EJERCICIOS = Path(__file__).resolve().parent
ANCHO = 96
USO = "Uso: uv run python ejercicios/verificar.py NUMERO [--solucion]   (por ejemplo, 04)"


def buscar_carpeta(numero: str) -> Path | None:
    """Devuelve la carpeta `NN_nombre` del ejercicio `numero` ("4" y "04" dan la misma)."""
    if not numero.isdigit():
        return None
    candidatas = sorted(EJERCICIOS.glob(f"{int(numero):02d}_*/"))
    return candidatas[0] if candidatas else None


def correr_revisiones(carpeta: Path, archivo: str) -> list[Revision]:
    """Carga `comprobar.py` y tu `archivo`, y devuelve las revisiones que hace sobre él."""
    comprobar = cargar_modulo(carpeta / "comprobar.py", f"comprobar_{carpeta.name}")
    modulo = cargar_modulo(carpeta / archivo, f"{Path(archivo).stem}_{carpeta.name}")
    try:
        return comprobar.comprobar(modulo)
    except NotImplementedError as hueco:
        # Una comprobación que no usa `revisar` deja subir el hueco: aquí se convierte en
        # una sola revisión, para que el mensaje sea el mismo que el de las demás.
        return [Revision(f"tu {archivo} está completo", False, str(hueco), sin_completar=True)]


def en_columna(texto: str, sangria: str = "    ") -> str:
    """Parte `texto` en líneas que caben en la terminal, con `sangria` al inicio de cada una."""
    return textwrap.fill(texto, width=ANCHO, initial_indent=sangria, subsequent_indent=sangria)


def imprimir(revisiones: Sequence[Revision]) -> None:
    """Imprime una línea por revisión y, si no pasó, lo que puedes hacer.

    Un hueco sin completar puede dejar varias revisiones sin pasar. Su mensaje sale una
    sola vez, bajo la primera, para que la lista no repita lo mismo.
    """
    huecos_ya_contados: set[str] = set()
    for revision in revisiones:
        if revision.paso:
            print(f"Bien: {revision.que_se_reviso}")
            continue
        etiqueta = "Te falta completar" if revision.sin_completar else "Te falta"
        print(f"{etiqueta}: {revision.que_se_reviso}")
        if not revision.que_hacer:
            continue
        if revision.sin_completar:
            if revision.que_hacer in huecos_ya_contados:
                continue
            huecos_ya_contados.add(revision.que_hacer)
        print(en_columna(revision.que_hacer))


def main(argv: Sequence[str] | None = None) -> int:
    argumentos = list(sys.argv[1:] if argv is None else argv)
    con_solucion = "--solucion" in argumentos
    numeros = [a for a in argumentos if a != "--solucion"]
    if len(numeros) != 1:
        print(USO, file=sys.stderr)
        return 2
    carpeta = buscar_carpeta(numeros[0])
    if carpeta is None:
        existentes = ", ".join(c.name for c in sorted(EJERCICIOS.glob("[0-9][0-9]_*/")))
        print(
            f"No encuentro el ejercicio {numeros[0]}. Los que hay: {existentes}.", file=sys.stderr
        )
        return 2

    archivo = "solucion.py" if con_solucion else "ejercicio.py"
    try:
        revisiones = correr_revisiones(carpeta, archivo)
    except SyntaxError as error:
        print(
            f"Tu archivo {archivo} tiene un error de escritura en la línea {error.lineno}: "
            f"{error.msg}. Corrígelo y vuelve a correr el verificador.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as error:
        if str(error) != MENSAJE_SIN_TARJETA:
            raise
        print(MENSAJE_SIN_TARJETA, file=sys.stderr)
        return 1
    if not revisiones:
        print(f"El comprobar.py de {carpeta.name} no devolvió ninguna revisión.", file=sys.stderr)
        return 1

    print(f"Reviso {carpeta.name}/{archivo}:\n")
    imprimir(revisiones)
    bien = sum(revision.paso for revision in revisiones)
    total = len(revisiones)
    print()
    if bien == total:
        quien = "La solución" if con_solucion else "Tu ejercicio"
        pasadas = "la única revisión" if total == 1 else f"las {total} revisiones"
        print(f"Resumen: {quien} pasa {pasadas}. ¡Bien hecho!")
        return 0
    faltan = total - bien
    te_faltan = f"te falta {faltan}" if faltan == 1 else f"te faltan {faltan}"
    resumen = (
        f"Resumen: {bien} de {total} revisiones en orden; {te_faltan}. "
        f"Lee las pistas del README de {carpeta.name} y vuelve a correr este comando."
    )
    print(textwrap.fill(resumen, width=ANCHO))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
