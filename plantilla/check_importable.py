"""Detecta el error más común al adaptar un `train.py`: código a nivel de módulo.

Si tu script hace `parser.parse_args()` o carga un dataset fuera de una función,
importarlo ya dispara ese código. `por_subproceso` no lo nota, porque ejecuta el script
completo, pero `en_proceso` sí importa el módulo. Esta revisión también exige que
`DEFAULTS` declare `dispositivo` como `'auto'` o `'cuda'` (con `':N'` opcional).

Importa el archivo en un subproceso aparte, con:
- `sys.argv = [ruta, "--help"]`: cualquier `ArgumentParser` a nivel de módulo imprime la
  ayuda y sale antes de reclamar argumentos. Un `ArgumentParser(add_help=False)` con
  `parse_known_args()` no se detecta.
- un timeout (`TIMEOUT_S`), por si el módulo hace trabajo pesado al importarse.
- la carpeta de `ruta` al principio de `sys.path`, como hace `python train.py`, para que
  los imports de módulos hermanos funcionen.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

TIMEOUT_S = 20.0

# `revisar` corre este fragmento con `python -c`, así el subproceso no hereda el
# sys.argv de quien llama.
_FRAGMENTO = """
import importlib.util
import os
import re
import sys

ruta = sys.argv[1]
sys.argv = [ruta, "--help"]
sys.path.insert(0, os.path.dirname(os.path.abspath(ruta)))

spec = importlib.util.spec_from_file_location("_candidato_check_importable", ruta)
if spec is None or spec.loader is None:
    print(f"el script no pudo cargar {ruta!r} como módulo", file=sys.stderr)
    sys.exit(4)

modulo = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(modulo)
except SystemExit:
    # "--help" lo reconoce cualquier ArgumentParser (parse_args() o
    # parse_known_args()) a nivel de módulo, y sale con SystemExit antes de
    # reclamar un posicional faltante: por eso este SystemExit (nunca el
    # resto de este fragmento, que no llama a sys.exit salvo más abajo) es la
    # señal de que algo parseó argumentos al importar.
    print("parse_args", file=sys.stderr)
    sys.exit(2)

faltantes = [
    nombre for nombre in ("entrenar", "main") if not callable(getattr(modulo, nombre, None))
]
defaults = getattr(modulo, "DEFAULTS", None)
# 'dispositivo' no solo debe existir: su valor debe tener la forma que el
# taller acepta ('auto' o 'cuda', opcionalmente con ':N').
if not isinstance(defaults, dict) or "dispositivo" not in defaults:
    faltantes.append("DEFAULTS['dispositivo']")
elif not re.fullmatch(r"auto|cuda(:\\d+)?", str(defaults["dispositivo"])):
    faltantes.append(
        f"DEFAULTS['dispositivo'] debe ser 'auto' o 'cuda' (opcionalmente con ':N'), "
        f"no {defaults['dispositivo']!r}"
    )
if faltantes:
    print("faltan: " + ", ".join(faltantes))
    sys.exit(3)
print("ok")
"""


def revisar(ruta: str, *, timeout_s: float = TIMEOUT_S) -> str | None:
    """None si `ruta` cumple el contrato; si no, un mensaje que dice cuál falló."""
    try:
        resultado = subprocess.run(
            [sys.executable, "-c", _FRAGMENTO, ruta],
            env={**os.environ, "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return (
            f"{ruta}: importarlo tardó más de {timeout_s:g} s. Hay trabajo pesado a "
            "nivel de módulo (dataset, entrenamiento): muévelo dentro de entrenar() "
            'o detrás de `if __name__ == "__main__":`'
        )

    if resultado.returncode == 2:
        return (
            f"{ruta}: algo llamó a `parse_args()` o `parse_known_args()` durante "
            "la importación (se detecta con `--help`, que argparse siempre "
            "reconoce): o hay un parser de argumentos a nivel de módulo, o falta "
            'la guarda `if __name__ == "__main__":` alrededor de main()'
        )
    if resultado.returncode == 3:
        return (
            f"{ruta}: {resultado.stdout.strip()} (el contrato exige entrenar(), main() y "
            "DEFAULTS con la clave 'dispositivo': el taller entrena siempre en la tarjeta "
            "de video, ver plantilla/README.md)"
        )
    if resultado.returncode != 0:
        detalle = (resultado.stderr or resultado.stdout).strip()
        return f"{ruta}: importarlo falló:\n{detalle}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Revisa que un train.py cumpla el contrato de plantilla/README.md"
    )
    parser.add_argument("ruta", help="script train.py a revisar")
    args = parser.parse_args(argv)

    problema = revisar(args.ruta)
    if problema is not None:
        print(problema, file=sys.stderr)
        return 1
    print(f"{args.ruta}: importable, define entrenar() y main(), y declara DEFAULTS['dispositivo']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
