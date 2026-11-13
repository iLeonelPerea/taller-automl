"""Mide cuánto tarda cada paso y llena la tabla "Tiempos medidos" del README.

Uso, en una máquina sin otros procesos del taller corriendo:

    uv run scripts/medir_tiempos.py [carpeta_de_salida]

- Usa el reloj monotónico (`time.perf_counter`), porque en WSL2 el reloj de pared se
  corrige con saltos a mitad de una corrida.
- Descarta una corrida de calentamiento, que paga la primera carga de torch y CUDA.
- Antes de cada celda espera a que la carga de 1 minuto baje de `UMBRAL_CARGA`, como
  mucho `ESPERA_MAX_S`, y anota el uso de la tarjeta de video.
- Repite cada celda tres veces y reporta la mediana, el mínimo y el máximo.
- Cada repetición usa carpetas nuevas de estudios y resultados, así ningún paso reanuda
  el estudio anterior. La carpeta de salida no puede estar en /mnt/*.
- Termina con código 1 si alguna corrida falla o escribe en stderr.

Cada fila queda en `tiempos.json` dentro de la carpeta de salida, con el stdout y el
stderr de cada corrida.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPETICIONES = 3
UMBRAL_CARGA = 0.5
ESPERA_MAX_S = 300

PASOS = sorted(p.name for p in (REPO / "pasos").glob("[0-9][0-9]_*.py"))
CELDAS: list[tuple[str, str, list[str]]] = []
for _paso in PASOS:
    CELDAS.append((_paso, "humo", []))
    CELDAS.append((_paso, "taller", []))
CELDAS.append(("anexo_paralelo.py", "humo", ["--lanzar", "3"]))
CELDAS.append(("anexo_paralelo.py", "taller", ["--lanzar", "3"]))


def carga() -> float:
    """Carga de 1 minuto del sistema (solo en Linux)."""
    return float(Path("/proc/loadavg").read_text().split()[0])


def uso_gpu() -> str:
    """Uso y memoria ocupada de la tarjeta de video según nvidia-smi ("" si no
    hay nvidia-smi)."""
    if shutil.which("nvidia-smi") is None:
        return ""
    resultado = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return resultado.stdout.strip().replace(" ", "")


def esperar_carga() -> tuple[float, float, bool]:
    """Espera a que la carga baje de UMBRAL_CARGA; devuelve (carga, espera, se_agoto)."""
    inicio = time.monotonic()
    while carga() >= UMBRAL_CARGA:
        if time.monotonic() - inicio >= ESPERA_MAX_S:
            return carga(), time.monotonic() - inicio, True
        time.sleep(5)
    return carga(), time.monotonic() - inicio, False


def entorno_de(presupuesto: str, estudios: Path, resultados: Path) -> dict:
    return {
        **os.environ,
        "PRESUPUESTO": presupuesto,
        "TALLER_ESTUDIOS": str(estudios),
        "TALLER_RESULTADOS": str(resultados),
    }


def correr(paso: str, presupuesto: str, extra: list[str], salida: Path, etiqueta: str) -> dict:
    estudios = salida / f"e_{etiqueta}"
    resultados = salida / f"r_{etiqueta}"
    for carpeta in (estudios, resultados):
        shutil.rmtree(carpeta, ignore_errors=True)
        carpeta.mkdir(parents=True)
    entorno = entorno_de(presupuesto, estudios, resultados)
    carga_antes = carga()
    t0 = time.perf_counter()
    resultado = subprocess.run(
        [sys.executable, str(REPO / "pasos" / paso), *extra],
        env=entorno,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    segundos = time.perf_counter() - t0
    (salida / f"{etiqueta}.out").write_text(resultado.stdout, encoding="utf-8")
    (salida / f"{etiqueta}.err").write_text(resultado.stderr, encoding="utf-8")
    # Basta el prefijo de la frase que imprime un paso que no entrena (el paso 01), para
    # que esta columna no quede vacía si cambia el final de la frase.
    motor_impreso = next(
        (
            linea.strip()
            for linea in resultado.stdout.splitlines()
            if linea.startswith(("Motor", "cash_paridad:", "Este paso no usa"))
        ),
        "",
    )
    return {
        "segundos": segundos,
        "exit": resultado.returncode,
        "stderr_bytes": len(resultado.stderr),
        "carga_antes": carga_antes,
        "motor_impreso": motor_impreso,
    }


def carpeta_de_salida() -> Path:
    if len(sys.argv) > 1:
        salida = Path(sys.argv[1]).resolve()
    else:
        salida = Path(tempfile.mkdtemp(prefix="tiempos_")).resolve()
    if salida.parts[:2] == ("/", "mnt"):
        raise SystemExit(
            f"la carpeta de salida {salida} está en /mnt/*: usa una del disco Linux "
            "(p. ej. /tmp/tiempos), porque ahí van los estudios de cada corrida"
        )
    salida.mkdir(parents=True, exist_ok=True)
    return salida


def main() -> None:
    salida = carpeta_de_salida()
    print(f"salida: {salida}", flush=True)
    print("calentamiento (el script lo descarta)...", flush=True)
    correr("00_linea_base.py", "humo", [], salida, "calentamiento")
    filas = []
    for paso, presupuesto, extra in CELDAS:
        carga_inicio, espera_s, se_agoto = esperar_carga()
        if se_agoto:
            print(
                f"AVISO: la carga no bajó de {UMBRAL_CARGA} en {ESPERA_MAX_S} s "
                f"(está en {carga_inicio:.2f}); el script mide la celda igual y anota "
                "en el resultado que la espera se agotó",
                flush=True,
            )
        gpu_inicio = uso_gpu()
        repeticiones = [
            correr(
                paso,
                presupuesto,
                extra,
                salida,
                f"{Path(paso).stem}_{presupuesto}_{i}",
            )
            for i in range(REPETICIONES)
        ]
        tiempos = [r["segundos"] for r in repeticiones]
        fila = {
            "paso": Path(paso).stem,
            "presupuesto": presupuesto,
            "mediana": statistics.median(tiempos),
            "min": min(tiempos),
            "max": max(tiempos),
            "tiempos": tiempos,
            "exits": [r["exit"] for r in repeticiones],
            "stderr_bytes": [r["stderr_bytes"] for r in repeticiones],
            "carga_inicio_celda": carga_inicio,
            "espera_s": espera_s,
            "espera_agotada": se_agoto,
            "gpu_inicio_celda": gpu_inicio,
            "carga_por_repeticion": [r["carga_antes"] for r in repeticiones],
            "motor_impreso": repeticiones[0]["motor_impreso"],
        }
        filas.append(fila)
        print(
            f"{fila['paso']:<22} {presupuesto:<6} mediana {fila['mediana']:6.1f} s "
            f"[{fila['min']:.1f}, {fila['max']:.1f}] exits={fila['exits']} "
            f"stderr={fila['stderr_bytes']} carga={carga_inicio:.2f} tarjeta={gpu_inicio}",
            flush=True,
        )
    (salida / "tiempos.json").write_text(json.dumps(filas, indent=1), encoding="utf-8")
    invalidas = [
        f"{f['paso']} {f['presupuesto']}"
        for f in filas
        if any(e != 0 for e in f["exits"]) or any(b > 0 for b in f["stderr_bytes"])
    ]
    if invalidas:
        raise SystemExit(f"celdas con exit != 0 o con stderr (no son tiempos válidos): {invalidas}")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
