"""Tu `entrenar` en otro proceso: otro entorno, otro intérprete, otra tarjeta de video.

`taller_hpo.objetivo.en_proceso` llama a tu `entrenar` en el mismo proceso que Optuna.
Eso no alcanza cuando tu modelo necesita otro venv, otro `python` o puede caerse sin
tumbar el estudio. Aquí, por cada trial, el módulo escribe un JSON de entrada, lanza
`comando` como subproceso y lee un JSON de salida. El contrato lo cumple quien escribe
`comando` (ver `plantilla/`).

Detalles que importan:
- `escribir_json_atomico` escribe en un temporal y lo mueve con `os.replace`, así que el
  archivo final está completo o no existe.
- El subproceso corre en su propia sesión (POSIX) o grupo (Windows), y al vencer el
  timeout el módulo mata el árbol completo (`os.killpg` o `taskkill /T`): un `uv run` o un
  `conda run` lanzan un nieto que sobreviviría a la muerte del hijo.
- `stdout` y `stderr` van a archivos (`trial_<n>_stdout.txt` y `trial_<n>_stderr.txt`) y
  no a tuberías, que un nieto podría dejar abiertas.
- Un Ctrl+C no le llega al hijo, así que una interrupción mata el árbol igual que un
  timeout. Un `kill` directo al proceso padre no lo hace: corre los estudios largos
  dentro de `tmux`, `screen` o `nohup`.
- `stdin=subprocess.DEVNULL`: un `input()` olvidado en tu `train.py` falla de inmediato
  con `EOFError` en vez de quedarse esperando.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna

from taller_hpo.objetivo import MetricaNoFinita
from taller_hpo.semillas import Semillas

Espacio = Callable[[optuna.Trial], dict[str, Any]]

_ES_WINDOWS = sys.platform == "win32"


def escribir_json_atomico(ruta: str | os.PathLike, obj: Any) -> None:
    """Escribe `obj` como JSON UTF-8 en `ruta` sin dejar nunca un archivo a medias.

    El archivo final queda con permisos 0644, para que `comando` pueda leerlo aunque
    corra con otro usuario sobre un directorio compartido.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(dir=ruta.parent, prefix=f".{ruta.name}.", suffix=".tmp")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, ruta)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def leer_json(ruta: str | os.PathLike) -> Any:
    """Lee un JSON UTF-8. Quien llama decide qué hacer con un archivo ausente o corrupto."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _matar_arbol(proceso: subprocess.Popen[bytes]) -> None:
    """Mata `proceso` y todos sus descendientes, no solo el hijo directo.

    En POSIX, un `ProcessLookupError` de `killpg` antes de que el proceso termine es un
    kill roto y se propaga. En Windows, si `taskkill` falla, el respaldo es
    `proceso.kill()`, que solo mata al líder del árbol.
    """
    if _ES_WINDOWS:
        try:
            resultado = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proceso.pid)],
                capture_output=True,
                text=True,
                errors="replace",
            )
            necesita_respaldo = resultado.returncode != 0
            motivo = (
                f"taskkill salió con código {resultado.returncode} ({resultado.stderr.strip()})"
            )
        except FileNotFoundError as error:
            necesita_respaldo = True
            motivo = f"taskkill no está disponible en el PATH ({error})"
        if necesita_respaldo and proceso.poll() is None:
            proceso.kill()
            try:
                proceso.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"el taller no pudo matar el árbol del proceso {proceso.pid}: {motivo} "
                    "y el respaldo proceso.kill() tampoco lo detuvo en 10 s"
                ) from error
            return
    else:
        try:
            os.killpg(proceso.pid, signal.SIGKILL)
        except ProcessLookupError:
            if proceso.returncode is None:
                raise
    proceso.wait()


def por_subproceso(
    comando: Sequence[str],
    espacio: Espacio,
    *,
    metricas: Sequence[str],
    semillas: Semillas | Callable[[optuna.Trial], Semillas],
    timeout_s: float,
    directorio: str | os.PathLike,
    fijos: Mapping[str, Any] | None = None,
    atributos: Sequence[str] = (),
) -> Callable[[optuna.Trial], float | tuple[float, ...]]:
    """Construye un objective de Optuna que corre `comando` por cada trial.

    `comando` es arbitrario: el python de OTRO entorno, o cualquier ejecutable
    que respete `--config <entrada> --salida <salida>` con JSON UTF-8 en ambos
    extremos (ver `plantilla/train_ejemplo.py`). `espacio(trial) -> dict` es,
    igual que en `en_proceso`, el único lugar con `suggest_*`.

    Sin poda: al no correr en este proceso, no hay forma de que tu modelo llame
    a `trial.report` a mitad de entrenamiento.
    """
    if len(metricas) < 1:
        raise ValueError("metricas no puede estar vacío")
    directorio = Path(directorio)

    def objetivo(trial: optuna.Trial) -> float | tuple[float, ...]:
        config = {**(fijos or {}), **espacio(trial)}
        semillas_trial = semillas(trial) if callable(semillas) else semillas
        # int() porque json no serializa los enteros de numpy (np.int64).
        semillas_dict = {k: int(v) for k, v in dataclasses.asdict(semillas_trial).items()}
        # Igual que en_proceso: los atributos quedan guardados antes de correr nada, así
        # un trial que falla o expira también queda identificado.
        trial.set_user_attr("config", dict(config))
        trial.set_user_attr("semillas", semillas_dict)

        entrada = directorio / f"trial_{trial.number}_entrada.json"
        salida = directorio / f"trial_{trial.number}_salida.json"
        stdout_ruta = directorio / f"trial_{trial.number}_stdout.txt"
        stderr_ruta = directorio / f"trial_{trial.number}_stderr.txt"
        escribir_json_atomico(entrada, {"config": config, "semillas": semillas_dict})

        # Borra la salida de una corrida anterior en el mismo directorio, para no leerla
        # como si fuera la de este trial.
        salida.unlink(missing_ok=True)

        env = {**os.environ, "PYTHONUTF8": "1"}
        popen_kwargs: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if _ES_WINDOWS
            else {"start_new_session": True}
        )
        with open(stdout_ruta, "wb") as out_f, open(stderr_ruta, "wb") as err_f:
            proceso = subprocess.Popen(
                [*comando, "--config", str(entrada), "--salida", str(salida)],
                stdin=subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                env=env,
                **popen_kwargs,
            )
            try:
                returncode = proceso.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired as error:
                _matar_arbol(proceso)
                raise TimeoutError(
                    f"trial {trial.number}: {comando!r} no terminó en {timeout_s} s"
                ) from error
            except BaseException:
                # El hijo vive en su propia sesión y no recibe la interrupción: el
                # módulo mata el árbol y relanza la excepción tal cual.
                _matar_arbol(proceso)
                raise

        if returncode != 0:
            # Decodifica el archivo entero, sin partir un carácter multibyte, y
            # `errors="replace"` cubre los bytes que no son UTF-8.
            contenido = stderr_ruta.read_bytes().decode("utf-8", errors="replace")
            final = contenido[-2000:]
            raise RuntimeError(
                f"trial {trial.number}: {comando!r} salió con código "
                f"{returncode}. Fin de stderr:\n{final}\n"
                f"(salida estándar completa en {stdout_ruta})"
            )

        try:
            metricas_obtenidas = leer_json(salida)
        except (OSError, ValueError) as error:
            # ValueError cubre json.JSONDecodeError y también UnicodeDecodeError, de un
            # `comando` que escribe con otro encoding.
            raise RuntimeError(
                f"trial {trial.number}: el taller no pudo leer {salida} como JSON: {error}"
            ) from error

        if not isinstance(metricas_obtenidas, dict):
            raise RuntimeError(
                f"trial {trial.number}: {salida} debe ser un objeto JSON "
                f"{{métrica: valor}}, no {type(metricas_obtenidas).__name__}"
            )

        for atributo in atributos:
            trial.set_user_attr(atributo, metricas_obtenidas[atributo])

        valores = []
        for metrica in metricas:
            valor = float(metricas_obtenidas[metrica])
            if not math.isfinite(valor):
                raise MetricaNoFinita(f"métrica {metrica!r} = {valor!r} no es finita")
            valores.append(valor)
        return valores[0] if len(valores) == 1 else tuple(valores)

    return objetivo
