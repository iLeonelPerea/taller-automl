"""La maldición del ganador: cuánto miente el mejor valor que reportó la búsqueda.

Cuando te quedas con la mejor de muchas configuraciones, su valor es el máximo de varias
mediciones ruidosas, y ese máximo está inflado en promedio. Este módulo mide esa
inflación:

- `plan_semillas` separa las semillas de la búsqueda, comunes a todas las
  configuraciones, de las de evaluación, disjuntas para no repetir la suerte que ya vio
  el sampler.
- `mejores_de` saca el ganador (y, si quieres, la línea base) de un estudio ya corrido.
- `reevaluar` vuelve a correr esas configuraciones fuera del estudio, con semillas nuevas
  e iguales para todas (comparación pareada).
- `inflacion_ganador` compara el valor que vio la búsqueda contra la media re-evaluada,
  con un intervalo de confianza. Para atribuirla a la selección hace falta un control,
  como hace el paso 06.
- `congelar` y `PruebaSellada` imponen la disciplina de un conjunto de prueba real: la
  configuración se congela antes de mirar la prueba, y la prueba se abre una sola vez.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from optuna.trial import TrialState

from taller_hpo import estadistica
from taller_hpo.objetivo import Entrenar
from taller_hpo.semillas import Semillas


@dataclass(frozen=True)
class PlanSemillas:
    busqueda: Semillas
    evaluacion: tuple[Semillas, ...]


def plan_semillas(base: int, n_evaluacion: int) -> PlanSemillas:
    """Una `Semillas` para la búsqueda y `n_evaluacion` para re-evaluar, con los mismos datos.

    Todas comparten `datos`, pero `orden` e `init` son disjuntos entre la búsqueda y la
    evaluación: re-evaluar con las semillas que ya vio el sampler repetiría su suerte.
    """
    if isinstance(n_evaluacion, bool) or not isinstance(n_evaluacion, numbers.Integral):
        raise ValueError(f"n_evaluacion debe ser un entero >= 1, no {n_evaluacion!r}")
    if n_evaluacion < 1:
        raise ValueError(f"n_evaluacion debe ser >= 1, no {n_evaluacion}")

    raiz_busqueda, raiz_evaluacion = np.random.SeedSequence(base).spawn(2)
    s_datos, s_orden, s_init = (int(v) for v in raiz_busqueda.generate_state(3))
    busqueda = Semillas(datos=s_datos, orden=s_orden, init=s_init)

    evaluacion = []
    for hijo in raiz_evaluacion.spawn(n_evaluacion):
        _, e_orden, e_init = (int(v) for v in hijo.generate_state(3))
        evaluacion.append(Semillas(datos=busqueda.datos, orden=e_orden, init=e_init))

    return PlanSemillas(busqueda=busqueda, evaluacion=tuple(evaluacion))


def congelar(params: Mapping[str, Any]) -> str:
    """sha256 del JSON canónico de `params`, con las claves ordenadas: el orden de las
    claves no cambia el hash."""
    canonico = json.dumps(dict(params), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def mejores_de(
    study: optuna.Study, k: int, *, incluir_linea_base: bool = True
) -> dict[str, dict[str, Any]]:
    """Los `k` mejores trials COMPLETE de `study` (respetando `study.direction`) y la línea base.

    Cada entrada es `{"params": ..., "valor": ..., "trial": ...}`, con la config completa
    que vio `entrenar` (`user_attrs["config"]`), el valor que reportó la búsqueda y el
    número de trial. Los `k` mejores excluyen la línea base sembrada. Con
    `incluir_linea_base=True` y sin línea base completa, es un ValueError.
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or k < 1:
        raise ValueError(f"k debe ser un entero >= 1, no {k!r}")

    completos = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    candidatos = [t for t in completos if t.user_attrs.get("rol") != "linea_base"]
    maximiza = study.direction == optuna.study.StudyDirection.MAXIMIZE
    candidatos.sort(key=lambda t: t.value, reverse=maximiza)
    if len(candidatos) < k:
        raise ValueError(
            f"se pidieron {k} mejores pero el estudio {study.study_name!r} solo tiene "
            f"{len(candidatos)} trial(s) COMPLETE sin contar la línea base"
        )

    resultado: dict[str, dict[str, Any]] = {}
    for i, trial in enumerate(candidatos[:k], start=1):
        resultado[f"top_{i}"] = {
            "params": dict(trial.user_attrs["config"]),
            "valor": trial.value,
            "trial": trial.number,
        }

    if incluir_linea_base:
        lineas_base = [t for t in completos if t.user_attrs.get("rol") == "linea_base"]
        if not lineas_base:
            raise ValueError(
                f"incluir_linea_base=True pero el estudio {study.study_name!r} no tiene "
                "ningun trial COMPLETE con user_attrs['rol'] == 'linea_base' "
                "(ver taller_hpo.estudio.sembrar_linea_base)"
            )
        # Determinista aunque hubiera más de una línea base completa: usa la de número
        # de trial más alto.
        base = max(lineas_base, key=lambda t: t.number)
        resultado["linea_base"] = {
            "params": dict(base.user_attrs["config"]),
            "valor": base.value,
            "trial": base.number,
        }
    return resultado


def reevaluar(
    configs: Mapping[str, Mapping[str, Any]],
    entrenar: Entrenar,
    semillas: Sequence[Semillas],
    *,
    fijos: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Corre `entrenar` para cada config en `configs`, con las mismas `semillas` para todas.

    Fuera de cualquier estudio y sin poda, porque re-evaluar es una medición. Las mismas
    semillas para todas las configs hacen la comparación pareada. `fijos` es la base de
    cada config, y la config gana si repite una clave.
    """
    if len(semillas) < 1:
        raise ValueError("semillas no puede estar vacía")

    resultado: dict[str, list[dict[str, float]]] = {}
    for etiqueta, config in configs.items():
        config_final = {**(fijos or {}), **config}
        resultado[etiqueta] = [entrenar(config_final, s, None) for s in semillas]
    return resultado


@dataclass(frozen=True)
class Inflacion:
    valor_busqueda: float
    media_reevaluada: float
    inflacion: float
    ic_bajo: float
    ic_alto: float


def inflacion_ganador(
    valor_busqueda: float,
    reevaluados: Sequence[float],
    *,
    direccion: str = "maximizar",
) -> Inflacion:
    """Cuánto infló la búsqueda el valor del ganador, con un intervalo de confianza
    sobre `reevaluados`.

    `inflacion = signo * (valor_busqueda - media_reevaluada)`, con `signo = 1.0` al
    maximizar y `-1.0` al minimizar, así que `inflacion > 0` siempre significa que la
    búsqueda vio un valor mejor del que hay de verdad. El intervalo sale de llevar el de
    la media re-evaluada a través de esa misma transformación.

    Por sí solo no prueba selección: una configuración que nadie eligió puede mostrar el
    mismo residuo. Compáralo con un control antes de atribuirlo a la selección.
    """
    if direccion not in ("maximizar", "minimizar"):
        raise ValueError(f"direccion debe ser 'maximizar' o 'minimizar', no {direccion!r}")
    if not math.isfinite(valor_busqueda):
        raise ValueError(f"valor_busqueda debe ser finito, no {valor_busqueda!r}")

    r = estadistica.resumen(reevaluados)
    signo = 1.0 if direccion == "maximizar" else -1.0
    inflacion = signo * (valor_busqueda - r.media)
    extremo_1 = signo * (valor_busqueda - r.ic_bajo)
    extremo_2 = signo * (valor_busqueda - r.ic_alto)
    ic_bajo, ic_alto = (extremo_1, extremo_2) if extremo_1 <= extremo_2 else (extremo_2, extremo_1)
    return Inflacion(
        valor_busqueda=valor_busqueda,
        media_reevaluada=r.media,
        inflacion=inflacion,
        ic_bajo=ic_bajo,
        ic_alto=ic_alto,
    )


class PruebaSellada:
    """Un conjunto de prueba que se abre una sola vez, atado a una config ya congelada.

    Congelas el ganador antes de mirar la prueba (`sellar`), y `abrir` solo acepta esos
    params exactos, una vez. Elegir después de ver varios resultados de prueba sería la
    maldición del ganador otra vez. Volver a sellar antes de abrir sí está permitido.
    """

    def __init__(self, evaluar: Callable[[Mapping[str, Any]], float]) -> None:
        self._evaluar = evaluar
        self._hash_sellado: str | None = None
        self._abierta = False

    def sellar(self, params: Mapping[str, Any]) -> str:
        """Congela `params` como la única config que `abrir` va a aceptar.

        RuntimeError si la prueba ya se abrió.
        """
        if self._abierta:
            raise RuntimeError(
                "PruebaSellada.sellar: ya abriste la prueba; no puedes sellar otra config"
            )
        self._hash_sellado = congelar(params)
        return self._hash_sellado

    def abrir(self, params: Mapping[str, Any]) -> float:
        """Evalúa `params` con `evaluar`, una sola vez, y solo si es la config sellada.

        RuntimeError si todavía no sellaste o si ya abriste la prueba, aunque `evaluar`
        haya fallado. ValueError si `params` no es la config sellada.
        """
        if self._hash_sellado is None:
            raise RuntimeError("PruebaSellada.abrir: primero llama a sellar(params)")
        if self._abierta:
            raise RuntimeError(
                "PruebaSellada.abrir: ya abriste la prueba una vez; no la abras otra vez"
            )
        hash_actual = congelar(params)
        if hash_actual != self._hash_sellado:
            raise ValueError(
                "PruebaSellada.abrir: params no coincide con la config sellada "
                f"(sellado={self._hash_sellado}, recibido={hash_actual})"
            )
        self._abierta = True
        return self._evaluar(params)
