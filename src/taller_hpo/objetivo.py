"""Envuelve tu entrenar(config, semillas, reportar) como objective de Optuna.

`espacio(trial) -> dict` es el único lugar que llama a `trial.suggest_*`. Todo lo demás
vive aquí: armar la config, guardar de qué corrió el trial, convertir `reportar` en poda
de Optuna y validar la métrica. Tu `entrenar` solo obedece el booleano que recibe de
`reportar` y nunca importa Optuna (ver las convenciones en docs/arquitectura.md).

Optuna 5 no poda en multiobjetivo (`Trial.report` y `Trial.should_prune` lanzan
NotImplementedError), así que con varias métricas `reportar=None` y el entrenamiento
siempre corre completo.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import optuna

from taller_hpo.semillas import Semillas

Reportar = Callable[[int, float], bool]
Entrenar = Callable[[dict[str, Any], Semillas, Reportar | None], dict[str, float]]
Espacio = Callable[[optuna.Trial], dict[str, Any]]


class MetricaNoFinita(ValueError):
    """La métrica que pides optimizar dio NaN o +-inf en vez de un número."""


def en_proceso(
    entrenar: Entrenar,
    espacio: Espacio,
    *,
    metricas: Sequence[str],
    semillas: Semillas | Callable[[optuna.Trial], Semillas],
    fijos: Mapping[str, Any] | None = None,
    atributos: Sequence[str] = (),
) -> Callable[[optuna.Trial], float | tuple[float, ...]]:
    """Construye el objective(trial) que `optuna.Study.optimize` necesita.

    `config = {**fijos, **espacio(trial)}`. Con una sola métrica, el objective reporta
    cada época a Optuna y, si `trial.should_prune()` dice que sí, `entrenar` para y el
    objective levanta `optuna.TrialPruned`. `atributos` son métricas que solo quedan
    como user_attrs y nunca se optimizan, como el oráculo `exactitud_verdadera`. Una
    clave en `metricas` y en `atributos` a la vez es un ValueError.
    """
    if isinstance(metricas, str):
        # Un str también es un Sequence[str]: sin este chequeo, metricas="acc" se
        # trataría como tres métricas, una por letra.
        raise TypeError(
            f"metricas debe ser una lista de str, p.ej. ['exactitud_val'], no {metricas!r}"
        )
    if isinstance(atributos, str):
        raise TypeError(
            f"atributos debe ser una lista de str, p.ej. ['exactitud_val'], no {atributos!r}"
        )
    if len(metricas) < 1:
        raise ValueError("metricas no puede estar vacío")
    solapadas = set(metricas) & set(atributos)
    if solapadas:
        # Sin esta guarda, el oráculo en las dos listas se optimizaría en silencio.
        raise ValueError(f"metricas y atributos no pueden compartir claves: {sorted(solapadas)}")
    multiobjetivo = len(metricas) > 1

    def objective(trial: optuna.Trial) -> float | tuple[float, ...]:
        config = {**(fijos or {}), **espacio(trial)}
        juego_semillas = semillas(trial) if callable(semillas) else semillas
        trial.set_user_attr("config", dict(config))
        # int() porque un storage persistente no guarda enteros de numpy (np.int64).
        trial.set_user_attr("semillas", {k: int(v) for k, v in asdict(juego_semillas).items()})

        podado = False
        reportar: Reportar | None = None
        if not multiobjetivo:

            def reportar(epoca: int, valor: float) -> bool:
                nonlocal podado
                if podado:
                    # La poda es monótona: una vez que should_prune() dijo que sí, el
                    # trial sigue podado aunque `entrenar` siga reportando.
                    return True
                trial.report(valor, epoca)
                podado = trial.should_prune()
                return podado

        resultado = entrenar(config, juego_semillas, reportar)

        # Los atributos quedan guardados antes de decidir la poda: un trial podado
        # también corrió, y su oráculo sirve para medir cuánto cómputo ahorró podar.
        for clave in atributos:
            trial.set_user_attr(clave, resultado[clave])

        if podado:
            raise optuna.TrialPruned(
                f"trial {trial.number} podado tras reportar should_prune()=True"
            )

        valores = [resultado[clave] for clave in metricas]
        for clave, valor in zip(metricas, valores, strict=True):
            if not math.isfinite(valor):
                raise MetricaNoFinita(f"métrica {clave!r} = {valor!r} no es finita")

        if multiobjetivo:
            return tuple(float(v) for v in valores)
        return float(valores[0])

    return objective
