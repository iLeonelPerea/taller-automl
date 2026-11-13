"""Ejemplo de `train.py`: un script autónomo que un tesista podría escribir.

No importa `taller_hpo` ni `optuna`: corre con
`python plantilla/train_ejemplo.py --config ... --salida ...` desde cualquier entorno con
numpy y torch y una tarjeta de video NVIDIA con CUDA, que es el escenario de
`taller_hpo.subproceso.por_subproceso`. Sin tarjeta de video visible, `entrenar` se
detiene con un `RuntimeError` explícito.

El problema es distinto al juguete del repo: dos lunas entrelazadas generadas con numpy,
más `"n_ruido"` columnas de puro ruido para que `DEFAULTS` no quede cerca del techo.
Compara `DEFAULTS` contra una buena configuración en promedio sobre varias semillas,
nunca en una sola corrida.

`get_config`, `DEFAULTS`, `entrenar` y `main` son la copia mínima de los cuatro cambios
de `plantilla/README.md`, y los valores de `DEFAULTS` coinciden con los que copia
`plantilla/buscar_por_subproceso.py`.

Cada fuente de azar tiene su propio generador, nunca el global:
- `semillas.datos` siembra un `numpy.random.default_rng` para las lunas y el ruido.
- `semillas.orden` siembra otro para el orden de los minibatches, con una permutación
  por época sobre datos que ya viven en la tarjeta de video.
- `semillas.init` siembra un `torch.Generator` propio para inicializar los pesos con la
  misma fórmula que `nn.Linear.reset_parameters`.

`entrenar` no es segura entre hilos, así que las búsquedas usan `n_jobs=1`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import namedtuple
from typing import Any

import numpy as np
import torch
from torch import nn

Semillas = namedtuple("Semillas", ["datos", "orden", "init"])

DEFAULTS: dict[str, Any] = {
    # El problema: fijo durante una búsqueda, no son hiperparámetros. Sus valores
    # coinciden con los que copia `plantilla/buscar_por_subproceso.py`.
    "n_train": 300,
    "n_val": 100,
    "ruido": 0.25,
    "n_ruido": 65,
    # Hiperparámetros. alpha es el weight_decay acoplado de torch.optim.Adam, que
    # penaliza también los sesgos.
    "n_capas": 1,
    "ancho": 16,
    "activacion": "relu",
    "learning_rate_init": 1e-2,
    "alpha": 1e-4,
    "batch_size": 32,
    "epocas": 15,
    "dispositivo": "auto",
}

_ACTIVACIONES: dict[str, Any] = {"relu": torch.relu, "tanh": torch.tanh}
# El taller entrena siempre en la tarjeta de video (ver el docstring del
# módulo): los valores válidos son "auto" y "cuda" (con ":N" opcional).
_PATRON_DISPOSITIVO = re.compile(r"auto|cuda(:\d+)?")


def get_config(defaults: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Copia corta de `taller_hpo.configuracion.get_config`: una clave desconocida es un
    KeyError."""
    desconocidas = sorted(set(overrides) - set(defaults))
    if desconocidas:
        raise KeyError(f"claves desconocidas: {desconocidas}")
    return {**defaults, **overrides}


def _resolver_dispositivo(dispositivo: str) -> str:
    """`'auto'`, `'cuda'` o `'cuda:N'`. Sin tarjeta de video visible, se detiene con un
    `RuntimeError` explícito."""
    if not _PATRON_DISPOSITIVO.fullmatch(dispositivo):
        raise ValueError(
            f"dispositivo debe ser 'auto' o 'cuda' (opcionalmente con ':N'), no {dispositivo!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "el taller necesita una tarjeta de video NVIDIA con CUDA y el controlador "
            "instalado; aquí no hay ninguna disponible"
        )
    return "cuda" if dispositivo == "auto" else dispositivo


def _lunas(
    n_total: int, ruido: float, n_ruido: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Dos lunas entrelazadas generadas con numpy, más `n_ruido` columnas de ruido
    gaussiano sin relación con la clase."""
    n_out = n_total // 2
    n_in = n_total - n_out
    angulo_out = np.linspace(0, np.pi, n_out)
    angulo_in = np.linspace(0, np.pi, n_in)
    x_out = np.stack([np.cos(angulo_out), np.sin(angulo_out)], axis=1)
    x_in = np.stack([1 - np.cos(angulo_in), 1 - np.sin(angulo_in) - 0.5], axis=1)
    x = np.concatenate([x_out, x_in], axis=0)
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)])
    x = x + rng.normal(scale=ruido, size=x.shape)
    if n_ruido > 0:
        x = np.concatenate([x, rng.normal(size=(n_total, n_ruido))], axis=1)
    orden = rng.permutation(n_total)
    return x[orden], y[orden]


def _capa(
    entradas: int, salidas: int, generador: torch.Generator
) -> tuple[nn.Parameter, nn.Parameter]:
    """Pesos y sesgo de una capa lineal con la fórmula de `nn.Linear.reset_parameters`
    (`bound = 1/sqrt(fan_in)`), sembrados con un `torch.Generator` propio."""
    limite = 1.0 / math.sqrt(entradas)
    peso = torch.empty(salidas, entradas).uniform_(-limite, limite, generator=generador)
    sesgo = torch.empty(salidas).uniform_(-limite, limite, generator=generador)
    return nn.Parameter(peso), nn.Parameter(sesgo)


class _MLP(nn.Module):
    """MLP simple con `n_capas` capas ocultas de ancho `ancho`, con los pesos ya
    inicializados por `_capa`."""

    def __init__(
        self, entradas: int, ancho: int, n_capas: int, activacion, generador: torch.Generator
    ):
        super().__init__()
        self._activacion = activacion
        self.pesos = nn.ParameterList()
        self.sesgos = nn.ParameterList()
        anterior = entradas
        for _ in range(n_capas):
            peso, sesgo = _capa(anterior, ancho, generador)
            self.pesos.append(peso)
            self.sesgos.append(sesgo)
            anterior = ancho
        peso, sesgo = _capa(anterior, 1, generador)
        self.pesos.append(peso)
        self.sesgos.append(sesgo)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        capas_ocultas = list(zip(self.pesos, self.sesgos, strict=True))[:-1]
        for peso, sesgo in capas_ocultas:
            x = self._activacion(nn.functional.linear(x, peso, sesgo))
        peso_salida, sesgo_salida = self.pesos[-1], self.sesgos[-1]
        return nn.functional.linear(x, peso_salida, sesgo_salida)


def entrenar(
    config: dict[str, Any],
    semillas: Semillas,
    reportar: Any = None,
) -> dict[str, float]:
    """Entrena un MLP de PyTorch sobre dos lunas ruidosas y devuelve métricas de
    validación: `exactitud_val` es lo único que una búsqueda puede ver.

    `semillas` es un objeto con `.datos`, `.orden` e `.init`, como `Semillas`.
    """
    c = get_config(DEFAULTS, **config)
    if c["activacion"] not in _ACTIVACIONES:
        raise ValueError(
            f"activacion debe ser una de {sorted(_ACTIVACIONES)}, no {c['activacion']!r}"
        )
    dispositivo = _resolver_dispositivo(c["dispositivo"])

    rng_datos = np.random.default_rng(semillas.datos)
    x, y = _lunas(c["n_train"] + c["n_val"], c["ruido"], c["n_ruido"], rng_datos)
    x_tr, y_tr = x[: c["n_train"]], y[: c["n_train"]]
    x_va, y_va = x[c["n_train"] :], y[c["n_train"] :]

    # int(): Generator.manual_seed exige un int nativo, y en_proceso reenvía
    # el Semillas tal cual (puede traer np.int64).
    generador_init = torch.Generator().manual_seed(int(semillas.init))
    modelo = _MLP(
        entradas=x.shape[1],
        ancho=c["ancho"],
        n_capas=c["n_capas"],
        activacion=_ACTIVACIONES[c["activacion"]],
        generador=generador_init,
    ).to(dispositivo)

    # Datos residentes en el dispositivo: el bucle indexa con `index_select` sobre la
    # permutación de cada época.
    x_tr_t = torch.from_numpy(x_tr.astype(np.float32)).to(dispositivo)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1).to(dispositivo)
    x_va_t = torch.from_numpy(x_va.astype(np.float32)).to(dispositivo)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).unsqueeze(1).to(dispositivo)

    perdida_fn = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=c["learning_rate_init"], weight_decay=c["alpha"]
    )

    rng_orden = np.random.default_rng(semillas.orden)
    n = c["n_train"]
    tam_lote = c["batch_size"]

    epocas = 0
    exactitud_val = float("nan")
    for epoca in range(c["epocas"]):
        orden = rng_orden.permutation(n)
        orden_t = torch.from_numpy(orden).to(dispositivo)
        modelo.train()
        for inicio in range(0, n, tam_lote):
            idx = orden_t[inicio : inicio + tam_lote]
            xb = x_tr_t.index_select(0, idx)
            yb = y_tr_t.index_select(0, idx)
            optimizador.zero_grad()
            perdida = perdida_fn(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
        epocas = epoca + 1

        modelo.eval()
        with torch.no_grad():
            logits_va = modelo(x_va_t)
            predicciones = (logits_va > 0).float()
            aciertos = int((predicciones == y_va_t).sum().item())
        exactitud_val = aciertos / len(y_va)
        if reportar is not None and reportar(epoca, exactitud_val):
            break

    return {"exactitud_val": exactitud_val, "epocas": float(epocas)}


def main(argv: list[str] | None = None) -> None:
    # Solo el primer párrafo del docstring, para `--help`.
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="JSON de entrada: {config, semillas}")
    parser.add_argument("--salida", required=True, help="dónde escribir el JSON de métricas")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        entrada = json.load(f)

    metricas = entrenar(entrada["config"], Semillas(**entrada["semillas"]))

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(metricas, f)


if __name__ == "__main__":
    main()
