"""El mismo juguete de mlp_paridad, pero en PyTorch y con los datos residentes en la
tarjeta de video.

Mismo contrato, mismas métricas, mismos datos y mismo `DEFAULTS` que `mlp_paridad`, así
que comparar los dos módulos muestra qué cambia al mover el mismo problema de un
framework a otro.

Puntos centrales del diseño:

1. Es un espejo de `mlp_paridad` hasta el redondeo: misma inicialización de pesos (bit a
   bit, ver `_inicializar_pesos`), mismo orden de minibatches, misma regularización L2 y
   mismo Adam. Las pruebas comprueban que las predicciones coinciden con las de
   scikit-learn.
2. Sube los datos a la tarjeta de video una sola vez. Cada época solo sube el orden de los
   minibatches e indexa los datos ya residentes.
3. Escribe el paso de Adam a mano (`_paso_adam_foreach`), porque la fórmula de
   `torch.optim.Adam` no coincide con la de scikit-learn y su `step()` agrega un costo
   fijo por paso.
4. No toca el generador aleatorio global de torch: los pesos iniciales salen de una
   `numpy.random.RandomState` propia y las capas se construyen con
   `torch.nn.utils.skip_init`.
5. No sincroniza con el host dentro del bucle de minibatches: la divergencia se acumula
   en un tensor y se revisa una vez al final de cada época.

La época 0 corre en modo eager y desde la época 1 cada época se repite como un CUDA graph,
con Adam por lotes (`torch._foreach_*`), porque el costo dominante es lanzar kernels desde
Python. Las dos rutas dan la misma trayectoria, y solo una captura de grafo corre a la
vez por proceso.

La precisión por defecto es float64, igual que scikit-learn, y `"precision": "float32"`
existe para medir su efecto. Con la misma config y las mismas semillas los resultados son
bit-idénticos sin activar `torch.use_deterministic_algorithms`, así que `entrenar` no
cambia ese estado global.

Una pérdida no finita lanza la misma `Divergencia` de `mlp_paridad`, y
`"dispositivo": "auto"` resuelve a `"cuda"` o falla con un ValueError que dice qué falta.
"""

from __future__ import annotations

import math
import numbers
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import skip_init

from taller_hpo.configuracion import get_config
from taller_hpo.problemas import mlp_contrato
from taller_hpo.problemas.mlp_contrato import SEMILLA_ORACULO, SEMILLA_PRUEBA, Divergencia
from taller_hpo.problemas.paridad import Predictor, _muestras_oraculo, generar, riesgo_verdadero
from taller_hpo.semillas import Semillas
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

Reportar = Callable[[int, float], bool]

DEFAULTS: dict[str, Any] = {**mlp_contrato.DEFAULTS, "dispositivo": "auto", "precision": "float64"}

# Tamaño de lote para las predicciones del oráculo, independiente de "batch_size": evita
# un tensor de activaciones gigante si `n_oraculo` crece mucho.
_LOTE_ORACULO = 65_536

# CUDA solo permite una captura de stream a la vez por proceso. Este candado serializa las
# capturas, nunca los replays: con `n_jobs>1` por hilos, como mucho un hilo captura a la
# vez, una vez por llamada a `entrenar`.
_CANDADO_CAPTURA = threading.Lock()

# Hiperparámetros fijos de Adam: los defaults de AdamOptimizer de scikit-learn, que
# mlp_paridad no cambia.
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPSILON = 1e-8

_ACTIVACIONES: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "logistic": nn.Sigmoid,
    "identity": nn.Identity,
}

_DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}
_NP_DTYPES: dict[str, type] = {"float32": np.float32, "float64": np.float64}
# Inverso de _DTYPES: de un dtype de torch a la perilla "precision", para derivar la
# precisión de un modelo ya construido.
_PRECISION_POR_DTYPE: dict[torch.dtype, str] = {v: k for k, v in _DTYPES.items()}


def subir(arr: np.ndarray, dtype: type, dispositivo: str) -> torch.Tensor:
    """`arr` (numpy) -> tensor en `dispositivo`, con una copia contigua y escribible propia.

    Copia siempre, porque `torch.from_numpy` sobre un array de solo lectura (como los de
    la caché del oráculo) da un tensor no escribible y un `UserWarning`.
    """
    return torch.from_numpy(np.array(arr, dtype=dtype, copy=True, order="C")).to(dispositivo)


def _resolver_dispositivo(dispositivo: str) -> str:
    """`"auto"` significa la tarjeta de video: sin CUDA visible, falla con un
    `ValueError` en español que dice qué falta."""
    if dispositivo == "auto":
        if not torch.cuda.is_available():
            raise ValueError(MENSAJE_SIN_TARJETA)
        return "cuda"
    return dispositivo


_PATRON_DISPOSITIVO = re.compile(r"auto|cuda(:\d+)?")


def _validar_dispositivo(dispositivo: str) -> None:
    if not _PATRON_DISPOSITIVO.fullmatch(dispositivo):
        raise ValueError(
            f"dispositivo debe ser 'auto' o 'cuda' (opcionalmente con ':N'), "
            f"no {dispositivo!r}: el taller entrena siempre en la tarjeta de video"
        )


def _validar_precision(precision: str) -> None:
    if precision not in _DTYPES:
        raise ValueError(f"precision debe ser una de {sorted(_DTYPES)}, no {precision!r}")


def _validar_hiperparametros_entrenamiento(
    *, batch_size: Any, learning_rate_init: Any, alpha: Any, epocas: Any
) -> None:
    """Valida los hiperparámetros numéricos de `entrenar_red` con los mismos dominios que
    exige scikit-learn: `batch_size` entero >= 1, `learning_rate_init` > 0, `alpha` >= 0
    y `epocas` entero >= 1. Acepta enteros y reales de numpy, pero no `bool` ni NaN."""
    if (
        not isinstance(batch_size, numbers.Integral)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(f"batch_size debe ser un entero >= 1, no {batch_size!r}")
    if (
        not isinstance(learning_rate_init, numbers.Real)
        or isinstance(learning_rate_init, bool)
        or math.isnan(learning_rate_init)
        or learning_rate_init <= 0
    ):
        raise ValueError(f"learning_rate_init debe ser > 0, no {learning_rate_init!r}")
    if (
        not isinstance(alpha, numbers.Real)
        or isinstance(alpha, bool)
        or math.isnan(alpha)
        or alpha < 0
    ):
        raise ValueError(f"alpha debe ser >= 0, no {alpha!r}")
    if not isinstance(epocas, numbers.Integral) or isinstance(epocas, bool) or epocas < 1:
        raise ValueError(f"epocas debe ser un entero >= 1, no {epocas!r}")


def _construir_red(
    anchos: Sequence[int], activacion: str, entradas: int, *, dispositivo: str, dtype: torch.dtype
) -> nn.Sequential:
    """Red de `len(anchos)` capas ocultas de `activacion` + una salida lineal de 1 logit.

    Las capas se construyen con `skip_init`, sin inicializar y sin tocar el generador
    global de torch: `_inicializar_pesos` las sobrescribe enseguida.
    """
    activacion_cls = _ACTIVACIONES[activacion]
    capas: list[nn.Module] = []
    anterior = entradas
    for ancho in anchos:
        capas.append(skip_init(nn.Linear, anterior, ancho, device=dispositivo, dtype=dtype))
        capas.append(activacion_cls())
        anterior = ancho
    capas.append(skip_init(nn.Linear, anterior, 1, device=dispositivo, dtype=dtype))
    return nn.Sequential(*capas)


def _inicializar_pesos(red: nn.Sequential, activacion: str, semilla_init: int) -> None:
    """Copia a `red` la misma inicialización que `MLPClassifier` de scikit-learn.

    Glorot uniforme con una `numpy.random.RandomState(semilla_init)`, sorteando por cada
    capa, en orden, primero el peso y luego el sesgo, igual que `_init_coef`. La cota es
    `sqrt(factor / (fan_in + fan_out))`, con `factor` 2.0 para "logistic" y 6.0 para las
    demás activaciones, también en la capa de salida. Los pesos coinciden bit a bit con
    los de scikit-learn. `nn.Linear.weight` tiene forma `(fan_out, fan_in)`, así que
    recibe la transpuesta.
    """
    factor = 2.0 if activacion == "logistic" else 6.0
    rng = np.random.RandomState(semilla_init)
    capas_lineales = [m for m in red if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for capa in capas_lineales:
            fan_in, fan_out = capa.in_features, capa.out_features
            cota = math.sqrt(factor / (fan_in + fan_out))
            coef = rng.uniform(-cota, cota, (fan_in, fan_out))
            intercepto = rng.uniform(-cota, cota, fan_out)
            capa.weight.copy_(torch.from_numpy(coef.T))
            capa.bias.copy_(torch.from_numpy(intercepto))


def _gen_batches(n: int, batch_size: int):
    """`(inicio, fin)` consecutivos de `batch_size` elementos sobre `range(n)`, como
    `sklearn.utils.gen_batches`: el último lote parcial siempre se incluye."""
    inicio = 0
    for _ in range(n // batch_size):
        fin = inicio + batch_size
        yield inicio, fin
        inicio = fin
    if inicio < n:
        yield inicio, n


@dataclass
class _EstadoAdam:
    """Primer y segundo momento (`m`, `v`) por parámetro, y el paso `t` compartido.

    `t` es un entero de Python, como en scikit-learn, así que elevar `beta1 ** t` no
    necesita traer nada del dispositivo.
    """

    ms: list[torch.Tensor] = field(default_factory=list)
    vs: list[torch.Tensor] = field(default_factory=list)
    t: int = 0

    @staticmethod
    def nuevo(parametros: Sequence[torch.Tensor]) -> _EstadoAdam:
        return _EstadoAdam(
            ms=[torch.zeros_like(p) for p in parametros],
            vs=[torch.zeros_like(p) for p in parametros],
        )


def _lr_schedule_epoca(
    t_inicial: int,
    n_pasos: int,
    lr_init: float,
    *,
    beta1: float = _ADAM_BETA1,
    beta2: float = _ADAM_BETA2,
) -> np.ndarray:
    """El `lr_t` de Adam (fórmula de scikit-learn) para los `n_pasos` minibatches de una
    época, desde el paso global `t_inicial + 1`.

    Un CUDA graph no vuelve a ejecutar Python en cada `replay()`, así que los `lr_t` de la
    época van en un tensor que se rellena antes de cada replay.
    """
    valores = np.empty(n_pasos, dtype=np.float64)
    for i in range(n_pasos):
        t = t_inicial + 1 + i
        valores[i] = lr_init * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
    return valores


def _paso_adam_foreach(
    parametros: Sequence[torch.Tensor],
    gradientes: Sequence[torch.Tensor],
    estado: _EstadoAdam,
    neg_lr_t: torch.Tensor,
    *,
    beta1: float = _ADAM_BETA1,
    beta2: float = _ADAM_BETA2,
    eps: float = _ADAM_EPSILON,
) -> None:
    """El paso de Adam de este módulo, idéntico al de `AdamOptimizer` de scikit-learn.

    Usa `torch._foreach_*` (un lanzamiento por operación para todos los parámetros) y
    recibe el learning rate efectivo como tensor (`neg_lr_t`, ya negado): un `float` de
    Python quedaría fijo dentro del CUDA graph capturado.
    """
    with torch.no_grad():
        torch._foreach_mul_(estado.ms, beta1)
        torch._foreach_add_(estado.ms, gradientes, alpha=1.0 - beta1)
        torch._foreach_mul_(estado.vs, beta2)
        torch._foreach_addcmul_(estado.vs, gradientes, gradientes, value=1.0 - beta2)
        denominador = torch._foreach_sqrt(estado.vs)
        torch._foreach_add_(denominador, eps)
        actualizacion = torch._foreach_div(estado.ms, denominador)
        torch._foreach_mul_(actualizacion, neg_lr_t)
        torch._foreach_add_(parametros, actualizacion)


@dataclass(frozen=True)
class ResultadoEntrenamiento:
    """Lo que devuelve `entrenar_red`: la red entrenada más sus métricas de costo.

    No incluye `exactitud_verdadera` ni `exactitud_prueba`, que dependen del oráculo del
    problema; `entrenar` las agrega después. `costo_captura_grafo_s` son los segundos de
    capturar, instanciar y correr el primer replay del CUDA graph de una época, o `None`
    si no hubo grafo.
    """

    modelo: nn.Sequential
    epocas: int
    exactitud_val: float
    n_parametros: int
    segundos: float
    costo_captura_grafo_s: float | None = None


def _ejecutar_epoca_cuda(
    red: nn.Sequential,
    capas_lineales: Sequence[nn.Linear],
    parametros: Sequence[torch.Tensor],
    estado_adam: _EstadoAdam,
    perdida_fn: nn.Module,
    x_tr_dev: torch.Tensor,
    y_tr_dev: torch.Tensor,
    x_va_dev: torch.Tensor,
    y_va_dev: torch.Tensor,
    orden_static: torch.Tensor,
    lr_static: torch.Tensor,
    contador_static: torch.Tensor,
    limites_lotes: Sequence[tuple[int, int]],
    alpha: float,
    dispositivo: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Una época completa (todos los minibatches y la validación) en tensores puros.

    No tiene control de flujo que dependa de un valor del dispositivo ni reasigna tensores
    que existan fuera de la función (`orden_static`, `lr_static`), así que la misma
    función corre en eager y capturada en un CUDA graph, con el mismo resultado bit a bit.

    Devuelve `torch.stack([aciertos_val, no_finito_epoca, contador_static])` en float64:
    el llamador hace un solo `.tolist()` por época, la única sincronización con el host.
    `contador_static` avanza 1 por época, y si un replay no lo avanza, la captura quedó
    vacía.
    """
    suma_perdida = torch.zeros((), dtype=dtype, device=dispositivo)
    pesos = [capa.weight for capa in capas_lineales]
    red.train()
    for i, (inicio, fin) in enumerate(limites_lotes):
        idx = orden_static[inicio:fin]
        xb = x_tr_dev.index_select(0, idx)
        yb = y_tr_dev.index_select(0, idx)

        for p in parametros:
            p.grad = None
        logits = red(xb)
        perdida = perdida_fn(logits, yb)
        # Sin sincronizar: acumula en un tensor del dispositivo y se revisa al final.
        suma_perdida = suma_perdida + perdida.detach()
        perdida.backward()

        n_en_lote = fin - inicio  # constante entre épocas: mismos límites siempre.
        with torch.no_grad():
            # L2 de scikit-learn en un solo lanzamiento: + alpha * W / n_en_lote, solo en
            # los pesos. `alpha / n_en_lote` es constante en esta llamada, así que puede
            # quedar fijo dentro del grafo, a diferencia de `lr_t`.
            torch._foreach_add_(
                [capa.weight.grad for capa in capas_lineales], pesos, alpha=alpha / n_en_lote
            )

        neg_lr_t = lr_static[i : i + 1].squeeze(0)
        _paso_adam_foreach(parametros, [p.grad for p in parametros], estado_adam, neg_lr_t)

    red.eval()
    with torch.no_grad():
        logits_va = red(x_va_dev)
        no_finito_epoca = ~torch.isfinite(suma_perdida) | ~torch.isfinite(logits_va).all()
        predicciones = (logits_va > 0).to(dtype)
        aciertos = (predicciones == y_va_dev).sum()
        contador_static.add_(1)
    return torch.stack(
        [aciertos.to(torch.float64), no_finito_epoca.to(torch.float64), contador_static]
    )


def entrenar_red(
    anchos: Sequence[int],
    *,
    activacion: str,
    learning_rate_init: float,
    alpha: float,
    batch_size: int,
    epocas: int,
    datos: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    semillas: Semillas,
    dispositivo: str,
    precision: str = "float64",
    reportar: Reportar | None = None,
    usar_cuda_graph: bool = True,
) -> ResultadoEntrenamiento:
    """Entrena un MLP espejo de scikit-learn sobre datos ya en numpy.

    Es la API que reusan `entrenar` y el CASH (`cash_paridad`), con un ancho por capa.
    Valida `anchos`, `activacion`, `precision`, `batch_size`, `learning_rate_init`,
    `alpha`, `epocas` y `dispositivo` antes de tocar la tarjeta de video, con un
    ValueError en español.

    Parameters
    ----------
    anchos:
        Un ancho por capa oculta, en orden.
    activacion:
        Una de `_ACTIVACIONES` ("relu", "tanh", "logistic", "identity"), para todas las
        capas ocultas.
    datos:
        `(x_tr, y_tr, x_va, y_va)`, arrays de numpy: `x_*` de forma `(n, n_features)`,
        `y_*` de forma `(n,)` en {0, 1}.
    dispositivo:
        "cuda", "cuda:N" o "auto" ("auto" es la tarjeta de video).
    precision:
        "float32" o "float64" (default).
    usar_cuda_graph:
        Interruptor interno, no es un hiperparámetro del taller. Con `True` (default)
        captura un CUDA graph desde la época 1; `False` corre todo en eager, para las
        pruebas y para depurar.

    Returns
    -------
    ResultadoEntrenamiento con la red ya en modo `eval()`.
    """
    for ancho in anchos:
        if not isinstance(ancho, numbers.Integral) or isinstance(ancho, bool) or ancho < 1:
            raise ValueError(f"cada ancho debe ser un entero >= 1, no {ancho!r}")
    if activacion not in _ACTIVACIONES:
        raise ValueError(f"activacion debe ser una de {sorted(_ACTIVACIONES)}, no {activacion!r}")
    _validar_precision(precision)
    _validar_hiperparametros_entrenamiento(
        batch_size=batch_size, learning_rate_init=learning_rate_init, alpha=alpha, epocas=epocas
    )
    dispositivo = _resolver_dispositivo(dispositivo)
    if not dispositivo.startswith("cuda"):
        raise ValueError(
            f"dispositivo debe resolver a la tarjeta de video ('cuda', opcionalmente con "
            f"':N'), no {dispositivo!r}: el taller entrena siempre en la tarjeta de video"
        )
    if not torch.cuda.is_available():
        # Sin esta comprobación, `subir` fallaría con un error de torch en inglés en vez
        # de este mensaje claro.
        raise ValueError(MENSAJE_SIN_TARJETA)
    dtype = _DTYPES[precision]
    np_dtype = _NP_DTYPES[precision]
    x_tr, y_tr, x_va, y_va = datos
    n_train = x_tr.shape[0]
    n_val = x_va.shape[0]
    entradas = x_tr.shape[1]

    # Datos residentes: UNA sola subida a `dispositivo`, antes de la primera época.
    x_tr_dev = subir(x_tr, np_dtype, dispositivo)
    y_tr_dev = subir(y_tr, np_dtype, dispositivo).unsqueeze(1)
    x_va_dev = subir(x_va, np_dtype, dispositivo)
    y_va_dev = subir(y_va, np_dtype, dispositivo).unsqueeze(1)

    red = _construir_red(anchos, activacion, entradas, dispositivo=dispositivo, dtype=dtype)
    _inicializar_pesos(red, activacion, semillas.init)
    capas_lineales = [m for m in red if isinstance(m, nn.Linear)]
    parametros = list(red.parameters())
    estado_adam = _EstadoAdam.nuevo(parametros)

    # Mismo generador que mlp_paridad, creado una vez y consumido con una permutación por
    # época: la secuencia de permutaciones es idéntica entre los dos módulos.
    rng_orden = np.random.default_rng(semillas.orden)
    # Recorta `batch_size` a `n_train` como scikit-learn, pero sin su aviso, que en las
    # pruebas con `n_train` chico sería ruido bajo `-W error`.
    batch_efectivo = min(batch_size, n_train)
    # Los límites de cada lote no dependen de la época, así que un solo CUDA graph sirve
    # para todas.
    limites_lotes = list(_gen_batches(n_train, batch_efectivo))
    n_lotes = len(limites_lotes)

    perdida_fn = nn.BCEWithLogitsLoss(reduction="mean")

    costo_captura_grafo_s: float | None = None
    inicio_reloj = time.perf_counter()
    epocas_hechas = 0
    exactitud_val = float("nan")

    # El dispositivo real con índice concreto: `torch.cuda.Stream` y `torch.cuda.device` lo
    # necesitan para capturar en el mismo dispositivo donde vive la red.
    dispositivo_torch = parametros[0].device
    # Buffers estáticos: siempre se rellenan con `.copy_()` y nunca se reasignan, porque un
    # CUDA graph exige la misma dirección de memoria en todas las épocas. `orden_host` y
    # `lr_host` son su contraparte en memoria fijada del host.
    orden_static = torch.empty(n_train, dtype=torch.int64, device=dispositivo)
    lr_static = torch.empty(n_lotes, dtype=dtype, device=dispositivo)
    orden_host = torch.empty(n_train, dtype=torch.int64, pin_memory=True)
    lr_host = torch.empty(n_lotes, dtype=dtype, pin_memory=True)
    # Contador de épocas grabado dentro del grafo: detecta una captura vacía, que sin él
    # repetiría en silencio las métricas de la época 1.
    contador_static = torch.zeros((), dtype=torch.float64, device=dispositivo)

    grafo: torch.cuda.CUDAGraph | None = None
    salida_epoca: torch.Tensor | None = None
    with torch.cuda.device(dispositivo_torch):
        for epoca in range(epocas):
            orden = rng_orden.permutation(n_train)
            orden_host.copy_(torch.from_numpy(orden))
            orden_static.copy_(orden_host, non_blocking=True)
            # Ya negado (`-lr_t`), así `_ejecutar_epoca_cuda` lo lee directo.
            lr_host.copy_(
                torch.from_numpy(-_lr_schedule_epoca(estado_adam.t, n_lotes, learning_rate_init))
            )
            lr_static.copy_(lr_host, non_blocking=True)
            estado_adam.t += n_lotes

            capturar_esta_epoca = usar_cuda_graph and epoca == 1 and grafo is None
            t_epoca_inicio = time.perf_counter() if capturar_esta_epoca else None
            args_epoca = (
                red,
                capas_lineales,
                parametros,
                estado_adam,
                perdida_fn,
                x_tr_dev,
                y_tr_dev,
                x_va_dev,
                y_va_dev,
                orden_static,
                lr_static,
                contador_static,
                limites_lotes,
                alpha,
                dispositivo,
                dtype,
            )
            if not usar_cuda_graph or epoca == 0:
                # Época 0 SIEMPRE eager (calienta cuBLAS y las asignaciones
                # perezosas), y CUALQUIER época si el interruptor está apagado.
                salida_epoca = _ejecutar_epoca_cuda(*args_epoca)
            elif capturar_esta_epoca:
                grafo = torch.cuda.CUDAGraph()
                # Stream de captura propio, en el dispositivo real de la red, en vez del
                # `default_capture_stream` compartido de `torch.cuda.graph`.
                flujo_captura = torch.cuda.Stream(device=dispositivo_torch)
                with (
                    _CANDADO_CAPTURA,
                    torch.cuda.graph(
                        grafo, stream=flujo_captura, capture_error_mode="thread_local"
                    ),
                ):
                    salida_epoca = _ejecutar_epoca_cuda(*args_epoca)
                # Capturar solo registra los kernels, así que el replay inmediato es el
                # que calcula de verdad la época 1.
                grafo.replay()
            else:
                grafo.replay()

            # Un solo `.item()`-equivalente por época (`.tolist()`): la única
            # sincronización dispositivo->host, para TODAS las ramas de arriba.
            aciertos_val, no_finito_val, epocas_en_dispositivo = salida_epoca.tolist()
            if epocas_en_dispositivo != epoca + 1:
                # El contador no avanzó: la captura quedó vacía o en un stream que nunca
                # corrió.
                raise RuntimeError(
                    f"el CUDA graph no avanzó en la época {epoca} (el dispositivo "
                    f"lleva {epocas_en_dispositivo:.0f} épocas): la captura quedó "
                    "vacía o en otro stream"
                )
            if capturar_esta_epoca:
                costo_captura_grafo_s = time.perf_counter() - t_epoca_inicio
            epocas_hechas = epoca + 1
            if no_finito_val:
                raise Divergencia(
                    f"pérdida o salidas no finitas en la época {epoca} con learning_rate_init="
                    f"{learning_rate_init}"
                )
            exactitud_val = aciertos_val / n_val
            if reportar is not None and reportar(epoca, exactitud_val):
                break
    segundos = time.perf_counter() - inicio_reloj

    # Suelta `.grad`, que puede apuntar al pool privado de memoria del CUDA graph, para
    # que el modelo devuelto no retrase su liberación.
    for p in parametros:
        p.grad = None

    red.eval()
    n_parametros = sum(p.numel() for p in parametros)
    return ResultadoEntrenamiento(
        modelo=red,
        epocas=epocas_hechas,
        exactitud_val=exactitud_val,
        n_parametros=n_parametros,
        segundos=segundos,
        costo_captura_grafo_s=costo_captura_grafo_s,
    )


def _resolver_dispositivo_y_precision(
    modelo: nn.Module, dispositivo: str | None, precision: str | None
) -> tuple[str, type]:
    """`(dispositivo, np_dtype)` compatibles con `modelo`, para `exactitud_en_dispositivo`
    y `riesgo_verdadero_en_dispositivo`.

    Con `None` los deriva de los parámetros del modelo; explícitos, los valida contra el
    dispositivo y el dtype reales del modelo y resuelve "auto". Un módulo sin parámetros
    usa "cuda" y float64, y un modelo fuera de la tarjeta de video es un ValueError.
    """
    parametro = next(modelo.parameters(), None)
    if dispositivo is None:
        if parametro is not None:
            if parametro.device.type != "cuda":
                raise ValueError(
                    f"el modelo vive en {parametro.device}, no en la tarjeta de video: "
                    "esta función siempre calcula en la tarjeta de video; entrena o "
                    "mueve el modelo ahí antes de llamarla"
                )
            dispositivo_final = str(parametro.device)
        else:
            dispositivo_final = "cuda"
    else:
        _validar_dispositivo(dispositivo)
        # "auto" vale como perilla del taller, pero no es un dispositivo real de torch.
        dispositivo_final = _resolver_dispositivo(dispositivo)
        if parametro is not None:
            pedido = torch.device(dispositivo_final)
            real = parametro.device
            # `pedido.index is None` cubre "cuda" sin índice (compatible con cualquier
            # tarjeta de video real del modelo, típicamente "cuda:0"); con índice
            # explícito ("cuda:1") sí se exige que coincida.
            if pedido.type != real.type or (
                pedido.index is not None and pedido.index != real.index
            ):
                raise ValueError(
                    f"dispositivo={dispositivo!r} no coincide con el dispositivo real del "
                    f"modelo ({real}); pasa dispositivo=None para tomarlo del modelo"
                )
    if precision is None:
        precision_final = (
            _PRECISION_POR_DTYPE[parametro.dtype] if parametro is not None else "float64"
        )
    else:
        _validar_precision(precision)
        if parametro is not None and _DTYPES[precision] != parametro.dtype:
            raise ValueError(
                f"precision={precision!r} no coincide con la precisión real del modelo "
                f"({_PRECISION_POR_DTYPE[parametro.dtype]}); pasa precision=None para tomarla "
                "del modelo"
            )
        precision_final = precision
    return dispositivo_final, _NP_DTYPES[precision_final]


def exactitud_en_dispositivo(
    modelo: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    dispositivo: str | None = None,
    precision: str | None = None,
) -> float:
    """Exactitud de `modelo` sobre `(x, y)` (numpy), calculada en `dispositivo`.

    Sube `x` y `y` una vez, corre `modelo` en `eval()` sin gradiente y hace un solo
    `.item()` al final. La predicción es `logit > 0`, la misma regla que
    `sigmoid(logit) > 0.5` salvo en una franja despreciable de ancho ~1.7e-16.
    `dispositivo` y `precision` en `None` se derivan de `modelo`.
    """
    dispositivo, np_dtype = _resolver_dispositivo_y_precision(modelo, dispositivo, precision)
    x_t = subir(x, np_dtype, dispositivo)
    y_t = subir(y, np_dtype, dispositivo).unsqueeze(1)
    modelo.eval()
    with torch.no_grad():
        logits = modelo(x_t)
        predicciones = (logits > 0).to(x_t.dtype)
        aciertos = (predicciones == y_t).sum()
    return (aciertos.to(torch.float64) / len(y)).item()


def riesgo_verdadero_en_dispositivo(
    modelo: nn.Module,
    *,
    d: int,
    a: float,
    k_ruido: int,
    rotar: bool,
    n: int,
    semilla: int,
    dispositivo: str | None = None,
    precision: str | None = None,
) -> float:
    """Riesgo verdadero de `modelo` (un `nn.Module`), calculado en `dispositivo`.

    Equivale a `paridad.riesgo_verdadero`, con la misma caché de muestras del oráculo,
    pero acumula las predicciones por lotes en el dispositivo y hace un solo `.item()` al
    final. `dispositivo` y `precision` en `None` se derivan de `modelo`.
    """
    dispositivo, np_dtype = _resolver_dispositivo_y_precision(modelo, dispositivo, precision)
    x, p = _muestras_oraculo(n, semilla, d, a, k_ruido, rotar)
    x_dev = subir(x, np_dtype, dispositivo)
    p_dev = subir(p, np_dtype, dispositivo)

    modelo.eval()
    total = torch.zeros((), dtype=torch.float64, device=dispositivo)
    with torch.no_grad():
        for inicio in range(0, n, _LOTE_ORACULO):
            xb = x_dev[inicio : inicio + _LOTE_ORACULO]
            pb = p_dev[inicio : inicio + _LOTE_ORACULO]
            logits = modelo(xb).squeeze(1)
            prediccion = (logits > 0).to(p_dev.dtype)
            total += torch.where(prediccion == 1, 1.0 - pb, pb).sum().to(torch.float64)
    return (total / n).item()


def entrenar(
    config: dict[str, Any], semillas: Semillas, reportar: Reportar | None = None
) -> dict[str, float]:
    c = get_config(DEFAULTS, **config)
    if c["activacion"] not in _ACTIVACIONES:
        raise ValueError(
            f"activacion debe ser una de {sorted(_ACTIVACIONES)}, no {c['activacion']!r}"
        )
    _validar_dispositivo(c["dispositivo"])
    _validar_precision(c["precision"])
    if c["epocas"] < 1:
        raise ValueError(f"epocas debe ser >= 1, no {c['epocas']}")
    problema = {k: c[k] for k in ("d", "a", "k_ruido", "rotar")}
    dispositivo = _resolver_dispositivo(c["dispositivo"])
    # Aquí `dispositivo` ya es "cuda" o "cuda:N".
    if not torch.cuda.is_available():
        raise ValueError(MENSAJE_SIN_TARJETA)
    indice = torch.device(dispositivo).index
    if indice is not None and indice >= torch.cuda.device_count():
        raise ValueError(
            f"dispositivo={c['dispositivo']!r} pero solo hay "
            f"{torch.cuda.device_count()} tarjetas de video con CUDA visibles"
        )

    x, y = generar(c["n_train"] + c["n_val"], semilla=semillas.datos, **problema)
    x_tr, y_tr = x[: c["n_train"]], y[: c["n_train"]]
    x_va, y_va = x[c["n_train"] :], y[c["n_train"] :]

    resultado = entrenar_red(
        (c["ancho"],) * c["n_capas"],
        activacion=c["activacion"],
        learning_rate_init=c["learning_rate_init"],
        alpha=c["alpha"],
        batch_size=c["batch_size"],
        epocas=c["epocas"],
        datos=(x_tr, y_tr, x_va, y_va),
        semillas=semillas,
        dispositivo=dispositivo,
        precision=c["precision"],
        reportar=reportar,
    )

    riesgo = riesgo_verdadero_en_dispositivo(
        resultado.modelo,
        n=c["n_oraculo"],
        semilla=SEMILLA_ORACULO,
        dispositivo=dispositivo,
        precision=c["precision"],
        **problema,
    )
    metricas = {
        "exactitud_val": resultado.exactitud_val,
        "exactitud_verdadera": 1.0 - riesgo,
        "n_parametros": float(resultado.n_parametros),
        "epocas": float(resultado.epocas),
        "segundos": resultado.segundos,
    }
    if c["n_prueba"] > 0:
        # Mismo conjunto de prueba que mlp_paridad (SEMILLA_PRUEBA): los dos modelos
        # del juguete se comparan sobre la misma prueba finita.
        x_pr, y_pr = generar(c["n_prueba"], semilla=SEMILLA_PRUEBA, **problema)
        metricas["exactitud_prueba"] = exactitud_en_dispositivo(
            resultado.modelo, x_pr, y_pr, dispositivo=dispositivo, precision=c["precision"]
        )
    return metricas


__all__ = [
    "DEFAULTS",
    "Divergencia",
    "Predictor",
    "ResultadoEntrenamiento",
    "entrenar",
    "entrenar_red",
    "exactitud_en_dispositivo",
    "riesgo_verdadero",
    "riesgo_verdadero_en_dispositivo",
    "subir",
]
