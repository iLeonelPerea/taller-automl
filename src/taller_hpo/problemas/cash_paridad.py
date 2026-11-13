"""CASH sobre la paridad gaussiana: el algoritmo también es un hiperparámetro.

CASH (selección conjunta de algoritmo e hiperparámetros) busca qué modelo entrenar,
además de sus perillas: `algoritmo` elige entre `logistica`, `bosque`, `hgb` y `mlp`, y
cada uno trae sus propios hiperparámetros con prefijo (`bosque_n_arboles`,
`hgb_learning_rate`, ...). El espacio es condicional: el paso 05 solo sugiere los
parámetros del algoritmo sorteado en cada trial. `knn` y `krr` son baselines con sus
defaults y nunca se buscan.

Mismo contrato que `mlp_paridad`:

    entrenar(config, semillas, reportar=None) -> dict de métricas

- `config`: overrides sobre DEFAULTS; una clave desconocida es un error.
- `semillas`: `Semillas(datos, orden, init)`. `datos` genera el dataset, `init` mueve los
  pesos iniciales de `mlp` y el submuestreo de `bosque`, y `orden` solo lo usa `mlp`.
  `logistica`, `knn` y `krr` no usan azar.
- `reportar(epoca, exactitud_val) -> bool`: solo se llama con `algoritmo == "mlp"`. Los
  demás algoritmos hacen un solo ajuste y reportan `epocas = 1`.
- Solo `mlp` puede lanzar `Divergencia`.

Devuelve las mismas métricas que `mlp_paridad`. `n_parametros` solo es comparable dentro
de cada algoritmo: pesos en `logistica` y `mlp`, nodos de los árboles en `bosque` y
`hgb`, el tamaño de la tabla guardada en `knn` y los coeficientes duales en `krr`.

`DEFAULTS` usa el mismo problema que `mlp_contrato`, pero con `rotar=True`: sin rotar,
los árboles tendrían ventaja porque las columnas informativas estarían alineadas con los
ejes. El oráculo comparte `SEMILLA_ORACULO` con el paso 00.

Todo entrena en una tarjeta de video NVIDIA con CUDA (`dispositivo` "auto", "cuda" o
"cuda:N"; "auto" siempre es "cuda"), y `bosque` y `hgb` además necesitan XGBoost con
soporte de CUDA. Sin eso, `entrenar` se detiene con un ValueError claro.

Implementaciones:
- `mlp` reusa `paridad_torch.entrenar_red`, con un ancho por capa como hiperparámetro.
- `logistica` resuelve con `torch.optim.LBFGS` el mismo objetivo que
  `LogisticRegression(solver="lbfgs")` de scikit-learn, con tolerancias más estrictas
  que las de torch para llegar al mismo óptimo.
- `knn` usa `torch.cdist` y `torch.topk` y cuenta los votos a mano, para desempatar
  igual que `KNeighborsClassifier` (a favor de la clase 0).
- `krr` es kernel ridge con núcleo RBF, resuelto con Cholesky en float64, en lugar de
  `SVC`, que no tiene implementación en la tarjeta de video. `krr_gamma="scale"` aplica
  la fórmula de `SVC`.
- `bosque` emula un bosque aleatorio con `XGBClassifier`: `n_estimators=1` con
  `num_parallel_tree` árboles, `subsample=0.632`, `colsample_bynode` de raíz cuadrada y
  `base_score=0.5`, para que `min_child_weight` cuente muestras exactas.
- `hgb` es boosting con `XGBClassifier(grow_policy="lossguide", max_depth=0)`, limitado
  solo por `max_leaves`, como `HistGradientBoostingClassifier`.

`logistica`, `knn` y `krr` calculan el oráculo en la tarjeta de video. `bosque` y `hgb`
predicen con un `DMatrix` de NumPy y `Booster.predict`, porque XGBoost no acepta tensores
de torch sin cupy y su `predict` de scikit-learn avisa por la diferencia de
dispositivos. Importar xgboost carga scikit-learn, así que solo `bosque` y `hgb` lo
cargan.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from taller_hpo.configuracion import get_config
from taller_hpo.problemas.mlp_contrato import SEMILLA_ORACULO, SEMILLA_PRUEBA, Divergencia
from taller_hpo.problemas.paridad import generar, riesgo_verdadero
from taller_hpo.semillas import Semillas
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

Reportar = Callable[[int, float], bool]
Predictor = Callable[[np.ndarray], np.ndarray]

# Techo de iteraciones de torch.optim.LBFGS para `logistica`; no es un hiperparámetro.
_LOGISTICA_MAX_ITER = 300

# Fracción de filas por árbol en `bosque`: 1 - 1/e, la fracción esperada de filas
# únicas en una muestra bootstrap.
_BOSQUE_SUBSAMPLE = 1.0 - 1.0 / math.e

# Tamaño de lote para predecir con knn y krr sin armar de una vez el tensor de
# distancias completo.
_LOTE_PREDICCION = 65_536

ALGORITMOS = ("logistica", "bosque", "hgb", "mlp", "knn", "krr")

DEFAULTS: dict[str, Any] = {
    # El problema: mismas claves que mlp_paridad, pero con rotar=True.
    "d": 3,
    "a": 1.5,
    "k_ruido": 10,
    "rotar": True,
    "n_train": 2000,
    "n_val": 500,
    # El paso 05 solo sugiere logistica, bosque, hgb y mlp; knn y krr son baselines.
    "algoritmo": "mlp",
    # "auto" siempre resuelve a una tarjeta de video con CUDA.
    "dispositivo": "auto",
    # logistica.
    "logistica_C": 1.0,
    # bosque (XGBClassifier con num_parallel_tree).
    "bosque_n_arboles": 100,
    "bosque_profundidad_max": 8,
    "bosque_min_muestras_hoja": 1,
    # hgb (XGBClassifier con grow_policy="lossguide").
    "hgb_learning_rate": 0.1,
    "hgb_max_hojas": 31,
    "hgb_max_iter": 100,
    # mlp: un ancho por capa (hasta 3); aquí la arquitectura completa es el hiperparámetro.
    "n_capas": 1,
    "ancho_1": 64,
    "ancho_2": 64,
    "ancho_3": 64,
    "mlp_activacion": "relu",
    "mlp_learning_rate_init": 1e-3,
    "mlp_alpha": 1e-4,
    "mlp_batch_size": 64,
    "mlp_epocas": 30,
    # knn y krr: baselines; el paso 05 los deja en sus defaults.
    "knn_n_vecinos": 5,
    # Default de KernelRidge.
    "krr_alpha": 1.0,
    # Único valor soportado: aplica la fórmula "scale" de SVC.
    "krr_gamma": "scale",
    # Control negativo: ningún algoritmo la lee.
    "placebo": 0.0,
    # Oráculo: muestras frescas de x para el riesgo verdadero.
    "n_oraculo": 200_000,
    # Conjunto de prueba finito, igual que mlp_paridad: 0 = no se genera.
    "n_prueba": 0,
}

_PATRON_DISPOSITIVO = re.compile(r"auto|cuda(:\d+)?")


def _validar_combinaciones(c: dict[str, Any]) -> None:
    if c["algoritmo"] not in ALGORITMOS:
        raise ValueError(f"algoritmo debe ser uno de {ALGORITMOS}, no {c['algoritmo']!r}")
    if not (1 <= c["n_capas"] <= 3):
        raise ValueError(f"n_capas debe estar entre 1 y 3, no {c['n_capas']!r}")


def _validar_dispositivo(dispositivo: str) -> None:
    if not _PATRON_DISPOSITIVO.fullmatch(dispositivo):
        raise ValueError(
            f"dispositivo debe ser 'auto' o 'cuda' (opcionalmente con ':N'), "
            f"no {dispositivo!r}: este taller entrena siempre en una tarjeta de "
            "video NVIDIA con CUDA"
        )


def _xgboost_con_cuda() -> bool:
    """`True` si `xgboost` se puede importar y su build trae soporte de CUDA.

    Solo revisa el build (`xgboost.build_info()["USE_CUDA"]`); la tarjeta de video ya la
    valida `torch.cuda.is_available()`.
    """
    try:
        import xgboost
    except ImportError:
        return False
    return bool(xgboost.build_info().get("USE_CUDA", False))


def _resolver_dispositivo(dispositivo: str) -> str:
    """Resuelve "auto" a "cuda"; "cuda" y "cuda:N" pasan tal cual."""
    return "cuda" if dispositivo == "auto" else dispositivo


def _validar_disponibilidad_gpu(dispositivo_cfg: str, dispositivo: str, algoritmo: str) -> None:
    """Comprueba que la tarjeta de video de `dispositivo` esté disponible de verdad.

    Revisa que torch esté instalado, que CUDA esté visible y que el índice esté en rango
    y, para `bosque` y `hgb`, que xgboost traiga soporte de CUDA.
    """
    try:
        import torch
    except ImportError as error:
        raise ValueError(
            f"dispositivo={dispositivo_cfg!r} pero torch no está instalado; corre "
            "'uv sync --locked' para instalar las dependencias fijas del taller "
            "(torch con CUDA)"
        ) from error
    if not torch.cuda.is_available():
        raise ValueError(f"dispositivo={dispositivo_cfg!r}: {MENSAJE_SIN_TARJETA}")
    indice = torch.device(dispositivo).index
    if indice is not None and indice >= torch.cuda.device_count():
        raise ValueError(
            f"dispositivo={dispositivo_cfg!r} pero esta máquina solo tiene "
            f"{torch.cuda.device_count()} tarjeta(s) de video con CUDA visibles"
        )
    if algoritmo in ("bosque", "hgb"):
        try:
            importlib.import_module("xgboost")
        except ImportError as error:
            raise ValueError(
                f"algoritmo={algoritmo!r} con dispositivo={dispositivo_cfg!r} necesita "
                "xgboost instalado; corre 'uv sync --locked' para instalar las "
                "dependencias fijas del taller"
            ) from error
        if not _xgboost_con_cuda():
            # Sin este chequeo, un xgboost sin CUDA fallaría más abajo con un error en inglés.
            raise ValueError(
                f"algoritmo={algoritmo!r} con dispositivo={dispositivo_cfg!r} necesita un "
                "xgboost compilado con soporte de CUDA (build_info()['USE_CUDA'] es "
                "False); revisa la instalación de xgboost (pyproject.toml fija la "
                "versión con soporte de CUDA)"
            )


def _gamma_krr(x_tr: np.ndarray, gamma_cfg: str) -> float:
    """Resuelve `krr_gamma` a un número; por ahora solo admite el literal "scale".

    Usa la fórmula de `SVC` de scikit-learn: `1.0 / (n_features * X.var())`, o `1.0` si
    la varianza da cero.
    """
    if gamma_cfg != "scale":
        raise ValueError(f"krr_gamma debe ser 'scale', no {gamma_cfg!r}")
    n_features = x_tr.shape[1]
    varianza = float(x_tr.var())
    return 1.0 / (n_features * varianza) if varianza != 0.0 else 1.0


# --- Tarjeta de video: torch (logistica/knn/krr) o xgboost (bosque/hgb) ----------------


def _contar_nodos_xgb(modelo: Any) -> float:
    """Nodos totales de todos los árboles de un modelo de xgboost ya entrenado.

    Recorre el volcado JSON de `get_dump`, porque `trees_to_dataframe` pide pandas.
    """
    dump = modelo.get_booster().get_dump(dump_format="json")

    def _contar(nodo: dict[str, Any]) -> int:
        return 1 + sum(_contar(hijo) for hijo in nodo.get("children", ()))

    return float(sum(_contar(json.loads(arbol)) for arbol in dump))


def _predecir_xgb_gpu(modelo: Any) -> Predictor:
    """`Predictor` para un modelo de xgboost entrenado en la tarjeta de video.

    Predice con `Booster.predict` sobre un `DMatrix`, porque `modelo.predict` cae a una
    ruta lenta y avisa por la diferencia de dispositivos. `nthread=1` respeta la
    convención `n_jobs=1` del taller.
    """
    import xgboost as xgb

    booster = modelo.get_booster()

    def predecir(x: np.ndarray) -> np.ndarray:
        dm = xgb.DMatrix(np.asarray(x, dtype=np.float64), nthread=1)
        probabilidades = booster.predict(dm)
        return (probabilidades > 0.5).astype(np.int64)

    return predecir


def _envolver_como_modulo(logit_fn: Callable[[Any], Any]) -> Any:
    """Envuelve `logit_fn` (tensor -> logits de forma `(n, 1)`) como un `nn.Module` sin
    estado propio, para reusar las funciones de oráculo de `paridad_torch`."""
    import torch

    class _Modulo(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            return logit_fn(x)

    return _Modulo()


def _predictor_desde_modulo(modulo: Any, dispositivo: str) -> Predictor:
    """`Predictor` de NumPy, por lotes de `_LOTE_PREDICCION`, a partir de un `nn.Module`
    que devuelve logits con la regla `logit > 0 -> clase 1`."""
    import torch

    from taller_hpo.problemas.paridad_torch import subir

    def predecir(x: np.ndarray) -> np.ndarray:
        etiquetas = np.empty(len(x), dtype=np.int64)
        with torch.no_grad():
            for inicio in range(0, len(x), _LOTE_PREDICCION):
                fin = inicio + _LOTE_PREDICCION
                xb = subir(x[inicio:fin], np.float64, dispositivo)
                logits = modulo(xb).squeeze(1)
                etiquetas[inicio:fin] = (logits > 0).cpu().numpy().astype(np.int64)
        return etiquetas

    return predecir


def _ajustar_logistica_gpu(
    x_tr: np.ndarray, y_tr: np.ndarray, capital_c: float, dispositivo: str
) -> tuple[Any, Any]:
    """Resuelve la regresión logística en `dispositivo` con `torch.optim.LBFGS`.

    Devuelve `(w, b)` en `dispositivo`, en float64. Vive aparte para que las pruebas
    comparen esta función real contra scikit-learn.
    """
    import torch

    from taller_hpo.problemas.paridad_torch import subir

    n_train, n_features = x_tr.shape
    xt = subir(x_tr, np.float64, dispositivo)
    yt = subir(y_tr, np.float64, dispositivo)
    w = torch.zeros(n_features, dtype=torch.float64, device=dispositivo, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=dispositivo, requires_grad=True)
    perdida_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizador = torch.optim.LBFGS(
        [w, b],
        max_iter=_LOGISTICA_MAX_ITER,
        line_search_fn="strong_wolfe",
        # Tolerancias más estrictas que los defaults de LBFGS, que paran antes del óptimo.
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )

    def cerradura() -> Any:
        optimizador.zero_grad()
        logits = xt @ w + b
        # L2 como scikit-learn: 0.5 / (C * n_train) * ||W||^2, nunca sobre el intercepto.
        perdida = perdida_fn(logits, yt) + 0.5 / (capital_c * n_train) * (w @ w)
        perdida.backward()
        return perdida

    optimizador.step(cerradura)
    return w.detach(), b.detach()


def _filas_por_bloque(n_referencia: int) -> int:
    """Filas de consulta por bloque para `_logit_knn` y `_logit_krr`.

    `2**24 // n_referencia` mantiene la matriz de distancias de cada bloque por debajo de
    unos 128 MiB en float64, sin importar cuántas filas de referencia haya.
    """
    return max(1, 2**24 // max(n_referencia, 1))


def _ajustar_baseline_gpu(
    c: dict[str, Any],
    semillas: Semillas,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    dispositivo: str,
) -> tuple[Predictor, float, float, Any]:
    """Ajusta `c["algoritmo"]` en `dispositivo` ("cuda" o "cuda:N").

    Devuelve `(predecir, exactitud_val, n_parametros, modulo)`. `modulo` es un
    `nn.Module` para `logistica`, `knn` y `krr`, que calculan el oráculo en la tarjeta de
    video, y `None` para `bosque` y `hgb`, que predicen con NumPy.
    """
    algoritmo = c["algoritmo"]
    if algoritmo == "logistica":
        w_final, b_final = _ajustar_logistica_gpu(x_tr, y_tr, c["logistica_C"], dispositivo)

        def _logit_logistica(x: Any) -> Any:
            return (x @ w_final + b_final).unsqueeze(1)

        modulo = _envolver_como_modulo(_logit_logistica)
        predecir = _predictor_desde_modulo(modulo, dispositivo)
        n_parametros = float(w_final.numel() + 1)
    elif algoritmo == "knn":
        import torch

        from taller_hpo.problemas.paridad_torch import subir

        k = int(c["knn_n_vecinos"])
        x_tr_dev = subir(x_tr, np.float64, dispositivo)
        y_tr_dev = subir(y_tr, np.int64, dispositivo)

        def _logit_knn(x: Any) -> Any:
            filas = _filas_por_bloque(x_tr_dev.shape[0])
            logits = []
            for inicio in range(0, x.shape[0], filas):
                xb = x[inicio : inicio + filas]
                distancias = torch.cdist(xb, x_tr_dev)
                _, indices = torch.topk(distancias, k, dim=1, largest=False)
                # Conteo de votos y no torch.mode: un empate (2*votos == k) da clase 0,
                # igual que el desempate de KNeighborsClassifier.
                votos = y_tr_dev[indices].sum(dim=1)
                logits.append((2.0 * votos - k).to(x.dtype).unsqueeze(1))
            return torch.cat(logits, dim=0)

        modulo = _envolver_como_modulo(_logit_knn)
        predecir = _predictor_desde_modulo(modulo, dispositivo)
        n_parametros = float(x_tr.shape[0] * x_tr.shape[1])
    elif algoritmo == "krr":
        import torch

        from taller_hpo.problemas.paridad_torch import subir

        gamma = _gamma_krr(x_tr, c["krr_gamma"])
        x_tr_dev = subir(x_tr, np.float64, dispositivo)
        y_pm1_dev = subir(2.0 * y_tr - 1.0, np.float64, dispositivo)
        distancias2 = torch.cdist(x_tr_dev, x_tr_dev) ** 2
        k_train = torch.exp(-gamma * distancias2)
        k_reg = k_train + c["krr_alpha"] * torch.eye(
            len(y_tr), dtype=torch.float64, device=dispositivo
        )
        factor_l = torch.linalg.cholesky(k_reg)
        dual_coef = torch.cholesky_solve(y_pm1_dev.unsqueeze(1), factor_l).squeeze(1)

        def _logit_krr(x: Any) -> Any:
            filas = _filas_por_bloque(x_tr_dev.shape[0])
            logits = []
            for inicio in range(0, x.shape[0], filas):
                xb = x[inicio : inicio + filas]
                distancias2_x = torch.cdist(xb, x_tr_dev) ** 2
                k_x = torch.exp(-gamma * distancias2_x)
                logits.append((k_x @ dual_coef).unsqueeze(1))
            return torch.cat(logits, dim=0)

        modulo = _envolver_como_modulo(_logit_krr)
        predecir = _predictor_desde_modulo(modulo, dispositivo)
        n_parametros = float(dual_coef.numel())
    else:  # "bosque" o "hgb", los únicos casos que faltan.
        modulo = None
        import xgboost as xgb

        if algoritmo == "bosque":
            n_features = x_tr.shape[1]
            colsample_bynode = math.sqrt(n_features) / n_features
            modelo = xgb.XGBClassifier(
                # Bosque aleatorio en XGBoost: una sola ronda de boosting con
                # num_parallel_tree árboles sin shrinkage (XGBRFClassifier está deprecado).
                n_estimators=1,
                num_parallel_tree=c["bosque_n_arboles"],
                learning_rate=1.0,
                max_depth=c["bosque_profundidad_max"],
                # Con base_score=0.5 cada fila pesa 0.25 en min_child_weight, así que
                # 0.25 * bosque_min_muestras_hoja exige ese número exacto de muestras.
                min_child_weight=0.25 * float(c["bosque_min_muestras_hoja"]),
                base_score=0.5,
                subsample=_BOSQUE_SUBSAMPLE,
                colsample_bynode=colsample_bynode,
                # reg_lambda=1e-5, el default de XGBRFClassifier: un bosque aleatorio no
                # regulariza sus árboles.
                reg_lambda=1e-5,
                tree_method="hist",
                device=dispositivo,
                random_state=semillas.init,
                n_jobs=1,
            )
        else:  # "hgb"
            modelo = xgb.XGBClassifier(
                learning_rate=c["hgb_learning_rate"],
                n_estimators=c["hgb_max_iter"],
                max_leaves=c["hgb_max_hojas"],
                grow_policy="lossguide",
                max_depth=0,
                tree_method="hist",
                device=dispositivo,
                random_state=semillas.init,
                n_jobs=1,
            )
        modelo.fit(x_tr, y_tr)
        predecir = _predecir_xgb_gpu(modelo)
        n_parametros = _contar_nodos_xgb(modelo)
    exactitud_val = float(np.mean(predecir(x_va) == y_va))
    return predecir, exactitud_val, n_parametros, modulo


def _empacar_metricas(
    exactitud_val: float, riesgo: float, n_parametros: float, epocas: float, segundos: float
) -> dict[str, float]:
    return {
        "exactitud_val": exactitud_val,
        "exactitud_verdadera": 1.0 - riesgo,
        "n_parametros": float(n_parametros),
        "epocas": float(epocas),
        "segundos": segundos,
    }


def entrenar(
    config: dict[str, Any], semillas: Semillas, reportar: Reportar | None = None
) -> dict[str, float]:
    c = get_config(DEFAULTS, **config)
    _validar_combinaciones(c)
    _validar_dispositivo(c["dispositivo"])
    dispositivo = _resolver_dispositivo(c["dispositivo"])
    # `dispositivo` siempre es "cuda" o "cuda:N", así que esta validación corre siempre.
    _validar_disponibilidad_gpu(c["dispositivo"], dispositivo, c["algoritmo"])

    problema = {k: c[k] for k in ("d", "a", "k_ruido", "rotar")}
    x, y = generar(c["n_train"] + c["n_val"], semilla=semillas.datos, **problema)
    x_tr, y_tr = x[: c["n_train"]], y[: c["n_train"]]
    x_va, y_va = x[c["n_train"] :], y[c["n_train"] :]

    # mlp reusa paridad_torch al completo (incluido su oráculo en el
    # dispositivo); los demás algoritmos usan el Predictor uniforme de abajo.
    inicio = time.perf_counter()
    if c["algoritmo"] == "mlp":
        from taller_hpo.problemas import paridad_torch

        anchos = tuple(int(c[f"ancho_{i}"]) for i in range(1, c["n_capas"] + 1))
        resultado = paridad_torch.entrenar_red(
            anchos,
            activacion=c["mlp_activacion"],
            learning_rate_init=c["mlp_learning_rate_init"],
            alpha=c["mlp_alpha"],
            batch_size=c["mlp_batch_size"],
            epocas=c["mlp_epocas"],
            datos=(x_tr, y_tr, x_va, y_va),
            semillas=semillas,
            dispositivo=dispositivo,
            reportar=reportar,
        )
        segundos = resultado.segundos
        epocas = resultado.epocas
        exactitud_val = resultado.exactitud_val
        n_parametros = float(resultado.n_parametros)
        riesgo = paridad_torch.riesgo_verdadero_en_dispositivo(
            resultado.modelo,
            n=c["n_oraculo"],
            semilla=SEMILLA_ORACULO,
            dispositivo=dispositivo,
            **problema,
        )
        metricas = _empacar_metricas(exactitud_val, riesgo, n_parametros, epocas, segundos)
        if c["n_prueba"] > 0:
            x_pr, y_pr = generar(c["n_prueba"], semilla=SEMILLA_PRUEBA, **problema)
            metricas["exactitud_prueba"] = paridad_torch.exactitud_en_dispositivo(
                resultado.modelo, x_pr, y_pr, dispositivo=dispositivo
            )
        return metricas

    predecir, exactitud_val, n_parametros, modulo = _ajustar_baseline_gpu(
        c, semillas, x_tr, y_tr, x_va, y_va, dispositivo
    )
    segundos = time.perf_counter() - inicio
    epocas = 1  # un solo fit: no hay entrenamiento iterativo por época.
    if modulo is not None:
        # logistica, knn y krr: oráculo y prueba finita en la tarjeta de video.
        from taller_hpo.problemas import paridad_torch

        riesgo = paridad_torch.riesgo_verdadero_en_dispositivo(
            modulo, n=c["n_oraculo"], semilla=SEMILLA_ORACULO, dispositivo=dispositivo, **problema
        )
        metricas = _empacar_metricas(exactitud_val, riesgo, n_parametros, epocas, segundos)
        if c["n_prueba"] > 0:
            x_pr, y_pr = generar(c["n_prueba"], semilla=SEMILLA_PRUEBA, **problema)
            metricas["exactitud_prueba"] = paridad_torch.exactitud_en_dispositivo(
                modulo, x_pr, y_pr, dispositivo=dispositivo
            )
        return metricas

    # bosque y hgb: XGBoost no acepta tensores de torch sin cupy, así que el oráculo va
    # con NumPy.
    riesgo = riesgo_verdadero(predecir, n=c["n_oraculo"], semilla=SEMILLA_ORACULO, **problema)
    metricas = _empacar_metricas(exactitud_val, riesgo, n_parametros, epocas, segundos)
    if c["n_prueba"] > 0:
        x_pr, y_pr = generar(c["n_prueba"], semilla=SEMILLA_PRUEBA, **problema)
        metricas["exactitud_prueba"] = float(np.mean(predecir(x_pr) == y_pr))
    return metricas


__all__ = [
    "ALGORITMOS",
    "DEFAULTS",
    "Divergencia",
    "entrenar",
]
