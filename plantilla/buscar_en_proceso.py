"""La ruta completa en proceso: `train_ejemplo.py` + `taller_hpo.estudio` + `taller_hpo.objetivo`.

`taller_hpo.objetivo.en_proceso` llama a tu `entrenar` en este mismo proceso: no paga el
arranque de un intérprete por trial, y es la única forma de que `reportar` pode de
verdad. Exige que tu modelo corra con el mismo intérprete y las mismas dependencias que
este repo; si necesita otro entorno, usa `buscar_por_subproceso.py`.

El script hace lo que un tesista haría al copiar la plantilla:
1. Crea o reabre un estudio con `crear_estudio`, con una fábrica de `TPESampler`, así
   reanudar no repite configuraciones.
2. Recupera los trials huérfanos de una corrida cortada (`recuperar_huerfanos`).
3. Siembra la línea base, los DEFAULTS de `train_ejemplo.py`, con `sembrar_linea_base`.
4. Corre `optimizar` hasta `n_total` trials.
5. Imprime la mejor configuración, con la advertencia de que su valor está inflado por
   selección hasta que se re-evalúe (paso 06).

Necesita una tarjeta de video NVIDIA con CUDA en este entorno (`uv sync --locked`).

Correr:  uv run plantilla/buscar_en_proceso.py
Presupuesto:  PRESUPUESTO=humo | taller (default) | completo
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import re
import time
import types
from pathlib import Path

import optuna

from taller_hpo.estudio import (
    abrir_storage,
    crear_estudio,
    optimizar,
    recuperar_huerfanos,
    sembrar_linea_base,
)
from taller_hpo.objetivo import MetricaNoFinita, en_proceso
from taller_hpo.semillas import Semillas
from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
TRAIN_EJEMPLO = Path(__file__).resolve().parent / "train_ejemplo.py"

PRESUPUESTOS = {"humo": 2, "taller": 15, "completo": 40}

# Semilla base del sampler: cada reapertura arranca TPESampler(seed=BASE + n)
# donde n son los trials que ya tiene el estudio (ver crear_estudio).
SEMILLA_BASE = 0
# Una sola Semillas para toda la búsqueda (números aleatorios comunes), para que la
# diferencia entre dos configs no se confunda con la de semilla.
SEMILLAS_BUSQUEDA = Semillas(datos=SEMILLA_BASE, orden=SEMILLA_BASE + 1, init=SEMILLA_BASE + 2)


def _importar_train_ejemplo(ruta: Path = TRAIN_EJEMPLO) -> types.ModuleType:
    """Importa `train_ejemplo.py` por RUTA, igual que `check_importable.py`:
    así este script no depende de que `plantilla/` esté en sys.path."""
    spec = importlib.util.spec_from_file_location("_train_ejemplo_en_proceso", ruta)
    if spec is None or spec.loader is None:
        raise ImportError(f"la plantilla no pudo cargar {ruta!r} como módulo")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# El taller entrena siempre en la tarjeta de video: los valores válidos son
# "auto" y "cuda" (con ":N" opcional).
_PATRON_DISPOSITIVO = re.compile(r"auto|cuda(:\d+)?")


def _resolver_dispositivo_auto(valor: str) -> str:
    """Resuelve `'auto'` a `cuda` para la huella del estudio, sin depender de funciones
    privadas de tu `train.py`. Valida primero el formato (`'auto'` o `'cuda'`, con `':N'`
    opcional) y después que haya una tarjeta de video, para cualquier valor."""
    if not _PATRON_DISPOSITIVO.fullmatch(valor):
        raise ValueError(
            f"dispositivo debe ser 'auto' o 'cuda' (opcionalmente con ':N'), no {valor!r}"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(MENSAJE_SIN_TARJETA)
    return "cuda" if valor == "auto" else valor


def espacio(trial: optuna.Trial) -> dict[str, object]:
    """El ÚNICO lugar con `suggest_*`: tres perillas de `train_ejemplo.py`."""
    return {
        "ancho": trial.suggest_int("ancho", 8, 64),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
    }


# Mismas claves que `espacio`, para `sembrar_linea_base`: sembrar_linea_base
# valida `params` contra estas distribuciones ANTES de encolar.
DISTRIBUCIONES = {
    "ancho": optuna.distributions.IntDistribution(8, 64),
    "learning_rate_init": optuna.distributions.FloatDistribution(1e-4, 1e-1, log=True),
    "alpha": optuna.distributions.FloatDistribution(1e-6, 1e-1, log=True),
}


def _dispositivo_resuelto_de_modulo(modulo: types.ModuleType, train_path: Path) -> str:
    """El dispositivo ya resuelto de `DEFAULTS`, que el contrato exige siempre. Se
    comprueba antes de crear el estudio, así un `train.py` sin la clave o sin tarjeta de
    video se detiene sin dejar archivos a medias."""
    if "dispositivo" not in modulo.DEFAULTS:
        raise ValueError(
            f"tu train.py ({train_path}) debe declarar 'dispositivo' en "
            "DEFAULTS (ver plantilla/README.md): el taller entrena siempre en la tarjeta "
            "de video, así que ningún train.py puede quedar fuera de esa comprobación"
        )
    return _resolver_dispositivo_auto(modulo.DEFAULTS["dispositivo"])


def buscar(
    *,
    n_total: int,
    destino: str | os.PathLike,
    train_path: Path = TRAIN_EJEMPLO,
    nombre: str = "buscar_en_proceso",
    modulo: types.ModuleType | None = None,
) -> optuna.Study:
    """Corre o reanuda la búsqueda completa en proceso y devuelve el `Study`.

    `modulo` es el `train.py` ya importado, para no importarlo dos veces; si no lo pasas,
    la función lo importa desde `train_path`.
    """
    if modulo is None:
        modulo = _importar_train_ejemplo(train_path)
    # Todas las claves de DEFAULTS que no están en el espacio de búsqueda quedan fijas en
    # `fijos`, porque también definen el problema y entran a la huella del estudio.
    # "dispositivo" va ya resuelto, nunca 'auto'.
    fijos = {k: v for k, v in modulo.DEFAULTS.items() if k not in DISTRIBUCIONES}
    fijos["dispositivo"] = _dispositivo_resuelto_de_modulo(modulo, train_path)

    study = crear_estudio(
        nombre,
        destino=destino,
        direcciones=["maximize"],
        # n_startup_trials chico: con pocas decenas de trials, el default de 10 dejaría
        # casi todo el presupuesto al azar.
        sampler=lambda n: optuna.samplers.TPESampler(seed=SEMILLA_BASE + n, n_startup_trials=2),
        pruner=optuna.pruners.NopPruner(),
        # La misma forma de protocolo que pasos/02_primer_estudio.py.
        protocolo={
            "sampler": "TPESampler",
            "sampler_kwargs": {"seed_base": SEMILLA_BASE, "n_startup_trials": 2},
            "pruner": "NopPruner",
            "politica_semillas": {
                "tipo": "CRN",
                "semillas": dataclasses.asdict(SEMILLAS_BUSQUEDA),
            },
            "metrica": "exactitud_val",
            # El índice de la tarjeta de video no entra al protocolo: cambiar de 'cuda:0'
            # a 'cuda:1' no debe romper la reapertura, aunque `fijos` conserve el índice.
            "fijos": {**fijos, "dispositivo": fijos["dispositivo"].split(":", 1)[0]},
        },
    )
    recuperar_huerfanos(study)
    linea_base = {clave: modulo.DEFAULTS[clave] for clave in DISTRIBUCIONES}
    sembrar_linea_base(study, linea_base, DISTRIBUCIONES)

    objetivo = en_proceso(
        modulo.entrenar,
        espacio,
        metricas=["exactitud_val"],
        semillas=SEMILLAS_BUSQUEDA,
        fijos=fijos,
    )
    # Una métrica no finita marca el trial FAIL y la búsqueda sigue, igual que en
    # buscar_por_subproceso.py.
    optimizar(study, objetivo, n_total=n_total, atrapar=(MetricaNoFinita,))
    return study


def _pluralizar(n: int, singular: str, plural: str) -> str:
    """`singular` si `n == 1`, si no `plural`."""
    return singular if n == 1 else plural


_ESTADOS_TERMINADOS = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)


def _trials_terminados(destino: str | os.PathLike, nombre: str) -> int:
    """Cuántos trials COMPLETE o PRUNED tiene ya el estudio en `destino`, o 0 si no
    existe, para medir cuántos corrió de verdad esta llamada."""
    try:
        existente = optuna.load_study(study_name=nombre, storage=abrir_storage(destino))
    except KeyError:
        return 0
    return len(existente.get_trials(deepcopy=False, states=_ESTADOS_TERMINADOS))


def main() -> None:
    presupuesto = os.environ.get("PRESUPUESTO", "taller")
    if presupuesto not in PRESUPUESTOS:
        raise SystemExit(f"PRESUPUESTO debe ser uno de {sorted(PRESUPUESTOS)}, no {presupuesto!r}")
    n_total = PRESUPUESTOS[presupuesto]

    estudios = Path(os.environ.get("TALLER_ESTUDIOS", RAIZ / "estudios"))
    destino = estudios / "buscar_en_proceso.log"
    nombre = "buscar_en_proceso"

    # Importa train_ejemplo.py y resuelve su dispositivo antes de abrir el Journal y de
    # arrancar el reloj: el `import torch` es costo de arranque, no de ningún trial, y
    # sin tarjeta de video el script se detiene sin dejar archivos.
    modulo_train = _importar_train_ejemplo(TRAIN_EJEMPLO)
    _dispositivo_resuelto_de_modulo(modulo_train, TRAIN_EJEMPLO)

    antes = _trials_terminados(destino, nombre)
    inicio = time.perf_counter()
    study = buscar(n_total=n_total, destino=destino, nombre=nombre, modulo=modulo_train)
    segundos = time.perf_counter() - inicio
    nuevos = _trials_terminados(destino, nombre) - antes

    if nuevos == 0:
        print(
            f"en_proceso: el estudio ya tenía {antes} trials terminados en {destino} "
            f"(pediste {n_total}): nada que correr en esta llamada (bórralo o pide más "
            "trials para ver progreso)"
        )
    else:
        print(
            f"en_proceso: {nuevos} {_pluralizar(nuevos, 'trial nuevo', 'trials nuevos')} en "
            f"{segundos:.2f} s ({segundos / nuevos:.3f} s/trial)"
        )

    mejor = study.best_trial
    print(f"mejor exactitud_val: {mejor.value:.3f} con {mejor.params}")
    print(
        "advertencia: ese valor está inflado por selección (maldición del ganador) -- "
        "no lo reportes tal cual. Re-evalúalo con semillas nuevas antes de "
        "confiar en él (ver paso 06, honestidad)."
    )


if __name__ == "__main__":
    main()
