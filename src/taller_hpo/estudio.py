"""Estudios de Optuna con las guardas que un tesista necesita y Optuna no da gratis.

Funciones delgadas sobre Study y Trial reales, sin clases que los envuelvan, así lo que
aprendes aquí te sirve al leer la documentación de Optuna. Las guardas:

- `abrir_storage` usa JournalFileOpenLock, porque el JournalFileSymlinkLock por defecto
  falla en Windows.
- `crear_estudio` escribe siempre sampler y pruner, porque Optuna 5 cambió los defaults,
  y guarda el protocolo de búsqueda. Reabrir con otro protocolo u otras direcciones es
  un error claro, no un estudio mezclado en silencio.
- `crear_estudio` acepta en `sampler` una fábrica `(n_previos: int) -> BaseSampler`:
  reabrir con una instancia sembrada repetiría las mismas configuraciones, y con una
  fábrica cada reapertura sigue explorando.
- `recuperar_huerfanos` existe porque JournalStorage no tiene heartbeat: un trial
  cortado a la mitad se queda RUNNING para siempre.
- `sembrar_linea_base` valida `params` contra `distribuciones` antes de encolar, porque
  Optuna solo avisa con un warning si un valor está fuera de rango.
- `optimizar` cuenta trials COMPLETE + PRUNED, así reanudar no repite trabajo.
- `importancia_con_piso` compara cada importancia contra la de un placebo, un parámetro
  que el modelo no lee: su importancia es la del puro ruido y sirve de piso.
"""

from __future__ import annotations

import json
import numbers
import os
import platform
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState

_PREFIJOS_URL_RDB = ("postgresql://", "postgresql+psycopg://", "sqlite:///")
_ESTADOS_LINEA_BASE_ACTIVA = (TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING)


def abrir_storage(destino: str | os.PathLike) -> optuna.storages.BaseStorage | str:
    """URL de RDBStorage tal cual, o JournalStorage sobre un archivo.

    "postgresql://", "postgresql+psycopg://" y "sqlite:///" se devuelven tal cual:
    Optuna arma el RDBStorage internamente a partir de la URL. Cualquier otra cosa
    se trata como una ruta de archivo: se crea la carpeta padre si falta y se abre
    con JournalFileOpenLock (el JournalFileSymlinkLock por defecto falla en
    Windows).
    """
    texto = str(destino)
    if texto.startswith(_PREFIJOS_URL_RDB):
        return texto

    ruta = Path(destino)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta_str = str(ruta)
    backend = JournalFileBackend(ruta_str, lock_obj=JournalFileOpenLock(ruta_str))
    return optuna.storages.JournalStorage(backend)


_DIRECCIONES_VALIDAS = ("MINIMIZE", "MAXIMIZE")


def _direcciones_de(valores: Sequence[str]) -> list[optuna.study.StudyDirection]:
    if isinstance(valores, str):
        # Un str también es un Sequence[str]: sin este chequeo, list("maximize")
        # lo parte en las letras 'm', 'a', 'x', ... y Optuna falla con un mensaje
        # que no dice nada de direcciones.
        raise TypeError(
            f"direcciones debe ser una lista de str, p.ej. ['maximize'], no {valores!r}"
        )
    resultado = []
    for valor in valores:
        clave = str(valor).upper()
        # Comparar contra la lista explícita, no con hasattr: StudyDirection
        # también tiene el miembro NOT_SET, que no es una dirección válida.
        if clave not in _DIRECCIONES_VALIDAS:
            raise ValueError(f"dirección inválida: {valor!r} (debe ser minimize o maximize)")
        resultado.append(optuna.study.StudyDirection[clave])
    return resultado


def _verificar_direcciones(study: optuna.Study, *, nombre: str, direcciones: Sequence[str]) -> None:
    esperadas = _direcciones_de(direcciones)
    if study.directions != esperadas:
        actuales = [d.name.lower() for d in study.directions]
        pedidas = [d.name.lower() for d in esperadas]
        raise ValueError(
            f"direcciones distintas al reabrir el estudio {nombre!r}: "
            f"tiene {actuales}, pediste {pedidas}"
        )


def _verificar_protocolo(study: optuna.Study, *, nombre: str, protocolo: dict[str, Any]) -> None:
    guardado = study.user_attrs.get("protocolo")
    if guardado != protocolo:
        raise ValueError(
            f"protocolo distinto al reabrir el estudio {nombre!r}: "
            f"tiene {guardado!r}, pediste {protocolo!r}"
        )


def _trials_existentes(storage: optuna.storages.BaseStorage | str, nombre: str) -> int:
    """Cuántos trials tiene ya el estudio `nombre` en `storage`, o 0 si no existe.

    `optuna.load_study` levanta KeyError en los dos backends cuando el nombre no existe.
    """
    try:
        existente = optuna.load_study(study_name=nombre, storage=storage)
    except KeyError:
        return 0
    return len(existente.get_trials(deepcopy=False))


def crear_estudio(
    nombre: str,
    *,
    destino: str | os.PathLike,
    direcciones: Sequence[str],
    sampler: optuna.samplers.BaseSampler | Callable[[int], optuna.samplers.BaseSampler],
    pruner: optuna.pruners.BasePruner,
    protocolo: Mapping[str, Any],
) -> optuna.Study:
    """Crea el estudio, o reabre uno existente verificando que sea compatible.

    `sampler` y `pruner` son keyword-only y sin default a propósito, así que siempre van
    explícitos. `protocolo` describe cómo se busca (sampler y sus kwargs, pruner,
    política de semillas, métrica) y queda en `study.user_attrs["protocolo"]`, junto con
    las versiones de optuna y python. Reabrir con otro protocolo u otras direcciones
    levanta ValueError diciendo qué cambió.

    `sampler` acepta una instancia de `BaseSampler` o una fábrica
    `(n_previos: int) -> BaseSampler`, como `lambda n: TPESampler(seed=BASE + n)`.
    Reabrir un estudio con trials usando una instancia levanta `UserWarning`, porque un
    sampler sembrado repetiría la misma secuencia.

    Si el estudio existe sin `protocolo` y sin trials, la función completa el registro:
    así N workers del anexo paralelo pueden crear el mismo estudio a la vez. Con trials y
    sin protocolo, es un estudio creado por fuera de esta función y levanta ValueError.
    """
    try:
        # Sin sort_keys: _verificar_protocolo compara dicts con `!=`, y ordenar fallaría
        # con un protocolo que mezcla claves int y str.
        protocolo_pedido = json.loads(json.dumps(dict(protocolo), allow_nan=False))
    except TypeError as error:
        raise TypeError(f"protocolo no es serializable a JSON: {error}") from error
    except ValueError as error:
        raise ValueError(f"protocolo no es serializable a JSON: {error}") from error

    # Validar siempre con _direcciones_de, también al crear: un str partido en letras
    # daría un error de Optuna que no dice nada de direcciones.
    direcciones_validadas = _direcciones_de(direcciones)
    storage = abrir_storage(destino)

    n_previos = _trials_existentes(storage, nombre)
    if isinstance(sampler, optuna.samplers.BaseSampler):
        if n_previos > 0:
            warnings.warn(
                f"el estudio {nombre!r} ya tiene {n_previos} trial(s) y crear_estudio "
                "recibió una INSTANCIA de sampler: si trae semilla fija, va a repetir "
                "exactamente la misma secuencia de configuraciones que ya corrió. Pasa "
                "una fábrica (n_previos: int) -> BaseSampler (p.ej. "
                "lambda n: TPESampler(seed=BASE + n)) para que reanudar sea "
                "determinista y no repita.",
                UserWarning,
                stacklevel=2,
            )
        sampler_final = sampler
    elif callable(sampler):
        sampler_final = sampler(n_previos)
        if not isinstance(sampler_final, optuna.samplers.BaseSampler):
            raise TypeError(
                f"la fábrica de sampler debe devolver un BaseSampler, no {sampler_final!r}"
            )
    else:
        raise TypeError(
            "sampler debe ser un optuna.samplers.BaseSampler o una fábrica "
            f"(n_previos: int) -> BaseSampler, no {sampler!r}"
        )

    try:
        study = optuna.create_study(
            study_name=nombre,
            storage=storage,
            sampler=sampler_final,
            pruner=pruner,
            directions=direcciones_validadas,
        )
    except optuna.exceptions.DuplicatedStudyError:
        study = optuna.load_study(
            study_name=nombre, storage=storage, sampler=sampler_final, pruner=pruner
        )
        _verificar_direcciones(study, nombre=nombre, direcciones=direcciones)

        if study.user_attrs.get("protocolo") is None:
            if study.get_trials(deepcopy=False):
                # from None: este ValueError es una validación propia, no una
                # consecuencia de la DuplicatedStudyError que abrió esta rama.
                raise ValueError(
                    f"el estudio {nombre!r} existe sin protocolo registrado: "
                    "¿se creó fuera de crear_estudio?"
                ) from None
            # Nadie corrió trials todavía: completar el registro con el mismo protocolo
            # hace idempotente la carrera de N workers creando este estudio a la vez.
            study.set_user_attr("protocolo", protocolo_pedido)
            study.set_user_attr("version_optuna", optuna.__version__)
            study.set_user_attr("version_python", platform.python_version())
            return study

        _verificar_protocolo(study, nombre=nombre, protocolo=protocolo_pedido)
        return study

    study.set_user_attr("protocolo", protocolo_pedido)
    study.set_user_attr("version_optuna", optuna.__version__)
    study.set_user_attr("version_python", platform.python_version())
    return study


def recuperar_huerfanos(study: optuna.Study) -> list[int]:
    """Pasa a FAIL los trials RUNNING de una corrida cortada y reencola sus params.

    Solo tiene sentido en modo secuencial, y se llama una vez al arrancar, antes de
    `optimizar`. Devuelve los números de trial recuperados.
    """
    huerfanos = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    numeros = []
    for trial in huerfanos:
        study.tell(trial.number, state=TrialState.FAIL)
        # Los params encolados viven en system_attrs["fixed_params"]: trial.params solo
        # trae lo que alcanzó a sugerirse antes del corte.
        fijos = trial.system_attrs.get("fixed_params", {})
        params = {**fijos, **trial.params}
        # Si el huérfano era la línea base, el reencolado conserva rol=linea_base, para
        # que sembrar_linea_base no encole una segunda copia.
        extra = {"rol": trial.user_attrs["rol"]} if "rol" in trial.user_attrs else {}
        study.enqueue_trial(params, user_attrs={**extra, "reintento_de": trial.number})
        numeros.append(trial.number)
    return numeros


def sembrar_linea_base(
    study: optuna.Study,
    params: Mapping[str, Any],
    distribuciones: Mapping[str, BaseDistribution],
) -> bool:
    """Encola `params` como línea base (user_attrs["rol"] = "linea_base").

    Valida `params` contra `distribuciones` antes de encolar: claves exactas y valores
    dentro de rango o de las opciones, con ValueError. Si ya hay una línea base COMPLETE,
    RUNNING o WAITING, no hace nada y devuelve False; una que falló o fue podada vuelve a
    encolarse. Los valores numéricos quedan como `float` o `int` nativos de Python,
    porque un escalar de numpy rompe un storage persistente.
    """
    claves_params = set(params)
    claves_dist = set(distribuciones)
    faltan = claves_dist - claves_params
    sobran = claves_params - claves_dist
    if faltan or sobran:
        detalles = []
        if faltan:
            detalles.append(f"faltan {sorted(faltan)}")
        if sobran:
            detalles.append(f"sobran {sorted(sobran)}")
        raise ValueError(f"params no coincide con distribuciones: {'; '.join(detalles)}")

    params_normalizados = dict(params)
    for clave, distribucion in distribuciones.items():
        valor = params[clave]
        if isinstance(distribucion, (FloatDistribution, IntDistribution)) and (
            isinstance(valor, bool) or not isinstance(valor, numbers.Real)
        ):
            # to_internal_repr solo hace float(v), así que un str como '0.5' o un bool
            # pasarían esa conversión con un tipo que `entrenar` no espera.
            raise ValueError(
                f"{clave!r}={valor!r} no es un número (recibió {type(valor).__name__})"
            )
        try:
            interno = distribucion.to_internal_repr(valor)
        except ValueError as error:
            raise ValueError(f"{clave!r}: {error}") from error
        # _contains es API privada de Optuna, revisada contra la versión 5.0.0.
        if not distribucion._contains(interno):
            raise ValueError(f"{clave!r}={valor!r} fuera de rango de {distribucion!r}")
        # to_external_repr devuelve un float o un int nativo de Python: un np.float32
        # llegaría intacto a enqueue_trial y fallaría al guardarse en JSON.
        params_normalizados[clave] = distribucion.to_external_repr(interno)

    ya_hay_linea_base_activa = any(
        trial.user_attrs.get("rol") == "linea_base" and trial.state in _ESTADOS_LINEA_BASE_ACTIVA
        for trial in study.get_trials(deepcopy=False)
    )
    if ya_hay_linea_base_activa:
        return False

    study.enqueue_trial(params_normalizados, user_attrs={"rol": "linea_base"})
    return True


def optimizar(
    study: optuna.Study,
    objetivo: Callable[[optuna.Trial], float | tuple[float, ...]],
    *,
    n_total: int,
    atrapar: tuple[type[BaseException], ...] = (),
    max_fallos: int = 10,
) -> None:
    """Corre trials de a uno hasta que COMPLETE + PRUNED sume `n_total`.

    Cuenta lo que ya había en el estudio, así reanudar no repite trabajo. Las
    excepciones en `atrapar` marcan el trial FAIL y la búsqueda sigue; si los FAIL de
    esta llamada superan `max_fallos`, levanta RuntimeError. `n_jobs=1` siempre: el
    paralelismo vive en el anexo.
    """
    estados_terminados = (TrialState.COMPLETE, TrialState.PRUNED)

    def _terminados() -> int:
        return len(study.get_trials(deepcopy=False, states=estados_terminados))

    fallos_en_esta_llamada = 0
    while _terminados() < n_total:
        fallos_antes = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))
        study.optimize(objetivo, n_trials=1, catch=atrapar, n_jobs=1)
        fallos_despues = len(study.get_trials(deepcopy=False, states=(TrialState.FAIL,)))

        if fallos_despues > fallos_antes:
            fallos_en_esta_llamada += 1
            if fallos_en_esta_llamada > max_fallos:
                raise RuntimeError(
                    f"{fallos_en_esta_llamada} trials fallidos en esta llamada a optimizar "
                    f"(max_fallos={max_fallos})"
                )


def importancia_con_piso(
    study: optuna.Study,
    *,
    evaluador: optuna.importance.BaseImportanceEvaluator,
    placebo: str,
) -> dict[str, dict[str, float | bool]]:
    """Importancia de cada hiperparámetro, marcando cuáles superan al placebo.

    `placebo` es un parámetro que el modelo no lee: su importancia es la que produce el
    puro ruido, y sirve de piso para decidir si otro hiperparámetro de verdad importa.
    """
    importancias = optuna.importance.get_param_importances(study, evaluator=evaluador)
    if placebo not in importancias:
        raise KeyError(
            f"placebo {placebo!r} no aparece entre los parámetros del estudio: "
            f"disponibles {sorted(importancias)}"
        )
    piso = float(importancias[placebo])
    return {
        param: {"importancia": float(valor), "sobre_el_piso": bool(valor > piso)}
        for param, valor in importancias.items()
    }
