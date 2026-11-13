"""Sin tarjeta de video, cada paso que entrena se detiene antes de escribir nada.

Este archivo ancla, a nivel de proceso completo (no solo a nivel de import,
que ya cubre tests/test_juguete.py), la promesa de las cabeceras "Correr:" de
los pasos que entrenan y del anexo: sin una tarjeta de video NVIDIA con CUDA
visible, el proceso termina con un código de salida distinto de 0, el
mensaje claro de `taller_hpo.tarjeta_de_video.MENSAJE_SIN_TARJETA` llega a
stderr, y ni TALLER_ESTUDIOS ni TALLER_RESULTADOS quedan con ningún archivo
dentro -- nunca a medias. El paso 01 es la única excepción documentada (el
paisaje analítico es una fórmula cerrada, sin nada que entrenar) y por eso
termina con código 0.

Las dos últimas pruebas fijan el texto LITERAL del mensaje y el de la copia que
lleva `plantilla/train_ejemplo.py`. Las demás pruebas del taller comparan contra
la constante `MENSAJE_SIN_TARJETA`, así que una frase nueva pasaría sin que nadie
la viera si alguien la cambia a la vez en la constante y en esa copia.

Ninguna prueba de este archivo lleva la marca `gpu`: fuerzan
`CUDA_VISIBLE_DEVICES=""` en el entorno de cada subproceso propio, así que
corren igual con o sin una tarjeta de video real en la máquina que las
ejecuta, y la integración continua (que no tiene una) sí las corre.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
PASO_01 = RAIZ / "pasos" / "01_grid_vs_random.py"
PASOS_QUE_ENTRENAN = [p for p in sorted((RAIZ / "pasos").glob("[0-9][0-9]_*.py")) if p != PASO_01]
ANEXO = RAIZ / "pasos" / "anexo_paralelo.py"
TRAIN_EJEMPLO = RAIZ / "plantilla" / "train_ejemplo.py"

# El texto que ve el tesista cuando falta la tarjeta de video. Si cambias una palabra
# del mensaje, cambia también esta constante: la prueba existe para que ese cambio
# sea una decisión que alguien toma a propósito.
TEXTO_DEL_MENSAJE = (
    "el taller necesita una tarjeta de video NVIDIA con CUDA y el controlador "
    "instalado; aquí no hay ninguna disponible"
)


def test_pasos_que_entrenan_es_la_lista_esperada():
    """Esta prueba ancla la lista que arma el glob de arriba: si un paso
    cambiara de nombre o desapareciera, ella lo nota con un mensaje claro, en vez
    de dejar que `PASOS_QUE_ENTRENAN` pierda cobertura en silencio o, en el
    peor caso, quede vacía y convierta la prueba parametrizada de más abajo
    en "skipped" (pytest marca así un parametrize sin ningún valor)."""
    assert [p.stem for p in PASOS_QUE_ENTRENAN] == [
        "00_linea_base",
        "02_primer_estudio",
        "03_poda",
        "04_pareto",
        "05_buscar_el_modelo",
        "06_honestidad",
        "07_conecta_tu_modelo",
    ]


def _entorno_sin_tarjeta(tmp_path: Path) -> dict[str, str]:
    entorno = dict(os.environ)
    entorno["CUDA_VISIBLE_DEVICES"] = ""
    entorno["PRESUPUESTO"] = "humo"
    entorno["TALLER_RESULTADOS"] = str(tmp_path / "resultados")
    entorno["TALLER_ESTUDIOS"] = str(tmp_path / "estudios")
    return entorno


def _no_existe(directorio: Path) -> bool:
    """Esta función comprueba que `directorio` no llegó a existir: cada paso
    que entrena crea TALLER_RESULTADOS/TALLER_ESTUDIOS (con `.mkdir`) recién
    DESPUÉS de importar `taller_hpo.problemas.juguete` (el paso 05 importa
    además `cash_paridad`, que NO importa `juguete` -- es el propio import de
    `juguete` en ese paso el que detiene la falta de tarjeta de video), así
    que sin tarjeta de video ese import falla antes de que el directorio
    llegue a existir. En el anexo, la detención llega por otra vía:
    `_exigir_tarjeta_de_video` (llamada antes de crear cualquier directorio),
    no un import. Un directorio vacío, creado y nunca llenado, no cuenta
    como "sin tarjeta": esta función lo rechaza, porque ningún punto de
    entrada del taller crea directorios sin tarjeta de video."""
    return not directorio.exists()


def _assert_sin_estudios_ni_resultados(tmp_path: Path) -> None:
    resultados = tmp_path / "resultados"
    estudios = tmp_path / "estudios"
    assert _no_existe(resultados), list(resultados.rglob("*"))
    assert _no_existe(estudios), list(estudios.rglob("*"))


@pytest.mark.parametrize("paso", PASOS_QUE_ENTRENAN, ids=[p.stem for p in PASOS_QUE_ENTRENAN])
def test_paso_que_entrena_falla_sin_tarjeta_y_no_escribe_nada(paso, tmp_path):
    entorno = _entorno_sin_tarjeta(tmp_path)
    resultado = subprocess.run(
        [sys.executable, str(paso)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0, resultado.stdout
    assert MENSAJE_SIN_TARJETA in resultado.stderr, resultado.stderr
    _assert_sin_estudios_ni_resultados(tmp_path)


def test_paso_01_no_necesita_tarjeta_de_video(tmp_path):
    entorno = _entorno_sin_tarjeta(tmp_path)
    resultado = subprocess.run(
        [sys.executable, str(PASO_01)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert MENSAJE_SIN_TARJETA not in resultado.stderr, resultado.stderr


def test_anexo_modo_por_defecto_falla_sin_tarjeta_y_no_escribe_nada(tmp_path):
    entorno = _entorno_sin_tarjeta(tmp_path)
    resultado = subprocess.run(
        [sys.executable, str(ANEXO)],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0, resultado.stdout
    assert MENSAJE_SIN_TARJETA in resultado.stderr, resultado.stderr
    _assert_sin_estudios_ni_resultados(tmp_path)


def test_anexo_lanzar_dos_falla_sin_tarjeta_y_no_escribe_nada(tmp_path):
    entorno = _entorno_sin_tarjeta(tmp_path)
    resultado = subprocess.run(
        [sys.executable, str(ANEXO), "--lanzar", "2"],
        env=entorno,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode != 0, resultado.stdout
    assert MENSAJE_SIN_TARJETA in resultado.stderr, resultado.stderr
    _assert_sin_estudios_ni_resultados(tmp_path)


def _importar(ruta: Path, nombre: str):
    """Importa un .py fuera de sys.path. Mismo truco que `_importar` de
    tests/test_plantilla.py: `plantilla/train_ejemplo.py` no es un paquete."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_mensaje_sin_tarjeta_dice_exactamente_el_texto_fijado():
    """Esta prueba compara `MENSAJE_SIN_TARJETA` con el texto literal de este
    archivo. Las demás pruebas comparan contra la constante misma, así que
    ninguna notaría que alguien reescribió la frase que ve el tesista."""
    assert MENSAJE_SIN_TARJETA == TEXTO_DEL_MENSAJE


def test_copia_del_mensaje_en_train_ejemplo_dice_exactamente_el_texto_fijado(monkeypatch):
    """`plantilla/train_ejemplo.py` es autónomo (el tesista lo copia sin
    `taller_hpo`) y lleva su propia copia del mensaje. Esta prueba fuerza la
    ausencia de la tarjeta de video para leer el `RuntimeError` real de esa
    copia y lo compara con el texto literal de este archivo, no con la
    constante de `taller_hpo`: así un cambio hecho solo en la copia, o a la vez
    en la copia y en la constante, falla igual."""
    train_ejemplo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_texto_fijado_sin_tarjeta")
    monkeypatch.setattr(train_ejemplo.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError) as excinfo:
        train_ejemplo._resolver_dispositivo("auto")
    assert str(excinfo.value) == TEXTO_DEL_MENSAJE
