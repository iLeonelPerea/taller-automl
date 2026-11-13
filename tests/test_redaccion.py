"""Guardas de redacción que ninguna otra prueba vigila.

Cada guarda fija un arreglo de redacción que el tesista lee: sin ella, un cambio
posterior vuelve a dejar una palabra sin tilde, un término en inglés o un mensaje con
"se" impersonal, y la suite sigue en verde. Las guardas no entrenan nada: leen los
archivos y comparan texto (la última importa una plantilla y reemplaza
`subprocess.run`), así que pasan igual con tarjeta de video y sin ella.

Las listas de palabras viven aquí y no en el código que vigilan: cada una nombra lo
que prohíbe, y por eso las guardas que recorren carpetas no leen este archivo (mismo
criterio que `tests/test_paso_03.py::_archivos_de_texto_del_taller`).
"""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
NUCLEO = ("estudio.py", "objetivo.py", "subproceso.py")

# Palabras que en español llevan tilde siempre y que ninguno de los tres módulos usa
# como nombre de variable, de parámetro o de función. Quedan fuera las que cambian de
# sentido con la tilde ("esta"/"está", "si"/"sí", "como"/"cómo", "tu"/"tú"), porque
# esas dependen del contexto y una lista de palabras no las distingue.
_SIN_TILDE = frozenset(
    {
        "abrio",
        "ademas",
        "ahi",
        "alcanzo",
        "aqui",
        "asi",
        "avisaria",
        "busqueda",
        "busquedas",
        "caracter",
        "codigo",
        "computo",
        "contaminacion",
        "conversion",
        "corrio",
        "creacion",
        "cuales",
        "cuantos",
        "demas",
        "despues",
        "direccion",
        "documentacion",
        "epoca",
        "estan",
        "explicita",
        "explicitamente",
        "explicito",
        "explicitos",
        "fabrica",
        "funcion",
        "habia",
        "habian",
        "hiperparametro",
        "huerfano",
        "huerfanos",
        "invalida",
        "jamas",
        "linea",
        "mas",
        "mayoria",
        "metrica",
        "modulo",
        "monotona",
        "numero",
        "numeros",
        "optimizaria",
        "oraculo",
        "parametro",
        "parametros",
        "pasarian",
        "pidio",
        "podria",
        "politica",
        "prohibe",
        "proposito",
        "recibio",
        "recien",
        "reintentaria",
        "salio",
        "sembrarian",
        "sesion",
        "tambien",
        "tipico",
        "todavia",
        "ultimo",
        "unica",
        "unico",
        "vacio",
        "validacion",
        "valido",
        "verificacion",
    }
)

_TIPOS_DE_TEXTO = (tokenize.STRING, tokenize.COMMENT, tokenize.FSTRING_MIDDLE)


def _borrar_codigo_entre_comillas_invertidas(texto: str) -> str:
    """Cambia por espacios lo que está entre comillas invertidas (los nombres de
    parámetros y de funciones que el texto cita) y deja intactos los saltos de línea,
    para que el número de línea de cada palabra no se corra."""
    return re.sub(r"`[^`]*`", lambda m: re.sub(r"[^\n]", " ", m.group()), texto)


def _palabras_sin_tilde(fuente: str) -> list[tuple[int, str]]:
    """Devuelve (línea, palabra) por cada palabra de `_SIN_TILDE` que aparece en un
    docstring, un comentario o un mensaje de `fuente`. Los nombres del código (variables,
    parámetros, funciones) no cuentan, porque `tokenize` los entrega como otro tipo de
    token."""
    halladas = []
    for token in tokenize.generate_tokens(io.StringIO(fuente).readline):
        if token.type not in _TIPOS_DE_TEXTO:
            continue
        texto = _borrar_codigo_entre_comillas_invertidas(token.string)
        for coincidencia in re.finditer(r"\w+", texto):
            palabra = coincidencia.group()
            if palabra.lower() in _SIN_TILDE:
                linea = token.start[0] + texto.count("\n", 0, coincidencia.start())
                halladas.append((linea, palabra))
    return halladas


@pytest.mark.parametrize("nombre", NUCLEO)
def test_el_nucleo_lleva_tildes_en_docstrings_comentarios_y_mensajes(nombre):
    """`estudio.py`, `objetivo.py` y `subproceso.py` son los módulos del núcleo que
    el tesista lee: sus docstrings, sus comentarios y sus mensajes de error llevan
    las tildes del español."""
    ruta = RAIZ / "src" / "taller_hpo" / nombre
    assert _palabras_sin_tilde(ruta.read_text(encoding="utf-8")) == []


def test_la_guarda_de_tildes_marca_las_palabras_sin_tilde_y_deja_pasar_el_resto():
    """Prueba la guarda en las dos direcciones: marca una palabra sin tilde en un
    docstring, en un comentario y en un mensaje con f-string, y deja pasar la misma
    palabra con tilde, un nombre del código y una palabra citada entre comillas
    invertidas."""
    mala = (
        '"""Busqueda de la mejor config."""\n'
        "numero = 1  # aqui va la funcion\n"
        'mensaje = f"no es un numero ({numero})"\n'
        "# la `funcion` va entre comillas invertidas\n"
    )
    assert _palabras_sin_tilde(mala) == [
        (1, "Busqueda"),
        (2, "aqui"),
        (2, "funcion"),
        (3, "numero"),
    ]
    buena = (
        '"""Búsqueda de la mejor config."""\n'
        "n_trials = 1  # aquí va la función\n"
        'mensaje = f"no es un número ({n_trials})"\n'
    )
    assert _palabras_sin_tilde(buena) == []


def _archivos_de_texto(carpetas: tuple[str, ...], *, con_readme: bool) -> list[Path]:
    """Los `.py`, `.md` y `.sh` de `carpetas`, más el `README.md` de la raíz si
    `con_readme`. Deja fuera este mismo archivo, que tiene que nombrar lo que busca."""
    archivos = [RAIZ / "README.md"] if con_readme else []
    for carpeta in carpetas:
        for patron in ("*.py", "*.md", "*.sh"):
            archivos.extend(sorted((RAIZ / carpeta).rglob(patron)))
    este_archivo = Path(__file__).resolve()
    archivos = [archivo for archivo in archivos if archivo.resolve() != este_archivo]
    assert len(archivos) > 10, archivos
    return archivos


def test_ningun_texto_del_taller_dice_el_termino_en_ingles_de_la_maldicion_del_ganador():
    """Los pasos dicen "maldición del ganador"; el módulo del contrato, el docstring de
    `en_proceso` y lo que imprimen las dos plantillas de búsqueda dicen lo mismo."""
    for archivo in _archivos_de_texto(
        ("src", "plantilla", "pasos", "docs", "scripts"), con_readme=True
    ):
        texto = archivo.read_text(encoding="utf-8").lower()
        assert "winner" not in texto, str(archivo.relative_to(RAIZ))


_JERGA_EN_INGLES = re.compile(
    r"\b(allocador(es)?|allocators?|snippets?|helpers?|flaky|flakiness)\b", re.IGNORECASE
)


def test_ningun_texto_del_taller_usa_jerga_en_ingles_con_palabra_en_espanol():
    """Busca jerga en inglés que ya tiene palabra en español: "allocator", "snippet",
    "helper", "flaky" y "flakiness"."""
    for archivo in _archivos_de_texto(
        ("pasos", "src", "plantilla", "scripts", "docs", "tests"), con_readme=True
    ):
        for numero, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
            assert not _JERGA_EN_INGLES.search(linea), (
                str(archivo.relative_to(RAIZ)),
                numero,
                linea,
            )


def test_los_titulos_de_arquitectura_no_llevan_el_estado_de_la_tarea():
    """Un título como "(E2, hecho)" cuenta el avance de quien escribió el documento,
    no lo que el módulo hace, y una etiqueta de etapa como "(E9)" no le dice nada
    a quien lee el documento."""
    texto = (RAIZ / "docs" / "arquitectura.md").read_text(encoding="utf-8")
    titulos_con_estado = re.findall(r"^#+ .*(?:\bhecho\b|\(E\d+[^)]*\)).*$", texto, re.MULTILINE)
    assert titulos_con_estado == []


# Una etiqueta de etapa como "E9" no le dice nada al tesista ni a quien mantiene el
# taller. `\b` delante de la E deja fuera los nombres que la contienen ("SEMILLA_E1",
# "1E5"), y `\b` detrás deja fuera "E2E".
_ETIQUETA_DE_ETAPA = re.compile(r"\bE\d+\b")


def test_ningun_texto_del_taller_usa_etiquetas_de_etapa():
    """Los `.py`, `.md` y `.sh` que lee el tesista (y las pruebas) nombran cada módulo
    por lo que hace, sin etiquetas de etapa como "E9"."""
    for archivo in _archivos_de_texto(
        ("src", "plantilla", "pasos", "docs", "scripts", "tests"), con_readme=True
    ):
        for numero, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
            assert not _ETIQUETA_DE_ETAPA.search(linea), (
                str(archivo.relative_to(RAIZ)),
                numero,
                linea,
            )


def test_la_guarda_de_etiquetas_de_etapa_marca_la_etiqueta_y_deja_pasar_el_resto():
    """Prueba el patrón en las dos direcciones: marca la etiqueta en un título, en una
    remisión y en un encabezado de prueba, y deja pasar los nombres que contienen una
    E seguida de dígitos."""
    for linea in (
        "### `taller_hpo.estudio` (E5), estudios de Optuna con guardas",
        "### `plantilla/buscar_*.py` (E10 parte 2)",
        "entrenar con `bosque` (ver E9).",
        '"""E1: el problema de juguete y su oráculo dicen la verdad.',
    ):
        assert _ETIQUETA_DE_ETAPA.search(linea), linea
    for linea in (
        "### `taller_hpo.estudio`, estudios de Optuna con guardas",
        "una prueba E2E del tesista",
        "SEMILLA_E1 y 1E5 muestras",
        "el paso 09 y la etapa final",
    ):
        assert not _ETIQUETA_DE_ETAPA.search(linea), linea


# (archivo, expresión regular de la frase vieja). Cada frase era bitácora, jerga en
# inglés, una sigla sin expandir o una descripción que no decía lo que mide ("reservar"
# para `torch.cuda.max_memory_allocated()`). La guarda mira archivo por archivo porque
# otros archivos del taller sí usan alguna de estas palabras con razón.
_FRASES_VIEJAS = (
    ("pasos/anexo_paralelo.py", r"test_pasos\.py-style"),
    ("pasos/anexo_paralelo.py", r"\bmatchea\b"),
    ("pasos/anexo_paralelo.py", r"\boverhead\b"),
    ("pasos/anexo_paralelo.py", r"common random numbers"),
    ("pasos/07_conecta_tu_modelo.py", r"Patrón copiado de"),
    ("tests/test_subproceso.py", r"ya no se descarta"),
    ("tests/test_pasos.py", r"se llegó a reservar"),
    ("tests/test_paso_05.py", r"\bgl\b"),
)


@pytest.mark.parametrize(("archivo", "patron"), _FRASES_VIEJAS)
def test_el_archivo_corregido_no_recupera_su_frase_vieja(archivo, patron):
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    assert re.search(patron, texto) is None, (archivo, patron)


# (archivo, sigla, palabras de su expansión): solo los archivos que expanden la sigla.
_SIGLAS_CON_EXPANSION = (
    ("pasos/01_grid_vs_random.py", "QMC", "cuasi Monte Carlo"),
    ("pasos/01_grid_vs_random.py", "TOST", "dos pruebas unilaterales"),
    ("pasos/anexo_paralelo.py", "CRN", "números aleatorios comunes"),
)


def _primera_aparicion_con_su_entorno(texto: str, sigla: str, ancho: int = 80) -> str:
    """Devuelve el fragmento de `texto` que rodea la primera aparición de `sigla` como
    palabra suelta (`QMCSampler` y `SEMILLA_QMC` no cuentan). Antes de buscar, junta las
    líneas de comentario de un mismo párrafo, para que la expansión no quede partida."""
    plano = " ".join(re.sub(r"\s*\n#\s*", " ", texto).split())
    coincidencia = re.search(rf"\b{sigla}\b", plano)
    assert coincidencia is not None, sigla
    return plano[max(0, coincidencia.start() - ancho) : coincidencia.end() + ancho]


@pytest.mark.parametrize(("archivo", "sigla", "expansion"), _SIGLAS_CON_EXPANSION)
def test_la_primera_aparicion_de_la_sigla_trae_su_expansion(archivo, sigla, expansion):
    """La primera vez que un paso nombra una sigla técnica, el texto la explica con
    palabras: `QMC` y `TOST` en el paso 01, `CRN` en el anexo."""
    texto = (RAIZ / archivo).read_text(encoding="utf-8")
    assert expansion in _primera_aparicion_con_su_entorno(texto, sigla), (archivo, sigla)


def test_la_busqueda_de_la_primera_aparicion_salta_los_nombres_que_contienen_la_sigla():
    texto = "# SEMILLA_QMC y QMCSampler no cuentan;\n# luego QMC (cuasi\n# Monte Carlo) sí."
    entorno = _primera_aparicion_con_su_entorno(texto, "QMC", ancho=20)
    assert " QMC (cuasi Monte Carlo)" in entorno
    assert "SEMILLA_QMC" not in entorno


def test_los_scripts_imprimen_mensajes_con_sujeto_explicito():
    """Los mensajes que imprimen los dos scripts dicen quién actúa ("el script no
    mide...") en vez de usar "se" impersonal."""
    docker = (RAIZ / "scripts" / "paralelo_docker.sh").read_text(encoding="utf-8")
    tiempos = (RAIZ / "scripts" / "medir_tiempos.py").read_text(encoding="utf-8")

    assert "(el script no escribe la contraseña en ningún archivo)" in docker
    # Dos avisos: nvidia-smi devuelve un valor no numérico, y nvidia-smi falla.
    assert docker.count("el script no mide la memoria") == 2
    assert "calentamiento (el script lo descarta)" in tiempos
    assert "el script mide la celda igual y anota" in tiempos


# Un fragmento de `echo` que termina en espacio, seguido de la continuación de línea y de
# otro fragmento entre comillas: `echo` junta los fragmentos con un espacio, así que el
# espacio final duplica el espacio entre las dos palabras.
_FRAGMENTO_CON_ESPACIO_FINAL = re.compile(r' " \\\n\s*"')


def test_paralelo_docker_no_duplica_espacios_entre_los_fragmentos_de_un_echo():
    texto = (RAIZ / "scripts" / "paralelo_docker.sh").read_text(encoding="utf-8")
    assert _FRAGMENTO_CON_ESPACIO_FINAL.findall(texto) == []

    con_defecto = 'echo "pico " \\\n    "(durante la espera)"\n'
    assert len(_FRAGMENTO_CON_ESPACIO_FINAL.findall(con_defecto)) == 1
    sin_defecto = 'echo "pico" \\\n    "(durante la espera)"\n'
    assert _FRAGMENTO_CON_ESPACIO_FINAL.findall(sin_defecto) == []


def _importar(ruta: Path, nombre: str):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_arquitectura_enumera_los_casos_en_que_detectar_dispositivo_lanza_valueerror(
    monkeypatch,
):
    """`docs/arquitectura.md` dice que `_detectar_dispositivo` lanza `ValueError` cuando
    el taller no puede lanzar el intérprete y cuando ese intérprete no puede importar
    `torch`, y `RuntimeError` cuando falta la tarjeta de video. Esta prueba comprueba
    las tres ramas contra el código real, con `subprocess.run` reemplazado, y que el
    documento nombre los dos casos."""
    modulo = _importar(
        RAIZ / "plantilla" / "buscar_por_subproceso.py", "_buscar_por_subproceso_redaccion"
    )

    def no_arranca(*args, **kwargs):
        raise FileNotFoundError("no existe el intérprete")

    monkeypatch.setattr(modulo.subprocess, "run", no_arranca)
    with pytest.raises(ValueError, match="el taller no pudo lanzar el intérprete"):
        modulo._detectar_dispositivo("python-que-no-existe")

    def sin_torch(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="sin torch")

    monkeypatch.setattr(modulo.subprocess, "run", sin_torch)
    with pytest.raises(ValueError, match="no pudo importar torch"):
        modulo._detectar_dispositivo("python-sin-torch")

    def sin_tarjeta(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="\n", stderr="")

    monkeypatch.setattr(modulo.subprocess, "run", sin_tarjeta)
    with pytest.raises(RuntimeError) as error:
        modulo._detectar_dispositivo("python-sin-tarjeta")
    assert str(error.value).startswith(MENSAJE_SIN_TARJETA)

    documento = " ".join((RAIZ / "docs" / "arquitectura.md").read_text(encoding="utf-8").split())
    assert "::_detectar_dispositivo` lo lanza además" in documento
    assert "(`OSError`)" in documento
    assert "código de salida distinto de 0" in documento
    assert "cuando ese intérprete no puede importar `torch`" in documento
