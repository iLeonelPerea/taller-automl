"""Los ejercicios: cada carpeta de `ejercicios/` está completa y su verificador discrimina.

`ejercicios/` tiene una carpeta por ejercicio (`NN_nombre/`) con `README.md`, `ejercicio.py`,
`solucion.py` y `comprobar.py`. Este archivo descubre las carpetas que existen y prueba cada
una, así que un ejercicio nuevo queda vigilado desde el momento en que existe su carpeta.
Una carpeta que todavía no existe no hace fallar nada, pero una carpeta a medias falla con
un mensaje que dice qué le falta.

Cuatro grupos de pruebas:

- La forma de cada carpeta (sin tarjeta de video): los cuatro archivos, las secciones del
  README en orden, los huecos de `ejercicio.py` contra los pasos del README, que
  `ejercicio.py` y `solucion.py` definan lo mismo y que ningún archivo haga trabajo al
  importarse.
- El verificador y lo que comparte (sin tarjeta de video): `verificar.py` y `_comun.py`
  contra ejercicios falsos escritos en un directorio temporal, para fijar cada mensaje y cada
  código de salida sin entrenar nada.
- La conducta de cada ejercicio (marca `gpu`): `solucion.py` pasa todas las revisiones, el
  `ejercicio.py` vacío no pasa ninguna y dice qué hueco falta, y las revisiones discriminan:
  una función que devuelve una constante, o cualquier cambio que declare el `comprobar.py`
  en `MUTANTES`, hace fallar alguna revisión.
- Los programas (marca `gpu`): `solucion.py` corre limpio y `ejercicio.py` vacío termina con
  un mensaje, sin traceback.

Para ver cuánto tarda cada ejercicio, corre `pytest tests/test_ejercicios.py --durations=0`.
Ninguna prueba escribe dentro del repo: los programas corren con su directorio de trabajo y
sus carpetas de estudios y resultados en un directorio temporal, y una fixture comprueba al
final que `estudios/`, `resultados/` y `ejercicios/` no ganaron archivos de resultados.
"""

from __future__ import annotations

import ast
import importlib
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from taller_hpo.tarjeta_de_video import MENSAJE_SIN_TARJETA

RAIZ = Path(__file__).resolve().parent.parent
EJERCICIOS = RAIZ / "ejercicios"

ARCHIVOS = ("README.md", "ejercicio.py", "solucion.py", "comprobar.py")
SECCIONES = (
    "Antes de empezar",
    "Lo que vas a hacer",
    "Pistas",
    "Comprueba tu trabajo",
    "Qué observar",
    "Siguiente ejercicio",
)

# Los trece nombres de la escalera. Una carpeta con otro nombre es un error de escritura o
# un ejercicio que no está en el plan, y `ejercicios/README.md` enlaza los ejercicios con
# estos nombres.
ESCALERA = (
    "01_primer_entrenamiento",
    "02_el_azar_importa",
    "03_la_validacion_se_equivoca",
    "04_primera_busqueda",
    "05_varias_perillas",
    "06_guarda_y_retoma",
    "07_empieza_desde_tu_configuracion",
    "08_compara_con_justicia",
    "09_cuantas_semillas",
    "10_la_ganadora_inflada",
    "11_corta_a_tiempo",
    "12_mejor_pero_mas_caro",
    "13_conecta_tu_modelo",
)
NIVELES = (
    "Nivel 1: conoce tu modelo",
    "Nivel 2: tu primer buscador",
    "Nivel 3: honestidad",
    "Nivel 4: búsquedas grandes",
)

CARPETAS = sorted(p for p in EJERCICIOS.glob("[0-9][0-9]_*") if p.is_dir())
IDS = [carpeta.name for carpeta in CARPETAS]

# Las constantes que sustituyen el cuerpo de una función con hueco: ninguna revisión puede
# dejar pasar una función que devuelve siempre lo mismo.
CONSTANTES = ("None", "0", "0.5", "{}")

PATRON_MENSAJE_DE_HUECO = re.compile(r"^Completa .+ \(ver el paso (\d+) del README\)$")

# Lo único que un archivo de ejercicio puede tener a nivel de módulo: definiciones,
# constantes y la guarda del programa. Todo lo demás correría al importarlo.
DEFINICIONES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
)

_NOMBRES_UNICOS = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def ejercicios_en_el_camino_de_python():
    """`verificar.py` y cada `comprobar.py` hacen `from _comun import ...`. Ese módulo vive
    en `ejercicios/`, que Python pone en su camino solo cuando corres `verificar.py` como
    programa. Esta fixture lo agrega mientras corren las pruebas de este archivo."""
    with pytest.MonkeyPatch.context() as parche:
        parche.syspath_prepend(str(EJERCICIOS))
        yield


def _instantanea_del_repo() -> dict[str, list[str]]:
    """Los archivos de resultados que los ejercicios podrían dejar en el repo: todo lo que
    haya en `estudios/` y `resultados/`, y en `ejercicios/` todo lo que no sea código o
    texto del propio ejercicio. Los `__pycache__` que Python escribe al importar no cuentan."""
    instantanea = {}
    for carpeta in ("estudios", "resultados", "ejercicios"):
        archivos = [
            p for p in (RAIZ / carpeta).rglob("*") if "__pycache__" not in p.parts and p.is_file()
        ]
        if carpeta == "ejercicios":
            archivos = [p for p in archivos if p.suffix not in (".py", ".md")]
        instantanea[carpeta] = sorted(str(p.relative_to(RAIZ)) for p in archivos)
    return instantanea


@pytest.fixture(scope="module", autouse=True)
def el_repo_queda_como_estaba():
    antes = _instantanea_del_repo()
    yield
    assert _instantanea_del_repo() == antes


def _verificar() -> ModuleType:
    return importlib.import_module("verificar")


def _comun() -> ModuleType:
    return importlib.import_module("_comun")


def _arbol(ruta: Path) -> ast.Module:
    return ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))


def _titulos(texto: str, marca: str) -> list[str]:
    """Los títulos de nivel `marca` ("#", "##", "###") de un texto en Markdown, sin mirar las
    líneas de un bloque de código: ahí un `#` es un comentario, no un título."""
    titulos = []
    en_bloque = False
    for linea in texto.splitlines():
        if linea.startswith("```"):
            en_bloque = not en_bloque
        elif not en_bloque and linea.startswith(marca + " "):
            titulos.append(linea[len(marca) + 1 :].strip())
    return titulos


def _seccion(carpeta: Path, titulo: str) -> str:
    """El cuerpo de la sección `## titulo` del README de `carpeta`."""
    cuerpo: list[str] | None = None
    en_bloque = False
    for linea in (carpeta / "README.md").read_text(encoding="utf-8").splitlines():
        if linea.startswith("```"):
            en_bloque = not en_bloque
        if not en_bloque and linea.startswith("## "):
            if cuerpo is not None:
                break
            cuerpo = [] if linea[3:].strip() == titulo else None
            continue
        if cuerpo is not None:
            cuerpo.append(linea)
    assert cuerpo is not None, f"{carpeta.name}/README.md no tiene la sección '## {titulo}'."
    return "\n".join(cuerpo)


def _lanza_hueco(nodo: ast.AST) -> bool:
    return (
        isinstance(nodo, ast.Raise)
        and isinstance(nodo.exc, ast.Call)
        and isinstance(nodo.exc.func, ast.Name)
        and nodo.exc.func.id == "NotImplementedError"
    )


def _huecos(ruta: Path) -> list[tuple[int, str]]:
    """(línea, mensaje) de cada `raise NotImplementedError(...)` del archivo, en orden."""
    huecos = []
    for nodo in ast.walk(_arbol(ruta)):
        if _lanza_hueco(nodo):
            argumento = nodo.exc.args[0] if nodo.exc.args else None
            mensaje = argumento.value if isinstance(argumento, ast.Constant) else ""
            huecos.append((nodo.lineno, mensaje))
    return sorted(huecos)


def _marcas_tu_turno(ruta: Path) -> list[int]:
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    return [n for n, linea in enumerate(lineas, 1) if re.match(r"\s*# TU TURNO: \S", linea)]


def _funciones_con_hueco(ruta: Path) -> list[str]:
    """Los nombres de las funciones de nivel superior que lanzan `NotImplementedError`."""
    return [
        nodo.name
        for nodo in _arbol(ruta).body
        if isinstance(nodo, ast.FunctionDef) and any(_lanza_hueco(n) for n in ast.walk(nodo))
    ]


def _es_la_guarda_de_programa(nodo: ast.stmt) -> bool:
    return (
        isinstance(nodo, ast.If)
        and isinstance(nodo.test, ast.Compare)
        and isinstance(nodo.test.left, ast.Name)
        and nodo.test.left.id == "__name__"
        and len(nodo.test.comparators) == 1
        and isinstance(nodo.test.comparators[0], ast.Constant)
        and nodo.test.comparators[0].value == "__main__"
    )


def _firmas(ruta: Path) -> tuple[dict[str, str], list[str]]:
    """(firma de cada función de nivel superior, nombres de las constantes asignadas)."""
    firmas = {}
    asignados = []
    for nodo in _arbol(ruta).body:
        if isinstance(nodo, ast.FunctionDef):
            retorno = ast.unparse(nodo.returns) if nodo.returns else ""
            firmas[nodo.name] = f"({ast.unparse(nodo.args)}) -> {retorno}"
        elif isinstance(nodo, ast.Assign):
            asignados += [t.id for t in nodo.targets if isinstance(t, ast.Name)]
        elif isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
            asignados.append(nodo.target.id)
    return firmas, sorted(asignados)


def _enlaces(texto: str) -> list[str]:
    """Los destinos relativos de los enlaces `[texto](destino)`, sin el `#ancla`."""
    return [
        destino.split("#")[0]
        for destino in re.findall(r"\]\(([^)\s]+)\)", texto)
        if not destino.startswith(("http://", "https://", "mailto:", "#"))
    ]


def _enlace_roto(readme: Path, destino: str) -> bool:
    """Dice si `destino` apunta a algo que no existe. Un enlace a una carpeta de la escalera
    que todavía no existe no cuenta como roto, porque los ejercicios nacen uno por uno."""
    if destino == "" or (readme.parent / destino).resolve().exists():
        return False
    carpeta = re.search(r"(?:^|/)([0-9][0-9]_[a-z_]+)(?:/|$)", destino)
    return not (carpeta and carpeta.group(1) in ESCALERA)


# ---------------------------------------------------------------------------------------
# 1. La forma de cada carpeta (sin tarjeta de video)
# ---------------------------------------------------------------------------------------


def test_existe_al_menos_el_ejercicio_01():
    assert "01_primer_entrenamiento" in IDS, IDS


def test_las_carpetas_de_ejercicios_llevan_los_nombres_de_la_escalera():
    desconocidas = [nombre for nombre in IDS if nombre not in ESCALERA]
    assert desconocidas == [], (
        f"Estas carpetas no están en la escalera de ejercicios/README.md: {desconocidas}. "
        f"Los nombres válidos son {list(ESCALERA)}."
    )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_la_carpeta_tiene_exactamente_los_cuatro_archivos(carpeta):
    presentes = {p.name for p in carpeta.iterdir() if p.name != "__pycache__"}
    faltan = sorted(set(ARCHIVOS) - presentes)
    sobran = sorted(presentes - set(ARCHIVOS))
    assert not faltan and not sobran, (
        f"{carpeta.name} debe tener exactamente {list(ARCHIVOS)}. "
        f"Le faltan {faltan} y le sobran {sobran}."
    )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_el_readme_tiene_un_titulo_y_las_secciones_en_orden(carpeta):
    texto = (carpeta / "README.md").read_text(encoding="utf-8")
    h1 = _titulos(texto, "#")
    assert len(h1) == 1 and texto.lstrip().startswith("# "), (
        f"El README necesita un solo título `# ...` en su primera línea; encontré {h1}."
    )
    numero = carpeta.name[:2]
    assert re.match(rf"Ejercicio {numero}: \S", h1[0]), (
        f"El título debe empezar con 'Ejercicio {numero}: ' y decir qué aprendes; es {h1[0]!r}."
    )

    todos = _titulos(texto, "##")
    en_orden = [titulo for titulo in todos if titulo in SECCIONES]
    assert en_orden == list(SECCIONES), (
        f"Las secciones del README van en este orden: {list(SECCIONES)}. "
        f"Encontré {en_orden} (títulos `##` del README: {todos})."
    )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_los_huecos_del_ejercicio_coinciden_con_los_pasos_del_readme(carpeta):
    huecos = _huecos(carpeta / "ejercicio.py")
    marcas = _marcas_tu_turno(carpeta / "ejercicio.py")
    n_huecos = len(huecos)
    assert n_huecos >= 1, "ejercicio.py necesita al menos un hueco: un `raise NotImplementedError`."
    assert len(marcas) == n_huecos, (
        f"ejercicio.py tiene {len(marcas)} comentarios `# TU TURNO: ...` y {n_huecos} "
        "`raise NotImplementedError`: cada hueco lleva uno de cada."
    )
    for numero, (marca, (lanzamiento, _)) in enumerate(zip(marcas, huecos, strict=True), 1):
        assert marca < lanzamiento, (
            f"El `# TU TURNO` del hueco {numero} (línea {marca}) va ANTES de su "
            f"`raise NotImplementedError` (línea {lanzamiento})."
        )
    for siguiente_marca, (lanzamiento, _) in zip(marcas[1:], huecos, strict=False):
        assert lanzamiento < siguiente_marca, (
            f"El `raise NotImplementedError` de la línea {lanzamiento} debe quedar antes del "
            f"siguiente `# TU TURNO` (línea {siguiente_marca})."
        )

    numeros = []
    for _, mensaje in huecos:
        coincidencia = PATRON_MENSAJE_DE_HUECO.match(mensaje)
        assert coincidencia, (
            f"El mensaje {mensaje!r} debe ser 'Completa <función> (ver el paso N del README)'."
        )
        numeros.append(int(coincidencia.group(1)))
    assert numeros == list(range(1, n_huecos + 1)), (
        f"Los huecos de ejercicio.py apuntan a los pasos {numeros}; deben ser 1..{n_huecos}, "
        "en el orden en que aparecen."
    )

    pasos = re.findall(r"^(\d+)\. ", _seccion(carpeta, "Lo que vas a hacer"), flags=re.MULTILINE)
    assert [int(p) for p in pasos] == list(range(1, n_huecos + 1)), (
        f"'Lo que vas a hacer' lista los pasos {pasos}; ejercicio.py tiene {n_huecos} huecos, "
        f"así que debe listar los pasos 1..{n_huecos}."
    )
    pistas = re.findall(r"^(?:\d+\.|-) ", _seccion(carpeta, "Pistas"), flags=re.MULTILINE)
    assert len(pistas) == n_huecos, (
        f"'Pistas' trae {len(pistas)} elementos de lista y debe traer una pista por paso "
        f"({n_huecos})."
    )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_la_solucion_no_trae_huecos(carpeta):
    assert _huecos(carpeta / "solucion.py") == [], (
        "solucion.py no puede lanzar NotImplementedError."
    )
    fuente = (carpeta / "solucion.py").read_text(encoding="utf-8")
    assert "TU TURNO" not in fuente, "solucion.py no lleva marcas `# TU TURNO`."


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_ejercicio_y_solucion_definen_las_mismas_funciones_y_constantes(carpeta):
    """`comprobar.py` revisa los dos archivos con las mismas llamadas, así que los dos deben
    ofrecer las mismas funciones (con la misma firma) y las mismas constantes."""
    assert _firmas(carpeta / "ejercicio.py") == _firmas(carpeta / "solucion.py")


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
@pytest.mark.parametrize("archivo", ["ejercicio.py", "solucion.py", "comprobar.py"])
def test_los_archivos_no_hacen_trabajo_al_importarse(carpeta, archivo):
    """Cargar un archivo solo define cosas: el programa vive bajo `if __name__ == "__main__":`.
    Sin eso, `verificar.py` correría el ejercicio entero al cargarlo."""
    arbol = _arbol(carpeta / archivo)
    sueltos = []
    for nodo in arbol.body:
        es_docstring = (
            isinstance(nodo, ast.Expr)
            and isinstance(nodo.value, ast.Constant)
            and isinstance(nodo.value.value, str)
        )
        if not (isinstance(nodo, DEFINICIONES) or es_docstring or _es_la_guarda_de_programa(nodo)):
            sueltos.append(f"línea {nodo.lineno}: {ast.unparse(nodo).splitlines()[0]}")
    assert sueltos == [], f"{archivo} corre código al importarse: {sueltos}. Muévelo a `main()`."
    if archivo != "comprobar.py":
        assert _es_la_guarda_de_programa(arbol.body[-1]), (
            f'{archivo} debe terminar con `if __name__ == "__main__":` y correr ahí su `main()`.'
        )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_comprobar_define_comprobar_con_el_parametro_modulo(carpeta):
    funciones = {
        nodo.name: nodo
        for nodo in _arbol(carpeta / "comprobar.py").body
        if isinstance(nodo, ast.FunctionDef)
    }
    assert "comprobar" in funciones, "comprobar.py debe definir `comprobar(modulo)`."
    assert [a.arg for a in funciones["comprobar"].args.args] == ["modulo"]


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_el_readme_dice_el_comando_exacto_del_verificador(carpeta):
    comando = f"uv run python ejercicios/verificar.py {carpeta.name[:2]}"
    seccion = _seccion(carpeta, "Comprueba tu trabajo")
    assert re.search(rf"^{re.escape(comando)}$", seccion, re.MULTILINE), (
        f"'Comprueba tu trabajo' debe traer, en su propia línea, el comando `{comando}`."
    )


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_los_enlaces_del_readme_apuntan_a_algo_que_existe(carpeta):
    readme = carpeta / "README.md"
    rotos = [d for d in _enlaces(readme.read_text(encoding="utf-8")) if _enlace_roto(readme, d)]
    assert rotos == [], f"Estos enlaces de {readme.relative_to(RAIZ)} no apuntan a nada: {rotos}"


@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_siguiente_ejercicio_enlaza_al_que_sigue_en_la_escalera(carpeta):
    enlaces = _enlaces(_seccion(carpeta, "Siguiente ejercicio"))
    posicion = ESCALERA.index(carpeta.name)
    if posicion + 1 < len(ESCALERA):
        esperado = f"../{ESCALERA[posicion + 1]}/README.md"
        assert esperado in enlaces, (
            f"'Siguiente ejercicio' debe enlazar a `{esperado}`; enlaza a {enlaces}."
        )
    else:
        assert enlaces, "El último ejercicio enlaza a dónde seguir (por ejemplo, ../README.md)."


def test_el_readme_de_ejercicios_enlaza_los_trece_en_cuatro_niveles():
    readme = EJERCICIOS / "README.md"
    texto = readme.read_text(encoding="utf-8")
    enlazados = [d for d in _enlaces(texto) if re.fullmatch(r"[0-9][0-9]_[a-z_]+/README\.md", d)]
    assert enlazados == [f"{nombre}/README.md" for nombre in ESCALERA], enlazados
    niveles = [t for t in _titulos(texto, "###") if t.startswith("Nivel")]
    assert len(niveles) == len(NIVELES) and all(
        titulo.startswith(prefijo) for titulo, prefijo in zip(niveles, NIVELES, strict=True)
    ), niveles
    rotos = [d for d in _enlaces(texto) if _enlace_roto(readme, d)]
    assert rotos == [], rotos


# ---------------------------------------------------------------------------------------
# 2. El verificador y lo que comparte (sin tarjeta de video)
# ---------------------------------------------------------------------------------------

FALSO_COMPROBAR = """
from _comun import Revision, revisar


def comprobar(modulo) -> list[Revision]:
    def suma_bien():
        obtenido = modulo.sumar(2, 3)
        if obtenido != 5:
            return f"Con 2 y 3 esperaba 5 y obtuve {obtenido!r}."
        return None

    def suma_otros_numeros():
        obtenido = modulo.sumar(10, 4)
        if obtenido != 14:
            return f"Con 10 y 4 esperaba 14 y obtuve {obtenido!r}."
        return None

    return [
        revisar("`sumar` suma sus dos argumentos", suma_bien, "Devuelve a + b."),
        revisar("`sumar` usa los dos argumentos", suma_otros_numeros, "Usa a y b."),
    ]
"""

FALSO_CORRECTO = "def sumar(a, b):\n    return a + b\n"
FALSO_VACIO = (
    'def sumar(a, b):\n    raise NotImplementedError("Completa sumar (ver el paso 1 del README)")\n'
)
FALSO_INCORRECTO = "def sumar(a, b):\n    return a\n"
FALSO_CON_ERROR = 'def sumar(a, b):\n    return a + "x"\n'
FALSO_CON_ERROR_DE_ESCRITURA = "def sumar(a, b)\n    return a + b\n"


class _Falso:
    """Un directorio de ejercicios con un solo ejercicio, `01_falso`, y el módulo
    `verificar` apuntando a él."""

    def __init__(self, raiz: Path, verificar: ModuleType) -> None:
        self.carpeta = raiz / "01_falso"
        self.verificar = verificar

    def escribir(self, archivo: str, fuente: str) -> None:
        (self.carpeta / archivo).write_text(fuente, encoding="utf-8")

    def main(self, *argumentos: str) -> int:
        return self.verificar.main(list(argumentos))


@pytest.fixture
def falso(tmp_path, monkeypatch):
    verificar = _verificar()
    monkeypatch.setattr(verificar, "EJERCICIOS", tmp_path)
    ejercicio_falso = _Falso(tmp_path, verificar)
    ejercicio_falso.carpeta.mkdir()
    ejercicio_falso.escribir("comprobar.py", FALSO_COMPROBAR)
    ejercicio_falso.escribir("ejercicio.py", FALSO_VACIO)
    ejercicio_falso.escribir("solucion.py", FALSO_CORRECTO)
    return ejercicio_falso


def test_el_verificador_aprueba_una_respuesta_correcta_con_codigo_0(falso, capsys):
    falso.escribir("ejercicio.py", FALSO_CORRECTO)
    assert falso.main("01") == 0
    salida = capsys.readouterr().out
    assert salida.count("Bien: ") == 2
    assert "Te falta" not in salida
    assert "Tu ejercicio pasa las 2 revisiones" in salida


def test_el_verificador_dice_te_falta_completar_con_codigo_1_y_sin_traceback(falso, capsys):
    assert falso.main("01") == 1
    capturado = capsys.readouterr()
    assert "Te falta completar: `sumar` suma sus dos argumentos" in capturado.out
    assert "Completa sumar (ver el paso 1 del README)" in capturado.out
    assert "Traceback" not in capturado.out + capturado.err
    assert "Bien:" not in capturado.out
    assert "0 de 2 revisiones en orden" in capturado.out


def test_el_verificador_dice_el_mensaje_de_un_hueco_una_sola_vez(falso, capsys):
    falso.main("01")
    assert capsys.readouterr().out.count("Completa sumar (ver el paso 1 del README)") == 1


def test_el_verificador_cuenta_que_obtuvo_y_que_hacer_ante_una_respuesta_incorrecta(falso, capsys):
    falso.escribir("ejercicio.py", FALSO_INCORRECTO)
    assert falso.main("01") == 1
    salida = capsys.readouterr().out
    assert "Te falta: `sumar` suma sus dos argumentos" in salida
    assert "Con 2 y 3 esperaba 5 y obtuve 2. Devuelve a + b." in salida
    assert "Te falta completar" not in salida


def test_el_verificador_cuenta_el_error_de_tu_codigo_en_vez_de_mostrar_un_traceback(falso, capsys):
    falso.escribir("ejercicio.py", FALSO_CON_ERROR)
    assert falso.main("01") == 1
    capturado = capsys.readouterr()
    assert "Tu código lanzó TypeError" in capturado.out
    assert "Traceback" not in capturado.out + capturado.err


def test_el_verificador_dice_en_que_linea_esta_el_error_de_escritura(falso, capsys):
    falso.escribir("ejercicio.py", FALSO_CON_ERROR_DE_ESCRITURA)
    assert falso.main("01") == 1
    capturado = capsys.readouterr()
    assert "error de escritura en la línea 1" in capturado.err
    assert "Traceback" not in capturado.out + capturado.err


def test_el_verificador_traduce_un_hueco_que_sube_sin_pasar_por_revisar(falso, capsys):
    falso.escribir(
        "comprobar.py", "def comprobar(modulo):\n    modulo.sumar(1, 2)\n    return []\n"
    )
    assert falso.main("01") == 1
    capturado = capsys.readouterr()
    assert "Te falta completar: tu ejercicio.py está completo" in capturado.out
    assert "Completa sumar (ver el paso 1 del README)" in capturado.out
    assert "Traceback" not in capturado.out + capturado.err


def test_el_verificador_rechaza_una_comprobacion_sin_revisiones(falso, capsys):
    falso.escribir("comprobar.py", "def comprobar(modulo):\n    return []\n")
    assert falso.main("01") == 1
    assert "no devolvió ninguna revisión" in capsys.readouterr().err


def test_el_verificador_deja_subir_un_error_que_no_es_del_tesista(falso):
    falso.escribir("comprobar.py", "def comprobar(modulo):\n    raise RuntimeError('fallo')\n")
    with pytest.raises(RuntimeError, match="fallo"):
        falso.main("01")


def test_con_solucion_el_verificador_revisa_solucion_py(falso, capsys):
    assert falso.main("01", "--solucion") == 0
    salida = capsys.readouterr().out
    assert "Reviso 01_falso/solucion.py" in salida
    assert "La solución pasa las 2 revisiones" in salida
    assert falso.main("--solucion", "1") == 0


def test_el_verificador_acepta_el_numero_con_o_sin_cero(falso):
    buscar = falso.verificar.buscar_carpeta
    assert buscar("1") == buscar("01") == falso.carpeta
    assert buscar("2") is None
    assert buscar("uno") is None


def test_el_verificador_explica_el_uso_y_los_ejercicios_que_hay(falso, capsys):
    assert falso.main() == 2
    assert "Uso: uv run python ejercicios/verificar.py NUMERO [--solucion]" in (
        capsys.readouterr().err
    )
    assert falso.main("01", "02") == 2
    capsys.readouterr()
    assert falso.main("07") == 2
    assert "No encuentro el ejercicio 07. Los que hay: 01_falso." in capsys.readouterr().err


def test_el_verificador_corre_como_programa_desde_cualquier_directorio(tmp_path):
    """`uv run python ejercicios/verificar.py` termina con el código que devuelve `main`.
    Sin argumentos no necesita la tarjeta de video."""
    resultado = subprocess.run(
        [sys.executable, "-W", "error", str(EJERCICIOS / "verificar.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert resultado.returncode == 2
    assert "Uso: uv run python ejercicios/verificar.py" in resultado.stderr
    assert "Traceback" not in resultado.stderr


def test_sin_tarjeta_de_video_el_verificador_dice_que_falta_y_no_muestra_traceback(tmp_path):
    entorno = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONIOENCODING": "utf-8"}
    resultado = subprocess.run(
        [sys.executable, "-W", "error", str(EJERCICIOS / "verificar.py"), "01"],
        cwd=tmp_path,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert resultado.returncode == 1, resultado.stdout + resultado.stderr
    assert MENSAJE_SIN_TARJETA in resultado.stderr
    assert "Traceback" not in resultado.stderr


def test_imprimir_marca_cada_revision_y_cuenta_cada_hueco_una_sola_vez(capsys):
    revision = _comun().Revision
    hueco_f = "Completa f (ver el paso 1 del README)"
    _verificar().imprimir(
        [
            revision("algo que pasa", True),
            revision("algo mal hecho", False, "Cambia esto."),
            revision("un hueco", False, hueco_f, True),
            revision("el mismo hueco", False, hueco_f, True),
            revision("otro hueco", False, "Completa g (ver el paso 2 del README)", True),
        ]
    )
    assert capsys.readouterr().out.splitlines() == [
        "Bien: algo que pasa",
        "Te falta: algo mal hecho",
        "    Cambia esto.",
        "Te falta completar: un hueco",
        f"    {hueco_f}",
        "Te falta completar: el mismo hueco",
        "Te falta completar: otro hueco",
        "    Completa g (ver el paso 2 del README)",
    ]


def test_revisar_aprueba_cuando_la_prueba_no_encuentra_nada_mal():
    revision = _comun().revisar("algo", lambda: None, "Haz esto.")
    assert (revision.que_se_reviso, revision.paso, revision.que_hacer) == ("algo", True, "")
    assert not revision.sin_completar


def test_revisar_une_lo_que_vio_con_lo_que_hay_que_hacer():
    revision = _comun().revisar("algo", lambda: "Esperaba 1 y obtuve 2.", "Haz esto.")
    assert not revision.paso
    assert revision.que_hacer == "Esperaba 1 y obtuve 2. Haz esto."
    assert not revision.sin_completar


def test_revisar_traduce_un_hueco_sin_completar():
    def hueco():
        raise NotImplementedError("Completa f (ver el paso 1 del README)")

    revision = _comun().revisar("algo", hueco, "Haz esto.")
    assert not revision.paso
    assert revision.sin_completar
    assert revision.que_hacer == "Completa f (ver el paso 1 del README)"


@pytest.mark.parametrize(
    "error",
    [
        TypeError("mal tipo"),
        KeyError("clave"),
        ZeroDivisionError("division"),
        AttributeError("sin atributo"),
        NameError("sin nombre"),
        IndexError("fuera de rango"),
        ValueError("mal valor"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_revisar_cuenta_los_errores_tipicos_del_tesista(error):
    def falla():
        raise error

    revision = _comun().revisar("algo", falla, "Haz esto.")
    assert not revision.paso
    assert not revision.sin_completar
    assert revision.que_hacer.startswith(f"Tu código lanzó {type(error).__name__}: ")
    assert revision.que_hacer.endswith("Haz esto.")


def test_revisar_deja_subir_un_error_que_no_es_del_tesista():
    def falla():
        raise RuntimeError("fallo de la comprobación")

    with pytest.raises(RuntimeError, match="fallo de la comprobación"):
        _comun().revisar("algo", falla, "Haz esto.")


def test_cargar_modulo_define_las_funciones_sin_correr_el_programa(tmp_path):
    ruta = tmp_path / "pieza.py"
    ruta.write_text(
        "def doble(x):\n    return 2 * x\n\n"
        'if __name__ == "__main__":\n    raise SystemExit("no debe correr")\n',
        encoding="utf-8",
    )
    modulo = _comun().cargar_modulo(ruta, "pieza_de_prueba")
    assert modulo.doble(21) == 42
    assert sys.modules["pieza_de_prueba"] is modulo


# ---------------------------------------------------------------------------------------
# 3. La conducta de cada ejercicio (necesita la tarjeta de video)
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def revisiones_de():
    """Las revisiones de `solucion.py` o de `ejercicio.py` de una carpeta, calculadas una
    sola vez por módulo: varias pruebas leen el mismo resultado."""
    cache: dict[tuple[str, str], list] = {}

    def obtener(carpeta: Path, archivo: str) -> list:
        clave = (carpeta.name, archivo)
        if clave not in cache:
            cache[clave] = _verificar().correr_revisiones(carpeta, archivo)
        return cache[clave]

    return obtener


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_la_solucion_pasa_todas_las_revisiones(carpeta, revisiones_de):
    revisiones = revisiones_de(carpeta, "solucion.py")
    assert len(revisiones) >= 2, "Un ejercicio necesita al menos dos revisiones."
    fallidas = [(r.que_se_reviso, r.que_hacer) for r in revisiones if not r.paso]
    assert fallidas == []


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_el_ejercicio_vacio_no_pasa_ninguna_revision_y_dice_que_hueco_falta(carpeta, revisiones_de):
    """Una revisión que pasa con el ejercicio vacío no revisa nada del trabajo del tesista.
    Cada revisión debe llamar a un hueco y, con el hueco vacío, decir cuál falta."""
    revisiones = revisiones_de(carpeta, "ejercicio.py")
    pasan_vacias = [r.que_se_reviso for r in revisiones if r.paso]
    assert pasan_vacias == [], (
        f"Estas revisiones pasan con el ejercicio vacío, así que no revisan nada: {pasan_vacias}."
    )
    for revision in revisiones:
        assert revision.sin_completar, (
            f"La revisión {revision.que_se_reviso!r} falla con el ejercicio vacío, pero no por "
            f"un hueco sin completar: {revision.que_hacer!r}."
        )
        assert PATRON_MENSAJE_DE_HUECO.match(revision.que_hacer), revision.que_hacer


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_verificar_devuelve_0_con_la_solucion_y_1_con_el_vacio(carpeta, capsys):
    verificar = _verificar()
    numero = carpeta.name[:2]

    assert verificar.main([numero, "--solucion"]) == 0
    con_solucion = capsys.readouterr()
    assert "Bien: " in con_solucion.out
    assert "Te falta" not in con_solucion.out
    assert "Traceback" not in con_solucion.out + con_solucion.err

    assert verificar.main([numero]) == 1
    con_el_vacio = capsys.readouterr()
    assert "Te falta completar: " in con_el_vacio.out
    assert "Bien:" not in con_el_vacio.out
    assert "Traceback" not in con_el_vacio.out + con_el_vacio.err
    assert re.search(r"\(ver el paso \d+ del README\)", con_el_vacio.out)


def _cargar_comprobar(carpeta: Path) -> ModuleType:
    return _comun().cargar_modulo(carpeta / "comprobar.py", f"comprobar_{carpeta.name}")


def _cargar_copia(carpeta: Path, fuente: str, tmp_path: Path) -> ModuleType:
    """Escribe `fuente` en un archivo nuevo de `tmp_path` y lo carga como módulo. Cada copia
    lleva su propio nombre: Python reutiliza el bytecode de un archivo cuyo tamaño y hora de
    modificación no cambiaron, y dos copias escritas en el mismo segundo se confundirían."""
    numero = next(_NOMBRES_UNICOS)
    destino = tmp_path / f"mutante_{numero}.py"
    destino.write_text(fuente, encoding="utf-8")
    return _comun().cargar_modulo(destino, f"mutante_{carpeta.name}_{numero}")


def _fuente_con_cuerpos_constantes(carpeta: Path, funciones: list[str], constante: str) -> str:
    """La fuente de `solucion.py` donde cada función de `funciones` devuelve `constante`."""
    arbol = _arbol(carpeta / "solucion.py")
    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name in funciones:
            nodo.body = ast.parse(f"def _f():\n    return {constante}").body[0].body
    ast.fix_missing_locations(arbol)
    return ast.unparse(arbol)


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_una_funcion_que_devuelve_una_constante_no_pasa_las_revisiones(carpeta, tmp_path):
    comprobar = _cargar_comprobar(carpeta)
    funciones = _funciones_con_hueco(carpeta / "ejercicio.py")
    assert funciones, "ejercicio.py no tiene funciones con hueco."
    for constante in CONSTANTES:
        for funcion in funciones:
            fuente = _fuente_con_cuerpos_constantes(carpeta, [funcion], constante)
            revisiones = comprobar.comprobar(_cargar_copia(carpeta, fuente, tmp_path))
            assert any(not r.paso for r in revisiones), (
                f"Con `{funcion}` devolviendo {constante}, ninguna revisión falló: "
                "esa función no está revisada."
            )


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_cada_cambio_que_declara_comprobar_hace_fallar_alguna_revision(carpeta, tmp_path):
    comprobar = _cargar_comprobar(carpeta)
    fuente = (carpeta / "solucion.py").read_text(encoding="utf-8")
    for descripcion, viejo, nuevo in getattr(comprobar, "MUTANTES", []):
        assert fuente.count(viejo) == 1, (
            f"El cambio {descripcion!r} busca {viejo!r}, que aparece {fuente.count(viejo)} "
            "veces en solucion.py; debe aparecer exactamente una."
        )
        modulo = _cargar_copia(carpeta, fuente.replace(viejo, nuevo), tmp_path)
        revisiones = comprobar.comprobar(modulo)
        assert any(not r.paso for r in revisiones), (
            f"Con el cambio {descripcion!r}, todas las revisiones pasaron: "
            "las revisiones no distinguen una solución rota."
        )


@pytest.mark.gpu
def test_el_ejercicio_01_declara_un_cambio_por_cada_manera_de_equivocarse():
    """El ejercicio de referencia muestra cómo se escribe `MUTANTES`: un cambio por cada
    manera típica de equivocarse (ignorar un argumento, cortar en vez de redondear, olvidar
    un término de la cuenta)."""
    comprobar = _cargar_comprobar(EJERCICIOS / "01_primer_entrenamiento")
    descripciones = [descripcion for descripcion, _, _ in comprobar.MUTANTES]
    for frase in (
        "ignora las semillas",
        "ignora la config",
        "int en vez de redondear",
        "ignora n_val",
        "olvida los sesgos",
        "olvida la capa final",
    ):
        assert any(frase in d for d in descripciones), (frase, descripciones)


# ---------------------------------------------------------------------------------------
# 4. Los programas (necesitan la tarjeta de video)
# ---------------------------------------------------------------------------------------


def _correr_programa(archivo: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    entorno = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "TALLER_ESTUDIOS": str(tmp_path / "estudios"),
        "TALLER_RESULTADOS": str(tmp_path / "resultados"),
    }
    return subprocess.run(
        [sys.executable, "-W", "error", str(archivo)],
        cwd=tmp_path,
        env=entorno,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=100,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_la_solucion_corre_como_programa_e_imprime_lo_que_hay_que_observar(carpeta, tmp_path):
    resultado = _correr_programa(carpeta / "solucion.py", tmp_path)
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    assert resultado.stdout.strip(), "solucion.py no imprimió nada."
    assert resultado.stderr == "", resultado.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("carpeta", CARPETAS, ids=IDS)
def test_el_ejercicio_vacio_corre_y_termina_con_un_mensaje_sin_traceback(carpeta, tmp_path):
    resultado = _correr_programa(carpeta / "ejercicio.py", tmp_path)
    salida = resultado.stdout + resultado.stderr
    assert resultado.returncode != 0, salida
    assert "Traceback" not in salida, salida
    assert re.search(r"Completa .+ \(ver el paso \d+ del README\)", salida), salida
