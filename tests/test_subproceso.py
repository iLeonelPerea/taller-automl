"""por_subproceso arma la entrada, corre `comando`, y traduce sus fallos.

`train_ejemplo.py` hace de "modelo del tesista" en la mayoría de las pruebas:
así se prueba contra el mismo contrato que usará quien copie la plantilla, no
contra un doble simplificado. Las pruebas de errores (returncode, timeout,
JSON inválido o ausente) usan scripts mínimos escritos en `tmp_path`: ahí sí
interesa un caso controlado, no el comportamiento real de un modelo.
"""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pytest

from taller_hpo import subproceso as subproceso_mod
from taller_hpo.objetivo import MetricaNoFinita
from taller_hpo.semillas import Semillas
from taller_hpo.subproceso import _matar_arbol, escribir_json_atomico, leer_json, por_subproceso

RAIZ = Path(__file__).resolve().parent.parent
TRAIN_EJEMPLO = RAIZ / "plantilla" / "train_ejemplo.py"
S = Semillas(datos=1, orden=2, init=3)
CONFIG_RAPIDO = {"n_train": 150, "n_val": 150, "epocas": 8, "ancho": 8}


def _sampler():
    return optuna.samplers.RandomSampler(seed=0)


def _importar(ruta: Path, nombre: str):
    """Importa un .py fuera de sys.path, como referencia directa de comparación."""
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# --- escribir_json_atomico / leer_json ---------------------------------------


def test_escribir_leer_json_atomico_roundtrip_con_acentos(tmp_path):
    ruta = tmp_path / "sub" / "archivo.json"  # el padre no existe todavía
    obj = {"nombre": "peña", "frase": "año niño", "n": 3, "lista": [1, 2, 3]}
    escribir_json_atomico(ruta, obj)
    assert leer_json(ruta) == obj


def test_escribir_json_atomico_no_deja_temporal_si_falla(tmp_path):
    ruta = tmp_path / "archivo.json"

    class NoSerializable:
        pass

    with pytest.raises(TypeError):
        escribir_json_atomico(ruta, {"x": NoSerializable()})
    assert not ruta.exists()
    assert list(tmp_path.iterdir()) == []


def test_leer_json_archivo_ausente_falla(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_json(tmp_path / "no_existe.json")


@pytest.mark.skipif(
    sys.platform == "win32", reason="permisos POSIX: chmod no aplica igual en Windows"
)
def test_escribir_json_atomico_deja_permisos_0644_pese_al_umask(tmp_path):
    # mkstemp crea el temporal en 0600 (fijo, sin importar el umask). Sin el
    # os.chmod explicito antes de os.replace, esos 0600 quedan en el archivo
    # final: un `comando` (en por_subproceso) que corre con OTRO usuario sobre
    # un `directorio` compartido no podria leer la entrada.
    umask_previo = os.umask(0o077)
    try:
        ruta = tmp_path / "entrada.json"
        escribir_json_atomico(ruta, {"x": 1})
    finally:
        os.umask(umask_previo)
    permisos = ruta.stat().st_mode & 0o777
    assert permisos == 0o644, f"permisos {oct(permisos)}, se esperaba 0o644"


# --- por_subproceso: camino feliz ---------------------------------------------


@pytest.mark.gpu
def test_valores_correctos_y_user_attrs_contra_un_study_real(tmp_path):
    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {"ancho": trial.suggest_int("ancho", 4, 8)},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos={"n_train": 60, "n_val": 40, "epocas": 3},
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=2)

    modulo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_ref")
    assert len(study.trials) == 2
    for t in study.trials:
        assert t.state == optuna.trial.TrialState.COMPLETE
        # El resultado del subproceso debe coincidir EXACTO con llamar a
        # entrenar() directo, con la misma config y semillas: por_subproceso
        # no debe alterar ni el cómputo ni el redondeo. train_ejemplo.entrenar
        # recibe semillas por atributo (igual que el Semillas del núcleo): el
        # dict que llega por user_attrs (vía JSON) se reconstruye primero.
        esperado = modulo.entrenar(
            t.user_attrs["config"], modulo.Semillas(**t.user_attrs["semillas"])
        )
        assert t.value == esperado["exactitud_val"]
        assert t.user_attrs["config"] == {
            "n_train": 60,
            "n_val": 40,
            "epocas": 3,
            "ancho": t.params["ancho"],
        }
        assert t.user_attrs["semillas"] == {"datos": 1, "orden": 2, "init": 3}
        assert (tmp_path / f"trial_{t.number}_entrada.json").exists()
        assert (tmp_path / f"trial_{t.number}_salida.json").exists()


@pytest.mark.gpu
def test_atributos_pedidos_se_guardan_como_user_attrs(tmp_path):
    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
        atributos=["epocas"],
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials
    # Sin reportar (por_subproceso no lo llama), las épocas configuradas en
    # CONFIG_RAPIDO corren completas: el atributo "epocas" de salida debe
    # reflejarlo.
    assert t.user_attrs["epocas"] == CONFIG_RAPIDO["epocas"]
    # La métrica optimizada NO se duplica como user_attr: solo lo hacen
    # "config", "semillas" y los `atributos` pedidos explícitamente.
    assert "exactitud_val" not in t.user_attrs


@pytest.mark.gpu
def test_semillas_como_callable_se_resuelve_por_trial(tmp_path):
    vistas = []

    def semillas_por_trial(trial):
        s = Semillas(datos=trial.number, orden=2, init=3)
        vistas.append(s)
        return s

    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=semillas_por_trial,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=2)

    assert [s.datos for s in vistas] == [0, 1]
    for t in study.trials:
        assert t.user_attrs["semillas"]["datos"] == t.number


@pytest.mark.gpu
def test_comando_arbitrario_con_flags_extra(tmp_path):
    # `comando` es cualquier lista que respete --config/--salida: aquí lleva
    # flags del intérprete además del ejecutable.
    objetivo = por_subproceso(
        [sys.executable, "-X", "utf8", str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert 0.0 <= t.value <= 1.0


def test_ruta_y_valores_con_acentos_y_espacios(tmp_path):
    # Un script de eco propio (no train_ejemplo.py): así se aísla el round-trip
    # JSON de acentos/espacios del comportamiento numérico de un modelo real.
    directorio = tmp_path / "carpeta con ñ y espacios"
    carpeta_script = tmp_path / "otro directorio con acentos áéí"
    carpeta_script.mkdir()
    script = carpeta_script / "eco.py"
    script.write_text(
        "import argparse\n"
        "import json\n"
        "\n"
        "\n"
        "def entrenar(config, semillas, reportar=None):\n"
        "    return {\n"
        "        'exactitud_val': 1.0,\n"
        "        'etiqueta': config['etiqueta'],\n"
        "        'semilla_datos': semillas['datos'],\n"
        "    }\n"
        "\n"
        "\n"
        "def main(argv=None):\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--config', required=True)\n"
        "    parser.add_argument('--salida', required=True)\n"
        "    args = parser.parse_args(argv)\n"
        "    with open(args.config, encoding='utf-8') as f:\n"
        "        entrada = json.load(f)\n"
        "    metricas = entrenar(entrada['config'], entrada['semillas'])\n"
        "    with open(args.salida, 'w', encoding='utf-8') as f:\n"
        "        json.dump(metricas, f, ensure_ascii=False)\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    etiqueta = "canción niño mañana áéíóú Ñoño"
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {"etiqueta": etiqueta},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=30.0,
        directorio=directorio,
        atributos=["etiqueta"],
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials

    assert t.value == 1.0
    assert t.user_attrs["etiqueta"] == etiqueta
    assert t.user_attrs["config"]["etiqueta"] == etiqueta
    entrada_en_disco = leer_json(directorio / "trial_0_entrada.json")
    assert entrada_en_disco["config"]["etiqueta"] == etiqueta


# --- por_subproceso: multiobjetivo y entorno del subproceso -------------------


@pytest.mark.gpu
def test_multiobjetivo_devuelve_tupla_en_el_orden_de_metricas(tmp_path):
    # Esta prueba falla si una implementacion siempre devolviera valores[0]:
    # con una sola metrica ninguna prueba anterior distingue devolver un
    # float de devolver siempre el primer elemento de una tupla.
    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val", "epocas"],
        semillas=S,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
    )
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials

    modulo = _importar(TRAIN_EJEMPLO, "_train_ejemplo_ref_multiobjetivo")
    esperado = modulo.entrenar(t.user_attrs["config"], modulo.Semillas(**t.user_attrs["semillas"]))
    assert t.values == [esperado["exactitud_val"], esperado["epocas"]]


def test_pythonutf8_esta_activado_en_el_subproceso(tmp_path):
    # Esta prueba falla si una implementacion pasara "env=dict(os.environ)"
    # sin agregar PYTHONUTF8: en Linux con locale UTF-8 la prueba de acentos
    # pasa igual con o sin la variable, asi que hace falta comprobar la
    # variable misma.
    script = tmp_path / "utf8_mode.py"
    script.write_text(
        "import json\n"
        "import sys\n"
        "\n"
        "salida = sys.argv[sys.argv.index('--salida') + 1]\n"
        "with open(salida, 'w', encoding='utf-8') as f:\n"
        "    json.dump({'exactitud_val': 1.0, 'utf8': sys.flags.utf8_mode}, f)\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
        atributos=["utf8"],
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials
    assert t.user_attrs["utf8"] == 1


@pytest.mark.gpu
def test_subproceso_se_lanza_con_stdin_devnull(tmp_path, monkeypatch):
    # Sin stdin=DEVNULL, el hijo hereda el stdin de este proceso: un
    # input()/breakpoint() olvidado en el train.py se queda esperando teclas
    # que nunca llegan (le roba teclas al padre de paso) hasta el timeout, en
    # vez de fallar de inmediato con EOFError.
    llamada = {}
    popen_original = subprocess.Popen

    def popen_espia(*args, **kwargs):
        llamada["kwargs"] = kwargs
        return popen_original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen_espia)

    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)

    assert "kwargs" in llamada, "Popen no se llamo (el espia no interceptó nada)"
    assert llamada["kwargs"].get("stdin") == subprocess.DEVNULL


# --- por_subproceso: errores ---------------------------------------------------


def test_returncode_no_cero_da_runtimeerror_con_stderr(tmp_path):
    script = tmp_path / "falla.py"
    script.write_text(
        "import sys\n"
        "print('mensaje de error especifico desde el subproceso', file=sys.stderr)\n"
        "sys.exit(3)\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="mensaje de error especifico"):
        study.optimize(objetivo, n_trials=1)
    # El taller guarda el stdout del subproceso en su propio archivo, aunque el
    # trial haya fallado por stderr o por returncode.
    assert (tmp_path / "trial_0_stdout.txt").exists()


def test_timeout_da_timeouterror(tmp_path):
    script = tmp_path / "duerme.py"
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=0.3,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    inicio = time.monotonic()
    with pytest.raises(TimeoutError):
        study.optimize(objetivo, n_trials=1)
    transcurrido = time.monotonic() - inicio
    # Si el arbol no se mata de verdad (kill roto), el proceso.wait() final de
    # _matar_arbol espera a que el hijo termine SOLO: los 5 s del sleep, no los
    # 0.3 s del timeout. Este limite (generoso: deja margen para arranque de
    # Python e I/O) es lo que detecta un kill roto sin correr una senal real.
    assert transcurrido < 3.0, f"tardo {transcurrido:.2f}s: el arbol no se mato a tiempo"


def test_json_invalido_da_runtimeerror(tmp_path):
    script = tmp_path / "json_invalido.py"
    script.write_text(
        "import sys\n"
        "salida = sys.argv[sys.argv.index('--salida') + 1]\n"
        "with open(salida, 'w', encoding='utf-8') as f:\n"
        "    f.write('{esto no es json valido')\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="JSON"):
        study.optimize(objetivo, n_trials=1)


def test_salida_no_utf8_da_runtimeerror(tmp_path):
    # `comando` es arbitrario (R, Julia, un python de otro entorno con
    # encoding='cp1252'): el JSON de --salida no siempre llega en UTF-8.
    # `leer_json` lo abre con encoding="utf-8" a secas, y antes de la
    # correccion el except de por_subproceso solo cubria (OSError,
    # json.JSONDecodeError): UnicodeDecodeError (subclase de ValueError, NO
    # de JSONDecodeError) se escapaba sin traducirse al RuntimeError del
    # contrato, y catch=(RuntimeError, TimeoutError) (ver README) no lo
    # atrapaba.
    script = tmp_path / "salida_no_utf8.py"
    script.write_text(
        "import sys\n"
        "salida = sys.argv[sys.argv.index('--salida') + 1]\n"
        'contenido = \'{"exactitud_val": 0.5, "nota": "canción"}\'\n'
        "with open(salida, 'wb') as f:\n"
        "    f.write(contenido.encode('cp1252'))\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="el taller no pudo leer"):
        study.optimize(objetivo, n_trials=1)


@pytest.mark.parametrize("contenido_salida", ["[0.9]", "null", "0.9"])
def test_json_valido_pero_no_es_objeto_da_runtimeerror(tmp_path, contenido_salida):
    # JSON valido (json.load no falla) pero no es {metrica: valor}: sin esta
    # validacion el error real era TypeError (indexado de lista/None/float),
    # no el RuntimeError del contrato, y un TypeError no lo atrapa
    # study.optimize(..., catch=(RuntimeError, TimeoutError)) (ver README).
    script = tmp_path / "salida_no_es_objeto.py"
    script.write_text(
        "import sys\n"
        "salida = sys.argv[sys.argv.index('--salida') + 1]\n"
        "with open(salida, 'w', encoding='utf-8') as f:\n"
        f"    f.write({contenido_salida!r})\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match=r"objeto JSON \{métrica: valor\}"):
        study.optimize(objetivo, n_trials=1)


def test_json_ausente_da_runtimeerror(tmp_path):
    script = tmp_path / "no_escribe_salida.py"
    script.write_text("pass\n", encoding="utf-8")
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="el taller no pudo leer"):
        study.optimize(objetivo, n_trials=1)


def test_metrica_no_finita_da_metricanofinita_con_el_nombre_de_la_metrica(tmp_path):
    # json.load acepta NaN: sin la guarda de `por_subproceso`, un trial con
    # exactitud_val=NaN terminaría COMPLETE con un valor que Optuna no puede ordenar.
    script = tmp_path / "salida_nan.py"
    script.write_text(
        "import sys\n"
        "salida = sys.argv[sys.argv.index('--salida') + 1]\n"
        "with open(salida, 'w', encoding='utf-8') as f:\n"
        "    f.write('{\"exactitud_val\": NaN}')\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(MetricaNoFinita, match=r"métrica 'exactitud_val' = nan no es finita"):
        study.optimize(objetivo, n_trials=1)


def test_metricas_vacio_lanza_valueerror(tmp_path):
    with pytest.raises(ValueError, match="metricas no puede estar vacío"):
        por_subproceso(
            [sys.executable, "-c", "pass"],
            espacio=lambda trial: {},
            metricas=[],
            semillas=S,
            timeout_s=10.0,
            directorio=tmp_path,
        )


def test_salida_rancia_no_se_reutiliza_si_el_comando_no_escribe_nada(tmp_path):
    # Si `directorio` se reusa (otro estudio, otra corrida) y el comando de
    # este trial no escribe salida, no debe leerse la salida de una corrida
    # anterior como si fuera la de este trial (ver docstring del modulo).
    (tmp_path / "trial_0_salida.json").write_text(
        json.dumps({"exactitud_val": 0.99}), encoding="utf-8"
    )
    script = tmp_path / "no_escribe_nada.py"
    script.write_text("pass\n", encoding="utf-8")
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="el taller no pudo leer"):
        study.optimize(objetivo, n_trials=1)


def test_final_de_stderr_se_preserva_no_el_inicio(tmp_path):
    # Esta prueba falla si una implementacion recortara con "[:2000]" en vez
    # de "[-2000:]": los tracebacks de Python ponen el error real al final,
    # no al principio.
    script = tmp_path / "stderr_largo.py"
    script.write_text(
        "import sys\nsys.stderr.write('INICIO' + 'x' * 5000 + 'MARCA_FINAL')\nsys.exit(1)\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError) as excinfo:
        study.optimize(objetivo, n_trials=1)
    mensaje = str(excinfo.value)
    assert "MARCA_FINAL" in mensaje
    assert "INICIO" not in mensaje


def test_stderr_no_utf8_da_runtimeerror_y_conserva_el_resto_del_mensaje(tmp_path):
    # Un ejecutable de otro entorno (u otra plataforma) puede escribir stderr
    # en un encoding que no es UTF-8. Debe seguir siendo RuntimeError (el
    # contrato), nunca UnicodeDecodeError perdiendo el mensaje real.
    script = tmp_path / "stderr_no_utf8.py"
    script.write_text(
        "import sys\nsys.stderr.buffer.write('José MARCA'.encode('cp1252'))\nsys.exit(1)\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(RuntimeError, match="MARCA"):
        study.optimize(objetivo, n_trials=1)


@pytest.mark.skipif(sys.platform == "win32", reason="usa process groups POSIX (setsid/killpg)")
def test_timeout_mata_tambien_al_nieto_no_solo_al_hijo_directo(tmp_path):
    # `comando` puede ser un lanzador ("uv run ...", un .sh) que a su vez crea
    # un nieto. Matar solo al hijo directo lo deja vivo (ver docstring del
    # modulo). La envoltura lanza al nieto en segundo plano y espera: ambos
    # quedan en el mismo grupo de procesos que crea por_subproceso.
    arranco = tmp_path / "nieto_arranco.marca"
    marca = tmp_path / "nieto_vivo.marca"
    nieto = tmp_path / "nieto.py"
    nieto.write_text(
        "import pathlib\n"
        "import time\n"
        "\n"
        f"pathlib.Path({str(arranco)!r}).write_text('vivo', encoding='utf-8')\n"
        "time.sleep(2)\n"
        f"pathlib.Path({str(marca)!r}).write_text('vivo', encoding='utf-8')\n",
        encoding="utf-8",
    )
    envoltura = tmp_path / "envoltura.sh"
    # shlex.quote: sin comillas, una ruta con espacios (tmp_path puede tenerlos
    # según el usuario o la integración continua) rompe la envoltura sin correr
    # al nieto, y la prueba pasaría aunque el kill estuviera roto (nunca habría
    # nieto que matar).
    envoltura.write_text(
        f"#!/bin/sh\n{shlex.quote(sys.executable)} {shlex.quote(str(nieto))} &\nwait\n",
        encoding="utf-8",
    )

    objetivo = por_subproceso(
        ["sh", str(envoltura)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=0.3,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(TimeoutError):
        study.optimize(objetivo, n_trials=1)

    # Control positivo: si el nieto nunca arranco (p.ej. la envoltura se rompio
    # por una ruta con espacios), "el nieto no esta vivo" es trivialmente
    # cierto y la prueba pasaria sin haber probado nada.
    assert arranco.exists(), "el nieto nunca arranco: la envoltura no lo lanzo"

    # Tiempo de sobra para que el nieto, si sobrevivio al timeout, escriba su
    # marca (duerme 2 s en total, el timeout fue de 0.3 s).
    time.sleep(2.5)
    assert not marca.exists()


def test_user_attrs_se_guardan_incluso_si_el_trial_falla(tmp_path):
    # en_proceso (objetivo.py) guarda "config"/"semillas" ANTES de llamar a
    # entrenar, asi que un trial FAIL tambien queda identificado. por_subproceso
    # debe comportarse igual: son justo los trials mas interesantes de
    # reproducir despues.
    script = tmp_path / "siempre_falla.py"
    script.write_text(
        "import sys\nprint('fallo a proposito', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {"x": trial.suggest_int("x", 0, 1)},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1, catch=(RuntimeError,))
    (t,) = study.trials
    assert t.state == optuna.trial.TrialState.FAIL
    assert t.user_attrs["config"] == {"x": t.params["x"]}
    assert t.user_attrs["semillas"] == {"datos": 1, "orden": 2, "init": 3}


@pytest.mark.gpu
def test_semillas_con_enteros_numpy_quedan_como_int_nativo(tmp_path):
    # dataclasses.asdict() deja los np.int64 tal cual (semillas_de, que usa
    # numpy.SeedSequence, produce justo eso). Sin convertir a int nativo,
    # escribir_json_atomico revienta con "Object of type int64 is not JSON
    # serializable" antes de correr el subproceso siquiera.
    s_numpy = Semillas(datos=np.int64(1), orden=np.int64(2), init=np.int64(3))
    objetivo = por_subproceso(
        [sys.executable, str(TRAIN_EJEMPLO)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=s_numpy,
        timeout_s=60.0,
        directorio=tmp_path,
        fijos=CONFIG_RAPIDO,
    )
    study = optuna.create_study(sampler=_sampler())
    study.optimize(objetivo, n_trials=1)
    (t,) = study.trials
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert t.user_attrs["semillas"] == {"datos": 1, "orden": 2, "init": 3}
    assert all(type(v) is int for v in t.user_attrs["semillas"].values())


# --- por_subproceso: interrupcion mientras se espera (no solo timeout) --------


def test_interrupcion_durante_la_espera_tambien_mata_el_arbol(tmp_path, monkeypatch):
    # Un Ctrl+C real (SIGINT a todo el grupo en primer plano) NO le llega al
    # hijo: esta en su propio grupo/sesion (ver docstring del modulo), asi que
    # sin este manejo el padre muere (o Optuna atrapa la excepcion y sigue) y
    # el hijo queda huerfano corriendo sin limite. Se simula sin senales
    # reales (no son portables ni deterministas): se reemplaza Popen.wait para
    # que la espera CON timeout (la unica que hace por_subproceso antes de
    # matar el arbol) lance KeyboardInterrupt justo despues de que el hijo
    # real escribio su marca de arranque (control positivo); la espera SIN
    # timeout que usa _matar_arbol para recolectar al proceso ya matado se deja
    # intacta delegando al wait() original.
    arranco = tmp_path / "arranco.marca"
    termino = tmp_path / "termino.marca"
    script = tmp_path / "tarda.py"
    script.write_text(
        "import pathlib\n"
        "import time\n"
        "\n"
        f"pathlib.Path({str(arranco)!r}).write_text('vivo', encoding='utf-8')\n"
        "time.sleep(2)\n"
        f"pathlib.Path({str(termino)!r}).write_text('vivo', encoding='utf-8')\n",
        encoding="utf-8",
    )

    wait_original = subprocess.Popen.wait

    def wait_falso(self, timeout=None):
        if timeout is None:
            # La espera de _matar_arbol tras matar el arbol: sin cambios.
            return wait_original(self, timeout)
        for _ in range(100):
            if arranco.exists():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("el hijo nunca escribio su marca de arranque")
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "wait", wait_falso)

    objetivo = por_subproceso(
        [sys.executable, str(script)],
        espacio=lambda trial: {},
        metricas=["exactitud_val"],
        semillas=S,
        timeout_s=10.0,
        directorio=tmp_path,
    )
    study = optuna.create_study(sampler=_sampler())
    with pytest.raises(KeyboardInterrupt):
        study.optimize(objetivo, n_trials=1)

    assert arranco.exists()
    # Tiempo de sobra para que el hijo, si sobrevivio a la interrupcion,
    # escriba su marca de termino (duerme 2 s en total).
    time.sleep(2.5)
    assert not termino.exists()


# --- _matar_arbol: ProcessLookupError solo es legitimo si ya se recolecto -----


@pytest.mark.skipif(sys.platform == "win32", reason="_matar_arbol usa killpg en POSIX")
def test_matar_arbol_relanza_processlookuperror_si_el_proceso_sigue_vivo(monkeypatch):
    # Si `killpg` falla con ESRCH pero el proceso NO ha terminado
    # (proceso.returncode is None, o sea nadie lo ha recolectado todavía), eso
    # no es "el proceso ya había terminado": es un kill roto. Si la función se
    # lo tragara en silencio, el `proceso.wait()` de más abajo esperaría a que
    # el hijo terminara SOLO, sin límite.
    proceso = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert proceso.returncode is None

        def killpg_roto(pid, sig):
            raise ProcessLookupError("simulado: PID reciclado o sin permisos")

        monkeypatch.setattr(os, "killpg", killpg_roto)
        with pytest.raises(ProcessLookupError):
            _matar_arbol(proceso)
    finally:
        proceso.kill()
        proceso.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="_matar_arbol usa killpg en POSIX")
def test_matar_arbol_no_relanza_si_el_proceso_ya_fue_recolectado(monkeypatch):
    # Direccion opuesta a la prueba anterior: si `proceso.returncode` ya no es
    # None (se recolecto en otro punto, p.ej. una carrera legitima), un
    # ProcessLookupError de killpg SI es el caso esperado y no debe propagarse.
    proceso = subprocess.Popen([sys.executable, "-c", "pass"])
    proceso.wait()
    assert proceso.returncode is not None

    def killpg_roto(pid, sig):
        raise ProcessLookupError("simulado: el grupo ya no existe")

    monkeypatch.setattr(os, "killpg", killpg_roto)
    _matar_arbol(proceso)  # no debe lanzar


# --- _matar_arbol: rama Windows (simulada, sin Windows real) -------------------


def test_matar_arbol_windows_sin_taskkill_en_el_path_usa_el_respaldo(monkeypatch):
    # Si `taskkill` no está en el PATH, `subprocess.run(["taskkill", ...])`
    # lanza FileNotFoundError ANTES de llegar al chequeo de
    # `resultado.returncode`; si esa excepción se escapara sin caer al
    # respaldo `proceso.kill()`, el árbol quedaría vivo. La prueba simula
    # Windows sin Windows real (este entorno es Linux/WSL): fuerza el flag
    # _ES_WINDOWS del módulo y hace que subprocess.run (la única llamada que
    # _matar_arbol hace a taskkill) lance FileNotFoundError, igual que si el
    # ejecutable no existiera.
    monkeypatch.setattr(subproceso_mod, "_ES_WINDOWS", True)

    def run_sin_taskkill(*args, **kwargs):
        raise FileNotFoundError("[WinError 2] no se encontro taskkill")

    monkeypatch.setattr(subprocess, "run", run_sin_taskkill)

    proceso = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert proceso.poll() is None
        _matar_arbol(proceso)
        assert proceso.poll() is not None, "el respaldo proceso.kill() no mato al proceso"
    finally:
        if proceso.poll() is None:
            proceso.kill()
            proceso.wait()


def test_matar_arbol_windows_respaldo_agotado_da_runtimeerror_encadenado(monkeypatch):
    # Direccion complementaria: si ni siquiera el respaldo logra matar al
    # proceso a tiempo, debe ser un RuntimeError propio (no un
    # TimeoutExpired crudo) que menciona la causa original de taskkill.
    monkeypatch.setattr(subproceso_mod, "_ES_WINDOWS", True)

    def run_sin_taskkill(*args, **kwargs):
        raise FileNotFoundError("[WinError 2] no se encontro taskkill")

    monkeypatch.setattr(subprocess, "run", run_sin_taskkill)

    proceso = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:

        def kill_que_no_mata():
            pass  # simula un kill() que no logra terminar al proceso

        def wait_que_agota_el_plazo(timeout=None):
            raise subprocess.TimeoutExpired("kill_que_no_mata", timeout)

        monkeypatch.setattr(proceso, "kill", kill_que_no_mata)
        monkeypatch.setattr(proceso, "wait", wait_que_agota_el_plazo)
        with pytest.raises(RuntimeError, match="taskkill no está disponible"):
            _matar_arbol(proceso)
    finally:
        monkeypatch.undo()
        if proceso.poll() is None:
            proceso.kill()
            proceso.wait()
