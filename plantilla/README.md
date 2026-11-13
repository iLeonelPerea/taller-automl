# Adaptar tu `train.py`

Cuatro cambios: envuelve tus hiperparámetros en `DEFAULTS`/`get_config` con
la clave `'dispositivo'` obligatoria y entrena en ese dispositivo (PyTorch
con `.to(dispositivo)`, XGBoost con su parámetro `device`, u otra biblioteca
que use la tarjeta de video); expón `entrenar`; expón `main` con
`--config`/`--salida`; y agrega la guarda `if __name__ == "__main__":`.
`train_ejemplo.py` es el resultado final; `check_importable.py` revisa que
los cumpliste.

## Requisito: una tarjeta de video NVIDIA con CUDA

El taller entrena siempre en la tarjeta de video. `train_ejemplo.py`
(y por lo tanto `buscar_en_proceso.py`, `buscar_por_subproceso.py` y todo
`pasos/07_conecta_tu_modelo.py`) necesita PyTorch con CUDA, que ya es una
dependencia fija del repo (ver `pyproject.toml`): `uv sync --locked` deja el
entorno listo. Sin una tarjeta de video NVIDIA con CUDA y el controlador
instalado, cualquiera de estos scripts se detiene con un mensaje claro.

## 1. De constantes de módulo a `get_config(defaults, **overrides)`

Si tus hiperparámetros son constantes (`LR = 1e-3`, `ANCHO = 64`), quedan
congelados al importar el módulo y ninguna búsqueda los puede variar. La
alternativa: un dict `DEFAULTS` y una función que construye la configuración de
cada corrida, con una clave mal escrita como error y no como valor ignorado en
silencio.

```python
DEFAULTS = {"ancho": 64, "learning_rate_init": 1e-3}


def get_config(defaults, **overrides):
    desconocidas = sorted(set(overrides) - set(defaults))
    if desconocidas:
        raise KeyError(f"claves desconocidas: {desconocidas}")
    return {**defaults, **overrides}
```

`DEFAULTS` debe declarar además `'dispositivo'`, sea cual sea la biblioteca de
tu modelo: el taller entrena siempre en la tarjeta de video (ver "Requisito"
arriba), así que la clave es obligatoria para cualquier `train.py`.
`buscar_en_proceso.buscar()` (la ruta que sí importa tu módulo, ver más
abajo) falla de entrada, con un mensaje claro, si `DEFAULTS` no la trae;
`check_importable.py` también revisa que su valor tenga la forma `'auto'` o
`'cuda'` (opcionalmente con `':N'`). Con PyTorch, resuélvela como hace
`train_ejemplo.py` (ver la sección siguiente); con XGBoost, con su propio
parámetro `device`; con otra biblioteca, de forma equivalente, siempre
hacia la tarjeta de video.

## 2. `entrenar(config, semillas, reportar=None) -> dict`

- `config`: el dict que arma `get_config`.
- `semillas`: un objeto con `.datos`, `.orden` e `.init`, con acceso por
  atributo y no por clave (así acepta sin cambios el `Semillas` real de
  `taller_hpo.semillas` cuando lo llama `en_proceso`). Una fuente de azar por
  separado, no una sola semilla para todo: si no puedes separar cuál varió
  qué, no puedes saber si tu resultado depende de la inicialización o del
  reparto de los datos. Si no quieres importar `taller_hpo` en tu script
  (ver `train_ejemplo.py`), define tu propia tupla con nombre equivalente:

  ```python
  from collections import namedtuple

  Semillas = namedtuple("Semillas", ["datos", "orden", "init"])
  ```
- `reportar(epoca, valor) -> bool`, opcional: llámalo al final de cada época.
  Si devuelve `True`, para el entrenamiento. Así la poda de Optuna vive fuera
  de tu modelo: `entrenar` nunca importa `optuna`.
- Devuelve un dict con solo métricas de **validación**, nunca la pérdida de
  entrenamiento cuando el tipo de pérdida es en sí un hiperparámetro (una
  búsqueda que puede elegir su propia vara de medir hace trampa).

### Si tu modelo es de PyTorch: una semilla por fuente, nunca el generador aleatorio global

`train_ejemplo.py` (ver el código, es el ejemplo de esta sección) entrena un
MLP de PyTorch para parecerse al modelo real de un tesista y para que todo el
entrenamiento corra en la tarjeta de video. Dos cambios más, además de
declarar `'dispositivo'` en `DEFAULTS` (ver el punto 1):

- **Los pesos iniciales salen de un `torch.Generator` propio**, sembrado con
  `semillas.init`, construido fuera de la tarjeta de video y movido ahí
  después (`torch.empty(...).uniform_(-b, b, generator=tu_generador)`, nunca
  `nn.Linear(...)` a secas: su `reset_parameters` consume el generador
  global de torch). Lo mismo para el orden de los minibatches: un
  `numpy.random.default_rng(semillas.orden)` propio, con `.permutation()` en
  cada época (no hace falta `DataLoader` si los datos ya viven enteros en el
  dispositivo). **Nunca `torch.manual_seed`** ni un `numpy.random.default_rng`
  compartido entre llamadas: si dos `entrenar()` con semillas distintas
  comparten generador, el resultado de la segunda depende de cuántos números
  consumió la primera, y dejas de poder reproducir una corrida sola.
- **Los datos viven en el dispositivo** (`.to(dispositivo)`) desde antes de
  empezar a entrenar, así que no los copias a la tarjeta de video en cada
  minibatch.

`train_ejemplo._resolver_dispositivo` resuelve `"auto"` siempre a `"cuda"`: el
taller entrena siempre ahí (ver "Requisito" arriba). Sin una tarjeta de video
NVIDIA con CUDA visible, se detiene con un mensaje claro. Una búsqueda puede
fijar el dispositivo a `"cuda"` (o `"cuda:N"`) a propósito si hace falta
elegir una tarjeta en particular.

### Por qué no basta `nn.Linear(...)` a secas

`Linear.reset_parameters` consume el generador global de torch para su
`kaiming_uniform_`, aunque el valor se descarte después. `train_ejemplo.py`
arma los pesos a mano con `torch.empty(...).uniform_(-b, b, generator=...)`,
que acepta un `torch.Generator` propio, con `b = 1/sqrt(fan_in)`: la misma
fórmula que usa `Linear.reset_parameters` (`kaiming_uniform_(weight,
a=sqrt(5))` equivale a `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`, y el sesgo
usa el mismo límite). Copia la fórmula, no el mecanismo, para conservar la
escala de init estándar de PyTorch. Arma los pesos fuera de la tarjeta de
video (un `torch.Generator()` sin argumentos no vive ahí) y mueve el módulo
completo después con `.to(dispositivo)`.

`taller_hpo.problemas.paridad_torch` resuelve lo mismo de otra forma: construye
las capas con `torch.nn.utils.skip_init` y saca los pesos de un
`numpy.random.RandomState(semillas.init)` propio.

### Determinismo entre procesos (en proceso y por subproceso)

`pasos/07_conecta_tu_modelo.py` (sección 2) y
`tests/test_plantilla.py::test_train_ejemplo_funciona_con_en_proceso_y_coincide_con_por_subproceso`
exigen que la misma config y las mismas semillas den el mismo `exactitud_val`
si llamas a `entrenar` directo o por `--config`/`--salida` en un proceso
nuevo. Ninguna fuente de azar depende del estado del proceso, y para una red
tan chica las operaciones que usa resultaron deterministas sin
`torch.use_deterministic_algorithms`. Si amplías la red (más capas, otros
optimizadores) y el determinismo deja de cumplirse, esa bandera, más
`CUBLAS_WORKSPACE_CONFIG`, es el primer lugar a revisar.

## 3. `main(argv=None)` con `--config`/`--salida` en JSON

Necesario para `taller_hpo.subproceso.por_subproceso`, que corre tu script como
proceso aparte (otro entorno, otro intérprete, o simplemente aislar un crash
del estudio completo). Lee `{"config": ..., "semillas": ...}` del JSON de
entrada y escribe el dict de métricas de `entrenar` como JSON de salida.

```python
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        entrada = json.load(f)

    # El JSON trae "semillas" como {"datos": ..., "orden": ..., "init": ...}:
    # reconstruye el objeto antes de llamar a entrenar (ver punto 2).
    metricas = entrenar(entrada["config"], Semillas(**entrada["semillas"]))

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(metricas, f)
```

Si solo vas a usar `en_proceso` (ver más abajo) todavía conviene escribir
`main`: es lo que te deja probar tu script suelto, sin Optuna de por medio, y
lo que revisa `check_importable.py`.

## 4. La guarda `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

Sin ella, `main()` se ejecuta al *importar* el archivo, no solo al correrlo.
Eso rompe `en_proceso` (que importa tu módulo) y confunde a
`check_importable.py` (que también lo importa). En Windows y macOS es además
obligatoria por cómo `multiprocessing` arranca procesos con `spawn`: cada
proceso hijo vuelve a importar el script que lo lanzó y, sin la guarda,
relanzaría el programa completo (con PyTorch, también reinicializaría CUDA).

## Revisar con `check_importable.py`

```bash
/ruta/a/tu/venv/bin/python plantilla/check_importable.py ruta/a/tu/train.py
```

Corre `check_importable.py` con el **python del entorno de tu modelo** (el
mismo con el que correrías `train.py`), no necesariamente el de este repo: el
script importa tu `train.py` con ese intérprete, así que si tu modelo usa una
librería (por ejemplo `jax`) que este repo no tiene instalada, correrlo con el
python de este repo reporta `ModuleNotFoundError` aunque tu script esté bien.

- Código de salida `0`: tu script es importable, define `entrenar` y `main`, y
  declara `DEFAULTS['dispositivo']` con un valor válido (`'auto'` o `'cuda'`,
  opcionalmente con `':N'`).
- Código de salida `1`: algo falló; el mensaje en stderr dice exactamente qué
  (`parse_args()` al importar, trabajo pesado a nivel de módulo, falta
  `entrenar`/`main`, o `DEFAULTS['dispositivo']` ausente o con un valor
  inválido).

## `en_proceso` o `por_subproceso`

- **`en_proceso`** (`taller_hpo.objetivo.en_proceso`): llama a tu `entrenar`
  directo, en el mismo proceso que Optuna. Más simple, sin el costo de
  arrancar un proceso por trial, y es la única forma de usar `reportar` para
  podar de verdad (`trial.report`/`should_prune` viven en ese mismo proceso).
  Exige que tu modelo corra con el mismo intérprete y las mismas dependencias
  que este repo.
- **`por_subproceso`** (`taller_hpo.subproceso.por_subproceso`): arranca
  `comando` (tu `python` de otro venv, u otro ejecutable) por cada trial y le
  pasa `--config`/`--salida`. Necesario cuando tu modelo vive en otro entorno u
  otra versión de una librería. Por sí solo no aísla el estudio: un trial que
  revienta (`RuntimeError`), que se cuelga (`TimeoutError`, hay `timeout_s`) o
  que devuelve una métrica no finita (`MetricaNoFinita`) levanta esa
  excepción y, sin atraparla, corta la búsqueda. Quien orquesta el estudio
  decide cuáles marcan el trial como FAIL, con `study.optimize(...,
  catch=(...))` o, mejor, con `taller_hpo.estudio.optimizar(...,
  atrapar=(...))`. El aislamiento completo pide las tres:

  ```python
  from taller_hpo.objetivo import MetricaNoFinita

  atrapar = (RuntimeError, TimeoutError, MetricaNoFinita)
  ```

  `KeyError` (tu `train.py` no escribió la métrica que le pediste en
  `metricas`) queda fuera de `atrapar` a propósito: es un error del script, no
  un trial malo, y debe detener la búsqueda para que lo notes. Sin poda, y más
  lento por trial por el costo de arrancar el proceso.

`TALLER_DISPOSITIVO=cuda(:N)` elige qué tarjeta de video usa cada trial de
`por_subproceso` (`buscar_por_subproceso.py::_resolver_dispositivo`); un
`dispositivo=` explícito distinto de `"auto"` gana sobre la variable, y un
valor fuera de `auto|cuda(:N)` es un `ValueError`. Antes de crear el estudio,
el script lanza el mismo intérprete que va a entrenar para preguntarle si ve
una tarjeta de video (`buscar_por_subproceso.py::_detectar_dispositivo`) y se
detiene con un `RuntimeError` si no ve ninguna. Esa comprobación no valida el
índice: con dos tarjetas visibles, `cuda:7` pasa igual que `cuda:0`, y el
error llega en el primer trial. El subproceso recibe el valor con su índice;
el protocolo del estudio se lo quita (`cuda:3` -> `cuda`), igual que
`taller_hpo.problemas.juguete.PROTOCOLO_MOTOR`.

Por defecto, usa `en_proceso`. Cambia a `por_subproceso` solo cuando la razón
de aislar el entorno sea real, no "por si acaso": **cada trial de
`por_subproceso` es un proceso nuevo que paga arrancar Python, `import torch`,
el primer `torch.optim.Adam(...)` (que importa `torch._dynamo` de forma
perezosa), crear el contexto de la tarjeta de video y cargar cuBLAS.** Cuánto
pesa depende de la carga de tu máquina, así que este archivo no da una cifra:
el paso 07 (sección 3) mide el sobrecosto fijo de `por_subproceso` frente a
`en_proceso` con una prueba pareada y da su propio veredicto.
