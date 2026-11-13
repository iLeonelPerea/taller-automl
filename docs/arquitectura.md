# Arquitectura del taller

El repo tiene tres capas:

1. **Tu modelo** cumple un contrato pequeño y no importa ni Optuna ni `taller_hpo`.
2. **El núcleo** (`src/taller_hpo/`) son funciones delgadas sobre la API pública de Optuna. No hay clases que la envuelvan: reciben y devuelven `Study` y `Trial` reales, así que lo que aprendes aquí sirve con la documentación oficial de Optuna.
3. **Los pasos** (`pasos/NN_*.py`) son scripts ejecutables sobre un problema de juguete con oráculo. Cada uno contesta preguntas concretas e imprime la respuesta con números.

## El contrato de tu modelo

```python
def entrenar(config: dict, semillas: Semillas, reportar=None) -> dict[str, float]
```

- `config`: los hiperparámetros. Una clave desconocida es un error, no un valor ignorado (`get_config`).
- `semillas`: `Semillas(datos, orden, init)`, una semilla por cada fuente de azar.
- `reportar(epoca, valor) -> bool`: se llama al final de cada época. Si devuelve `True`, el entrenamiento para. La poda de Optuna vive fuera de tu modelo.
- Devuelve solo métricas de **validación**, externas y comparables entre configuraciones. Nunca la pérdida de entrenamiento cuando el tipo de pérdida es un hiperparámetro.

## Convenciones (todas obligatorias)

- Identificadores propios en español ASCII (`crear_estudio`, `reevaluar`). La API de Optuna se queda en inglés. Documentación y mensajes en español.
- Nada de `# noqa`, `# type: ignore`, `|| true`, `except: pass`, ni `except Exception` que se trague errores. Si ruff se queja, se arregla la causa.
- ruff: `line-length = 100`, reglas `E, F, W, I, B, UP`.
- Gráficas con `matplotlib.figure.Figure` (nunca `pyplot`): guardan PNG sin backend.
- Todo azar pasa por `Semillas` o por un `numpy.random.Generator` con semilla explícita. Nada de generador aleatorio global.
- `n_jobs=1` en la ruta principal. El paralelismo solo existe en el anexo.

### Interpretaciones y cifras

- Una frase que interpreta un resultado ("es más lento", "está inflado", "se distingue") sale de
  una prueba medida en esa misma corrida (`taller_hpo.estadistica.diferencia_pareada` o un intervalo
  de confianza), con rama "no concluyente" cuando el intervalo cruza el umbral; su texto cambia según
  el resultado. Un dato descriptivo ("tardó X s") no necesita prueba; interpretarlo sí.
- Cada conclusión impresa tiene una prueba que falla con un cambio real que la invierte o la vacía.
- Ninguna cifra de tiempo en código, docstrings ni markdown de los pasos: los tiempos viven solo en
  la tabla `<!-- TIEMPOS:INICIO -->`/`<!-- TIEMPOS:FIN -->` del README raíz. En el código solo van
  afirmaciones cualitativas que no dependan de la carga de la máquina.
- El oráculo (`exactitud_verdadera`/`riesgo_verdadero`) solo revela la verdad: nunca es objetivo de
  la búsqueda ni criterio para elegir entre configuraciones.

### Pasos

- Nombre: `pasos/NN_nombre.py`, con celdas `# %%` y `# %% [markdown]`.
- `PRESUPUESTO = humo | taller | completo` (variable de entorno, default `taller`). Si llega otro valor: `SystemExit("PRESUPUESTO debe ser uno de [...]...")`.
  - `humo`: 20 s o menos (lo corre la suite de pruebas).
  - `taller`: 3 min o menos en la máquina de referencia.
- `TALLER_RESULTADOS` (default `<repo>/resultados`): cada paso guarda ahí al menos un PNG.
- `TALLER_ESTUDIOS` (default `<repo>/estudios`): archivos de estudio de Optuna.
- Cada paso imprime sus respuestas numeradas (`1. ...`, `2. ...`) con números medidos.
- `tests/test_pasos.py` corre cada `pasos/[0-9][0-9]_*.py` con `PRESUPUESTO=humo` una vez por módulo
  (`corridas_cache`, con `scope="module"`) y exige código de salida 0, un PNG y stderr sin
  `"Warning"`. `TALLER_ESTUDIOS` apunta a un directorio propio de cada corrida, para que una corrida
  no reabra el estudio de otra. Los pasos que entrenan en la tarjeta de video (`PASOS_CON_GPU_REAL`)
  corren con un lanzador que al final imprime `torch.cuda.max_memory_allocated()`: así se ancla el
  uso real de la tarjeta, no solo el texto de `DESCRIPCION`. Esas pruebas llevan la marca `gpu`; el
  único paso sin ella es el 01, que no entrena nada. `PREFIJOS_MOTOR` y `EXENTOS` clasifican cada
  paso (con qué prefijo imprime la `DESCRIPCION` del juguete, o por qué está exento), y
  `test_todos_los_pasos_estan_clasificados` falla si un paso nuevo no queda en ninguno.
  `test_pasos_con_juguete_no_entrenan_via_mlp_paridad_directo` revisa con `ast` que ningún paso
  importe `mlp_paridad` o `paridad_torch` por su nombre.

### Pruebas

- `tests/test_<modulo>.py`, con semillas fijas y sin pruebas intermitentes.
- Cada afirmación estadística se prueba en las dos direcciones: pasa cuando debe y falla cuando debe.
- Toda prueba que necesita la tarjeta de video lleva `@pytest.mark.gpu` (o `pytestmark` a nivel de
  módulo). `tests/conftest.py` registra la marca y, sin una tarjeta de video visible, hace FALLAR
  esas pruebas con un mensaje claro -- nunca las salta. La integración continua
  (`.github/workflows/pruebas.yml`) corre `pytest -m "not gpu"` porque sus runners no tienen tarjeta
  de video.
- `tests/test_conftest_gpu.py` comprueba, en un mini-proyecto aparte, que una prueba marcada `gpu`
  sin tarjeta de video visible FALLA (nunca se salta), y que una prueba sin la marca no se confunde
  con una que sí la necesita aunque su id de parametrize se llame exactamente `"gpu"`.
  `tests/test_sin_tarjeta.py`, sin la marca `gpu` (fuerza `CUDA_VISIBLE_DEVICES=""` en cada
  subproceso propio, así que corre igual con o sin tarjeta real), comprueba que sin tarjeta de video
  cada paso que entrena y el anexo terminan con `MENSAJE_SIN_TARJETA` y sin escribir ningún archivo en
  `TALLER_ESTUDIOS` ni `TALLER_RESULTADOS`.
- `tests/test_redaccion.py` vigila la redacción que ninguna otra prueba lee: las tildes de
  `estudio.py`, `objetivo.py` y `subproceso.py`, el nombre en español de la maldición del ganador, la
  jerga en inglés que ya tiene palabra en español, las siglas que cada paso expande la primera vez
  que las nombra, los mensajes de los dos scripts de `scripts/`, las etiquetas de etapa del plan de
  trabajo y los casos de `ValueError` de `_detectar_dispositivo` que este documento enumera. Lee
  archivos y no entrena nada, así que corre igual con o sin tarjeta de video.

## Módulos

### `taller_hpo.tarjeta_de_video`
`MENSAJE_SIN_TARJETA`: la frase única que un tesista ve cuando su máquina no tiene una tarjeta de
video NVIDIA con CUDA. Este módulo no importa `torch`, así que la suite puede comparar el mensaje
aunque la máquina no tenga tarjeta de video. Cada punto que exige la tarjeta de video toma el mensaje
de aquí en vez de escribir su propia frase: `taller_hpo.problemas.juguete` (al importarse),
`cash_paridad.entrenar`, `paridad_torch.entrenar`, `plantilla/buscar_en_proceso.py`,
`plantilla/buscar_por_subproceso.py`, `pasos/anexo_paralelo.py` y `tests/conftest.py` (para las
pruebas marcadas `gpu`). `plantilla/train_ejemplo.py` conserva una copia idéntica del mensaje, porque
el tesista lo copia sin `taller_hpo`, y `tests/test_plantilla.py` exige que esa copia coincida con la
fuente única. Un tesista lee siempre el mismo texto, sea cual sea el punto que lo detuvo.

`taller_hpo.problemas.juguete` (al importarse) y las dos plantillas
(`plantilla/buscar_en_proceso.py::_resolver_dispositivo_auto`,
`plantilla/buscar_por_subproceso.py::_detectar_dispositivo`) lanzan `RuntimeError(MENSAJE_SIN_TARJETA)`,
porque un módulo de biblioteca no debe terminar el intérprete al importarse: `SystemExit` es para el
punto de entrada de un programa, no para código que otro módulo importa; las dos plantillas lanzan
`ValueError` ante un valor de dispositivo con forma inválida (ni `"auto"` ni `"cuda(:N)?"`), y
`plantilla/buscar_por_subproceso.py::_detectar_dispositivo` lo lanza además cuando el taller no puede
lanzar el intérprete de `TALLER_PYTHON` (`OSError`) o cuando ese intérprete no puede importar `torch`
(código de salida distinto de 0): en esos dos casos falla el intérprete, no la tarjeta de video.
`cash_paridad.entrenar` y `paridad_torch.entrenar` tratan la falta de tarjeta de video como un valor inválido de
`config["dispositivo"]` -- incluido el default `"auto"`, que no es algo que el tesista haya pedido
explícito -- y lanzan `ValueError(MENSAJE_SIN_TARJETA)`. `pasos/anexo_paralelo.py`, que sí es un punto
de entrada (un script que el tesista corre, no algo que otro código importa), lanza
`SystemExit(MENSAJE_SIN_TARJETA)`. Estas pruebas lo fijan:
`tests/test_juguete.py::test_subproceso_sin_gpu_visible_falla_al_importar` y, en la plantilla,
`tests/test_plantilla_busqueda.py::test_resolver_dispositivo_auto_sin_tarjeta_de_video_falla_con_mensaje_claro`
y `::test_buscar_dispositivo_auto_falla_claro_si_el_interprete_hijo_no_ve_la_tarjeta`.

### `taller_hpo.configuracion`
`get_config(defaults, **overrides) -> dict`: KeyError con sugerencia ante claves desconocidas y TypeError ante tipos incompatibles.

### `taller_hpo.semillas`
`Semillas(datos, orden, init)` (frozen). `semillas_de(base, n, *, datos=None) -> list[Semillas]` con `numpy.SeedSequence`.

### `taller_hpo.problemas.paridad`
`generar`, `error_bayes`, `posterior`, `riesgo_verdadero`: la paridad gaussiana y su oráculo.
`riesgo_verdadero` toma `x` frescas y su posterior de `_muestras_oraculo`, cacheada con
`functools.lru_cache(maxsize=8, typed=True)` por `(n, semilla, d, a, k_ruido, rotar)`: con
`SEMILLA_ORACULO` fija, todos los modelos de una misma corrida se evalúan sobre las MISMAS `x` sin
regenerarlas. `typed=True` evita que `n=100` y `n=100.0` compartan entrada (`generar` valida tipos;
sin `typed`, el segundo cache-hit se saltaría esa validación). Los arreglos que devuelve son los
mismos que guarda la cache: no se mutan.

### `taller_hpo.problemas.mlp_contrato`, el contrato del MLP del juguete, sin scikit-learn
`DEFAULTS`, `Divergencia`, `SEMILLA_ORACULO` y `SEMILLA_PRUEBA`: el problema, los hiperparámetros por
defecto y el error de divergencia que comparten `mlp_paridad` (el modelo de referencia, scikit-learn)
y `paridad_torch` (el motor real, en la tarjeta de video), para que la comparación entre los dos sea
directa. Este módulo no importa scikit-learn ni threadpoolctl: `paridad_torch` toma de aquí
`DEFAULTS`, `Divergencia`, `SEMILLA_ORACULO` y `SEMILLA_PRUEBA`. `taller_hpo.problemas.juguete` toma
`DEFAULTS` vía `paridad_torch`, y `Divergencia`, `SEMILLA_ORACULO` y `SEMILLA_PRUEBA` directo de aquí,
igual que `taller_hpo.problemas.cash_paridad` -- las dos corren en la tarjeta de video y ninguna de las
dos toma esos nombres de `mlp_paridad`: el motor de la tarjeta de video no depende de su modelo de
referencia.
`mlp_paridad` reexpone estos cuatro nombres para las pruebas. `tests/test_contrato_mlp.py` comprueba,
en un subproceso limpio, que importar `juguete`/`paridad_torch`/`cash_paridad` no carga `sklearn`; al
entrenar con `algoritmo` en `("bosque", "hgb")`, `cash_paridad` sí lo carga (ver su sección más abajo).

### `taller_hpo.problemas.mlp_paridad`
`DEFAULTS`, `entrenar(config, semillas, reportar=None)`, `Divergencia`, `SEMILLA_ORACULO`, `SEMILLA_PRUEBA`.

`mlp_paridad` es el modelo de referencia contra el que las pruebas comprueban que la tarjeta de video
da los resultados correctos: ningún paso entrena con él. Los pasos que entrenan el MLP del juguete lo
hacen con `paridad_torch` (vía `taller_hpo.problemas.juguete`, más abajo); el paso 05 entrena con
`cash_paridad` (que para `algoritmo="mlp"` también reusa `paridad_torch.entrenar_red`, y para
`bosque`/`hgb` usa XGBoost, ver su sección más abajo) y el paso 07 entrena con `plantilla/train_ejemplo.py`, un problema
distinto al juguete. `mlp_paridad` cumple el mismo contrato que el taller le exige al modelo del
tesista, para que la comparación sea directa.
- **Métricas:** `exactitud_val`, `exactitud_verdadera` (el oráculo; NUNCA se optimiza), `n_parametros`, `epocas`, `segundos`.
- **Prueba finita:** con `n_prueba > 0` agrega `exactitud_prueba`, un conjunto de prueba finito con semilla propia (lo que tendría un tesista sin oráculo; se abre una vez con `PruebaSellada`).
- **Placebo:** incluye la perilla `placebo`, que el modelo no lee.
- **Problema por defecto:** d=3, a=1.5, 10 columnas de ruido, 2000/500 datos. Techo 0.825; línea
  base con `semillas_de(0, 10)` sobre `exactitud_verdadera`: media 0.7024, de 0.0318, rango
  0.6283-0.7248 (idéntico entre sklearn y torch).
- `paridad_torch` hereda los mismos DEFAULTS y respeta `n_prueba` igual.
- Todo el cómputo numérico (`partial_fit` por época, `score`, el `predict` del oráculo) corre dentro
  de un límite de 1 hilo BLAS/OpenMP (`threadpoolctl.ThreadpoolController` cacheado con
  `functools.cache`, para no re-escanear las librerías cargadas en cada llamada; al salir se
  restauran los límites previos): `mlp_paridad`, el modelo de referencia de las pruebas, limita así
  sus hilos de cálculo para que los procesos de la suite no compitan entre sí por ellos.

### `taller_hpo.problemas.juguete`, el único punto de entrada al juguete, siempre en la tarjeta de video
Este módulo reexporta el contrato de `paridad_torch` (delega internamente en él), para que los pasos
entrenen el juguete sin repetir cada uno por su cuenta la comprobación de la tarjeta de video:

```python
entrenar(config, semillas, reportar=None) -> dict de métricas   # delega en paridad_torch.entrenar
```

- Al importar este módulo, si `torch.cuda.is_available()` es falso, la importación falla de entrada
  con `RuntimeError(MENSAJE_SIN_TARJETA)` (ver `taller_hpo.tarjeta_de_video`).
- `MOTOR` vale siempre `"torch"` y `DISPOSITIVO` siempre `"cuda"`: son constantes fijadas al importar,
  una vez que la comprobación de arriba ya pasó.
- `DEFAULTS`, `DESCRIPCION` (la línea que cada paso imprime al arrancar, con la versión de PyTorch y
  el nombre de la tarjeta de video) y `PROTOCOLO_MOTOR` (`{"motor": "torch", "dispositivo": "cuda"}`,
  con cualquier `"cuda:N"` normalizado a `"cuda"`) se calculan al importar, una sola vez.
- `PROTOCOLO_MOTOR` entra al `protocolo` que guarda `taller_hpo.estudio.crear_estudio`: reabrir un
  estudio con un protocolo distinto es un `ValueError`. Cambiar de `cuda:0` a `cuda:1` en la misma
  máquina no rompe la reapertura, porque el índice de tarjeta se normaliza antes de entrar al
  protocolo.
- Los pasos que entrenan el MLP del juguete (00, 02, 03, 04, 06 y el anexo) entrenan vía este
  módulo e imprimen `DESCRIPCION` al arrancar (el anexo, como `"Motor: ..."`, una vez por worker).
  El 07 lo importa solo para imprimir `DESCRIPCION` y heredar la comprobación de la tarjeta: entrena
  `plantilla/train_ejemplo.py`, un problema distinto. El 01 no entrena el juguete (usa
  `taller_hpo.problemas.analitico`) y el 05 resuelve su propio dispositivo vía `cash_paridad`,
  aunque importa este módulo para `_frase_tamano_problema` y así también hereda la comprobación.
- `HIPERPARAMETROS` (tupla `("n_capas", "ancho", "activacion", "learning_rate_init", "alpha",
  "batch_size")`): las claves del espacio de búsqueda del MLP del juguete, para que un paso las
  recorra sin repetirlas a mano. El módulo también reexporta `Divergencia`, `SEMILLA_ORACULO` y
  `SEMILLA_PRUEBA`, que trae de `taller_hpo.problemas.mlp_contrato` (ver más arriba), así un paso que
  solo necesita el contrato del juguete no importa `mlp_paridad` (que sí carga scikit-learn) para
  nada.

### `taller_hpo.estadistica`, numpy + scipy, sin Optuna

```python
@dataclass(frozen=True)
class Resumen:  n: int; media: float; de: float; ee: float; ic_bajo: float; ic_alto: float; confianza: float
def resumen(valores, *, confianza=0.95) -> Resumen            # intervalo t de Student

@dataclass(frozen=True)
class Pareado:
    n: int; diferencia_media: float; de_dif: float; ic_bajo: float; ic_alto: float
    p_valor: float                 # t pareada, bilateral
    wilcoxon_p: float | None       # solo si hay >= 6 diferencias no nulas; si no, None
    rho: float                     # correlación entre a y b (lo que gana el emparejamiento)
    veredicto: str                 # "mejor" | "peor" | "equivalente" | "no concluyente"
    delta: float; direccion: str; confianza: float
def diferencia_pareada(a: Mapping[int, float], b: Mapping[int, float], *, delta: float,
                       direccion: str = "maximizar", confianza: float = 0.95) -> Pareado
    # confianza en (0.5, 1): las dos pruebas unilaterales de equivalencia (TOST) usan el nivel
    # 1 - 2*alfa, que debe ser > 0.
    # Empareja por clave (la semilla). Claves distintas -> ValueError. Diferencia = a - b
    # (con "minimizar" se invierte el signo para que "mejor" siempre signifique que a gana).
    # Veredicto: "mejor" si el intervalo entero está sobre 0; "peor" si está bajo 0;
    # "equivalente" si el intervalo de nivel (1 - 2*alfa) de TOST cae dentro de (-delta, +delta);
    # si no, "no concluyente". El veredicto NUNCA sale de Wilcoxon: con n=5, Wilcoxon exacto
    # no puede bajar de p = 0.0625.
def semillas_necesarias(de_dif: float, delta: float, *, alfa=0.05, potencia=0.8) -> int
    # n mínimo (>= 2) de la t pareada bilateral, calculado con la t no central (scipy.stats.nct).
    # Parte de la aproximación normal (z_{1-alfa/2} + z_potencia)^2 · (de_dif/delta)^2, que con
    # los defaults es 7.8489·r², y la corrige por la t. Con r = de_dif/delta menor que ~41 queda
    # >= ceil(7.85·r²). Más arriba, el redondeo de 7.85 pesa más que la corrección de la t y el n
    # exacto (el mínimo correcto) puede quedar unas semillas por debajo.
    # Dominio: alfa en [1e-3, 0.5] (ValueError fuera). Con alfas menores, scipy.stats.nct no
    # converge con pocas semillas; si aun así avisa que no convergió -> FloatingPointError.
def ic_proporcion(k: int, n: int, *, metodo="wilson", confianza=0.95) -> tuple[float, float]
    # scipy.stats.binomtest(k, n).proportion_ci. Nunca Wald: con k=0 da un intervalo de ancho cero.
def maximo_esperado_normal(t: int) -> float    # E[max de t N(0,1) iid]; 1.539 / 2.043 / 2.508 / 3.241 con t = 10/30/100/1000
def holm(p_valores: Sequence[float]) -> list[float]   # p ajustados, monótonos, en el orden de entrada
```

### `taller_hpo.problemas.analitico`, el paisaje del paso 01

```python
@dataclass(frozen=True)
class Paisaje:  n_ejes: int; eje_importante: int; optimo: float     # invariantes validadas
    def regret(self, params: Mapping[str, float]) -> float          # (x[eje_importante] - optimo)**2
    def espacio(self) -> dict[str, tuple[float, float]]             # {"x0": (0, 1), ...}
def crear_paisaje(semilla: int, *, n_ejes: int = 3) -> Paisaje     # eje y óptimo al azar por réplica
def regret_minimo_grid(paisaje: Paisaje, niveles: Sequence[float]) -> float
```

Un solo eje importa y los demás no afectan nada, por construcción. Con 27 evaluaciones, una rejilla 3×3×3 prueba solo 3 valores del eje que importa, mientras que 27 puntos al azar prueban 27.

### `taller_hpo.estudio`, estudios de Optuna con guardas

```python
def abrir_storage(destino: str | os.PathLike) -> optuna.storages.BaseStorage | str
    # "postgresql://", "postgresql+psycopg://", "sqlite:///" -> la URL tal cual (RDBStorage).
    # Cualquier otra cosa es una ruta de archivo -> JournalStorage(JournalFileBackend(ruta,
    # lock_obj=JournalFileOpenLock(ruta))). Crea la carpeta padre. (OpenLock: el SymlinkLock
    # por defecto falla en Windows.)
def crear_estudio(nombre: str, *, destino, direcciones: Sequence[str], sampler, pruner,
                  protocolo: Mapping[str, Any]) -> optuna.Study
    # sampler y pruner: keyword-only y SIN default (Optuna 5 cambió los defaults; aquí se escriben siempre).
    # sampler: una instancia, o una FÁBRICA (n_previos: int) -> BaseSampler. Con la fábrica,
    #   reanudar es determinista y no repite configuraciones:
    #       sampler=lambda n: optuna.samplers.TPESampler(seed=BASE + n, ...)
    #   Con una instancia sembrada y un estudio que ya tiene trials, avisa (UserWarning): el sampler
    #   volvería a proponer las mismas configuraciones de la primera sesión.
    # protocolo: dict JSON que describe CÓMO se busca (sampler y sus kwargs, pruner, política de
    # semillas, métrica). Se guarda en study.user_attrs["protocolo"] junto con las versiones de
    # optuna y python. Al reabrir: protocolo o direcciones distintos -> ValueError que dice qué cambió.
    # Si el estudio existe sin protocolo y sin trials, lo registra (así N workers que lo crean a la
    # vez no chocan). Sin protocolo pero con trials -> ValueError.
def recuperar_huerfanos(study) -> list[int]
    # SOLO en modo secuencial. Los trials RUNNING de una corrida cortada pasan a FAIL y sus params se
    # reencolan con user_attrs {"reintento_de": number}. Devuelve los números. (JournalStorage no
    # tiene heartbeat: sin esto, un trial cortado queda RUNNING para siempre.)
def sembrar_linea_base(study, params: Mapping[str, Any],
                       distribuciones: Mapping[str, BaseDistribution]) -> bool
    # Valida contra `distribuciones`: claves exactas (ValueError si sobra o falta alguna) y valores
    # dentro del rango o de las opciones (ValueError; Optuna solo avisaría). Si ya hay un trial de
    # línea base COMPLETE, RUNNING o WAITING (user_attrs["rol"] == "linea_base") -> False y no hace
    # nada. Si no -> enqueue_trial(params, user_attrs={"rol": "linea_base"}) -> True.
    # (No se usa skip_if_exists: cuenta los FAIL y nunca reintentaría una línea base que falló.)
def optimizar(study, objetivo, *, n_total: int, atrapar: tuple[type[BaseException], ...] = (),
              max_fallos: int = 10) -> None
    # n_jobs=1. Corre hasta que COMPLETE + PRUNED >= n_total, contando lo que ya había (reanudar
    # no repite trabajo). Las excepciones de `atrapar` marcan el trial FAIL y la búsqueda sigue;
    # si los FAIL de esta llamada superan max_fallos -> RuntimeError. Las demás se propagan.
def importancia_con_piso(study, *, evaluador, placebo: str) -> dict[str, dict[str, float | bool]]
    # optuna.importance.get_param_importances(study, evaluator=evaluador). Devuelve
    # {param: {"importancia": v, "sobre_el_piso": v > importancia del placebo}}. El placebo es un
    # parámetro que el modelo no lee: su importancia es la que produce el puro ruido.
```

### `taller_hpo.objetivo`, tu `entrenar` como objective de Optuna

```python
class MetricaNoFinita(ValueError)
def en_proceso(entrenar, espacio, *, metricas: Sequence[str],
               semillas: Semillas | Callable[[optuna.Trial], Semillas],
               fijos: Mapping[str, Any] | None = None, atributos: Sequence[str] = ())
               -> Callable[[optuna.Trial], float | tuple[float, ...]]
    # config = {**fijos, **espacio(trial)}. `espacio(trial) -> dict` es el ÚNICO lugar con suggest_*.
    # user_attrs: "config" (la config completa) y "semillas" (dict).
    # Una métrica: reportar(epoca, v) hace trial.report(v, epoca) y devuelve trial.should_prune().
    #   Si entrenar regresa después de una poda -> raise optuna.TrialPruned (fuera de tu código).
    # Varias métricas: reportar=None (Optuna no poda en multiobjetivo: Trial.report lanza error).
    # Métrica ausente -> KeyError. Métrica no finita -> MetricaNoFinita.
    # `atributos`: otras métricas que se guardan como user_attrs y NUNCA se devuelven
    #   (p. ej. "exactitud_verdadera": el oráculo se registra, no se optimiza).
```

### `taller_hpo.subproceso`, tu `train.py` en su propio entorno

```python
def escribir_json_atomico(ruta, obj) -> None   # UTF-8, archivo temporal + os.replace
def leer_json(ruta) -> Any
def por_subproceso(comando: Sequence[str], espacio, *, metricas: Sequence[str],
                   semillas: Semillas | Callable[[optuna.Trial], Semillas],
                   timeout_s: float, directorio: str | os.PathLike, fijos=None,
                   atributos: Sequence[str] = ()) -> Callable[[optuna.Trial], float | tuple]
    # Por trial: escribe {"config": {...}, "semillas": {...}} en <directorio>/trial_<n>_entrada.json,
    # ejecuta [*comando, "--config", entrada, "--salida", salida] con PYTHONUTF8=1 y lee la salida
    # JSON {metrica: valor}. `comando` es arbitrario: el python de OTRO entorno, o cualquier
    # ejecutable que respete --config/--salida. Sin poda.
    # returncode != 0 -> RuntimeError con el final de stderr. Timeout -> TimeoutError.
    # JSON ausente o inválido (o no UTF-8) -> RuntimeError.
    # stdout y stderr de cada trial quedan en <directorio>/trial_<n>_stdout.txt y _stderr.txt.
    # El hijo corre en su propio grupo de procesos, con stdin cerrado: un timeout o una interrupción
    # matan el árbol completo (incluidos los nietos, como los workers de un DataLoader), y un
    # input() o breakpoint() olvidado no cuelga la búsqueda.
```

### `plantilla/`, lo que el tesista copia

Cualquier `train.py` que un tesista conecte debe declarar `DEFAULTS["dispositivo"]` (`"auto"`,
`"cuda"` o `"cuda:N"`): es la clave con la que la plantilla comprueba, antes de entrenar nada, que el
intérprete ve una tarjeta de video. `buscar_en_proceso.buscar()` falla de entrada, con un mensaje
claro, si `DEFAULTS` no la trae; `check_importable.py` además valida que su valor tenga esa forma.

- `train_ejemplo.py`: script **autónomo** (numpy y torch; no importa `taller_hpo` ni Optuna) con su propia copia corta de `get_config`, `DEFAULTS`, `entrenar(config, semillas, reportar=None)`, `main(argv=None)` con `--config/--salida` en JSON, y la guarda `if __name__ == "__main__":`. El script entrena con `"dispositivo": "auto"` (solo acepta `"auto"`, `"cuda"` o `"cuda:N"`; otro valor es `ValueError`, igual que en `paridad_torch`) y resuelve un problema distinto al juguete (dos lunas entrelazadas más columnas de ruido). El script usa la misma instalación fija del taller.
- `buscar_en_proceso.py` y `buscar_por_subproceso.py`: las dos rutas completas contra `train_ejemplo.py`. `buscar_por_subproceso.py` copia a mano, como fijos del problema y línea base, todas las claves de `train_ejemplo.DEFAULTS` salvo `dispositivo`; `tests/test_plantilla_busqueda.py` falla si esa copia diverge de las `DEFAULTS` reales.
- `check_importable.py`: `python plantilla/check_importable.py ruta/train.py` detecta `parse_args` al importar, trabajo pesado a nivel de módulo, si falta `entrenar` o `main`, y si `DEFAULTS["dispositivo"]` falta o no tiene la forma `"auto"`/`"cuda"` (opcionalmente con `":N"`).
- `README.md`: los cuatro cambios que necesita un `train.py` existente.

### `taller_hpo.problemas.paridad_torch`, el mismo juguete en PyTorch, con CUDA graphs
Espejo de `mlp_paridad` hasta el redondeo de punto flotante (misma inicialización de pesos, mismo
orden de minibatches, misma regularización L2, mismo Adam), no una red "parecida": `DEFAULTS` reusa
los de `taller_hpo.problemas.mlp_contrato` (los mismos que reexpone `mlp_paridad`) más
`"dispositivo": "auto"` y `"precision": "float64"`.

- Los datos (`x_tr, y_tr, x_va, y_va`) quedan residentes en el dispositivo desde antes de la primera
  época (`subir`); cada época solo sube el ORDEN de los minibatches, nunca vuelve a mover `x`/`y`.
- Los pesos iniciales salen de una `numpy.random.RandomState(semillas.init)` propia (la MISMA fuente
  y el mismo orden de sorteo que usa sklearn), y las capas se construyen con
  `torch.nn.utils.skip_init`: nunca toca el generador global de torch (nada de `torch.manual_seed` ni
  `torch.random.fork_rng`). El orden de los minibatches sale de `numpy.random.default_rng(semillas.orden)`,
  también propio. No hay `DataLoader`.
- Entrena siempre en `cuda`: `dispositivo` solo acepta `"auto"`, `"cuda"` o `"cuda:N"`; otro valor es
  un `ValueError` explícito. Un CUDA graph por ÉPOCA: la época 0 corre en eager (calienta cuBLAS), la época 1 se CAPTURA completa y las
  siguientes la REPRODUCEN con `.replay()`. El paso de Adam se escribe a mano con
  `torch._foreach_*` (nunca `torch.optim.Adam`: su fórmula no coincide con la de sklearn, y además
  paga un chequeo de acelerador en cada `step()`).
- API pública, reusable sin pasar por el contrato del taller:
  `entrenar_red(anchos, *, activacion, learning_rate_init, alpha, batch_size, epocas, datos,
  semillas, dispositivo, precision="float64", reportar=None, usar_cuda_graph=True) ->
  ResultadoEntrenamiento`; `riesgo_verdadero_en_dispositivo(modelo, *, d, a, k_ruido, rotar, n,
  semilla, dispositivo=None, precision=None) -> float`; `exactitud_en_dispositivo(modelo, x, y, *,
  dispositivo=None, precision=None) -> float`; `subir(arr, dtype, dispositivo) -> torch.Tensor`.
  Reexporta además `riesgo_verdadero` (el de `taller_hpo.problemas.paridad`, en numpy puro, para
  llamar sobre un `Predictor` que ya vive en la memoria de tu computadora) junto a la variante nativa del dispositivo.
  `cash_paridad` reusa `entrenar_red` directo para su algoritmo `mlp` en la tarjeta de video.
- Si la pérdida o los logits de validación no son finitos -> `Divergencia` (la MISMA clase que
  `mlp_paridad`, para que poda y honestidad la reconozcan igual).
- **Seguridad entre hilos:** CUDA permite una sola captura de stream a la vez por proceso.
  `_CANDADO_CAPTURA` (un `threading.Lock` de módulo) serializa solo el instante de captura -- una
  vez por llamada a `entrenar`, nunca los `replay()` de las épocas siguientes -- así que varios
  hilos entrenando en paralelo (`n_jobs>1` de Optuna, que usa hilos) solo esperan turno en ese
  instante puntual, no durante todo el entrenamiento. La ruta principal del taller usa `n_jobs=1`
  por convención (`taller_hpo.estudio.optimizar`), así que esto solo importa a un llamador directo
  de `entrenar_red`/`entrenar` fuera de esa función.
- Con dos dispositivos (uno pedido con `dispositivo="cuda:N"`, otro activo en el proceso en ese
  instante), la captura podría quedar sobre el stream del dispositivo equivocado (con el
  `default_capture_stream` de `torch.cuda.graph`; por eso `entrenar_red` captura en un stream propio
  del dispositivo real de la red, ver arriba). Con una sola tarjeta de video visible el módulo no puede reproducir
  ese escenario, así que lo que sigue es una hipótesis, no una medición: si la captura quedara
  VACÍA, un contador grabado dentro del propio grafo lo detecta -> `RuntimeError`, nunca una métrica
  de la época en que se capturó reportada en silencio para siempre; si en cambio la época completa
  corriera en otro stream del mismo dispositivo, la captura fallaría RUIDOSAMENTE con
  `AcceleratorError` en vez de quedar vacía.
- En esta red chica, el costo fijo de arrancar la tarjeta de video (inicializar CUDA, capturar el
  primer grafo) pesa proporcionalmente más cuanto más corto es el entrenamiento; cuánto, depende de
  la carga de la máquina, y este documento no da una cifra. La tabla del README raíz solo registra el
  tiempo total de cada paso del taller (humo y taller), sin una comparación por tamaño de red.

### `taller_hpo.honestidad`

```python
@dataclass(frozen=True)
class PlanSemillas: busqueda: Semillas; evaluacion: tuple[Semillas, ...]
def plan_semillas(base: int, n_evaluacion: int) -> PlanSemillas
    # La búsqueda usa UNA Semillas (números aleatorios comunes). La evaluación usa los MISMOS datos
    # (lo que puede hacer un tesista con un dataset fijo) con orden e init nuevos, disjuntos de los
    # de la búsqueda.
def congelar(params: Mapping[str, Any]) -> str      # sha256 del JSON canónico
def mejores_de(study, k: int, *, incluir_linea_base: bool = True) -> dict[str, dict[str, Any]]
def reevaluar(configs: Mapping[str, Mapping], entrenar, semillas: Sequence[Semillas], *,
              fijos=None) -> dict[str, list[dict[str, float]]]
    # Fuera de cualquier estudio. Mismas semillas para todas las configs, así la comparación es pareada.
@dataclass(frozen=True)
class Inflacion: valor_busqueda: float; media_reevaluada: float; inflacion: float; ic_bajo: float; ic_alto: float
def inflacion_ganador(valor_busqueda: float, reevaluados: Sequence[float], *,
                      direccion: str = "maximizar") -> Inflacion
class PruebaSellada:
    # La prueba se abre UNA vez, con la config congelada. Volver a sellar ANTES de la
    # primera apertura reemplaza la config; sellar DESPUÉS de que ya se abrió -> RuntimeError
    # (ninguna config nueva vale una vez que ya se miró la prueba).
    def __init__(self, evaluar: Callable[[Mapping[str, Any]], float]) -> None
    def sellar(self, params) -> str   # RuntimeError si la prueba ya se abrió
    def abrir(self, params) -> float   # error si no se selló, si el hash no coincide o si ya se abrió
```

### `taller_hpo.problemas.cash_paridad`
El mismo contrato y las mismas métricas, con `algoritmo` como hiperparámetro:

- `logistica`, `bosque`, `hgb` y `mlp` entran en la búsqueda.
- `knn` y `krr` son los baselines competitivos, con sus defaults.
- Cada algoritmo tiene sus parámetros con prefijo (`bosque_n_arboles`, `hgb_learning_rate`, ...).
- El MLP usa anchos por capa (`ancho_1`, `ancho_2`, `ancho_3`), y `n_capas` decide cuántos se usan.
- El espacio del paso 05 es condicional: solo se sugieren los parámetros del algoritmo elegido.
- El problema usa `rotar=True`, para quitarles a los árboles la ventaja de cortar por ejes.
- `reportar` solo aplica al MLP (una llamada por época).
- `n_parametros` es un tamaño de modelo comparable DENTRO de cada algoritmo (entre algoritmos las
  unidades no son la misma cosa): pesos entrenables en `logistica`/`mlp`; nodos totales de los
  árboles en `bosque`/`hgb`; `n_train * n_features` (el tamaño de la tabla que guarda) en `knn`, que
  no tiene parámetros entrenables; coeficientes duales (`n_train`) en `krr`. El código de
  `cash_paridad` documenta esta cuenta junto a cada algoritmo.
- `dispositivo`: `"auto"` | `"cuda"` | `"cuda:N"` (default `"auto"`, que resuelve a `"cuda"`);
  cualquier otro valor da `ValueError` explícito, antes de tocar datos. `bosque` y `hgb` usan XGBoost
  (`taller_hpo` no tiene árboles propios), así que además de una tarjeta de video con CUDA necesitan
  un XGBoost compilado con soporte de CUDA (`xgboost.build_info()["USE_CUDA"]`); `logistica`, `mlp`,
  `knn` y `krr` solo necesitan torch. `torch` y `xgboost`, los dos con CUDA, son dependencias FIJAS
  del taller (`pyproject.toml`): sin una tarjeta de video con CUDA visible, `entrenar` se detiene con
  un `ValueError` que trae `MENSAJE_SIN_TARJETA` (ver `taller_hpo.tarjeta_de_video`); con `algoritmo`
  en `("bosque", "hgb")` y sin XGBoost compilado con soporte de CUDA, el error es del mismo tipo, con
  su propio mensaje.
- Con `algoritmo` en `("bosque", "hgb")`, `entrenar` importa `xgboost`, y ese import carga
  scikit-learn y threadpoolctl (`XGBClassifier` hereda de `sklearn.base`). `logistica`, `knn`, `krr`
  y `mlp` no importan `xgboost` y no cargan scikit-learn. El paso 02 calcula las importancias solo
  con `PedAnovaImportanceEvaluator`, pero importar `optuna.importance` ya carga scikit-learn y
  threadpoolctl, porque su `__init__` importa `FanovaImportanceEvaluator` y
  `MeanDecreaseImpurityImportanceEvaluator`
  (`tests/test_paso_02.py::test_paso_02_humo_si_carga_sklearn_por_optuna_importance`).

### `plantilla/buscar_*.py`
- `buscar_en_proceso.py` importa `train_ejemplo.py` por ruta.
- `buscar_por_subproceso.py` lo ejecuta con `comando = [TALLER_PYTHON o sys.executable, train_ejemplo.py]`.

Los dos usan `crear_estudio` con una fábrica de sampler, `sembrar_linea_base` con los DEFAULTS del script, `optimizar` y `PRESUPUESTO`, e imprimen la mejor configuración con la advertencia de que ese valor está inflado hasta que se re-evalúe (paso 06).

### Anexo: paralelo
- **Worker:** `pasos/anexo_paralelo.py` abre o crea el estudio (`TALLER_DESTINO`: un Journal o `postgresql+psycopg://...`) y llama a `optimizar` hasta `TALLER_N_TOTAL` entre todos los workers. Cada worker usa `TPESampler(constant_liar=True, seed=BASE + 1000*worker + n_previos)`, así que dos workers no proponen lo mismo a la vez. La semilla por worker queda fuera del protocolo para que todos compartan el mismo.
- **Modo local:** `--lanzar N` arranca N procesos sobre un mismo Journal y reporta los trials por worker y el tiempo de reloj.
- **Docker:** `compose.yaml` tiene el perfil `paralelo`, con `postgres` (sin puertos publicados y contraseña desde `TALLER_PG_PASSWORD`; nunca en el repo) y `worker` escalable con `--scale worker=N`, con tarjeta de video. `scripts/paralelo_docker.sh` genera una contraseña aleatoria por corrida si no existe, levanta todo, espera a los workers, resume el estudio y baja todo.
- **Reproducibilidad:** con varios workers el orden de los trials depende de quién termina primero, así que el estudio NO se reproduce exactamente. El anexo lo dice y lo mide.
- **Guarda de WSL:** antes de tocar el storage, si `TALLER_DESTINO` es un Journal (no una URL) sobre
  un montaje `9p`/`v9fs`/`drvfs` (`/mnt/c`, `/mnt/d`, ...), el que WSL2 usa para exponer los discos
  de Windows, falla con `SystemExit` explícito con cualquier número de workers, incluido
  `--lanzar 1`. `--preparar`, que crea el estudio en un solo proceso sin optimizar, no llama a esta
  guarda. El Journal de Optuna asume que varios procesos pueden abrir y escribir el mismo archivo a
  la vez, y ese montaje no lo sostiene (fallas intermitentes de `OSError` al leer el journal). El
  mensaje sugiere `TALLER_ESTUDIOS` en el disco de Linux (p. ej. `~/estudios_taller`) o
  `TALLER_DESTINO=postgresql+psycopg://...`.

## Pasos

| Paso | Qué contesta |
|---|---|
| 00 | ¿Cuánto varía tu línea base solo por la semilla? ¿La validación dice la verdad? ¿Cuánto falta para el techo? |
| 01 | ¿Grid o random? El resultado de Bergstra y Bengio como distribución sobre réplicas, no como anécdota |
| 02 | Primer estudio con método: sampler escrito, storage, reanudar, la línea base dentro del estudio, importancias contra un placebo |
| 03 | Poda: cuánto cómputo ahorra, si mató trials que iban a mejorar tarde, y por qué no es early stopping |
| 04 | Pareto: mejor pero más caro, sin sumas ponderadas; ¿el frente también se infla? (medido contra un control) |
| 05 | Buscar el modelo, no solo sus perillas: arquitectura y algoritmo como hiperparámetros; siempre contra random y contra un baseline competitivo |
| 06 | Honestidad: la maldición del ganador medida contra el oráculo, comparación pareada, cuántas semillas, prueba sellada |
| 07 | Conecta tu modelo: en proceso y por subproceso, con la plantilla |
| Anexo | Paralelo: N workers sobre un mismo estudio (Journal local o PostgreSQL en Docker) y por qué deja de ser reproducible |

Tiempos medidos por paso: ver la tabla del README raíz.
