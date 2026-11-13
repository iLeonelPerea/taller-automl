# taller-automl

Búsqueda de hiperparámetros **con método**: Optuna 5, diseño de experimentos y honestidad estadística, para conectar a tu propio modelo de investigación.

El taller tiene ocho pasos (`00` a `07`) y un anexo de paralelismo. Los que entrenan usan un
problema de juguete (paridad gaussiana con oráculo) en tu tarjeta de video. `docs/arquitectura.md`
documenta los contratos exactos de cada módulo.

## Pasos

| Paso | Qué contesta | Correr |
|---|---|---|
| 00 | ¿Cuánto varía tu línea base solo por la semilla? ¿La validación dice la verdad? ¿Cuánto falta para el techo? | `uv run pasos/00_linea_base.py` |
| 01 | ¿Grid o random? El resultado de Bergstra y Bengio como distribución sobre réplicas, no como anécdota | `uv run pasos/01_grid_vs_random.py` |
| 02 | Primer estudio con método: sampler escrito, storage, reanudar, la línea base dentro del estudio, importancias contra un placebo | `uv run pasos/02_primer_estudio.py` |
| 03 | Poda: cuánto cómputo ahorra, si mató trials que iban a mejorar tarde, y por qué no es early stopping | `uv run pasos/03_poda.py` |
| 04 | Pareto: mejor pero más caro, sin sumas ponderadas; ¿el frente también se infla? (medido contra un control) | `uv run pasos/04_pareto.py` |
| 05 | Buscar el modelo, no solo sus perillas: arquitectura y algoritmo como hiperparámetros; siempre contra random y contra un baseline competitivo | `uv run pasos/05_buscar_el_modelo.py` |
| 06 | Honestidad: la maldición del ganador medida contra el oráculo, comparación pareada, cuántas semillas, prueba sellada | `uv run pasos/06_honestidad.py` |
| 07 | Conecta tu modelo: en proceso y por subproceso, con la plantilla | `uv run pasos/07_conecta_tu_modelo.py` |
| Anexo | Paralelo: N workers sobre un mismo estudio (Journal local o PostgreSQL en Docker) y por qué deja de ser reproducible | `TALLER_ESTUDIOS=~/estudios_taller uv run pasos/anexo_paralelo.py --lanzar 3` (con el repo en `/mnt/c` o `/mnt/d` hace falta `TALLER_ESTUDIOS`: ver la advertencia de WSL más abajo) |

El taller necesita una tarjeta de video NVIDIA con CUDA (ver "Instalación"). Si no ve una, cada paso
que entrena se detiene al arrancar y la última línea dice exactamente qué falta; el `01` no entrena
nada (compara con una fórmula cerrada cómo repartir las pruebas) y corre igual.

Cada paso acepta `PRESUPUESTO=humo | taller (default) | completo`, guarda su figura en
`resultados/` (`TALLER_RESULTADOS` para cambiar la carpeta) y, si persiste un estudio, lo hace en
`estudios/` (`TALLER_ESTUDIOS`). Cada paso que entrena imprime primero una línea con la tarjeta de
video que usa (`Motor: ...` en 00, 02, 03, 04, 06 y el anexo; `cash_paridad: dispositivo=...` en el
05; `Motor resuelto para el juguete de este repo: ...` en el 07): así ves, sin buscar más, qué
tarjeta de video usó esa corrida.

## Instalación

Solo necesitas [uv](https://docs.astral.sh/uv/) y una tarjeta de video NVIDIA con CUDA, con su
controlador instalado, en Linux (incluido WSL2) o en Windows con WSL2. uv instala Python 3.12 y crea
el venv del proyecto; nada se instala de forma global.

```bash
uv sync --locked
```

`torch` (con CUDA) y `xgboost` (con CUDA) son dependencias fijas de `pyproject.toml`.
`plantilla/train_ejemplo.py` (y por lo tanto `plantilla/buscar_*.py` y el **paso 07** completo) usan
la misma instalación.

Un extra, `postgres` (`psycopg[binary]`), agrega el controlador que necesita el anexo paralelo para correr
contra PostgreSQL en Docker:

```bash
uv sync --locked --extra postgres
```

Pruebas:

```bash
uv run pytest -W error
```

Toda prueba que necesita la tarjeta de video lleva la marca `gpu` (`tests/conftest.py`): sin una
tarjeta de video visible, esas pruebas **fallan** con un mensaje claro, en vez de saltarse. La
integración continua (`.github/workflows/pruebas.yml`), que no tiene tarjeta de video, las excluye a
propósito con `pytest -m "not gpu"`.

### WSL2 con el repo en un disco de Windows (`/mnt/c`, `/mnt/d`)

Antes de cualquier comando de `uv`, en cada terminal nueva:

```bash
source scripts/wsl/entorno.sh
```

Este script exporta `UV_PROJECT_ENVIRONMENT` y pone así el venv en el disco de Linux (`~/.venvs/taller-hpo`), mucho más rápido que `/mnt/*`. `uv sync`, `uv add` y `uv run` respetan esa variable; `uv pip` la ignora.

### Advertencia: el anexo paralelo y `/mnt/c` o `/mnt/d`

El Journal de Optuna (el storage de archivo que usan los pasos por defecto) asume que varios
procesos pueden abrir y escribir el mismo archivo a la vez. Ese supuesto **no se sostiene** sobre
el montaje `9p`/`v9fs`/`drvfs` que WSL2 usa para exponer los discos de Windows (`/mnt/c`, `/mnt/d`,
...): con varios workers del anexo paralelo, el Journal falla de forma intermitente (3 de 6 corridas
de `--lanzar 3` sobre `/mnt/c`, 0 de 6 sobre `/tmp`). Por eso `pasos/anexo_paralelo.py` se niega a
abrir un Journal sobre ese montaje con cualquier número de workers, incluido `--lanzar 1`, y sale con
un `SystemExit` antes de tocar el storage (ver `_verificar_journal_no_9p`). `--preparar`, un solo
proceso que crea el estudio sin optimizar, no pasa por esa guarda. Hay dos salidas:
`TALLER_ESTUDIOS`/`TALLER_DESTINO` apuntando al disco de Linux (p. ej. `~/estudios_taller`, fuera de
`/mnt/*`), o PostgreSQL (`TALLER_DESTINO=postgresql+psycopg://...`, el perfil `paralelo` de Docker
de abajo).

## Docker con tarjeta de video

Una sola vez por máquina, en WSL2 Ubuntu 24.04. Corre el script sin `sudo`: él lo pide donde hace falta:

```bash
bash scripts/wsl/instalar_docker_gpu.sh
```

Al terminar, **cierra y vuelve a abrir** la terminal de Ubuntu. Después:

```bash
docker compose build
docker compose run --rm taller
```

`taller` corre las pruebas dentro del contenedor con la tarjeta de video, y falla si no la ve.
`--rm` borra solo el contenedor: la red vacía que `docker compose run` crea queda en tu
máquina hasta que ejecutas `docker compose down`.

`compose.yaml` define dos imágenes del mismo `Dockerfile`, en dos stages encadenados (el segundo
deriva del primero, así que comparten casi todas sus capas):

- **`taller`** (`docker compose run --rm taller`, arriba): corre la suite completa dentro del
  contenedor, con la tarjeta de video, que siempre exige. La imagen `taller` trae el grupo `dev`
  (pytest, ruff).
- **Perfil `paralelo`** (`scripts/paralelo_docker.sh N`): PostgreSQL, un servicio `preparar` (crea
  el estudio una sola vez, sin correr ninguna búsqueda) y N réplicas de `worker` (imagen
  `taller-hpo-worker`, sin el grupo `dev`) que optimizan el mismo estudio en PostgreSQL. El script
  genera `TALLER_PG_PASSWORD` al vuelo, levanta todo, espera a los workers, resume el estudio y baja
  todo. Úsalo en vez de `docker compose --profile paralelo up` directo: sin esa variable exportada,
  Postgres 18 no arranca. Sin puertos publicados; la contraseña nunca vive en el repo.

## Versiones fijadas

| Pieza | Versión |
|---|---|
| Python | 3.12 (`.python-version`) |
| Optuna | 5.0.0 |
| optuna-dashboard | 0.21.0 |
| PyTorch | 2.14.0+cu130 (siempre con CUDA) |
| scikit-learn | 1.9.1 (modelo de referencia de las pruebas, `src/taller_hpo/problemas/mlp_paridad.py`) |
| XGBoost | 3.4.1 (siempre con CUDA; entrena los algoritmos de árboles del paso 05 en la tarjeta de video) |
| threadpoolctl | 3.7.0 |
| psycopg | 3.3.6 (extra `postgres`) |
| SQLAlchemy | 2.0.x (`pyproject.toml` fija `sqlalchemy<2.1`: Optuna 5.0.0 salió antes que la 2.1 y no la prueba) |
| uv | 0.12.19 (fijado en el `Dockerfile`; `pyproject.toml` solo exige `>=0.12` para `uv sync` local) |

Todo queda fijado en `uv.lock`: `uv sync --locked` reproduce el entorno exacto.

## Tiempos medidos

Medidos sin otros procesos del taller y con la máquina en reposo: una cifra tomada en una máquina
ocupada no es comparable entre pasos ni reproducible. En la máquina de referencia, la tarjeta de
video también atiende el escritorio de Windows, así que nunca está del todo libre (ver el uso
anotado abajo). El script que las mide está en `scripts/medir_tiempos.py`.

<!-- TIEMPOS:INICIO -->
Medido en la máquina de referencia (RTX 4090, WSL2 Ubuntu 24.04, torch 2.14.0+cu130), sin otros procesos del taller: cada celda arrancó con carga de 1 minuto menor a 0.5 (de 0.46 a 0.49) y con la tarjeta de video entre 17% y 19% de uso y unos 1.95 GB ocupados por el escritorio de Windows. Cada celda es la mediana de 3 corridas, con el mínimo y el máximo entre paréntesis; hubo una única corrida de calentamiento al inicio, que se descartó. El tiempo es el transcurrido por el proceso completo, incluidos el arranque de Python y de CUDA, medido con un reloj monotónico (`time.perf_counter`) y no con el reloj de pared, que en WSL2 salta cuando se corrige la hora. Las corridas medidas terminaron con código de salida 0 y sin nada en stderr. Los límites son 20 s para humo y 3 min para taller.

| Paso | humo (s) | taller (s) | motor |
|---|---|---|---|
| 00 línea base | 3.9 (3.9-4.3) | 6.1 (6.1-6.3) | PyTorch en cuda |
| 01 grid contra random | 1.2 (1.2-1.2) | 2.9 (2.8-2.9) | no entrena (paisaje analítico) |
| 02 primer estudio | 7.0 (6.9-7.3) | 25.5 (25.2-25.9) | PyTorch en cuda |
| 03 poda | 13.3 (13.2-13.3) | 57.2 (56.9-57.5) | PyTorch en cuda |
| 04 Pareto | 6.9 (6.9-7.2) | 67.6 (67.6-67.7) | PyTorch en cuda |
| 05 buscar el modelo | 9.5 (9.4-10.0) | 53.4 (53.3-53.8) | cash_paridad en cuda (torch + XGBoost) |
| 06 honestidad | 11.7 (11.5-12.0) | 67.7 (67.6-68.0) | PyTorch en cuda |
| 07 conecta tu modelo | 11.1 (11.0-11.3) | 57.0 (56.4-57.2) | PyTorch en cuda |
| anexo paralelo (`--lanzar 3`) | 8.7 (8.5-8.9) | 68.2 (67.8-69.2) | PyTorch en cuda, 3 procesos sobre una tarjeta de video |

Con la tarjeta de video, cada proceso paga un costo fijo al arrancar: importar torch, inicializar CUDA y capturar el primer grafo. Con presupuestos tan chicos como humo, ese costo puede pesar más que el propio entrenamiento; en taller se reparte entre muchos más entrenamientos, y por eso pesa menos por corrida.
<!-- TIEMPOS:FIN -->
