#!/usr/bin/env bash
# scripts/paralelo_docker.sh N
#
# Levanta PostgreSQL, el servicio "preparar" (crea el estudio una vez y termina) y N
# workers sobre el mismo estudio (perfil "paralelo" de compose.yaml). Espera a que los N
# terminen, imprime el resumen del estudio y baja todo con `down -v`, pase lo que pase.
#
# Si TALLER_PG_PASSWORD no está en el entorno, el script genera una contraseña aleatoria
# solo para esta corrida y no la escribe en ningún archivo. PRESUPUESTO, TALLER_N_TOTAL y
# TALLER_ESTUDIO se heredan del entorno si existen; si no, el anexo usa sus defaults.
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "uso: $0 N" >&2
    exit 2
fi
N="$1"
case "$N" in
    '' | *[!0-9]*)
        echo "N debe ser un entero positivo, no '$N'" >&2
        exit 2
        ;;
esac
if [ "$N" -lt 1 ]; then
    echo "N debe ser >= 1, no $N" >&2
    exit 2
fi

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"

PROYECTO="${TALLER_COMPOSE_PROJECT:-taller-hpo-paralelo}"
COMPOSE=(docker compose --profile paralelo -p "$PROYECTO")

# El script busca "python3" y, si no existe, "python", porque calcula la contraseña y el
# resumen de FALLOS en la máquina que corre Docker, no dentro de un contenedor.
PYTHON_HOST=""
if command -v python3 >/dev/null 2>&1; then
    PYTHON_HOST="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_HOST="$(command -v python)"
fi
if [ -z "$PYTHON_HOST" ]; then
    echo "el script no encontró python3 ni python en la máquina que corre Docker" >&2
    exit 2
fi

GENERADA=0
if [ -z "${TALLER_PG_PASSWORD:-}" ]; then
    TALLER_PG_PASSWORD="$("$PYTHON_HOST" -c 'import secrets; print(secrets.token_hex(16))')"
    export TALLER_PG_PASSWORD
    GENERADA=1
    echo "TALLER_PG_PASSWORD generada solo para esta corrida (el script no escribe la contraseña en ningún archivo)"
fi

# compose.yaml mete la contraseña tal cual en la URL de TALLER_DESTINO, sin escaparla,
# así que solo acepta letras, números y '.', '_', '~' o '-'. La contraseña generada
# arriba es hexadecimal y siempre cumple.
case "$TALLER_PG_PASSWORD" in
    *[!A-Za-z0-9._~-]*)
        echo "TALLER_PG_PASSWORD solo puede tener letras, números, '.', '_', '~' o '-':" >&2
        echo "cualquier otro caracter ('@', ':', '/', '%', '?', '#', ...) puede romper la URL" >&2
        echo "postgresql+psycopg://taller:<password>@postgres:5432/taller que arma compose.yaml." >&2
        echo "Usa una contraseña que cumpla esa lista (la generada automáticamente ya lo hace)." >&2
        exit 2
        ;;
esac

limpiar() {
    codigo=$?
    if [ "$codigo" -ne 0 ]; then
        # Si "up -d --build" falló, el bloque de más abajo que imprime el estado de
        # "preparar" nunca se alcanza, y "down -v" borraría sus logs junto con la causa.
        echo "=== fallo detectado (código $codigo): estado y logs antes de bajar ===" >&2
        if ! "${COMPOSE[@]}" ps -a; then
            echo "advertencia: 'docker compose ps -a' salió con error" >&2
        fi
        # postgres incluido: si falla su healthcheck, sus logs son la única pista.
        if ! "${COMPOSE[@]}" logs --no-color postgres preparar worker; then
            echo "advertencia: 'docker compose logs' salió con error" >&2
        fi
    fi
    echo "=== bajando postgres + workers (down -v --remove-orphans) ==="
    # Avisa y sigue, sin pisar el código de salida de la corrida con el de `down`.
    if ! "${COMPOSE[@]}" down -v --remove-orphans; then
        echo "advertencia: 'docker compose down' salió con error (ver arriba)" >&2
    fi
    if [ "$GENERADA" -eq 1 ]; then
        unset TALLER_PG_PASSWORD
    fi
    exit "$codigo"
}
# Un solo trap real (EXIT), que siempre limpia. INT y TERM solo fijan el código de
# salida (128 + señal) con un "exit", que dispara el trap EXIT, así `down` corre una vez.
trap limpiar EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Memoria de toda la tarjeta de video vista desde la máquina que corre Docker
# (nvidia-smi, memory.used). La base se toma justo después de "up -d --build", que no
# regresa hasta que "preparar" termina, y el pico durante la espera de más abajo.
# Devuelve "" si nvidia-smi no existe o falla, y siempre sale con 0, para que una
# métrica opcional no aborte el script bajo `set -euo pipefail`.
_memoria_gpu_host_mib() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    local salida primera_linea
    if salida="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)"; then
        primera_linea="${salida%%$'\n'*}"
        case "$primera_linea" in
            '' | *[!0-9]*)
                echo "aviso: nvidia-smi devolvió un valor no numérico, el script no mide la memoria" \
                    "de la máquina que corre Docker: '$primera_linea'" >&2
                return 0
                ;;
        esac
        printf '%s\n' "$primera_linea"
    else
        echo "aviso: nvidia-smi falló, el script no mide la memoria de la máquina que corre Docker: $salida" >&2
    fi
    return 0
}
echo "=== arrancando postgres + preparar + $N worker(s) (proyecto $PROYECTO) ==="
# --build: "docker compose build" a secas no construye los servicios del perfil
# "paralelo", así que sin esto los workers correrían una imagen vieja.
"${COMPOSE[@]}" up -d --build --scale "worker=$N" worker

# La base de memoria va aquí, el punto más cercano al arranque de los workers.
MEM_HOST_BASE="$(_memoria_gpu_host_mib)"
MEM_HOST_PICO="$MEM_HOST_BASE"

# Defensa en profundidad: si "preparar" falla, normalmente "up -d --build" ya falló
# antes y el trap ya imprimió sus logs.
echo "=== estado de 'preparar' (corre una vez, antes de escalar workers) ==="
"${COMPOSE[@]}" ps -a preparar
"${COMPOSE[@]}" logs preparar
PREPARAR_CODIGO="$("${COMPOSE[@]}" ps -a --format json preparar | "$PYTHON_HOST" -c '
import json
import sys

for linea in sys.stdin:
    linea = linea.strip()
    if linea:
        print(json.loads(linea).get("ExitCode", 0))
        break
else:
    print(0)
')"
if [ "$PREPARAR_CODIGO" -ne 0 ]; then
    echo "preparar salió con código $PREPARAR_CODIGO (ver logs arriba)" >&2
    exit 1
fi

echo "=== esperando a que los $N worker(s) terminen ==="
LIMITE="${TALLER_PARALELO_TIMEOUT_S:-600}"
# Muestreo cada 0.5 s: con humo un worker vive poco y un muestreo grueso perdería el
# pico de memoria. LIMITE sigue en segundos (TALLER_PARALELO_TIMEOUT_S).
INTERVALO_MUESTREO_S=0.5
LIMITE_TICKS=$((LIMITE * 2))
ticks=0
while true; do
    activos="$("${COMPOSE[@]}" ps --status running --format json worker | wc -l)"
    actual="$(_memoria_gpu_host_mib)"
    if [ -n "$actual" ] && { [ -z "$MEM_HOST_PICO" ] || [ "$actual" -gt "$MEM_HOST_PICO" ]; }; then
        MEM_HOST_PICO="$actual"
    fi
    if [ "$activos" -eq 0 ]; then
        break
    fi
    if [ "$ticks" -ge "$LIMITE_TICKS" ]; then
        echo "timeout esperando a los workers ($LIMITE s, todavía $activos activo(s))" >&2
        exit 1
    fi
    sleep "$INTERVALO_MUESTREO_S"
    ticks=$((ticks + 1))
done

echo "=== estado final de los workers ==="
"${COMPOSE[@]}" ps -a worker
# Los logs de los servicios siempre, no solo si algo falló: "ps -a" da el código de
# salida y los logs dan el porqué.
echo "=== logs de los workers ==="
"${COMPOSE[@]}" logs worker
FALLOS="$("${COMPOSE[@]}" ps -a --format json worker | "$PYTHON_HOST" -c '
import json
import sys

fallos = 0
for linea in sys.stdin:
    linea = linea.strip()
    if not linea:
        continue
    fila = json.loads(linea)
    codigo = fila.get("ExitCode", 0)
    if codigo != 0:
        fallos += 1
print(fallos)
')"
if [ "$FALLOS" -ne 0 ]; then
    echo "$FALLOS worker(s) salieron con código != 0 (ver arriba)" >&2
    exit 1
fi

echo "=== memoria de la tarjeta de video (máquina que corre Docker, nvidia-smi) ==="
if [ -n "$MEM_HOST_BASE" ] && [ -n "$MEM_HOST_PICO" ]; then
    HUELLA_HOST_TOTAL=$((MEM_HOST_PICO - MEM_HOST_BASE))
    HUELLA_HOST_POR_WORKER="$("$PYTHON_HOST" -c "print($HUELLA_HOST_TOTAL / $N)")"
    echo "base (justo tras 'up', preparar ya terminado): ${MEM_HOST_BASE} MiB; pico" \
        "(durante la espera): ${MEM_HOST_PICO} MiB; huella total~=${HUELLA_HOST_TOTAL} MiB" \
        "(~${HUELLA_HOST_POR_WORKER} MiB/worker de $N -- memory.used es de toda la tarjeta" \
        "de video de la máquina que corre Docker: otros procesos pueden SUBIR o BAJAR este" \
        "número, en cualquier dirección," \
        "mientras los workers corren; el desglose interno que cada worker imprime en" \
        "'logs de los workers' arriba es por proceso y no tiene ese problema)"
    echo "MEMORIA_HOST_JSON $("$PYTHON_HOST" -c "
import json
print(json.dumps({
    'base_mib': $MEM_HOST_BASE,
    'pico_mib': $MEM_HOST_PICO,
    'huella_total_mib': $HUELLA_HOST_TOTAL,
    'huella_por_worker_mib': $HUELLA_HOST_TOTAL / $N,
}))
")"
else
    echo "no aplica (nvidia-smi no disponible en la máquina que corre Docker)"
fi

echo "=== resumen del estudio (contenedor de una sola vez) ==="
"${COMPOSE[@]}" run --rm --no-deps worker python -c "
import json
import os

import optuna

from taller_hpo.estudio import abrir_storage

destino = os.environ['TALLER_DESTINO']
nombre = os.environ.get('TALLER_ESTUDIO') or 'anexo_paralelo'
storage = abrir_storage(destino)
study = optuna.load_study(study_name=nombre, storage=storage)
estados_terminados = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
terminados = study.get_trials(deepcopy=False, states=estados_terminados)
corriendo = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
por_worker: dict[str, int] = {}
for t in terminados:
    w = str(t.user_attrs.get('worker'))
    por_worker[w] = por_worker.get(w, 0) + 1
print(f'trials totales: {len(terminados)}')
print(f'trials por worker: {por_worker}')
print(f'RUNNING sin terminar: {len(corriendo)}')
# Línea aparte en JSON, para que una prueba no tenga que leer el repr() de arriba.
resumen = {'trials_totales': len(terminados), 'por_worker': por_worker, 'running': len(corriendo)}
print('RESUMEN_JSON ' + json.dumps(resumen))
"

echo "=== listo ==="
