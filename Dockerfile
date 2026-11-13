# Imagen del taller, con el patrón oficial de uv en Docker:
# https://docs.astral.sh/uv/guides/integration/docker/
#
# torch trae sus propias librerías de CUDA como ruedas (nvidia-*), así que la base
# no necesita CUDA. El NVIDIA Container Toolkit inyecta el controlador desde tu computadora.
#
# Hay dos imágenes finales con una sola cadena de capas: "worker", sin el grupo dev, para
# los workers del anexo paralelo, y "taller", que deriva de "worker" y solo agrega el
# grupo dev (pytest y ruff). Así las dos comparten en disco las ruedas de CUDA.
#
# Las imágenes base van fijadas por digest, así que construir no depende de la red. Para
# renovarlas, corre `docker buildx imagetools inspect python:3.12-slim` (y lo mismo para
# la imagen de uv) y copia aquí el "Digest" que imprime.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS base

COPY --from=ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# --- worker: perfil "paralelo" de compose.yaml, sin el grupo dev ---------------
FROM base AS worker

# Capa de dependencias: solo se reconstruye si cambian el lock o el pyproject.
# --extra postgres instala psycopg para el anexo paralelo, y --no-dev deja fuera
# pytest y ruff, que un worker nunca corre.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev --extra postgres

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra postgres

CMD ["python", "pasos/anexo_paralelo.py"]

# --- taller: `docker compose run --rm taller` corre la suite completa ---------
# Deriva de "worker" y solo agrega el grupo dev (pytest, pytest-timeout y ruff).
FROM worker AS taller

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra postgres

CMD ["pytest", "-W", "error"]
