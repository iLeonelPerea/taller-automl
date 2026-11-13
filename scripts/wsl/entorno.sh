# Uso (en WSL, desde la raíz del repo):  source scripts/wsl/entorno.sh
#
# Pone el venv del proyecto en el disco de Linux, porque /mnt/c y /mnt/d son lentos.
# Vale solo para esta sesión de la terminal y para este proyecto. No va en ~/.bashrc
# porque ahí todos los proyectos con uv compartirían este venv.
# Ojo: `uv pip` ignora esta variable. Usa `uv sync`, `uv add` o `uv run`.
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/taller-hpo"
echo "venv del proyecto: $UV_PROJECT_ENVIRONMENT"
