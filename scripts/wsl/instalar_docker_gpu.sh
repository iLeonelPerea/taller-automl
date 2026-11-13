#!/usr/bin/env bash
# Instala Docker Engine + NVIDIA Container Toolkit en WSL2 Ubuntu 24.04 (una sola vez por máquina).
# Uso:  bash scripts/wsl/instalar_docker_gpu.sh     (SIN sudo: lo pide solo donde hace falta)
#
# Fuentes oficiales:
#   https://docs.docker.com/engine/install/ubuntu/
#   https://docs.docker.com/engine/install/linux-postinstall/
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html
#
# En WSL2 NO se instala ningún controlador NVIDIA: lo provee el controlador
# de Windows (https://docs.nvidia.com/cuda/wsl-user-guide/).
set -euo pipefail

# Versión que fija la guía de NVIDIA para que sus 4 paquetes coincidan.
NVIDIA_CONTAINER_TOOLKIT_VERSION="1.20.1-1"

falla() { echo "ERROR: $*" >&2; exit 1; }

# apt espera hasta 10 min al lock de dpkg en vez de fallar: al arrancar WSL,
# unattended-upgrades suele tenerlo tomado. Nunca borres el lock, porque apt advierte
# que eso rompe el sistema.
apt_() { sudo apt-get -o DPkg::Lock::Timeout=600 "$@"; }

echo "== 0/4 Comprobaciones previas"
# Con sudo, $USER sería root y el usuario real quedaría fuera del grupo docker.
[ "$(id -u)" -ne 0 ] || falla "córrelo SIN sudo: bash $0"
[ "$(ps -p 1 -o comm=)" = "systemd" ] || falla "systemd no es el PID 1. Agrega [boot] systemd=true en /etc/wsl.conf y reinicia WSL (wsl --shutdown)."
nvidia-smi >/dev/null 2>&1 || falla "nvidia-smi no responde en WSL: revisa el controlador NVIDIA de Windows."
if command -v docker >/dev/null 2>&1 && ! dpkg -s docker-ce >/dev/null 2>&1; then
    falla "ya hay un 'docker' que no es docker-ce ($(command -v docker)); ¿integración de Docker Desktop? Desactívala primero."
fi
[ ! -f /etc/apt/sources.list.d/docker.list ] || falla "existe /etc/apt/sources.list.d/docker.list (formato viejo); bórralo para evitar un Signed-By en conflicto."
CONFLICTIVOS=""
for paquete in docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc; do
    if dpkg -s "$paquete" >/dev/null 2>&1; then
        CONFLICTIVOS="$CONFLICTIVOS $paquete"
    fi
done
if [ -n "$CONFLICTIVOS" ]; then
    echo "Quitando paquetes que chocan con docker-ce (paso de la guía oficial): $CONFLICTIVOS"
    # shellcheck disable=SC2086
    apt_ remove -y $CONFLICTIVOS
fi

echo "== 1/4 Repositorio oficial de Docker"
apt_ update
apt_ install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt_ update
apt_ install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "== 2/4 Docker sin sudo (grupo docker)"
sudo groupadd -f docker
sudo usermod -aG docker "$USER"

echo "== 3/4 NVIDIA Container Toolkit ${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
apt_ update
apt_ install -y \
    nvidia-container-toolkit="${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    nvidia-container-toolkit-base="${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    libnvidia-container-tools="${NVIDIA_CONTAINER_TOOLKIT_VERSION}" \
    libnvidia-container1="${NVIDIA_CONTAINER_TOOLKIT_VERSION}"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "== 4/4 Verificación: la tarjeta de video dentro de un contenedor"
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

echo
echo "Listo. CIERRA y vuelve a abrir la terminal de Ubuntu para usar 'docker' sin sudo."
