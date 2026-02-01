#!/usr/bin/env bash
set -euo pipefail

NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda13-archive.tar.xz"
CUDA_VERSION="${CUDA_VERSION:-13.0}"
CUDA_INSTALL="${CUDA_INSTALL:-1}"

if [[ -z "${NVCOMP_URL:-}" ]]; then
  echo "NVCOMP_URL is not set. Provide a direct nvCOMP SDK URL or preinstall nvCOMP in the image."
  exit 1
fi

if [[ "$CUDA_INSTALL" == "1" ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "CUDA installation requires root. Re-run as root or set CUDA_INSTALL=0 to skip."
    exit 1
  fi

  if [[ ! -r /etc/os-release ]]; then
    echo "Cannot detect OS version. Set CUDA_INSTALL=0 to skip CUDA install."
    exit 1
  fi

  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || -z "${VERSION_ID:-}" ]]; then
    echo "Unsupported OS for CUDA install: ${ID:-unknown}. Set CUDA_INSTALL=0 to skip."
    exit 1
  fi

  ubuntu_version="${VERSION_ID//./}"
  cuda_pkg_version="${CUDA_VERSION//./-}"

  echo "Installing CUDA Toolkit ${CUDA_VERSION} for Ubuntu ${VERSION_ID}"
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates wget gnupg

  keyring_deb="/tmp/cuda-keyring.deb"
  wget -O "$keyring_deb" "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu_version}/x86_64/cuda-keyring_1.1-1_all.deb"
  dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"

  apt-get update
  apt-get install -y --no-install-recommends "cuda-toolkit-${cuda_pkg_version}"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

echo "Downloading nvCOMP from ${NVCOMP_URL}"
curl -L "$NVCOMP_URL" -o "$tmp_dir/nvcomp.tar.xz"

mkdir -p /opt/nvcomp
tar -xJf "$tmp_dir/nvcomp.tar.xz" -C /opt/nvcomp --strip-components=1

echo "Installed nvCOMP to /opt/nvcomp"
