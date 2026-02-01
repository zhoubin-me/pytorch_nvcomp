#!/usr/bin/env bash
set -euo pipefail

NVCOMP_URL="https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda13-archive.tar.xz"

if [[ -z "${NVCOMP_URL:-}" ]]; then
  echo "NVCOMP_URL is not set. Provide a direct nvCOMP SDK URL or preinstall nvCOMP in the image."
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

echo "Downloading nvCOMP from ${NVCOMP_URL}"
curl -L "$NVCOMP_URL" -o "$tmp_dir/nvcomp.tar.xz"

mkdir -p /opt/nvcomp
tar -xJf "$tmp_dir/nvcomp.tar.xz" -C /opt/nvcomp --strip-components=1

echo "Installed nvCOMP to /opt/nvcomp"
