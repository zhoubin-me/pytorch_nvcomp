#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NVCOMP_URL:-}" ]]; then
  echo "NVCOMP_URL is not set. Provide a direct nvCOMP SDK URL or preinstall nvCOMP in the image."
  exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

echo "Downloading nvCOMP from ${NVCOMP_URL}"
curl -L "$NVCOMP_URL" -o "$tmp_dir/nvcomp.tgz"

mkdir -p /opt/nvcomp
tar -xzf "$tmp_dir/nvcomp.tgz" -C /opt/nvcomp --strip-components=1

echo "Installed nvCOMP to /opt/nvcomp"
