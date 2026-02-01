# pytorch_nvcomp

Minimal PyTorch CUDA extension that wraps nvCOMP codecs (LZ4, Zstd, Cascaded).

## Build/install

This extension depends on your existing PyTorch install. Use `--no-build-isolation`
so pip doesn't create a separate build env without `torch`.

```bash
cd pytorch_nvcomp
NVCOMP_ROOT=/opt/nvcomp-cuda13 NVCC_ALLOW_UNSUPPORTED=1 pip install -e . --no-build-isolation
```

If `libnvcomp.so` is not on your runtime linker path:

```bash
export LD_LIBRARY_PATH=/opt/nvcomp-cuda13/lib:$LD_LIBRARY_PATH
```

## Download nvCOMP

Download the nvCOMP SDK from NVIDIA and extract it locally. The main downloads
page links to the latest release and an archive of previous releases:

```text
https://developer.nvidia.com/nvcomp-downloads
```

If you use a different install path, set `NVCOMP_ROOT` to that location.

## Publish to PyPI

Build a wheel locally (Linux):

```bash
NVCOMP_ROOT=/opt/nvcomp-cuda13 NVCC_ALLOW_UNSUPPORTED=1 python -m build --wheel --no-isolation
```

Upload with twine:

```bash
twine upload dist/*
```

Note: the wheel does **not** bundle `libnvcomp.so`. Users must install nvCOMP
and set `LD_LIBRARY_PATH` (or rpath) so the loader can find the library.
