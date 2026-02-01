# pytorch_nvcomp

Minimal PyTorch CUDA extension that wraps nvCOMP LZ4 batched compress/decompress.

## Build/install

This extension depends on your existing PyTorch install. Use `--no-build-isolation`
so pip doesn't create a separate build env without `torch`.

```bash
cd pytorch_nvcomp
NVCOMP_ROOT=../nvcomp-cuda13 pip install -e . --no-build-isolation
```

If `libnvcomp.so` is not on your runtime linker path:

```bash
export LD_LIBRARY_PATH=/home/bzhou/repo/nvcomp/nvcomp-cuda13/lib:$LD_LIBRARY_PATH
```

## Download nvCOMP

Download the nvCOMP SDK from NVIDIA and extract it locally. The main downloads
page links to the latest release and an archive of previous releases:

```text
https://developer.nvidia.com/nvcomp-downloads
```

If you use a different install path, set `NVCOMP_ROOT` to that location.
