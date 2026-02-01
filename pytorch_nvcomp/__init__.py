try:
    from ._nvcomp_ext import (
        lz4_compress,
        lz4_decompress,
        zstd_compress,
        zstd_decompress,
        cascaded_compress,
        cascaded_decompress,
    )
except OSError as exc:
    raise OSError(
        "Failed to load pytorch_nvcomp extension. "
        "Ensure libnvcomp.so is installed and on your library path. "
        "Download nvCOMP: https://developer.nvidia.com/nvcomp-downloads\n"
        "Example:\n"
        "  export LD_LIBRARY_PATH=/opt/nvcomp-cuda13/lib:$LD_LIBRARY_PATH"
    ) from exc

__all__ = [
    "lz4_compress",
    "lz4_decompress",
    "zstd_compress",
    "zstd_decompress",
    "cascaded_compress",
    "cascaded_decompress",
]
