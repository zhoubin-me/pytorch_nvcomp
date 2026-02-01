import argparse
from pathlib import Path

import torch
from PIL import Image

from pytorch_nvcomp import (
    lz4_compress,
    lz4_decompress,
    zstd_compress,
    zstd_decompress,
    cascaded_compress,
    cascaded_decompress,
)


def load_image_bytes(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    data = img.tobytes()  # H * W * 3 bytes
    # Use frombuffer to avoid numpy dependency
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return tensor, img.size, img.mode


def main() -> None:
    parser = argparse.ArgumentParser(description="nvCOMP LZ4 roundtrip on an image")
    parser.add_argument("image", type=Path, help="Path to an image file")
    parser.add_argument(
        "--codec",
        type=str,
        default="lz4",
        choices=["lz4", "zstd", "cascaded"],
        help="Compression codec",
    )
    parser.add_argument("--chunk-size", type=int, default=65536, help="Chunk size in bytes")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    cpu_bytes, (w, h), mode = load_image_bytes(args.image)
    x = cpu_bytes.cuda(non_blocking=True)

    if args.codec == "lz4":
        compressed, comp_sizes, orig_sizes, max_out, chunk_sz = lz4_compress(x, args.chunk_size)
        x2 = lz4_decompress(compressed, comp_sizes, orig_sizes, max_out, chunk_sz)
    elif args.codec == "zstd":
        compressed, comp_sizes, orig_sizes, max_out, chunk_sz = zstd_compress(x, args.chunk_size)
        x2 = zstd_decompress(compressed, comp_sizes, orig_sizes, max_out, chunk_sz)
    else:
        compressed, comp_sizes, orig_sizes, max_out, chunk_sz = cascaded_compress(x, args.chunk_size)
        x2 = cascaded_decompress(compressed, comp_sizes, orig_sizes, max_out, chunk_sz)

    ok = torch.equal(x, x2)
    actual_comp = int(comp_sizes.sum().item())
    orig_bytes = int(x.numel())
    ratio = (actual_comp / orig_bytes) if orig_bytes else 0.0
    print(f"Image: {args.image} ({mode} {w}x{h})")
    print(f"Codec: {args.codec}")
    print(f"Original bytes: {orig_bytes}")
    print(f"Compressed bytes (actual): {actual_comp}")
    print(f"Compressed buffer bytes (max): {compressed.numel()}")
    print(f"Compression ratio (actual/original): {ratio:.4f}")
    print(f"Roundtrip ok: {ok}")


if __name__ == "__main__":
    main()
