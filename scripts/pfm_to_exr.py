#!/usr/bin/env python3
"""Convert the renderer's Portable Float Map (.pfm) output into OpenEXR (.exr).

Only numpy is required: the EXR writer here is self-contained (struct and zlib
are stdlib), so no image library has to be installed. It emits plain scanline
OpenEXR files with FLOAT (default) or HALF pixels, ZIP- or uncompressed, which
is what OpenImageIO, the OpenEXR distribution, Nuke, Blender and ffmpeg read.

Both formats are scene-linear float, so the conversion is lossless unless
--half is asked for.

Examples
--------
    # every .pfm under build/, writing .exr next to each source
    python scripts/pfm_to_exr.py build

    # same set, mirrored under a single output directory
    python scripts/pfm_to_exr.py build -o exr

    # one file, half float, overwriting whatever is already there
    python scripts/pfm_to_exr.py render.pfm --half --overwrite
"""

from __future__ import annotations

import argparse
import glob as globlib
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

MAGIC = 20000630
VERSION = 2

PIXEL_HALF = 1
PIXEL_FLOAT = 2

COMPRESSION_NONE = 0
COMPRESSION_ZIP = 3

# Scanlines per chunk, as fixed by the OpenEXR compression scheme.
LINES_PER_BLOCK = {COMPRESSION_NONE: 1, COMPRESSION_ZIP: 16}

CHANNEL_NAMES = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}


def read_pfm(path):
    """Read a PFM file into a float32 array of shape (h, w, c), row 0 = top."""
    with open(path, "rb") as handle:
        magic = handle.readline().rstrip()
        color = magic == b"PF"
        if not color and magic != b"Pf":
            raise ValueError(f"not a PFM file (magic {magic[:8]!r})")

        try:
            width, height = (int(token) for token in handle.readline().split())
        except ValueError:
            raise ValueError("malformed PFM size line") from None
        if width <= 0 or height <= 0:
            raise ValueError(f"bad PFM dimensions {width}x{height}")

        # A negative scale means the raster is little-endian; the magnitude is
        # the value scale factor and is almost always 1.0.
        scale = float(handle.readline())
        endian = "<" if scale < 0 else ">"

        channels = 3 if color else 1
        count = width * height * channels
        raster = np.frombuffer(handle.read(count * 4), dtype=endian + "f4")
        if raster.size != count:
            raise ValueError(f"truncated raster: {raster.size} of {count} floats")

    # PFM stores scanlines bottom-up, so flip to the top-down EXR convention.
    image = np.ascontiguousarray(
        raster.reshape(height, width, channels)[::-1], dtype=np.float32
    )
    if abs(scale) != 1.0:
        image = image * abs(scale)
    return image


def _attribute(name, type_name, payload):
    return (
        name.encode("ascii")
        + b"\x00"
        + type_name.encode("ascii")
        + b"\x00"
        + struct.pack("<i", len(payload))
        + payload
    )


def _chlist(channels):
    out = bytearray()
    for name, pixel_type in channels:
        out += name.encode("ascii") + b"\x00"
        out += struct.pack("<i", pixel_type)
        out += b"\x00\x00\x00\x00"  # pLinear plus three reserved bytes
        out += struct.pack("<ii", 1, 1)  # xSampling, ySampling
    out += b"\x00"
    return bytes(out)


def _zip_pack(raw):
    """OpenEXR's ZIP filter: byte shuffle, delta predictor, then deflate.

    The shuffle splits the buffer into its even- and odd-indexed bytes, which
    groups the low and high halves of each float; the predictor then replaces
    every byte with its difference from the previous one, which is what makes
    the result compress well.
    """
    source = np.frombuffer(raw, dtype=np.uint8)
    count = source.size
    if count == 0:
        return zlib.compress(b"", 6)

    half = (count + 1) // 2
    shuffled = np.empty(count, dtype=np.uint8)
    shuffled[:half] = source[0::2]
    shuffled[half:] = source[1::2]

    if count > 1:
        delta = shuffled[1:].astype(np.int32) - shuffled[:-1].astype(np.int32)
        shuffled[1:] = (delta + 128) & 0xFF

    return zlib.compress(shuffled.tobytes(), 6)


def write_exr(path, image, half=False, compression="zip"):
    """Write a scanline OpenEXR file. `image` is (h, w) or (h, w, c), row 0 = top."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError("expected a 2D or 3D image array")

    height, width, channels = image.shape
    if channels not in CHANNEL_NAMES:
        raise ValueError(f"unsupported channel count {channels}")
    if width <= 0 or height <= 0:
        raise ValueError(f"bad dimensions {width}x{height}")

    names = CHANNEL_NAMES[channels]
    # OpenEXR keeps the channel list sorted by name.
    order = sorted(range(channels), key=names.__getitem__)
    pixel_type = PIXEL_HALF if half else PIXEL_FLOAT
    codec = COMPRESSION_ZIP if compression == "zip" else COMPRESSION_NONE

    header = bytearray()
    header += _attribute(
        "channels", "chlist", _chlist([(names[c], pixel_type) for c in order])
    )
    header += _attribute("compression", "compression", struct.pack("<B", codec))
    window = struct.pack("<4i", 0, 0, width - 1, height - 1)
    header += _attribute("dataWindow", "box2i", window)
    header += _attribute("displayWindow", "box2i", window)
    header += _attribute("lineOrder", "lineOrder", struct.pack("<B", 0))
    header += _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attribute("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
    header += _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\x00"

    # Scanline payload is stored y-major, then channel, then x, so transpose the
    # per-pixel channels into their own planes before serialising.
    dtype = np.float16 if half else np.float32
    planar = np.ascontiguousarray(
        image[:, :, order].transpose(0, 2, 1), dtype=dtype
    )

    lines_per_block = LINES_PER_BLOCK[codec]
    blocks = []
    for first in range(0, height, lines_per_block):
        raw = planar[first : first + lines_per_block].tobytes()
        if codec == COMPRESSION_ZIP:
            packed = _zip_pack(raw)
            # A compressed chunk may never be larger than its uncompressed size:
            # when deflate does not pay off (noisy renders often do not), the raw
            # bytes are stored and readers recognise that by the equal size.
            blocks.append((first, packed if len(packed) < len(raw) else raw))
        else:
            blocks.append((first, raw))

    offset = 8 + len(header) + 8 * len(blocks)
    offsets = []
    for _, data in blocks:
        offsets.append(offset)
        offset += 8 + len(data)

    with open(path, "wb") as handle:
        handle.write(struct.pack("<II", MAGIC, VERSION))
        handle.write(header)
        handle.write(struct.pack(f"<{len(offsets)}Q", *offsets))
        for first, data in blocks:
            handle.write(struct.pack("<ii", first, len(data)))
            handle.write(data)

    return len(offsets)


def collect(inputs):
    """Expand files, directories and glob patterns into a de-duplicated list."""
    found = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            matches = sorted(path.rglob("*.pfm"))
        elif path.is_file():
            matches = [path]
        else:
            matches = sorted(
                Path(m) for m in globlib.glob(item, recursive=True) if Path(m).is_file()
            )
        if not matches:
            print(f"warning: nothing matched {item!r}", file=sys.stderr)
        found.extend(matches)

    unique, seen = [], set()
    for path in found:
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Examples") :],
    )
    parser.add_argument(
        "inputs", nargs="+", help="PFM files, directories to scan recursively, or globs"
    )
    parser.add_argument(
        "-o", "--outdir", help="output directory (default: alongside each source)"
    )
    parser.add_argument(
        "--half", action="store_true", help="store HALF instead of FLOAT pixels"
    )
    parser.add_argument(
        "--compression",
        choices=("zip", "none"),
        default="zip",
        help="EXR compression (default: zip)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .exr files"
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="list what would be written"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report totals")
    args = parser.parse_args(argv)

    sources = collect(args.inputs)
    if not sources:
        print("no PFM files found", file=sys.stderr)
        return 1

    # Mirror each source's path relative to the common parent of all inputs, so
    # a recursive run keeps its directory structure instead of colliding names.
    try:
        root = Path(os.path.commonpath([str(p.resolve().parent) for p in sources]))
    except ValueError:
        root = Path.cwd()
    outdir = Path(args.outdir) if args.outdir else root

    converted = skipped = failed = 0
    written_bytes = 0

    for source in sources:
        relative = source.resolve().parent.relative_to(root)
        target = outdir / relative / (source.stem + ".exr")

        if args.dry_run:
            print(f"{source} -> {target}")
            continue
        if target.exists() and not args.overwrite:
            skipped += 1
            if not args.quiet:
                print(f"skip {target} (exists)")
            continue

        try:
            image = read_pfm(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_exr(target, image, half=args.half, compression=args.compression)
        except Exception as error:  # keep going through the rest of the batch
            failed += 1
            print(f"FAIL {source}: {error}", file=sys.stderr)
            continue

        converted += 1
        written_bytes += target.stat().st_size
        if not args.quiet:
            kind = "half" if args.half else "float"
            print(
                f"{source} -> {target} "
                f"({image.shape[1]}x{image.shape[0]}x{image.shape[2]}, "
                f"{kind}, {target.stat().st_size / 1024:.1f} KiB)"
            )

    if args.dry_run:
        print(f"{len(sources)} file(s) would be converted")
        return 0

    print(
        f"converted {converted}, skipped {skipped}, failed {failed} "
        f"({written_bytes / 1024:.1f} KiB written)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
