#!/usr/bin/env python3
"""Fetch two pinned NVFP4 projections and independent CPU reference outputs."""

import argparse
import json
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

BLOCK_SIZE = 16
VALUES_PER_BYTE = 2
HEADER_PREFIX_BYTES = 8
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_PROJECTION_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30
FETCH_WORKERS = 4
INPUT_DIM = 512
OUTPUT_DIM = 2048
INPUT_MULTIPLIER = 7
INPUT_MODULUS = 19
INPUT_OFFSET = 9
INPUT_DIVISOR = 8
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CT_INPUT_VALUES = (-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0)


@dataclass(frozen=True)
class Source:
    stem: str
    repo: str
    revision: str
    shard: str
    prefix: str
    modelopt: bool

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.shard}"
        )

    @property
    def names(self) -> tuple[str, ...]:
        if self.modelopt:
            return "weight", "weight_scale", "weight_scale_2", "input_scale"
        return (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        )


SOURCES = (
    Source(
        "modelopt-qwen3.6-expert-down",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "1355db6a052410cfd62085d94b58866fd0f2c3c5",
        "model-00001-of-00003.safetensors",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
        True,
    ),
    Source(
        "ct-moe-expert0-down",
        "RedHatAI/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "b6787aea86799bf50a6e98ee61cd9398b5875578",
        "model-00001-of-00010.safetensors",
        "model.layers.0.mlp.experts.0.down_proj",
        False,
    ),
)


def read_range(url: str, start: int, end: int) -> bytes:
    byte_range = f"{start}-{end - 1}"
    request = urllib.request.Request(
        f"{url}?mistralrs_nvfp4_fixture={byte_range}",
        headers={"Range": f"bytes={byte_range}"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if response.status != 206 or not response.headers.get(
            "Content-Range", ""
        ).startswith(f"bytes {byte_range}/"):
            raise RuntimeError(f"server did not honor byte range {byte_range}")
        data = response.read(end - start + 1)
    if len(data) != end - start:
        raise RuntimeError(f"unexpected response length for byte range {byte_range}")
    return data


def fetch_projection(source: Source) -> tuple[dict, dict[str, bytes]]:
    prefix = read_range(source.url, 0, HEADER_PREFIX_BYTES)
    header_size = struct.unpack("<Q", prefix)[0]
    if header_size > MAX_HEADER_BYTES:
        raise ValueError(f"safetensors header exceeds {MAX_HEADER_BYTES} bytes")
    data_start = HEADER_PREFIX_BYTES + header_size
    header = json.loads(read_range(source.url, HEADER_PREFIX_BYTES, data_start))
    selected = {name: header[f"{source.prefix}.{name}"] for name in source.names}
    tensor_bytes = sum(
        item["data_offsets"][1] - item["data_offsets"][0] for item in selected.values()
    )
    if tensor_bytes > MAX_PROJECTION_BYTES:
        raise ValueError(f"selected projection exceeds {MAX_PROJECTION_BYTES} bytes")
    weight_name, scale_name, *global_names = source.names
    expected_shapes = {
        weight_name: ("U8", [OUTPUT_DIM, INPUT_DIM // VALUES_PER_BYTE]),
        scale_name: ("F8_E4M3", [OUTPUT_DIM, INPUT_DIM // BLOCK_SIZE]),
        **{name: ("F32", [] if source.modelopt else [1]) for name in global_names},
    }
    for name, (dtype, shape) in expected_shapes.items():
        if selected[name]["dtype"] != dtype or selected[name]["shape"] != shape:
            raise ValueError(f"unexpected dtype or shape for {source.prefix}.{name}")

    def fetch(name: str) -> tuple[str, bytes]:
        start, end = selected[name]["data_offsets"]
        return name, read_range(source.url, data_start + start, data_start + end)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        tensors = dict(pool.map(fetch, source.names))
    return selected, tensors


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp4(bits: int) -> float:
    return FP4_VALUES[bits & 7] * (-1 if bits & 8 else 1)


def fp8(bits: int) -> float:
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    magnitude = (
        mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
    )
    return magnitude * (-1 if bits & 128 else 1)


def reference(source: Source, tensors: dict[str, bytes]) -> dict:
    weight_name, scale_name, global_name, input_name = source.names
    weight_global = struct.unpack("<f", tensors[global_name])[0]
    input_global = struct.unpack("<f", tensors[input_name])[0]
    if source.modelopt:
        inputs = [
            ((index * INPUT_MULTIPLIER) % INPUT_MODULUS - INPUT_OFFSET) / INPUT_DIVISOR
            for index in range(INPUT_DIM)
        ]
    else:
        weight_global = f32(1 / weight_global)
        input_global = f32(1 / input_global)
        # Exact E2M1 levels make each activation block's E4M3 scale equal to one.
        inputs = [f32(value * input_global) for value in CT_INPUT_VALUES] * (
            INPUT_DIM // BLOCK_SIZE
        )
    packed = tensors[weight_name]
    scales = [fp8(value) for value in tensors[scale_name]]
    outputs = []
    for row in range(OUTPUT_DIM):
        result = 0.0
        for column, value in enumerate(inputs):
            byte = packed[
                row * INPUT_DIM // VALUES_PER_BYTE + column // VALUES_PER_BYTE
            ]
            nibble = (byte >> (4 * (column % VALUES_PER_BYTE))) & 15
            block_scale = scales[row * INPUT_DIM // BLOCK_SIZE + column // BLOCK_SIZE]
            result += fp4(nibble) * block_scale * weight_global * value
        outputs.append(result)
    return {
        "repo": source.repo,
        "revision": source.revision,
        "source_shard": source.shard,
        "prefix": source.prefix,
        "in_dim": INPUT_DIM,
        "out_dim": OUTPUT_DIM,
        "algorithm": "W4A16_NVFP4" if source.modelopt else "NVFP4",
        "input": inputs,
        "expected": outputs,
    }


def save_projection(source: Source, directory: Path) -> None:
    metadata, tensors = fetch_projection(source)
    output_header = {}
    data = bytearray()
    for name in source.names:
        start = len(data)
        data.extend(tensors[name])
        output_header[f"{source.prefix}.{name}"] = {
            "dtype": metadata[name]["dtype"],
            "shape": metadata[name]["shape"],
            "data_offsets": [start, len(data)],
        }
    header = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % HEADER_PREFIX_BYTES)
    path = directory / f"{source.stem}.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    manifest = reference(source, tensors)
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {path} ({path.stat().st_size} bytes) and reference manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_directory",
        type=Path,
        help="directory for both extracted projections and manifests",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        save_projection(source, args.output_directory)


if __name__ == "__main__":
    main()
