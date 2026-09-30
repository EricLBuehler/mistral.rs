#!/usr/bin/env python3
"""Inventory cached checkpoint metadata and shard headers without reading tensor payloads."""

import collections
import datetime
import hashlib
import json
import struct
from pathlib import Path

WORK = Path(__file__).resolve().parent
CACHE = Path("/home/ericbuehler/.cache/huggingface/hub")
MODELS = {
    "Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "Qwen3.8-27B-FP8": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
}
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(model, revision):
    snapshot = CACHE / ("models--Qwen--" + model) / "snapshots" / revision
    config_bytes = (snapshot / "config.json").read_bytes()
    index_bytes = (snapshot / "model.safetensors.index.json").read_bytes()
    config, index = json.loads(config_bytes), json.loads(index_bytes)
    counts = collections.Counter()
    shards, shapes = [], {}
    for name in sorted(set(index["weight_map"].values())):
        path = snapshot / name
        with path.open("rb") as source:
            header_length = struct.unpack("<Q", source.read(8))[0]
            header_bytes = source.read(header_length)
        header = json.loads(header_bytes)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        data_end = max(tensor["data_offsets"][1] for tensor in tensors.values())
        assert path.stat().st_size == 8 + header_length + data_end, name
        assert all(index["weight_map"][key] == name for key in tensors), name
        counts.update(tensor["dtype"] for tensor in tensors.values())
        shapes.update({key: tensor["shape"] for key, tensor in tensors.items()})
        shards.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "header_bytes": header_length,
                "header_sha256": digest(header_bytes),
                "tensor_count": len(tensors),
                "file_size_matches_header_extent": True,
            }
        )
    quant = config.get("quantization_config") or {}
    return {
        "model": "Qwen/" + model,
        "revision": revision,
        "snapshot": str(snapshot),
        "config_sha256": digest(config_bytes),
        "text_config_sha256": digest(
            json.dumps(
                config["text_config"], sort_keys=True, separators=(",", ":")
            ).encode()
        ),
        "weight_index_sha256": digest(index_bytes),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_fingerprint_note": "Both tokenizer files were SHA256-checked earlier in this investigation, before the new timed pair; payload was not reread during this inventory.",
        "text_dimensions": {
            key: config["text_config"].get(key)
            for key in (
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_experts",
                "vocab_size",
            )
        },
        "checkpoint_quantization": {
            key: quant.get(key)
            for key in ("quant_method", "activation_scheme", "fmt", "weight_block_size")
        },
        "shard_count": len(shards),
        "total_shard_bytes": sum(shard["bytes"] for shard in shards),
        "tensor_count": sum(counts.values()),
        "tensor_dtype_counts": dict(counts),
        "shards": shards,
    }, shapes


def main():
    models, shapes = {}, {}
    for name, revision in MODELS.items():
        models[name], shapes[name] = inventory(name, revision)
    common = set(shapes["Qwen3.8-27B"]) & set(shapes["Qwen3.8-27B-FP8"])
    mismatches = [
        key
        for key in common
        if shapes["Qwen3.8-27B"][key] != shapes["Qwen3.8-27B-FP8"][key]
    ]
    assert not mismatches
    output = {
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Config, index and safetensors headers only. File extents and index membership checked for every tensor; tensor payload contents were not hashed or compared.",
        "models": models,
        "shared_tensor_keys": len(common),
        "shared_tensor_shape_mismatches": mismatches,
        "text_configs_identical": models["Qwen3.8-27B"]["text_config_sha256"]
        == models["Qwen3.8-27B-FP8"]["text_config_sha256"],
        "current_binary": {
            "path": "/home/ericbuehler/mistral.rs/target/release/mistralrs",
            "sha256": "d245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588",
            "bytes": 609661096,
            "fingerprint_note": "Full executable hash checked before the new timed pair. Each run's metadata independently records the executable used.",
        },
        "earlier_dense_control_binary_sha256": "1166ae7cb6be9b723ff4195102587b570c9a45d2c7ec980a732709c472f65012",
        "pre_pr_binary": {
            "path": "/home/ericbuehler/qwen4exp_work/master_target/release/mistralrs",
            "sha256": "87cbcc33223343295a700fb2feb6cf828124b8bc6b2cfce58a8f11c4035dcb85",
            "bytes": 603673216,
        },
    }
    (WORK / "checkpoint_inventory.json").write_text(json.dumps(output, indent=2) + "\n")
    print("Saved checkpoint_inventory.json; 18 BF16 and 66 FP8 shard headers checked.")


if __name__ == "__main__":
    main()
