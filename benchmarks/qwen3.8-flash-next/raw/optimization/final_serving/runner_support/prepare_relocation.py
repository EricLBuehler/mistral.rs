"""Prepare or apply the cuTile GGUF prototype move into shared test support."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path("/home/ericbuehler/mistral.rs")
OLD = "mistralrs-quant/src/cutile/gguf_moe.rs"
NEW = "mistralrs-quant/tests/support/gguf_moe.rs"
MODULE = "mistralrs-quant/src/cutile/mod.rs"
PROJECTION = "mistralrs-quant/tests/cutile_gguf_moe_tests.rs"
REPLAY = "mistralrs-quant/tests/support/cutile_gguf_replay.rs"
PARENT = "mistralrs-quant/tests/moe_dispatch_bench.rs"


def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    before = {
        name: (ROOT / name).read_text()
        for name in (OLD, MODULE, PROJECTION, REPLAY, PARENT)
    }
    source_hash = hashlib.sha256(before[OLD].encode()).hexdigest()
    assert source_hash == args.expected_source_sha256, source_hash
    assert not (ROOT / NEW).exists(), NEW
    after = dict(before)
    source = replace(
        before[OLD],
        "use candle_core::cuda::cudarc::driver::CudaSlice;",
        "use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};",
    )
    source = replace(
        source,
        "use super::{catch_cutile_panic, context};\nuse crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};",
        "use mistralrs_quant::cutile::context;",
    )
    source = replace(
        source,
        "let (input_ptr, _input_guard) = slice_ptr_on_stream(input, layout.start_offset(), &stream);",
        "let (input_ptr, _input_guard) = input.device_ptr(&stream);\n    let input_ptr = input_ptr + (layout.start_offset() * std::mem::size_of::<bf16>()) as u64;",
    )
    source = replace(
        source,
        "slice_ptr_mut_on_stream(&mut output, 0, &stream)",
        "output.device_ptr_mut(&stream)",
    )
    for name in ("sorted_token_ids", "expert_ids", "num_tokens_post_pad"):
        source = replace(
            source,
            f"slice_ptr_on_stream(args.{name}, 0, &stream)",
            f"args.{name}.device_ptr(&stream)",
        )
    helper = """fn catch_cutile_panic<T>(
    operation: &str,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            candle_core::bail!("cuTile {operation} panicked: {message}")
        }
    }
}

"""
    source = replace(source, "#[cutile::module]", helper + "#[cutile::module]")
    del after[OLD]
    after[NEW] = source
    after[MODULE] = replace(before[MODULE], "pub mod gguf_moe;\n", "")
    after[PROJECTION] = replace(
        before[PROJECTION],
        "use mistralrs_quant::cutile::gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection};",
        '#[path = "support/gguf_moe.rs"]\nmod gguf_moe;\nuse gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection};',
    )
    after[PARENT] = replace(
        before[PARENT],
        '#[cfg(feature = "cutile")]\n#[path = "support/cutile_gguf_replay.rs"]',
        '#[cfg(feature = "cutile")]\n#[path = "support/gguf_moe.rs"]\nmod gguf_moe;\n\n#[cfg(feature = "cutile")]\n#[path = "support/cutile_gguf_replay.rs"]',
    )
    after[REPLAY] = replace(
        before[REPLAY],
        "use mistralrs_quant::{\n    cutile::gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection},\n    fused_glu,\n    moe::cuda::moe_align,\n};",
        "use super::gguf_moe::{gguf_moe_projection, GgufMoeConfig, GgufMoeProjection};\nuse mistralrs_quant::{fused_glu, moe::cuda::moe_align};",
    )
    args.output.mkdir(parents=True, exist_ok=False)
    patch = []
    for name in sorted(before.keys() | after.keys()):
        patch.extend(
            difflib.unified_diff(
                before.get(name, "").splitlines(keepends=True),
                after.get(name, "").splitlines(keepends=True),
                fromfile=f"a/{name}" if name in before else "/dev/null",
                tofile=f"b/{name}" if name in after else "/dev/null",
            )
        )
        for kind, files in (("before", before), ("after", after)):
            if name in files:
                target = args.output / kind / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(files[name])
    (args.output / "relocation.patch").write_text("".join(patch))
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "original_source_sha256": source_hash,
                "changed_files": sorted(before.keys() | after.keys()),
                "applied": args.apply,
                "semantic_scope": "Module location, equivalent guarded pointer access, and local panic conversion only; device kernel and launch configuration unchanged.",
            },
            indent=2,
        )
        + "\n"
    )
    if args.apply:
        for name, contents in after.items():
            (ROOT / name).write_text(contents)
        (ROOT / OLD).unlink()
    print(args.output / "relocation.patch")


if __name__ == "__main__":
    main()
