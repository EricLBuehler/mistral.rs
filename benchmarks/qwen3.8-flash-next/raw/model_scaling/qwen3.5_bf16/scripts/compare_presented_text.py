"""Compare decoded text with explicitly bounded presentation normalization."""

import json
from pathlib import Path
import statistics

from compare_bf16 import ROOT, digest, require

SPECIAL_MARKERS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
PREFIX_LENGTHS = (32, 64, 128, 256)


def normalize(text):
    for marker in SPECIAL_MARKERS:
        text = text.replace(marker, "")
    return text.lstrip()


def common_prefix(a, b):
    return next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def summarize(rows):
    return {
        "compared": len(rows),
        "raw_identical": sum(row["raw_identical"] for row in rows),
        "leading_whitespace_stripped_identical": sum(
            row["lstrip_identical"] for row in rows
        ),
        "normalized_identical": sum(row["normalized_identical"] for row in rows),
        "normalized_matching_prefixes": {
            str(n): sum(row["normalized_common_prefix_characters"] >= n for row in rows)
            for n in PREFIX_LENGTHS
        },
        "normalized_common_prefix_characters": {
            "min": min(row["normalized_common_prefix_characters"] for row in rows),
            "median": statistics.median(
                row["normalized_common_prefix_characters"] for row in rows
            ),
            "max": max(row["normalized_common_prefix_characters"] for row in rows),
        },
    }


def main():
    comparison = json.loads((ROOT / "comparison/comparison.json").read_text())
    require(comparison["complete"], "Complete protocol/statistics validation first")
    output = ROOT / "comparison/presented_text.json"
    require(not output.exists(), "Refusing to overwrite text diagnostic")
    rows, sources = [], {}
    for concurrency in (1, 6, 8):
        paths = [
            Path(comparison["run_directories"][engine])
            / f"concurrency.c{concurrency}.json"
            for engine in ("mistralrs", "vllm")
        ]
        sources.update({str(path): digest(path) for path in paths})
        runs = [
            json.loads(path.read_text())["cases"][f"closed-loop_c{concurrency}"]["runs"]
            for path in paths
        ]
        for native, vllm in zip(*runs, strict=True):
            require(native["warmup"] == vllm["warmup"], "Warmup mismatch")
            if native["warmup"]:
                continue
            for a, b in zip(native["requests"], vllm["requests"], strict=True):
                require(a["request"] == b["request"], "Request mismatch")
                x, y = (item["response"]["choices"][0]["text"] for item in (a, b))
                nx, ny = normalize(x), normalize(y)
                rows.append(
                    {
                        "concurrency": concurrency,
                        "trial": native["trial"],
                        "request_index": a["request_index"],
                        "prompt_name": a["prompt_name"],
                        "raw_identical": x == y,
                        "lstrip_identical": x.lstrip() == y.lstrip(),
                        "normalized_identical": nx == ny,
                        "normalized_common_prefix_characters": common_prefix(nx, ny),
                        "normalized_lengths": [len(nx), len(ny)],
                    }
                )
    result = {
        "complete": True,
        "source_sha256": sources,
        "analyzer_sha256": digest(Path(__file__)),
        "comparison_sha256": digest(ROOT / "comparison/comparison.json"),
        "normalization": {
            "remove_exact_markers": list(SPECIAL_MARKERS),
            "then": "str.lstrip(): remove leading whitespace only",
            "preserved": "All other characters, internal whitespace, indentation, and case",
        },
        "limitations": [
            "Decoded text presentation differs across engines; this is not generated-token equality or expert-route evidence.",
            "Removing visible markers can make distinct strings equal; no tokenizer roundtrip or token identity is inferred.",
            "Fixed prefix lengths count Unicode characters, not tokens, and do not assess quality or semantic equivalence.",
            "Only five measured trials are compared; both warmups remain excluded.",
        ],
        "all": summarize(rows),
        "by_concurrency": {
            str(c): summarize([row for row in rows if row["concurrency"] == c])
            for c in (1, 6, 8)
        },
        "by_prompt": {
            name: summarize([row for row in rows if row["prompt_name"] == name])
            for name in sorted({row["prompt_name"] for row in rows})
        },
        "requests": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["all"]))


if __name__ == "__main__":
    main()
