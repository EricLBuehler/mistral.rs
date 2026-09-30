#!/usr/bin/env python3
"""Build portable release data and static figures from committed benchmark evidence."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
import numpy
import PIL

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RELEASE = Path(__file__).resolve().parents[1]
BENCH = Path("benchmarks/qwen3.8-flash-next")
GRAPH = BENCH / "raw/model_scaling/graph_candidate"
BF16 = BENCH / "raw/model_scaling/qwen3.5_bf16"
PROMPTS = (
    "python",
    "rust",
    "prose",
    "json",
    "primes",
    "math",
    "translation",
    "quicksort",
)
PREFILLS = ("pp512", "pp2048", "pp8192")
CONCURRENCIES = (1, 6, 8)
MEASURED_TRIALS = 5
WARMUPS = 2
OUTPUT_TOKENS = 128
COLORS = {"mistralrs": "#2563eb", "llama_patched": "#6b7280", "vllm": "#6b7280"}
LABELS = {
    "mistralrs": "mistral.rs",
    "llama_patched": "llama.cpp (patched)",
    "vllm": "vLLM",
}
SECONDARY = (
    "latency_weighted_request_output_tokens_per_second",
    "mean_active_requests",
    "mean_request_wall_seconds",
)
DPI = 220


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    require(
        math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
        f"Value mismatch: {actual} != {expected}",
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Evidence:
    def __init__(self, repo):
        self.repo = repo
        self.sources = {}
        self.cells = []
        self.trials = []

    def read(self, relative, role):
        path = self.repo / relative
        self.sources[str(relative)] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
        return (
            json.loads(path.read_text()) if path.suffix == ".json" else path.read_text()
        )

    def add(self, spec, trials, expected):
        require(len(trials) == MEASURED_TRIALS, "Expected five measured trials")
        values = [row["value"] for row in trials]
        require(all(math.isfinite(x) and x > 0 for x in values), "Invalid throughput")
        mean, deviation = statistics.mean(values), statistics.stdev(values)
        close(mean, expected.get("mean", expected.get("mean_tokens_per_second")))
        close(deviation, expected["sample_stddev"])
        for actual, saved in zip(values, expected["samples"], strict=True):
            close(actual, saved)
        cell_id = f"{spec['family']}:{spec['engine']}:{spec['workload']}"
        cell = {
            "id": cell_id,
            **spec,
            "mean": mean,
            "sample_stddev": deviation,
            "samples": values,
            "measured_trials": MEASURED_TRIALS,
            "discarded_warmups": WARMUPS,
        }
        if all(name in trials[0] for name in SECONDARY):
            cell["secondary"] = {
                name: {
                    "mean": statistics.mean(row[name] for row in trials),
                    "sample_stddev": statistics.stdev(row[name] for row in trials),
                }
                for name in SECONDARY
            }
        self.cells.append(cell)
        self.trials.extend(
            {"cell_id": cell_id, "source_path": spec["source_path"], **row}
            for row in trials
        )


def measured(rows):
    require(
        [row["warmup"] for row in rows] == [True] * WARMUPS + [False] * MEASURED_TRIALS,
        "Unexpected warmup/trial sequence",
    )
    return list(enumerate(rows))[WARMUPS:]


def input_identity(requests):
    return [
        (row["prompt_sha256"], row["response"]["usage"]["prompt_tokens"])
        for row in requests
    ]


def serving_trials(data, workload):
    trials, identities = [], []
    for index, row in measured(data["cases"][workload]):
        requests = row["requests"] if workload == "concurrency8" else [row]
        outputs = [
            request["response"]["usage"]["completion_tokens"] for request in requests
        ]
        inputs = [request["response"]["usage"]["prompt_tokens"] for request in requests]
        expected_output = 1 if workload in PREFILLS else OUTPUT_TOKENS
        require(
            all(n == expected_output for n in outputs), "Incorrect completion count"
        )
        require(
            len(requests) == (8 if workload == "concurrency8" else 1),
            "Incorrect burst size",
        )
        numerator = inputs[0] if workload in PREFILLS else sum(outputs)
        if workload in PREFILLS:
            require(inputs == [int(workload[2:])], "Incorrect prefill count")
        wall = row["wall_seconds"]
        require(math.isfinite(wall) and wall > 0, "Invalid trial wall time")
        trials.append(
            {
                "trial": index - WARMUPS,
                "source_pointer": f"/cases/{workload}/{index}",
                "value": numerator / wall,
                "wall_seconds": wall,
                "numerator_tokens": numerator,
                "input_tokens": sum(inputs),
                "output_tokens": sum(outputs),
                "request_count": len(requests),
            }
        )
        identities.append(input_identity(requests))
    return trials, identities


def closed_loop_trials(data, concurrency):
    name = f"closed-loop_c{concurrency}"
    require(data["complete"], "Incomplete closed-loop result")
    trials, identities = [], []
    for index, row in measured(data["cases"][name]["runs"]):
        require(row["complete"], "Incomplete trial")
        requests = row["requests"]
        require(
            len(requests) == (8 if concurrency == 1 else 24),
            "Incorrect closed-loop request count",
        )
        require(
            all(
                request["response"]["usage"]["completion_tokens"] == OUTPUT_TOKENS
                for request in requests
            ),
            "Incorrect completion count",
        )
        tokens = sum(
            request["response"]["usage"]["completion_tokens"] for request in requests
        )
        latency = sum(request["wall_seconds"] for request in requests)
        wall = row["wall_seconds"]
        require(
            math.isfinite(wall) and wall > 0 and latency > 0,
            "Invalid closed-loop duration",
        )
        trial = {
            "trial": index - WARMUPS,
            "source_pointer": f"/cases/{name}/runs/{index}",
            "value": tokens / wall,
            "wall_seconds": wall,
            "numerator_tokens": tokens,
            "input_tokens": sum(
                request["response"]["usage"]["prompt_tokens"] for request in requests
            ),
            "output_tokens": tokens,
            "request_count": len(requests),
            "latency_weighted_request_output_tokens_per_second": tokens / latency,
            "mean_active_requests": latency / wall,
            "mean_request_wall_seconds": latency / len(requests),
        }
        close(trial["value"], row["aggregate_output_tokens_per_second"])
        for field in SECONDARY:
            close(trial[field], row[field])
        trials.append(trial)
        identities.append(input_identity(requests))
    return trials, identities


def collect(repo):
    evidence = Evidence(repo)
    reference = evidence.read(
        BENCH / "summary.json", "Validated same-GGUF serving summary"
    )
    require(reference["validation"]["complete"], "Serving validation incomplete")
    gguf_inputs = {}
    for engine, filename in (
        ("llama_patched", "llama_patched.json"),
        ("mistralrs", "gguf.json"),
    ):
        source = BENCH / "raw" / filename
        data = evidence.read(source, "Raw serving repetitions")
        require(
            data["settings"]["iterations"] == MEASURED_TRIALS
            and data["settings"]["warmup"] == WARMUPS,
            "Serving protocol changed",
        )
        for workload in (*PREFILLS, *PROMPTS, "concurrency8"):
            trials, identities = serving_trials(data, workload)
            key = "gguf" if engine == "mistralrs" else engine
            prefill = workload in PREFILLS
            evidence.add(
                {
                    "family": "flash_next_gguf",
                    "engine": engine,
                    "workload": workload,
                    "concurrency": 8 if workload == "concurrency8" else 1,
                    "protocol": "prefill_one_output"
                    if prefill
                    else "finite_burst"
                    if workload == "concurrency8"
                    else "single_request",
                    "unit": "input_tokens_per_second"
                    if prefill
                    else "output_tokens_per_second",
                    "source_path": str(source),
                },
                trials,
                reference["throughput"][key][workload],
            )
            if workload in gguf_inputs:
                require(identities == gguf_inputs[workload], "Same-GGUF inputs differ")
            gguf_inputs[workload] = identities
    stage = evidence.read(
        GRAPH / "analysis/stage_comparison/comparison.json",
        "Validated final adaptive stage summary; use after only",
    )
    require(stage["validation"]["complete"], "Final stage validation incomplete")
    for concurrency in CONCURRENCIES:
        source = (
            GRAPH
            / "run"
            / ("concurrency.c1.json" if concurrency == 1 else "concurrency.c6_c8.json")
        )
        trials, _ = closed_loop_trials(
            evidence.read(source, "Raw final adaptive closed-loop trials"), concurrency
        )
        expected = next(
            row
            for row in stage["concurrency_changes"]
            if row["concurrency"] == concurrency
        )["after"]["metrics"]
        evidence.add(
            {
                "family": "flash_next_mtp",
                "engine": "mistralrs",
                "workload": f"c{concurrency}",
                "concurrency": concurrency,
                "protocol": "closed_loop",
                "unit": "output_tokens_per_second",
                "source_path": str(source),
            },
            trials,
            expected["aggregate_output_tokens_per_second"],
        )
        for name in SECONDARY:
            close(evidence.cells[-1]["secondary"][name]["mean"], expected[name]["mean"])
            close(
                evidence.cells[-1]["secondary"][name]["sample_stddev"],
                expected[name]["sample_stddev"],
            )
    bf16 = evidence.read(
        BF16 / "comparison.json", "Validated same-checkpoint BF16 comparison"
    )
    require(bf16["complete"], "BF16 comparison incomplete")
    bf16_inputs = {}
    for engine in ("mistralrs", "vllm"):
        for concurrency in CONCURRENCIES:
            source = BF16 / engine / f"concurrency.c{concurrency}.json"
            trials, identities = closed_loop_trials(
                evidence.read(source, "Raw same-checkpoint BF16 trials"), concurrency
            )
            expected = bf16["engines"][engine]["concurrencies"][str(concurrency)][
                "statistics"
            ]
            evidence.add(
                {
                    "family": "qwen35_bf16",
                    "engine": engine,
                    "workload": f"c{concurrency}",
                    "concurrency": concurrency,
                    "protocol": "closed_loop",
                    "unit": "output_tokens_per_second",
                    "source_path": str(source),
                },
                trials,
                expected["aggregate_output_tokens_per_second"],
            )
            for name in SECONDARY:
                close(
                    evidence.cells[-1]["secondary"][name]["mean"],
                    expected[name]["mean"],
                )
                close(
                    evidence.cells[-1]["secondary"][name]["sample_stddev"],
                    expected[name]["sample_stddev"],
                )
            if concurrency in bf16_inputs:
                require(
                    identities == bf16_inputs[concurrency],
                    "BF16 input hashes/counts differ",
                )
            bf16_inputs[concurrency] = identities
    provenance = {}
    for name, path in {
        "gguf": BENCH / "raw/gguf.metadata.json",
        "llama_patched": BENCH / "raw/llama_patched.metadata.json",
        "final_adaptive": GRAPH / "run/metadata.json",
        "bf16_mistralrs": BF16 / "mistralrs/metadata.json",
        "bf16_vllm": BF16 / "vllm/metadata.json",
    }.items():
        data = evidence.read(
            path, "Run command, binary/image and environment provenance"
        )
        provenance[name] = {
            "path": str(path),
            "sha256": evidence.sources[str(path)]["sha256"],
            "identity": {
                key: data[key]
                for key in (
                    "binary_sha256",
                    "git_head",
                    "source_diff_sha256",
                    "patch_sha256",
                    "original_shared_library_sha256",
                    "patched_library_sha256",
                )
                if key in data
            },
        }
        if name in ("final_adaptive", "bf16_mistralrs", "bf16_vllm"):
            require(data["complete"], f"Incomplete run: {name}")
    for path in (
        BENCH / "raw/llama-indexed-mmq-padding.patch",
        GRAPH / "SHA256SUMS.json",
        BF16 / "SHA256SUMS.json",
    ):
        evidence.read(path, "Patch or source archive manifest")
    identities = {
        "final_adaptive_binary_sha256": stage["provenance"]["after_binary_sha256"],
        "bf16": {engine: row["identity"] for engine, row in bf16["engines"].items()},
    }
    return evidence, provenance, identities


def cell(evidence, family, engine, workload):
    return next(
        row
        for row in evidence.cells
        if (row["family"], row["engine"], row["workload"]) == (family, engine, workload)
    )


def draw_bars(ax, evidence, family, *, engines, workloads, labels, annotate=True):
    width = 0.7 / len(engines)
    maximum = 0
    for index, engine in enumerate(engines):
        rows = [cell(evidence, family, engine, workload) for workload in workloads]
        positions = [
            x + (index - (len(engines) - 1) / 2) * width for x in range(len(rows))
        ]
        bars = ax.bar(
            positions,
            [row["mean"] for row in rows],
            width,
            color=COLORS[engine],
            label=LABELS[engine],
            yerr=[row["sample_stddev"] for row in rows],
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
        if annotate:
            ax.bar_label(
                bars,
                labels=[f"{row['mean']:.1f}" for row in rows],
                padding=5,
                fontsize=9,
            )
        maximum = max(maximum, *(row["mean"] + row["sample_stddev"] for row in rows))
    ax.set_xticks(range(len(workloads)), labels)
    ax.set_ylim(0, maximum * 1.23)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig, output, name):
    fig.savefig(
        output / f"{name}.png",
        dpi=DPI,
        metadata={"Software": "mistral.rs release benchmark generator"},
    )
    svg = output / f"{name}.svg"
    fig.savefig(
        svg,
        metadata={"Date": None, "Creator": "mistral.rs release benchmark generator"},
    )
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def plot(evidence, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "svg.hashsalt": "mistralrs-v0.9.5",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.8), gridspec_kw={"width_ratios": [3.2, 7.1, 1.8]}
    )
    engines = ("mistralrs", "llama_patched")
    draw_bars(
        axes[0],
        evidence,
        "flash_next_gguf",
        engines=engines,
        workloads=PREFILLS,
        labels=["512", "2,048", "8,192"],
    )
    axes[0].set(title="(a) Prefill", xlabel="Input tokens", ylabel="Input tokens/s")
    draw_bars(
        axes[1],
        evidence,
        "flash_next_gguf",
        engines=engines,
        workloads=PROMPTS,
        labels=["JSON" if name == "json" else name.title() for name in PROMPTS],
        annotate=False,
    )
    axes[1].set(title="(b) Ordinary single requests", ylabel="Output tokens/s")
    axes[1].tick_params(axis="x", rotation=30)
    draw_bars(
        axes[2],
        evidence,
        "flash_next_gguf",
        engines=engines,
        workloads=("concurrency8",),
        labels=["C8"],
    )
    axes[2].set(title="(c) Finite burst", ylabel="Aggregate output tokens/s")
    fig.suptitle("Qwen3.8-Flash-Next: same GGUF on NVIDIA GB10", fontsize=17, y=0.97)
    fig.text(
        0.5,
        0.915,
        "UD-Q4_K_XL | Target-only | Patched llama.cpp serving reference",
        ha="center",
        fontsize=11,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.045,
        "Mean +/- sample SD of 5 measured repetitions; 2 warmups excluded. HTTP timings include prompt processing and request overhead.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.76, wspace=0.34)
    save_figure(fig, output, "flash_next_gguf")
    for family, engines, title, subtitle in (
        (
            "flash_next_mtp",
            ("mistralrs",),
            "Qwen3.8-Flash-Next: Q4K + adaptive MTP",
            "NVIDIA GB10 | Mixed prompts | Replenished closed-loop traffic",
        ),
        (
            "qwen35_bf16",
            ("mistralrs", "vllm"),
            "Qwen3.5-35B-A3B BF16: same checkpoint",
            "NVIDIA GB10 | Target-only | Replenished closed-loop traffic",
        ),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        draw_bars(
            ax,
            evidence,
            family,
            engines=engines,
            workloads=[f"c{n}" for n in CONCURRENCIES],
            labels=[f"C{n}" for n in CONCURRENCIES],
        )
        ax.set(xlabel="Client concurrency", ylabel="Aggregate output tokens/s")
        ax.legend(frameon=False, loc="upper left")
        fig.suptitle(title, fontsize=16, y=0.96)
        fig.text(0.5, 0.9, subtitle, ha="center", fontsize=11)
        fig.text(
            0.5,
            0.07,
            "Mean +/- sample SD of 5 measured trials; 2 warmups excluded. 128 output tokens/request.",
            ha="center",
            fontsize=10,
        )
        footnote = "C1: 8 requests/trial; C6/C8: 24. Prompt processing and trial drain included."
        if family == "qwen35_bf16":
            footnote += " Cache and graph policies differ."
        fig.text(0.5, 0.03, footnote, ha="center", fontsize=9)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.83, bottom=0.2)
        save_figure(fig, output, family)


def tables(evidence):
    lines = [
        "# Release benchmark tables",
        "",
        "All cells are mean +/- sample standard deviation across five measured repetitions after two excluded warmups.",
        "",
        "## Same GGUF: target-only serving",
        "",
        "| Workload | Unit | mistral.rs | llama.cpp (patched) |",
        "| --- | --- | ---: | ---: |",
    ]
    for workload in (*PREFILLS, *PROMPTS, "concurrency8"):
        rows = [
            cell(evidence, "flash_next_gguf", engine, workload)
            for engine in ("mistralrs", "llama_patched")
        ]
        values = [f"{row['mean']:.3f} +/- {row['sample_stddev']:.3f}" for row in rows]
        unit = (
            "input tok/s"
            if workload in PREFILLS
            else "aggregate output tok/s"
            if workload == "concurrency8"
            else "output tok/s"
        )
        lines.append(f"| {workload} | {unit} | {' | '.join(values)} |")
    means = [
        statistics.mean(
            cell(evidence, "flash_next_gguf", engine, name)["mean"] for name in PROMPTS
        )
        for engine in ("mistralrs", "llama_patched")
    ]
    lines += [
        "",
        f"The arithmetic mean of the eight ordinary-prompt means is {means[0]:.3f} tok/s for mistral.rs and {means[1]:.3f} for patched llama.cpp. This is not a pooled throughput measurement; no repetition uncertainty is assigned to this cross-prompt mean.",
    ]
    for family, title, engines in (
        ("flash_next_mtp", "Final Flash-Next Q4K + adaptive MTP", ("mistralrs",)),
        (
            "qwen35_bf16",
            "Same-checkpoint Qwen3.5 BF16, target-only",
            ("mistralrs", "vllm"),
        ),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Engine | Concurrency | Aggregate output tok/s | Per-active output tok/s |",
            "| --- | ---: | ---: | ---: |",
        ]
        for concurrency in CONCURRENCIES:
            for engine in engines:
                row = cell(evidence, family, engine, f"c{concurrency}")
                active = row["secondary"][SECONDARY[0]]
                lines.append(
                    f"| {LABELS[engine]} | {concurrency} | {row['mean']:.3f} +/- {row['sample_stddev']:.3f} | {active['mean']:.3f} +/- {active['sample_stddev']:.3f} |"
                )
    return "\n".join(lines) + "\n"


def build(repo, output):
    evidence, provenance, identities = collect(repo)
    raw, figures = output / "raw", output / "figures"
    raw.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "cells": evidence.cells,
        "definitions": {
            "prefill_one_output": "Input tokens divided by HTTP request wall time for a one-token completion.",
            "single_request": "128 output tokens divided by complete HTTP request wall time, including prompt processing.",
            "finite_burst": "Eight requests started together; total output tokens divided by whole-burst wall time.",
            "closed_loop": "Completed request slots replenished; total output tokens divided by whole-trial wall time, including prompt processing and trial drain. Model loading is excluded.",
            "sample_stddev": "Sample standard deviation of five measured trial rates; not a confidence interval.",
            "per_active": "Output tokens divided by summed request latency; not aggregate rate divided by requested concurrency.",
        },
        "selection": "Same-GGUF baseline serving; final adaptive Flash-Next mixed closed-loop trials; same-checkpoint BF16 engine comparison. Homogeneous finite-wave controls and profiled timings are excluded.",
    }
    write_json(raw / "summary.json", summary)
    (raw / "results.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in evidence.trials
        )
    )
    fields = (
        "id",
        "family",
        "engine",
        "workload",
        "concurrency",
        "protocol",
        "unit",
        "mean",
        "sample_stddev",
        "measured_trials",
        "discarded_warmups",
        "source_path",
    )
    with (raw / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in fields} for row in evidence.cells)
    (raw / "tables.md").write_text(tables(evidence))
    plot(evidence, figures)
    outputs = {
        str(path.relative_to(output)): sha(path)
        for folder in (raw, figures)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    write_json(
        raw / "run_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "generator": "releases/v0.9.5/scripts/build_release.py",
            "generator_sha256": sha(Path(__file__)),
            "sources": evidence.sources,
            "run_provenance": provenance,
            "identities": identities,
            "outputs": outputs,
            "render_environment": {
                "python": platform.python_version(),
                "matplotlib": matplotlib.__version__,
                "numpy": numpy.__version__,
                "pillow": PIL.__version__,
                "freetype": matplotlib.ft2font.__freetype_version__,
                "backend": "Agg",
                "font": "DejaVu Sans",
                "dpi": DPI,
            },
            "validation": {
                "summary_cells": len(evidence.cells),
                "measured_trial_rows": len(evidence.trials),
                "raw_rates_recomputed": True,
                "saved_summary_means_sd_samples_match": True,
                "paired_input_hashes_and_counts_match": True,
            },
            "limitations": [
                "Different workload protocols are kept separate; no cross-workload ratio is generated.",
                "Same-GGUF comparison uses the documented llama.cpp allocation-padding patch and earlier baseline binaries.",
                "Final Flash-Next rates precede the completion-whitespace rendering fix; frozen binary identity remains recorded.",
                "BF16 engines use different physical cache allocations and graph policies despite matching checkpoint and request protocol.",
                "Source archives retain complete requests, validation, logs and large-artifact provenance; this release duplicates only compact derived data.",
                "Byte reproducibility of figures is checked in the recorded rendering environment; other library/font versions may render differently.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "cells": len(evidence.cells),
                "trials": len(evidence.trials),
                "output": str(output),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path, default=RELEASE)
    args = parser.parse_args()
    build(args.repo_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
