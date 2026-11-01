#!/usr/bin/env python3
"""Regenerate the v0.9.5 throughput figures from the aggregate results."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

RELEASE = Path(__file__).resolve().parents[1]
FIGURE_SIZE = (13.2, 6.6)
FIGURE_DPI = 180
BAR_WIDTH = 0.31
BAR_OFFSET = 0.18
COLORS = {"mistralrs": "#2563eb", "baseline": "#6b7280"}
METRICS = (
    ("prefill_tokens_s", "Effective prefill", 0),
    ("decode_tokens_s", "Decode", 2),
)


@dataclass(frozen=True)
class Comparison:
    key: str
    title: str
    model: str
    baseline_label: str
    footer: tuple[str, str]


COMPARISONS = (
    Comparison(
        "gguf",
        "GGUF serving throughput",
        "Qwen/Qwen3.8-Flash-Next",
        "llama.cpp (best recorded controls)",
        (
            "Same UD-Q4_K_XL checkpoint. Medians of three measured requests after warmup.",
            "Each llama.cpp bar selects its strongest recorded control independently by metric and prompt length.",
        ),
    ),
    Comparison(
        "fp4",
        "Official FP4 serving throughput",
        "nvidia/Qwen3.8-Flash-Next-FP4-Local",
        "vLLM (optimized)",
        (
            "Same checkpoint. A4 expert prefill in both; expert decode uses A4 in mistral.rs and A16 in vLLM.",
            "18 requests per length. Prefill = prompt tokens / mean TTFT; decode = 1,000 / mean TPOT (ms).",
        ),
    ),
)


def plot_comparison(cells: list[dict], spec: Comparison, figure_dir: Path) -> None:
    cells = sorted(cells, key=lambda cell: cell["prompt_tokens"])
    positions = list(range(len(cells)))
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.74, bottom=0.23, wspace=0.22)
    figure.suptitle(spec.title, x=0.525, y=0.97, fontsize=18, weight="bold")
    figure.text(
        0.525,
        0.91,
        f"{spec.model}  |  NVIDIA GB10  |  C1  |  No speculative decoding",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    for axis, (metric, title, decimals) in zip(axes, METRICS):
        values = {
            engine: [cell[engine][metric] for cell in cells]
            for engine in ("mistralrs", "baseline")
        }
        maximum = max(value for series in values.values() for value in series)
        for engine, offset, label in (
            ("mistralrs", -BAR_OFFSET, "mistral.rs"),
            ("baseline", BAR_OFFSET, spec.baseline_label),
        ):
            bars = axis.bar(
                [position + offset for position in positions],
                values[engine],
                width=BAR_WIDTH,
                color=COLORS[engine],
                label=label,
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:,.{decimals}f}" for value in values[engine]],
                padding=4,
                fontsize=10,
                color="#111827",
            )
        for position, ours, baseline in zip(
            positions, values["mistralrs"], values["baseline"]
        ):
            advantage = 100 * (ours / baseline - 1)
            axis.text(
                position,
                max(ours, baseline) + maximum * 0.105,
                f"{advantage:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                color=COLORS["mistralrs"],
                weight="bold",
                bbox={"boxstyle": "round,pad=0.25", "fc": "#eff6ff", "ec": "none"},
            )
        axis.set_title(title, loc="left", fontsize=13, weight="bold", pad=12)
        axis.set_xticks(positions, [f"{cell['prompt_tokens']:,}" for cell in cells])
        axis.set_xlabel("Prompt tokens", labelpad=10)
        axis.set_ylabel("Tokens/s", labelpad=8)
        axis.set_ylim(0, maximum * 1.25)
        axis.set_xlim(-0.6, len(cells) - 0.4)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.grid(axis="y", alpha=0.2, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.855),
        ncol=2,
        frameon=False,
        fontsize=11,
        columnspacing=2.4,
    )
    figure.text(0.065, 0.105, spec.footer[0], fontsize=9.5, color="#374151")
    figure.text(0.065, 0.065, spec.footer[1], fontsize=9.5, color="#374151")
    path = figure_dir / f"{spec.key}_throughput.png"
    figure.savefig(
        path, dpi=FIGURE_DPI, facecolor="white", metadata={"Software": "matplotlib"}
    )
    plt.close(figure)
    print(path)


def plot_concurrency(cells: list[dict], figure_dir: Path) -> None:
    cells = sorted(cells, key=lambda cell: cell["concurrency"])
    positions = list(range(len(cells)))
    figure, axes = plt.subplots(1, 3, figsize=(17.6, 6.6))
    figure.subplots_adjust(left=0.05, right=0.985, top=0.74, bottom=0.23, wspace=0.25)
    figure.suptitle(
        "Official FP4 concurrency", x=0.52, y=0.97, fontsize=18, weight="bold"
    )
    figure.text(
        0.52,
        0.91,
        "nvidia/Qwen3.8-Flash-Next-FP4-Local  |  NVIDIA GB10  |  2,048 input / 128 output tokens",
        ha="center",
        fontsize=11,
        color="#374151",
    )
    metrics = (
        ("output_tokens_s", "Aggregate output (higher is better)", "Tokens/s", 2),
        ("ttft_ms", "Time to first token (lower is better)", "Milliseconds", 0),
        (
            "decode_tokens_s",
            "Per-request decode (higher is better)",
            "Tokens/s/request",
            2,
        ),
    )
    for axis, (metric, title, unit, decimals) in zip(axes, metrics):
        values = {engine: [cell[engine][metric] for cell in cells] for engine in COLORS}
        maximum = max(value for series in values.values() for value in series)
        for engine, offset, label in (
            ("mistralrs", -BAR_OFFSET, "mistral.rs"),
            ("baseline", BAR_OFFSET, "vLLM (optimized)"),
        ):
            bars = axis.bar(
                [position + offset for position in positions],
                values[engine],
                width=BAR_WIDTH,
                color=COLORS[engine],
                label=label,
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:,.{decimals}f}" for value in values[engine]],
                padding=4,
                fontsize=9,
                color="#111827",
            )
        for position, cell in zip(positions, cells):
            ratio = cell["mistralrs"][metric] / cell["baseline"][metric]
            gain = 100 * (1 - ratio if metric == "ttft_ms" else ratio - 1)
            label = f"{gain:.1f}% lower" if metric == "ttft_ms" else f"{gain:+.1f}%"
            axis.text(
                position,
                max(cell["mistralrs"][metric], cell["baseline"][metric])
                + maximum * 0.105,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                color=COLORS["mistralrs"] if gain >= 0 else "#b91c1c",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "fc": "#eff6ff" if gain >= 0 else "#fef2f2",
                    "ec": "none",
                },
            )
        axis.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        axis.set_xticks(positions, [f"C{cell['concurrency']}" for cell in cells])
        axis.set_xlabel("Concurrency", labelpad=10)
        axis.set_ylabel(unit, labelpad=8)
        axis.set_ylim(0, maximum * 1.28)
        axis.set_xlim(-0.6, len(cells) - 0.4)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.grid(axis="y", alpha=0.2, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.855),
        ncol=2,
        frameon=False,
        fontsize=11,
        columnspacing=2.4,
    )
    figure.text(
        0.05,
        0.105,
        "Output: median of three round rates. TTFT: request mean. Decode: 1,000 / mean TPOT (ms). No speculation or prefix cache.",
        fontsize=9.5,
        color="#374151",
    )
    figure.text(
        0.05,
        0.065,
        "Same checkpoint. A4 expert prefill in both; expert decode uses A4 in mistral.rs and A16 in vLLM. C8 per-request decode remains 5.1% lower.",
        fontsize=9.5,
        color="#374151",
    )
    path = figure_dir / "fp4_concurrency.png"
    figure.savefig(
        path, dpi=FIGURE_DPI, facecolor="white", metadata={"Software": "matplotlib"}
    )
    plt.close(figure)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RELEASE / "results.json")
    parser.add_argument("--output-dir", type=Path, default=RELEASE / "figures")
    args = parser.parse_args()
    results = json.loads(args.data.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "axes.axisbelow": True}):
        for spec in COMPARISONS:
            plot_comparison(results[spec.key], spec, args.output_dir)
        plot_concurrency(results["fp4_concurrency"], args.output_dir)


if __name__ == "__main__":
    main()
