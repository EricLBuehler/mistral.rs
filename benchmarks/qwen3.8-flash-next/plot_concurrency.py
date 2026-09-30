#!/usr/bin/env python3
"""Plot validated closed-loop concurrency results as standalone SVG and PNG files."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (
    ("aggregate_output_tokens_per_second", "Aggregate throughput", "Output tokens/s"),
    (
        "latency_weighted_request_output_tokens_per_second",
        "Throughput per active request",
        "Output tokens/s per active request",
    ),
)
FIGURE_SIZE = (11, 4.8)
PNG_DPI = 180
DISPLAY_LABELS = {
    "before_batched_draft_sampling": "Before GPU draft sampling",
    "isq_mtp_batched": "Before depth exploration",
    "isq_mtp_tuner": "Final ISQ + MTP",
}


def plot_summary(summary, output_prefix):
    if summary.get("validation", {}).get("complete") is not True:
        raise ValueError("Use a complete report from summarize_concurrency.py")
    groups = defaultdict(list)
    for row in summary["rows"]:
        if row["mode"] == "closed-loop":
            groups[row["label"]].append(row)
    if not groups:
        raise ValueError("No closed-loop results to plot")
    for label, rows in groups.items():
        counts = [row["concurrency"] for row in rows]
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError(f"Invalid concurrency for {label}")
        if 1 not in counts or len(counts) != len(set(counts)):
            raise ValueError(
                f"{label} needs measured C1 and distinct concurrency values"
            )
        rows.sort(key=lambda row: row["concurrency"])
        for row in rows:
            for metric, _, _ in METRICS:
                value = row["metrics"][metric]
                if not math.isfinite(value["mean"]) or value["mean"] <= 0:
                    raise ValueError(f"Invalid {metric} mean for {label}")
                if (
                    not math.isfinite(value["sample_stddev"])
                    or value["sample_stddev"] < 0
                ):
                    raise ValueError(f"Invalid {metric} standard deviation for {label}")

    with plt.rc_context({"svg.fonttype": "none", "font.size": 10}):
        fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
        ticks = sorted({row["concurrency"] for rows in groups.values() for row in rows})
        for index, (label, rows) in enumerate(groups.items()):
            display_label = DISPLAY_LABELS.get(label, label)
            concurrency = [row["concurrency"] for row in rows]
            color = f"C{index % 10}"
            for panel, (metric, _, _) in enumerate(METRICS):
                values = [row["metrics"][metric]["mean"] for row in rows]
                deviations = [row["metrics"][metric]["sample_stddev"] for row in rows]
                axes[panel].errorbar(
                    concurrency,
                    values,
                    yerr=deviations,
                    color=color,
                    marker="o",
                    capsize=3,
                    label=f"{display_label}: measured",
                )
                reference = [
                    values[0] * (count if panel == 0 else 1) for count in concurrency
                ]
                axes[panel].plot(
                    concurrency,
                    reference,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label=f"{display_label}: C1 scaling reference",
                )
        for axis, (_, title, ylabel) in zip(axes, METRICS):
            axis.set(
                title=title, xlabel="Concurrent requests", ylabel=ylabel, ylim=(0, None)
            )
            axis.set_xticks(ticks)
            axis.grid(axis="y", alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
        )
        fig.suptitle("Qwen3.8-Flash-Next: closed-loop concurrency", y=0.99)
        fig.text(
            0.5,
            0.02,
            "Whole-trial means +/- sample SD. Dashed lines are references from measured C1, not measured scaling.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.85))
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        paths = [Path(f"{output_prefix}.{extension}") for extension in ("svg", "png")]
        for path in paths:
            fig.savefig(path, dpi=PNG_DPI)
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    try:
        paths = plot_summary(json.loads(args.summary.read_text()), args.output_prefix)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
