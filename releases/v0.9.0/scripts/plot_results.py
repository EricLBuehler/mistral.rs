#!/usr/bin/env python3
"""Generate v0.9.0 CPU bench figures from results jsonl files."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path(__file__).resolve().parents[1] / "raw"
FIG = Path(__file__).resolve().parents[1] / "figures"

QUANTS = ["q4k", "q6k", "q8_0"]
QUANT_LABEL = {"q4k": "Q4_K", "q6k": "Q6_K", "q8_0": "Q8_0"}
ENGINE_LABEL = {"mistralrs": "mistral.rs", "llamacpp": "llama.cpp"}
ENGINE_COLOR = {"mistralrs": "#1f77b4", "llamacpp": "#d62728"}
QUANT_STYLE = {"q4k": "-", "q6k": "--", "q8_0": ":"}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index(rows):
    idx = defaultdict(dict)  # (engine, model, quant, mode) -> {length: tps}
    for r in rows:
        idx[(r["engine"], r["model"], r["quant"], r["mode"])][r["length"]] = r["tps"]
    return idx


def plot_model(idx, model, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode, xlabel in [
        (axes[0], "prefill", "prompt length (tokens)"),
        (axes[1], "decode", "context depth (tokens)"),
    ]:
        for engine in ["mistralrs", "llamacpp"]:
            for quant in QUANTS:
                pts = idx.get((engine, model, quant, mode))
                if not pts:
                    continue
                xs = sorted(pts)
                ax.plot(xs, [pts[x] for x in xs], QUANT_STYLE[quant],
                        color=ENGINE_COLOR[engine], marker="o", ms=3,
                        label=f"{ENGINE_LABEL[engine]} {QUANT_LABEL[quant]}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("tokens/s")
        ax.set_title(f"{model} {mode} (CPU)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_speedup(idx, models, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in [(axes[0], "prefill"), (axes[1], "decode")]:
        labels, values = [], []
        for model in models:
            for quant in QUANTS:
                m = idx.get(("mistralrs", model, quant, mode))
                l = idx.get(("llamacpp", model, quant, mode))
                if not m or not l:
                    continue
                common = sorted(set(m) & set(l))
                if not common:
                    continue
                sp = [m[x] / l[x] for x in common]
                labels.append(f"{model}\n{QUANT_LABEL[quant]}")
                values.append(sum(sp) / len(sp))
        ax.bar(range(len(values)), values, color=["#2ca02c" if v >= 1 else "#d62728" for v in values])
        ax.axhline(1.0, color="black", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("mistral.rs speedup vs llama.cpp (x)")
        ax.set_title(f"mean {mode} speedup (CPU)")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_affinity(rows, out):
    # group: (engine, quant, mode) -> {affinity: tps}
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["engine"], r["quant"], r["mode"])][r["affinity"]] = r["tps"]
    keys = sorted(groups)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, engine in [(axes[0], "mistralrs"), (axes[1], "llamacpp")]:
        cases = [k for k in keys if k[0] == engine]
        modes = sorted({aff for k in cases for aff in groups[k]})
        width = 0.8 / max(len(modes), 1)
        for j, aff in enumerate(modes):
            xs, ys = [], []
            for i, k in enumerate(cases):
                if aff in groups[k]:
                    xs.append(i + j * width)
                    ys.append(groups[k][aff])
            ax.bar(xs, ys, width=width, label=aff)
        ax.set_xticks([i + 0.4 for i in range(len(cases))])
        ax.set_xticklabels([f"{k[1]}\n{k[2]}" for k in cases], fontsize=8)
        ax.set_ylabel("tokens/s")
        ax.set_title(f"{ENGINE_LABEL[engine]}: affinity strategies (qwen3-4b)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    full = RAW / "results_full.jsonl"
    aff = RAW / "results_affinity.jsonl"
    if full.exists():
        rows = load(full)
        idx = index(rows)
        models = sorted({r["model"] for r in rows})
        for model in models:
            plot_model(idx, model, FIG / f"{model.replace('.', '_')}_cpu_throughput.png")
        plot_speedup(idx, models, FIG / "cpu_speedup_bars.png")
        print(f"full figures written for {models}")
    if aff.exists():
        plot_affinity(load(aff), FIG / "affinity_study.png")
        print("affinity figure written")


if __name__ == "__main__":
    main()
