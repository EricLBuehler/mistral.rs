#!/usr/bin/env python3
"""Parse mistral.rs bench txt + vLLM latency json into results.json, report.md, and plots."""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
LENS = [128, 512, 2048, 4096, 8192, 16384]

PREFILL_RE = re.compile(r"Prefill \((\d+) tokens\)\s*┆\s*([\d.]+)")
DECODE_RE = re.compile(r"Decode \(256 tokens @ d(\d+)\)\s*┆\s*([\d.]+)")


def parse_mrs(path, regex):
    out = {}
    if not path.exists():
        return out
    for m in regex.finditer(path.read_text()):
        out[int(m.group(1))] = float(m.group(2))
    return out


def parse_vllm(path):
    prompt, decode = {}, {}
    if not path.exists():
        return prompt, decode
    d = json.loads(path.read_text())
    for k, v in d.items():
        prompt[int(k)] = round(v["prefill_tps_approx"], 1)
        decode[int(k)] = round(v["decode_tps_delta"], 2)
    return prompt, decode


def load_baseline_table(report_path, header):
    """Pull a markdown table that follows a given `## header` from a prior report.md."""
    if not report_path.exists():
        return {}
    text = report_path.read_text()
    idx = text.find(header)
    if idx < 0:
        return {}
    rows = {}
    started = False
    for line in text[idx:].splitlines()[1:]:
        s = line.strip()
        if s.startswith("##"):
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) >= 2 and cells[0].lstrip("-").isdigit():
            rows[int(cells[0])] = float(cells[1])
            started = True
        elif started:
            break
    return rows


def table(title, mrs, vllm, baseline=None, baseline_name="before-cutile"):
    if baseline:
        head = f"| tokens | mistral.rs (cutile) | {baseline_name} | vLLM | vs vLLM |\n| ---: | ---: | ---: | ---: | ---: |"
    else:
        head = "| tokens | mistral.rs | vLLM | vs vLLM |\n| ---: | ---: | ---: | ---: |"
    lines = [f"## {title}", head]
    for L in LENS:
        m, v = mrs.get(L), vllm.get(L)
        if m is None or v is None:
            continue
        delta = f"{(m / v - 1) * 100:+.0f}%"
        if baseline:
            b = baseline.get(L)
            bcell = f"{b:.1f}" if b is not None else "-"
            lines.append(f"| {L} | {m:.1f} | {bcell} | {v:.1f} | {delta} |")
        else:
            lines.append(f"| {L} | {m:.1f} | {v:.1f} | {delta} |")
    return "\n".join(lines)


def main():
    data = {}
    for tag in ["e4b", "26b_a4b"]:
        p = parse_mrs(OUT / f"mistralrs_{tag}_bf16_prompt.txt", PREFILL_RE)
        d = parse_mrs(OUT / f"mistralrs_{tag}_bf16_decode.txt", DECODE_RE)
        vp, vd = parse_vllm(OUT / f"vllm_{tag}_bf16_latency_delta.json")
        data[tag] = {"mrs_prompt": p, "mrs_decode": d, "vllm_prompt": vp, "vllm_decode": vd}

    (OUT / "results.json").write_text(json.dumps(data, indent=2) + "\n")

    sweep = OUT.parent
    base_26b_prompt = load_baseline_table(
        sweep / "26b_bf16_vllm_resweep_20260526" / "report.md", "## Prompt T/s")
    base_26b_decode = load_baseline_table(
        sweep / "26b_bf16_vllm_resweep_20260526" / "report.md", "## Decode T/s")

    rev = (OUT / "git_rev.txt").read_text().strip() if (OUT / "git_rev.txt").exists() else "?"
    md = [
        "# Gemma 4 BF16 mistral.rs (cutile MoE) vs vLLM",
        "",
        f"Branch `cuda_graphs_v1` @ `{rev}` (cutile MoE backend, auto-default on bf16/CUDA). "
        "CUDA graphs + FlashInfer decode default-on. vLLM 0.21.0, same methodology as "
        "`26b_bf16_vllm_resweep_20260526` (language_model_only, prefix caching off, max_model_len 20000; "
        "E4B gpu-mem-util 0.85, 26B 0.60). vLLM prompt = input_len/latency(out=1); decode = 256/(lat(257)-lat(1)).",
        "",
        "# E4B BF16 (dense - cutile does not apply; control)",
        table("E4B Prompt T/s", data["e4b"]["mrs_prompt"], data["e4b"]["vllm_prompt"]),
        "",
        table("E4B Decode T/s", data["e4b"]["mrs_decode"], data["e4b"]["vllm_decode"]),
        "",
        "# 26B-A4B BF16 (cutile MoE path)",
        table("26B-A4B Prompt T/s", data["26b_a4b"]["mrs_prompt"], data["26b_a4b"]["vllm_prompt"],
              base_26b_prompt),
        "",
        table("26B-A4B Decode T/s", data["26b_a4b"]["mrs_decode"], data["26b_a4b"]["vllm_decode"],
              base_26b_decode),
        "",
        "## Notes",
        "- `before-cutile` column = `26b_bf16_vllm_resweep_20260526` (commit 1bd863a62, the pre-cutile "
        "commit, graphs+FlashInfer on). Same machine/methodology, so it isolates the cutile MoE delta.",
        "- 26B-A4B was re-benched after the cuTile warmup fix (warmup runs on the engine thread + MoE "
        "shapes registered in the weight loaders), eliminating per-shape JIT during measured iterations. "
        "This corrected the previously JIT-depressed p128/p2048 means (659->947, 2811->3822) and collapsed "
        "their stddev (+/-419->+/-32, +/-1456->+/-50). Verified zero JIT via CUTILE_JIT_TIMING.",
        "- cutile is bf16-only and auto-selected on the unquantized CUDA MoE path; E4B is dense so it is a "
        "control (warmup-3, unaffected by the MoE warmup fix; matches prior BF16 numbers).",
        "- Remaining: fix C (pad M) to make prefill robust to arbitrary un-warmed lengths; warmup currently "
        "covers the bucket set {1..16384}.",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print((OUT / "report.md").read_text())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for tag in ["e4b", "26b_a4b"]:
            for kind, mk, vk in [("prompt", "mrs_prompt", "vllm_prompt"),
                                 ("decode", "mrs_decode", "vllm_decode")]:
                m, v = data[tag][mk], data[tag][vk]
                xs = [L for L in LENS if L in m and L in v]
                if not xs:
                    continue
                plt.figure(figsize=(7, 4))
                plt.plot(xs, [m[x] for x in xs], "o-", label="mistral.rs")
                plt.plot(xs, [v[x] for x in xs], "s-", label="vLLM")
                plt.xscale("log", base=2)
                plt.xlabel("prompt tokens" if kind == "prompt" else "decode depth")
                plt.ylabel("T/s")
                plt.title(f"{tag} BF16 {kind}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(OUT / f"{tag}_bf16_{kind}_tps.png", dpi=110)
                plt.close()
        print("plots written")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
