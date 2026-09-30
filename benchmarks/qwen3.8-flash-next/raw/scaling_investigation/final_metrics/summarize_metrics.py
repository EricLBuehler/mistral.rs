#!/usr/bin/env python3
"""Summarize saved server counters without querying the running server."""

import hashlib
import json
import math
from pathlib import Path
import re


OUT = Path(__file__).resolve().parent
COUNTERS = {
    "sequence_proposals": "mistralrs_speculative_drafts_total",
    "proposed_draft_tokens": "mistralrs_speculative_draft_tokens_proposed_total",
    "accepted_draft_tokens": "mistralrs_speculative_draft_tokens_accepted_total",
}
ACCEPTED_POSITION = "mistralrs_speculative_draft_tokens_accepted_per_pos_total"
GRAPH_DISPATCH = "mistralrs_cuda_graph_dispatch_total"
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$")
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)=("(?:[^"\\]|\\.)*")(?:,|$)')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse(text):
    types = {}
    samples = {}
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            types[name] = kind
            continue
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.fullmatch(line)
        if match is None:
            raise ValueError(f"Invalid metric sample: {line}")
        name, raw_labels, raw_value = match.groups()
        labels = []
        if raw_labels:
            remaining = raw_labels[1:-1]
            while remaining:
                label = LABEL.match(remaining)
                if label is None:
                    raise ValueError(f"Invalid metric labels: {raw_labels}")
                labels.append((label[1], json.loads(label[2])))
                remaining = remaining[label.end():]
        key = (name, tuple(sorted(labels)))
        if key in samples:
            raise ValueError(f"Duplicate metric sample: {key}")
        value = float(raw_value)
        samples[key] = int(value) if math.isfinite(value) and value.is_integer() else value
    return types, samples


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def counter_deltas(before, after):
    before_types, before_samples = parse(before)
    after_types, after_samples = parse(after)
    for metric in COUNTERS.values():
        if (metric, ()) not in after_samples or after_types.get(metric) != "counter":
            raise ValueError(f"Required speculative counter is missing: {metric}")
    types = dict(before_types)
    for name, kind in after_types.items():
        if name in types and types[name] != kind:
            raise ValueError(f"Metric type changed: {name}")
        types[name] = kind
    deltas = {}
    for key in before_samples.keys() | after_samples.keys():
        if types.get(key[0]) != "counter":
            continue
        old = before_samples.get(key, 0)
        new = after_samples.get(key, 0)
        if not math.isfinite(old) or not math.isfinite(new) or min(old, new) < 0 or new < old:
            raise ValueError(f"Counter reset/nonfinite/negative sample: {key}: {old} -> {new}")
        deltas[key] = new - old
    return deltas


def summarize(name, deltas):
    totals = {label: deltas.get((metric, ()), 0) for label, metric in COUNTERS.items()}
    drafts = totals["sequence_proposals"]
    proposed = totals["proposed_draft_tokens"]
    accepted = totals["accepted_draft_tokens"]
    if not 0 <= accepted <= proposed or (drafts == 0 and proposed != 0):
        raise ValueError(f"Inconsistent speculative counts: {totals}")
    positions = {}
    graph = []
    for (metric, labels), delta in sorted(deltas.items()):
        labels = dict(labels)
        if metric == ACCEPTED_POSITION:
            if not 0 <= delta <= drafts:
                raise ValueError(f"Invalid per-position accepted count: {labels}: {delta}")
            positions[labels["position"]] = {
                "accepted_drafts": delta,
                "fraction_of_sequence_proposals": ratio(delta, drafts),
            }
        if metric == GRAPH_DISPATCH:
            graph.append({"labels": labels, "dispatches": delta})
    if sum(item["accepted_drafts"] for item in positions.values()) != accepted:
        raise ValueError(f"Accepted-position counts do not sum to accepted total for {name}")
    return {
        "name": name,
        "scope": "Whole subprocess, including its warmups and all trials; concurrency_c6_c8 combines C6 and C8.",
        "counts": totals,
        "draft_token_acceptance_fraction": ratio(accepted, proposed),
        "mean_proposed_depth_per_sequence_proposal": ratio(proposed, drafts),
        "mean_accepted_drafts_per_sequence_proposal": ratio(accepted, drafts),
        "theoretical_accepted_drafts_plus_one_continuation_per_sequence_proposal": 1 + accepted / drafts if drafts else None,
        "accepted_positions": positions,
        "graph_dispatches": graph,
        "all_counter_deltas": [
            {"name": metric, "labels": dict(labels), "delta": delta}
            for (metric, labels), delta in sorted(deltas.items())
        ],
    }


def main():
    if not (OUT / "measurements-complete").exists():
        raise RuntimeError("Wait for measurements-complete before reading final snapshots")
    command_path = OUT / "measurement_commands.json"
    commands = json.loads(command_path.read_text())
    if not commands or any(command["returncode"] != 0 for command in commands):
        raise ValueError("Measurement commands are missing or include a failed subprocess")
    snapshots = {}
    results = []
    for command in commands:
        name = command["name"]
        texts = {}
        for phase in ("before", "after"):
            path = OUT / f"{name}.metrics.{phase}.txt"
            data = path.read_bytes()
            text = data.decode()
            snapshots[path.name] = {"sha256": sha(data), "content": text}
            texts[phase] = text
        result = summarize(name, counter_deltas(texts["before"], texts["after"]))
        result["command"] = command
        result["snapshots"] = {
            phase: {"file": f"{name}.metrics.{phase}.txt", "sha256": snapshots[f"{name}.metrics.{phase}.txt"]["sha256"]}
            for phase in ("before", "after")
        }
        results.append(result)
    payload = {
        "source_directory": str(OUT),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "measurement_commands_sha256": sha(command_path.read_bytes()),
        "validation": "All saved counter deltas are finite and nonnegative; accepted counts are bounded and match per-position sums.",
        "limitations": [
            "Sequence proposals are verified sequence rows, not batch steps.",
            "The theoretical accepted-plus-continuation quantity is not committed output length; EOS/limits can discard verified lookahead and omit continuation.",
            "Per-position fractions are unconditional over sequence proposals, not conditional acceptance; proposed-per-position counts are unavailable.",
            "No depth histogram, batch-size histogram, or actual expert-occupancy counters are recorded.",
            "Graph dispatch counts do not measure kernel-time coverage or percentage of requests using graphs.",
            "Before/after running-sequence gauges cannot determine time-averaged batching; client request intervals provide mean active requests separately.",
            "Snapshots bracket whole subprocesses, not individual trials; the combined C6/C8 subprocess cannot be split by concurrency from these counters.",
        ],
        "subprocesses": results,
    }
    (OUT / "metrics.snapshots.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    (OUT / "metrics.summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"subprocesses": len(results), "snapshots": len(snapshots), "summary": str(OUT / "metrics.summary.json")}, indent=2))


if __name__ == "__main__":
    main()
