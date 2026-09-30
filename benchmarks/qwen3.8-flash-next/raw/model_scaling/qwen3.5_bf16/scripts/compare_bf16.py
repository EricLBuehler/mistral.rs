"""Compare only completed, validated same-checkpoint BF16 concurrency runs."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
CONCURRENCIES = (1, 6, 8)
TRIALS = 5
WARMUPS = 2
OUTPUT_TOKENS = 128
SEED = 20260929
FIELDS = (
    "aggregate_output_tokens_per_second",
    "latency_weighted_request_output_tokens_per_second",
    "mean_active_requests",
    "mean_request_wall_seconds",
    "mean_request_output_tokens_per_second",
)
HASH_FIELDS = (
    "harness_sha256",
    "shared_prompt_harness_sha256",
    "tokenizer_sha256",
    "prompts_sha256",
)
LIMITS = [
    "Five measured closed-loop trials follow two excluded warmups at each concurrency; finite trial startup and drain are included.",
    "Per-active throughput is output tokens divided by summed request latency, not aggregate throughput divided by requested concurrency.",
    "Reported variability is sample standard deviation across five trial rates; engine and scaling ratios divide means and have no inferred confidence interval.",
    "The ordered prompts and request bodies match, but engine arithmetic, generated text, expert routes, and scheduling can differ.",
    "Physical KV-cache allocation differs despite the same 16384-token limit and eight-sequence cap; report both capacities and memory observations.",
    "Runs are sequential, not randomized; clocks, cache state, and memory pressure may differ.",
    "These target-only BF16 results do not measure Flash-Next low-bit experts or adaptive MTP performance.",
    "Counter snapshots include warmups and are engine-specific; absent counters are unavailable, not zero.",
    "Clock/power/temperature ranges use each whole command window including warmups; frozen trial records lack absolute per-trial boundaries.",
    "Swap deltas are global system counters; process VmSwap snapshots do not identify GPU paging or its timing cost.",
]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, field):
    require(
        math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
        field,
    )


def stats(values):
    require(
        len(values) == TRIALS and all(math.isfinite(x) and x > 0 for x in values),
        "Invalid measured trials",
    )
    return {
        "mean": statistics.mean(values),
        "sample_stddev": statistics.stdev(values),
        "samples": values,
    }


def validate_phase(data, concurrency):
    require(data["complete"] is True, "Phase is incomplete")
    settings = data["settings"]
    count = 8 if concurrency == 1 else 24
    expected = {
        "concurrencies": [concurrency],
        "modes": ["closed-loop"],
        "trials": TRIALS,
        "warmup": WARMUPS,
        "requests": count,
        "max_tokens": OUTPUT_TOKENS,
    }
    for key, value in expected.items():
        require(settings[key] == value, f"Unexpected setting: {key}")
    prompt_counts = settings["expected_prompt_tokens"]
    require(len(prompt_counts) == 8, "Expected eight canonical prompts")
    case_key = f"closed-loop_c{concurrency}"
    require(set(data["cases"]) == {case_key}, "Unexpected cases")
    case = data["cases"][case_key]
    require(
        case["mode"] == "closed-loop" and case["concurrency"] == concurrency,
        "Case identity",
    )
    runs = case["runs"]
    require(len(runs) == WARMUPS + TRIALS, "Missing or extra trials")
    protocol, output_text, measured = {}, {}, []
    cached_fields = 0
    for index, run in enumerate(runs):
        require(
            run["complete"] is True
            and run["trial"] == index
            and run["warmup"] == (index < WARMUPS),
            "Incomplete or reordered trial",
        )
        requests = run["requests"]
        require(len(requests) == count, "Missing request")
        active_seconds = 0.0
        rates = []
        for request_index, request in enumerate(requests):
            require(
                request["request_index"] == request_index and request["error"] is None,
                "Failed or reordered request",
            )
            body = request["request"]
            expected_body = {
                "model": "default",
                "prompt": body["prompt"],
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0.0,
                "seed": SEED,
                "ignore_eos": True,
                "cache_prompt": False,
            }
            require(body == expected_body, "Unexpected request body")
            require(
                hashlib.sha256(body["prompt"].encode()).hexdigest()
                == request["prompt_sha256"],
                "Prompt hash mismatch",
            )
            response = request["response"]
            usage = response["usage"]
            require(
                "error" not in response and usage["completion_tokens"] == OUTPUT_TOKENS,
                "Response output count",
            )
            require(
                usage["prompt_tokens"] == prompt_counts[request["prompt_name"]],
                "Response prompt count",
            )
            details = usage.get("prompt_tokens_details") or {}
            cached_fields += int("cached_tokens" in details)
            require(details.get("cached_tokens", 0) == 0, "Unexpected prefix reuse")
            require(
                len(response["choices"]) == 1
                and isinstance(response["choices"][0]["text"], str),
                "Expected text completion",
            )
            wall = request["wall_seconds"]
            require(wall > 0 and math.isfinite(wall), "Invalid request latency")
            close(
                request["finished_seconds"] - request["started_seconds"],
                wall,
                "Request time consistency",
            )
            require(
                request["started_seconds"] >= 0
                and request["finished_seconds"] <= run["wall_seconds"] + 1e-8,
                "Request outside trial",
            )
            close(
                request["output_tokens_per_second"],
                OUTPUT_TOKENS / wall,
                "Request rate consistency",
            )
            key = f"{index}:{request_index}"
            protocol[key] = {
                "body": body,
                "prompt_name": request["prompt_name"],
                "prompt_tokens": usage["prompt_tokens"],
            }
            if index >= WARMUPS:
                output_text[key] = response["choices"][0]["text"]
            active_seconds += wall
            rates.append(OUTPUT_TOKENS / wall)
        tokens = count * OUTPUT_TOKENS
        require(run["completed_output_tokens"] == tokens, "Trial token count")
        elapsed = run["wall_seconds"]
        require(elapsed > 0 and math.isfinite(elapsed), "Invalid trial duration")
        calculated = {
            "aggregate_output_tokens_per_second": tokens / elapsed,
            "latency_weighted_request_output_tokens_per_second": tokens
            / active_seconds,
            "mean_active_requests": active_seconds / elapsed,
            "mean_request_wall_seconds": active_seconds / count,
            "mean_request_output_tokens_per_second": statistics.mean(rates),
        }
        require(
            calculated["mean_active_requests"] <= concurrency + 1e-8,
            "Concurrency cap exceeded",
        )
        for field in FIELDS:
            close(run[field], calculated[field], f"Trial arithmetic: {field}")
        if index >= WARMUPS:
            measured.append(calculated)
    summary = {field: stats([row[field] for row in measured]) for field in FIELDS}
    return (
        {
            "statistics": summary,
            "measured_requests": count * TRIALS,
            "measured_output_tokens": count * TRIALS * OUTPUT_TOKENS,
            "validated_requests_including_warmups": count * (TRIALS + WARMUPS),
            "responses_reporting_cached_tokens": cached_fields,
        },
        protocol,
        output_text,
    )


def flag(command, name):
    require(name in command, f"Missing server option {name}")
    return command[command.index(name) + 1]


def operating_conditions(directory, metadata, engine):
    timezone_path = Path("/etc/localtime")
    with timezone_path.open("rb") as stream:
        timezone = ZoneInfo.from_file(stream)
    samples = []
    with (directory / "gpu.csv").open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            timestamp = datetime.strptime(
                row["timestamp"], "%Y/%m/%d %H:%M:%S.%f"
            ).replace(tzinfo=timezone)
            numeric = {}
            for key, value in row.items():
                if key == "timestamp":
                    continue
                try:
                    number = float(value.strip().replace(" %", ""))
                except ValueError:
                    continue
                if math.isfinite(number):
                    numeric[key.strip()] = number
            samples.append((timestamp, numeric))
    phases = {}
    for phase in metadata["phases"]:
        name = phase["name"]
        start = datetime.fromisoformat(phase["started"]["utc"])
        end = datetime.fromisoformat(
            phase.get("process_finished", phase["finished"])["utc"]
        )
        selected = [row for stamp, row in samples if start <= stamp <= end]
        fields = sorted({key for row in selected for key in row})
        ranges = {}
        for key in fields:
            values = [row[key] for row in selected if key in row]
            ranges[key] = {
                "samples": len(values),
                "min": min(values),
                "max": max(values),
                "median": statistics.median(values),
            }
        before = load(directory / f"{name}.memory.before.json")
        after = load(directory / f"{name}.memory.after.json")
        require(
            before["page_size_bytes"] == after["page_size_bytes"], "Page size changed"
        )
        page_size = before["page_size_bytes"]
        swap = {}
        for counter in ("pswpin_pages", "pswpout_pages"):
            delta = after[counter] - before[counter]
            require(delta >= 0, f"Global counter reset: {counter}")
            swap[counter] = {"delta_pages": delta, "delta_bytes": delta * page_size}
        process_swap = {
            point: (snapshot.get("model") or {}).get("VmSwap_KiB")
            for point, snapshot in (("before_KiB", before), ("after_KiB", after))
        }
        phases[name] = {
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "gpu_samples": len(selected),
            "gpu_ranges": ranges,
            "unavailable_numeric_fields_are_not_zero": True,
            "global_swap": swap,
            "process_VmSwap": process_swap,
            "process_scope": "Model server process"
            if engine == "mistralrs"
            else "Container init process; model worker memory can reside in child processes",
            "system_MemAvailable_KiB": {
                "before": before["system"]["MemAvailable_KiB"],
                "after": after["system"]["MemAvailable_KiB"],
            },
        }
    return {
        "scope": "Whole phase command including two warmups and five measured trials; no exact measured-only GPU clock window is available",
        "gpu_timestamp_timezone_source": str(timezone_path.resolve()),
        "phases": phases,
    }


def read_engine(directory, engine):
    metadata = load(directory / "metadata.json")
    require(metadata["complete"] is True, f"{engine}: lifecycle incomplete")
    require(
        metadata["target_only"] is True and metadata["profiled"] is False,
        "Expected unprofiled target-only run",
    )
    require(
        not metadata.get("error") and not metadata.get("monitor_errors"),
        "Lifecycle error",
    )
    require(
        {p["name"] for p in metadata["phases"]} == {"c1", "c6", "c8"},
        "Unexpected phases",
    )
    require(
        all(
            p["complete"] is True and p.get("returncode") == 0
            for p in metadata["phases"]
        ),
        "Incomplete phase subprocess",
    )
    manifest = load(directory / "SHA256SUMS.json")
    required_files = {
        "metadata.json",
        "checkpoint/config.json",
        "checkpoint/model.safetensors.index.json",
        "server.log",
        "gpu.csv",
    }
    required_files.update(f"concurrency.c{c}.json" for c in CONCURRENCIES)
    required_files.update(
        f"c{c}.memory.{point}.json"
        for c in CONCURRENCIES
        for point in ("before", "after")
    )
    require(required_files <= set(manifest), "Required source missing from manifest")
    for name, expected in manifest.items():
        require(digest(directory / name) == expected, f"Changed input: {engine}/{name}")
    require(
        digest(directory / "checkpoint/config.json")
        == metadata["expected_config_sha256"],
        "Checkpoint config hash",
    )
    command = metadata["command"]
    require(flag(command, "--max-model-len") == "16384", "Context limit")
    require(
        not any(
            x in command
            for x in (
                "--mtp",
                "--speculative-config",
                "--isq",
                "--quantization",
                "--enforce-eager",
            )
        ),
        "Unexpected engine mode",
    )
    if engine == "mistralrs":
        require(
            metadata.get("isq") is None and flag(command, "--dtype") == "bf16",
            "Expected native BF16",
        )
        require(
            flag(command, "--max-seqs") == "8"
            and flag(command, "--prefix-cache-n") == "0",
            "Scheduler or prefix configuration",
        )
        require(
            load(directory / "startup_validation.json")["complete"] is True,
            "Startup validation",
        )
        shutdown = metadata["server_shutdown"]
        require(
            shutdown["forced_kill"] is False and shutdown["returncode"] in (0, -15),
            "Unclean native shutdown",
        )
        require(
            metadata["binary_sha256"] == metadata["binary_sha256_after"],
            "Binary changed",
        )
    else:
        require(
            flag(command, "--dtype") == "bfloat16"
            and flag(command, "--max-num-seqs") == "8",
            "Expected BF16/scheduler cap",
        )
        require("--no-enable-prefix-caching" in command, "Prefix caching enabled")
        require(
            load(directory / "startup_evidence.json")["complete"] is True,
            "Startup evidence",
        )
        cleanup = metadata["container_shutdown"]
        require(cleanup["removed"] is True, "Container remains")
        require(
            cleanup["state"]["Running"] is False
            and cleanup["state"]["OOMKilled"] is False
            and cleanup["state"]["ExitCode"] in (0, 143),
            "Unclean container shutdown",
        )
        require(
            metadata["reported_gpu_kv_cache_tokens"] >= 16384,
            "Insufficient cache capacity",
        )
    reports, protocols, texts, hashes = {}, {}, {}, {}
    for concurrency in CONCURRENCIES:
        data = load(directory / f"concurrency.c{concurrency}.json")
        result, protocol, output = validate_phase(data, concurrency)
        (
            reports[str(concurrency)],
            protocols[str(concurrency)],
            texts[str(concurrency)],
        ) = result, protocol, output
        current = {key: data[key] for key in HASH_FIELDS}
        require(all(current.values()), "Missing protocol hash")
        require(not hashes or current == hashes, "Phase protocol hashes differ")
        hashes = current
    c1 = reports["1"]["statistics"][FIELDS[0]]["mean"]
    scaling = {
        f"C{c}/C1": reports[str(c)]["statistics"][FIELDS[0]]["mean"] / c1
        for c in (6, 8)
    }
    cache_lines = [
        line
        for line in (directory / "server.log").read_text().splitlines()
        if re.search(
            r"GPU KV cache size|KV blocks|GPU blocks|PagedAttention KV cache|cache size is|num_gpu_blocks",
            line,
        )
    ]
    identity = {
        key: metadata[key]
        for key in (
            "binary_sha256",
            "git_head",
            "binary_version",
            "image_id",
            "image_commit",
            "versions",
        )
        if key in metadata
    }
    return (
        {
            "identity": identity,
            "concurrencies": reports,
            "scaling_ratios_of_means": scaling,
            "protocol_hashes": hashes,
            "config_sha256": metadata["expected_config_sha256"],
            "checkpoint_path": metadata["checkpoint"],
            "checkpoint_index_sha256": digest(
                directory / "checkpoint/model.safetensors.index.json"
            ),
            "checkpoint_gdn_recurrent_dtype": load(
                directory / "checkpoint/config.json"
            )["text_config"]["mamba_ssm_dtype"],
            "physical_cache": {
                "explicit_bytes": metadata.get("kv_cache_bytes"),
                "reported_tokens": metadata.get("reported_gpu_kv_cache_tokens"),
                "startup_lines": cache_lines,
            },
            "operating_conditions": operating_conditions(directory, metadata, engine),
            "metadata_sha256": digest(directory / "metadata.json"),
            "manifest_sha256": digest(directory / "SHA256SUMS.json"),
        },
        protocols,
        texts,
    )


def compare(root, native=None, vllm=None):
    reports, protocols, texts = {}, {}, {}
    directories = {
        "mistralrs": native or root / "mistralrs_bf16",
        "vllm": vllm or root / "vllm_bf16",
    }
    for engine, directory in directories.items():
        reports[engine], protocols[engine], texts[engine] = read_engine(
            directory, engine
        )
    a, b = reports["mistralrs"], reports["vllm"]
    require(a["config_sha256"] == b["config_sha256"], "Checkpoint config differs")
    require(
        a["checkpoint_path"] == b["checkpoint_path"]
        and a["checkpoint_index_sha256"] == b["checkpoint_index_sha256"],
        "Checkpoint source/index differs",
    )
    require(
        a["protocol_hashes"] == b["protocol_hashes"], "Harness/tokenizer/prompts differ"
    )
    require(
        protocols["mistralrs"] == protocols["vllm"],
        "Request body/order/token counts differ",
    )
    precision = load(ROOT / "precision_evidence.json")
    require(
        a["config_sha256"] == precision["checkpoint_config_sha256"],
        "Precision evidence checkpoint differs",
    )
    require(
        b["identity"]["image_id"] == precision["vllm"]["image_id"],
        "Precision evidence image differs",
    )
    native_sources = load(directories["mistralrs"] / "source_files.sha256.json")
    require(
        all(
            native_sources.get(name) == value
            for name, value in precision["mistralrs"]["source_hashes"].items()
        ),
        "Precision evidence native source differs",
    )
    ratios, agreements = {}, {}
    for c in map(str, CONCURRENCIES):
        ratios[c] = {
            field: a["concurrencies"][c]["statistics"][field]["mean"]
            / b["concurrencies"][c]["statistics"][field]["mean"]
            for field in FIELDS
        }
        left, right = texts["mistralrs"][c], texts["vllm"][c]
        agreements[c] = {
            "text_identical": sum(left[k] == right[k] for k in left),
            "compared": len(left),
            "scope": "Same trial/request indices, measured trials only; text equality is not a quality test.",
        }
    return {
        "complete": True,
        "run_directories": {
            key: str(path.resolve()) for key, path in directories.items()
        },
        "engines": reports,
        "mistralrs_over_vllm_ratios_of_means": ratios,
        "measured_output_text_agreement": agreements,
        "precision": precision,
        "limitations": LIMITS,
        "summarizer_sha256": digest(Path(__file__)),
    }


def markdown(report):
    lines = [
        "# Qwen3.5-35B-A3B BF16 engine comparison",
        "",
        "Five measured trials after two warmups per concurrency; values are mean +/- sample standard deviation.",
        "",
        "| Engine | C | Aggregate tok/s | Per-active tok/s | Mean active requests | Mean request latency, s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, data in report["engines"].items():
        for c, row in data["concurrencies"].items():
            values = [row["statistics"][field] for field in FIELDS[:4]]
            lines.append(
                f"| {engine} | {c} | "
                + " | ".join(
                    f"{x['mean']:.3f} +/- {x['sample_stddev']:.3f}" for x in values
                )
                + " |"
            )
    lines.extend(["", "| Comparison | C1 | C6 | C8 |", "| --- | ---: | ---: | ---: |"])
    ratios = report["mistralrs_over_vllm_ratios_of_means"]
    lines.append(
        "| mistral.rs / vLLM aggregate rate | "
        + " | ".join(f"{ratios[str(c)][FIELDS[0]]:.3f}x" for c in CONCURRENCIES)
        + " |"
    )
    for engine, data in report["engines"].items():
        scaling = data["scaling_ratios_of_means"]
        lines.append(
            f"| {engine}, relative to own C1 | 1.000x | {scaling['C6/C1']:.3f}x | {scaling['C8/C1']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "Operating ranges below cover whole phase commands, including warmups. Swap counters are system-wide, not GPU page-in measurements.",
            "",
            "| Engine | Phase | SM MHz range | Power W range | GPU C range | Global swap-in/out MiB | Process VmSwap before/after MiB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for engine, data in report["engines"].items():
        for name, phase in data["operating_conditions"]["phases"].items():
            values = []
            for prefix in ("clocks.current.sm", "power.draw", "temperature.gpu"):
                metric = next(
                    (
                        value
                        for key, value in phase["gpu_ranges"].items()
                        if key.startswith(prefix)
                    ),
                    None,
                )
                values.append(
                    f"{metric['min']:.1f}-{metric['max']:.1f}"
                    if metric
                    else "unavailable"
                )
            swap = phase["global_swap"]
            global_value = f"{swap['pswpin_pages']['delta_bytes'] / 1024**2:.2f} / {swap['pswpout_pages']['delta_bytes'] / 1024**2:.2f}"
            process = phase["process_VmSwap"]
            process_value = " / ".join(
                "unavailable" if process[key] is None else f"{process[key] / 1024:.2f}"
                for key in ("before_KiB", "after_KiB")
            )
            lines.append(
                f"| {engine} | {name} | "
                + " | ".join(values + [global_value, process_value])
                + " |"
            )
    lines.extend(
        [
            "",
            "vLLM process VmSwap refers to its container init process and can omit model-worker children. The precision contract is BF16 weights/attention KV/convolution state and F32 GDN recurrent state, supported by the checkpoint and pinned implementation sources.",
        ]
    )
    lines.extend(
        [
            "",
            "All request bodies, ordered prompts, prompt counts, checkpoint config, tokenizer, and frozen harness hashes match. Output text agreement is recorded in comparison.json without requiring identical continuations.",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in LIMITS)
    lines.extend(
        [
            "",
            "Physical cache settings and observed startup capacities, engine identities, raw trial samples, and input hashes are recorded in comparison.json.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native", type=Path, help="Completed native run directory")
    parser.add_argument(
        "--vllm",
        type=Path,
        help="Completed vLLM run directory, including an explicitly selected retry",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "comparison")
    args = parser.parse_args()
    report = compare(args.root, args.native, args.vllm)
    require(not args.output.exists(), "Refusing to overwrite comparison")
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "comparison.md").write_text(markdown(report))
    print(json.dumps({"complete": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
