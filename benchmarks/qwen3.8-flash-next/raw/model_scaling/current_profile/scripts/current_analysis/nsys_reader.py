#!/usr/bin/env python3
"""Read an Nsight SQLite export without changing it; write auditable JSON summaries."""

import argparse
import collections
import dataclasses
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics


@dataclasses.dataclass(slots=True)
class Kernel:
    start: int
    end: int
    name: str
    pid: int
    context: int
    stream: int
    correlation: int
    graph: bool
    grid: tuple
    block: tuple
    category: str = "other"


def summary(values):
    values = sorted(values)
    if not values:
        return {"count": 0}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "p99": percentile(0.99),
        "min": values[0],
        "max": values[-1],
    }


def merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def interval_total(intervals):
    return sum(end - start for start, end in merge_intervals(intervals))


def classify(name, grouped):
    lower = name.lower()
    if any(part in lower for part in ("moe_gemv", "moe_gemm", "moe_grouped_gemm")):
        return "moe_gemv" if "gemv" in lower else "moe_grouped_gemm"
    if "mul_mat_q" in lower:
        return "moe_grouped_mmq_context" if grouped else "mmq_outside_grouped_context"
    if any(part in lower for part in ("gdn", "gated_delta_rule", "causal_conv1d", "save_conv_state")):
        return "gdn"
    if any(part in lower for part in ("qsa", "attention", "flash_fwd", "flashinfer", "qk_rms_norm_rope", "reshape_and_cache", "gather_kv_cache")):
        return "attention"
    if "q4_hc_" in lower:
        return "hyper_connections"
    if "q4_ple_" in lower:
        return "ple"
    if "moe_" in lower:
        return "moe_dispatch_routing_reduce"
    if "quantize" in lower:
        return "grouped_moe_quantization_context" if grouped else "other_quantization"
    if any(part in lower for part in ("mmvq", "gemv", "gemm", "cutlass", "nvjet_", "splitkreduce")):
        return "other_matrix_multiply"
    return "other"


def annotate_categories(kernels):
    groups = collections.defaultdict(list)
    for kernel in kernels:
        groups[(kernel.pid, kernel.context, kernel.stream)].append(kernel)
    completed, abandoned = 0, 0
    for sequence in groups.values():
        pending = []
        for kernel in sorted(sequence, key=lambda item: item.start):
            kernel.category = classify(kernel.name, False)
            if "moe_dispatch_count_kernel" in kernel.name:
                abandoned += bool(pending)
                pending = [kernel]
            elif pending:
                pending.append(kernel)
                if "moe_weighted_reduce" in kernel.name:
                    for member in pending:
                        member.category = classify(member.name, True)
                    completed += 1
                    pending = []
        abandoned += bool(pending)
    return {"completed_dispatch_reduce_regions": completed, "incomplete_regions": abandoned}


def read_rows(connection, table):
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]


def read_kernels(connection, tables, device):
    table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if table not in tables:
        return []
    columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    graph = "COALESCE(k.graphNodeId,0) != 0" if "graphNodeId" in columns else "0"
    if "graphId" in columns:
        graph += " OR COALESCE(k.graphId,0) != 0"
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    query = f"""SELECT k.start,k.end,k.demangledName,k.globalPid,k.contextId,k.streamId,
        k.correlationId,({graph}),k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ
        FROM {table} k
        WHERE k.deviceId=? ORDER BY k.start"""
    result = []
    for row in connection.execute(query, (device,)):
        result.append(Kernel(row[0], row[1], strings[row[2]], *row[3:8],
                             tuple(row[8:11]), tuple(row[11:14])))
    return result


def read_apis(connection, tables):
    result = []
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        condition = "WHERE a.callchainId IS NULL" if "callchainId" in columns else ""
        query = f"""SELECT a.start,a.end,a.globalTid,a.correlationId,a.nameId
            FROM {table} a {condition}"""
        for start, end, tid, correlation, name_id in connection.execute(query):
            result.append((start, end, tid, correlation, strings[name_id], table))
    return sorted(result)


def api_kind(name):
    if re.search(r"(?:cu|cuda)GraphLaunch", name):
        return "graph_launch"
    if re.search(r"(?:cu|cuda)Launch(?:Kernel|CooperativeKernel)", name):
        return "kernel_launch"
    if "Synchronize" in name:
        return "synchronization"
    if "Graph" in name or "Capture" in name:
        return "graph_management"
    return "other"


def analyze_window(connection, tables, kernels, apis, device, start, end, metric_metadata):
    selected = [kernel for kernel in kernels if kernel.start < end and kernel.end > start]
    categories = collections.defaultdict(list)
    names = collections.defaultdict(list)
    grids = collections.Counter()
    kernel_intervals = []
    for kernel in selected:
        duration = min(end, kernel.end) - max(start, kernel.start)
        categories[kernel.category].append(duration)
        names[(kernel.category, kernel.name)].append(duration)
        kernel_intervals.append((max(start, kernel.start), min(end, kernel.end)))
        grids[(kernel.category, kernel.name, kernel.grid, kernel.block, kernel.graph)] += 1
    denominator = sum(sum(values) for values in categories.values())
    category_summary = {
        category: dict(summary(values), share_of_summed_kernel_time_pct=100 * sum(values) / denominator)
        for category, values in sorted(categories.items(), key=lambda item: -sum(item[1]))
    } if denominator else {}

    activity_intervals = list(kernel_intervals)
    transfer_summary = {}
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in tables:
            continue
        intervals, byte_count = [], 0
        for row in connection.execute(
            f'SELECT start,end,bytes FROM "{table}" WHERE deviceId=? AND start<? AND end>?',
            (device, end, start),
        ):
            intervals.append((max(start, row[0]), min(end, row[1])))
            byte_count += row[2]
        activity_intervals.extend(intervals)
        transfer_summary[table] = {"count": len(intervals), "bytes_for_overlapping_events": byte_count,
                                   "summed_duration_ns": sum(b - a for a, b in intervals)}
    busy = merge_intervals(activity_intervals)
    busy_ns = sum(b - a for a, b in busy)
    gaps = [(a[1], b[0]) for a, b in zip(busy, busy[1:]) if b[0] > a[1]]
    window_apis = [api for api in apis if api[0] < end and api[1] > start]
    api_stats = collections.defaultdict(list)
    api_kinds = collections.Counter()
    per_thread = collections.defaultdict(list)
    for api in window_apis:
        a, b, tid, correlation, name, table = api
        api_stats[(table, name)].append(min(end, b) - max(start, a))
        api_kinds[api_kind(name)] += 1
        if api_kind(name) in ("kernel_launch", "graph_launch"):
            per_thread[(table, tid)].append(api)
    launch_gaps = []
    thread_launch_gaps = []
    for (table, tid), sequence in per_thread.items():
        values = [max(0, current[0] - previous[1])
                  for previous, current in zip(sequence, sequence[1:])]
        launch_gaps.extend(values)
        thread_launch_gaps.append({"table": table, "global_tid": tid, "launch_count": len(sequence),
                                   "inter_launch_api_gap_ns": summary(values)})
    by_correlation = collections.defaultdict(list)
    for api in apis:
        if api[2] is not None and api[3]:
            by_correlation[(api[2] & 0xFFFFFFFFFF000000, api[3])].append(api)
    latency, graph_latency = [], []
    matched = 0
    for kernel in selected:
        matches = by_correlation.get((kernel.pid, kernel.correlation), [])
        if not matches:
            continue
        launch = min(matches, key=lambda api: abs(kernel.start - api[1]))
        if api_kind(launch[4]) not in ("kernel_launch", "graph_launch"):
            continue
        matched += 1
        (graph_latency if kernel.graph else latency).append(kernel.start - launch[1])

    metrics = []
    if "GPU_METRICS" in tables and metric_metadata:
        for metadata in metric_metadata:
            if metadata["device_id"] != device:
                continue
            samples = connection.execute(
                "SELECT timestamp,value FROM GPU_METRICS WHERE typeId=? AND metricId=? "
                "AND timestamp>=? AND timestamp<? ORDER BY timestamp",
                (metadata["typeId"], metadata["metricId"], start, end),
            ).fetchall()
            if not samples:
                continue
            values = [row[1] for row in samples if row[1] is not None and math.isfinite(row[1])]
            spacing = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
            metrics.append(dict(metadata, sample_values=summary(values),
                                sample_spacing_ns=summary(spacing),
                                first_sample_ns=samples[0][0], last_sample_ns=samples[-1][0],
                                nonfinite_samples=len(samples) - len(values)))
    return {
        "start_ns": start, "end_ns": end, "duration_ns": end - start,
        "kernel_timeline_available": "CUPTI_ACTIVITY_KIND_KERNEL" in tables,
        "kernel_count": len(selected), "summed_kernel_time_ns": denominator,
        "kernel_busy_union_ns": interval_total(kernel_intervals),
        "gpu_activity_busy_union_ns": busy_ns,
        "gpu_activity_busy_pct": 100 * busy_ns / (end - start),
        "busy_time_is_lower_bound_only": "CUPTI_ACTIVITY_KIND_KERNEL" not in tables,
        "gpu_internal_idle_gap_ns": summary([b - a for a, b in gaps]) if "CUPTI_ACTIVITY_KIND_KERNEL" in tables else None,
        "longest_gpu_internal_idle_gaps": [
            {"start_ns": a, "end_ns": b, "duration_ns": b - a}
            for a, b in sorted(gaps, key=lambda interval: interval[1] - interval[0], reverse=True)[:20]
        ] if "CUPTI_ACTIVITY_KIND_KERNEL" in tables else None,
        "boundary_idle_ns": (((busy[0][0] - start) + (end - busy[-1][1])) if busy else end - start) if "CUPTI_ACTIVITY_KIND_KERNEL" in tables else None,
        "categories": category_summary,
        "top_kernels": [dict(category=category, name=name, duration_ns=summary(values))
                        for (category, name), values in sorted(names.items(), key=lambda item: -sum(item[1]))[:60]],
        "kernel_shapes": [dict(category=key[0], name=key[1], grid=key[2], block=key[3],
                               graph_node=key[4], count=count) for key, count in grids.most_common()],
        "graph_kernel_count": sum(kernel.graph for kernel in selected),
        "graph_kernel_time_ns": sum(min(end, kernel.end) - max(start, kernel.start)
                                    for kernel in selected if kernel.graph),
        "transfers": transfer_summary,
        "api_call_kinds": dict(api_kinds),
        "api_functions": [dict(table=table, name=name, duration_ns=summary(values))
                          for (table, name), values in sorted(api_stats.items(), key=lambda item: -sum(item[1]))],
        "cpu_inter_launch_api_gap_ns": summary(launch_gaps),
        "cpu_inter_launch_api_gaps_by_thread": thread_launch_gaps,
        "correlated_kernel_count": matched,
        "nongraph_kernel_start_minus_launch_api_end_ns": summary(latency),
        "graph_node_start_minus_launch_api_end_ns": summary(graph_latency),
        "gpu_metrics_status": "present" if metrics else "no samples in window or counters absent",
        "gpu_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--windows", type=Path,
                        help='JSON list [{"name":"c6_decode","start_ns":...,"end_ns":...}]; trace timestamps')
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = read_kernels(connection, tables, args.device)
    classification = annotate_categories(kernels)
    apis = read_apis(connection, tables)
    if kernels:
        default_start, default_end = kernels[0].start, max(kernel.end for kernel in kernels)
    elif "GPU_METRICS" in tables:
        default_start, default_end = connection.execute(
            "SELECT min(timestamp),max(timestamp) FROM GPU_METRICS").fetchone()
        default_end += 1
    elif apis:
        default_start, default_end = apis[0][0], max(api[1] for api in apis)
    else:
        raise SystemExit("No kernels, GPU samples or CUDA APIs were recorded.")
    metric_metadata = []
    if "TARGET_INFO_GPU_METRICS" in tables:
        for row in read_rows(connection, "TARGET_INFO_GPU_METRICS"):
            name = row.get("metricName", row.get("name", ""))
            if re.search(r"SM.? Active|DRAM|VRAM|SM.? Issue|Tensor Active|Compute Warps", name, re.IGNORECASE):
                metric_metadata.append(dict(row, device_id=row["typeId"] & 0xFF))
    windows = json.loads(args.windows.read_text()) if args.windows else [
        {"name": "recorded_kernel_span" if kernels else "recorded_counter_or_api_span",
         "start_ns": default_start, "end_ns": default_end}
    ]
    result = {
        "source": str(args.sqlite.resolve()), "device": args.device,
        "metadata": {table: read_rows(connection, table) for table in
                     ("META_DATA_EXPORT", "TARGET_INFO_SESSION_START_TIME", "TARGET_INFO_GPU") if table in tables},
        "diagnostics": read_rows(connection, "DIAGNOSTIC_EVENT") if "DIAGNOSTIC_EVENT" in tables else [],
        "available_metric_metadata": read_rows(connection, "TARGET_INFO_GPU_METRICS")
                                     if "TARGET_INFO_GPU_METRICS" in tables else [],
        "classification": classification,
        "kernel_timeline_status": "present" if kernels else "MISSING: GPU busy time and kernel shares cannot be determined",
        "methodology": [
            "All duration fields are nanoseconds. Window-clipped summed kernel time is not wall time; kernels can overlap.",
            "GPU activity busy union includes traced kernels, memcpy and memset on the selected GPU; untraced work is unknown.",
            "Grouped MMQ is inferred only inside complete same-process/context/stream dispatch-count to weighted-reduce regions.",
            "MMQ outside a recognized grouped region is kept separate; symbols alone do not distinguish dense and expert matrices.",
            "Graph node membership uses graphId/graphNodeId. Kernel-launch API counts may include capture-time calls.",
            "CPU inter-launch API gaps include application work, waits and scheduler time; they are not proven GPU idle time.",
            "Correlation joins include process and correlation ID. Positive launch-to-kernel latency can mean useful queueing.",
            "GPU metrics are point-filtered sample means and quantiles, without assumed start/end semantics for each sample period.",
            "SM Active measures cycles with at least one resident warp averaged over SMs, not arithmetic utilization.",
            "DRAM/VRAM read/write bandwidth percentages are interface-active cycles, not measured bytes/s or total-memory saturation.",
            "Node tracing and GPU counters add profiling overhead; compare diagnostics separately from ordinary benchmark rates.",
        ],
        "windows": {},
    }
    for window in windows:
        if window["end_ns"] <= window["start_ns"]:
            raise SystemExit(f"Invalid window: {window}")
        result["windows"][window["name"]] = analyze_window(
            connection, tables, kernels, apis, args.device,
            window["start_ns"], window["end_ns"], metric_metadata,
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, window in result["windows"].items():
        if window["kernel_timeline_available"]:
            print(f"{name}: {window['kernel_count']} kernels, GPU traced busy {window['gpu_activity_busy_pct']:.2f}%")
        else:
            print(f"{name}: MISSING kernel timeline; only API/counter analysis is valid")
        for category, values in window["categories"].items():
            print(f"  {category}: {values['share_of_summed_kernel_time_pct']:.2f}% summed kernel time")
    print(args.output)


if __name__ == "__main__":
    main()
