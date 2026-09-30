#!/usr/bin/env python3
import sqlite3

from analyze_nsys import Kernel, analyze_window, annotate_categories, classify, interval_total


def main():
    assert classify("moe_grouped_gemm_q4_1", False) == "moe_grouped_gemm"
    assert classify("vmajor_chunked_gated_delta_rule_recurrence", False) == "gdn"
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE GPU_METRICS(timestamp INTEGER,typeId INTEGER,metricId INTEGER,value REAL)")
    connection.executemany("INSERT INTO GPU_METRICS VALUES(?,?,?,?)", [
        (15, 0, 1, 20), (30, 0, 1, 40), (15, 1, 1, 99), (15, 256, 1, 75),
    ])
    pid = 1 << 24
    kernels = [
        Kernel(0, 10, "moe_dispatch_count_kernel", pid, 0, 1, 1, False, (1, 1, 1), (32, 1, 1)),
        Kernel(10, 90, "mul_mat_q", pid, 0, 1, 2, True, (48, 1, 1), (32, 4, 1)),
        Kernel(80, 100, "moe_weighted_reduce_flat_kernel", pid, 0, 1, 3, True, (1, 1, 1), (32, 1, 1)),
        Kernel(25, 50, "gdn_decode", pid, 0, 2, 4, False, (48, 1, 1), (32, 4, 1)),
    ]
    assert annotate_categories(kernels) == {"completed_dispatch_reduce_regions": 1, "incomplete_regions": 0}
    assert kernels[1].category == "moe_grouped_mmq_context"
    apis = [(0, 2, pid | 1, 2, "cuGraphLaunch", "CUPTI_ACTIVITY_KIND_RUNTIME")]
    metadata = [dict(typeId=kind, metricId=1, metricName="SMs Active", device_id=kind & 0xff)
                for kind in [0, 1, 256]]
    result = analyze_window(connection, {"GPU_METRICS", "CUPTI_ACTIVITY_KIND_KERNEL"}, kernels, apis, 0, 0, 120, metadata)
    assert result["summed_kernel_time_ns"] == 135
    assert result["gpu_activity_busy_union_ns"] == 100
    assert result["boundary_idle_ns"] == 20
    assert result["graph_kernel_count"] == 2
    assert result["api_call_kinds"]["graph_launch"] == 1
    assert len(result["gpu_metrics"]) == 2
    assert result["gpu_metrics"][0]["sample_values"]["mean"] == 30
    assert result["gpu_metrics"][1]["sample_values"]["mean"] == 75
    assert interval_total([(0, 10), (5, 20), (30, 35)]) == 25
    clipped = analyze_window(connection, {"GPU_METRICS", "CUPTI_ACTIVITY_KIND_KERNEL"}, kernels, apis, 0, 12, 35, metadata)
    assert clipped["gpu_activity_busy_union_ns"] == 23
    assert clipped["summed_kernel_time_ns"] == 33
    assert clipped["gpu_activity_busy_pct"] == 100
    missing = analyze_window(connection, {"GPU_METRICS"}, [], apis, 0, 0, 120, metadata)
    assert missing["boundary_idle_ns"] is None
    assert missing["gpu_internal_idle_gap_ns"] is None
    assert missing["busy_time_is_lower_bound_only"]
    print("PASS: interval union/clipping, grouped context, graph flags/API, metric type+ID isolation")


if __name__ == "__main__":
    main()
