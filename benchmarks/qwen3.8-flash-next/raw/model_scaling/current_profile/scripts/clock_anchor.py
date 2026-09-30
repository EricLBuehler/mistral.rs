"""Bound the mapping between client monotonic timestamps and the trace epoch."""

import time


def anchor():
    mono_before = time.perf_counter_ns()
    realtime = time.time_ns()
    mono_after = time.perf_counter_ns()
    raw_before = time.perf_counter_ns()
    raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    raw_after = time.perf_counter_ns()
    return {
        'realtime_ns': realtime,
        'realtime_perf_counter_lower_ns': mono_before,
        'realtime_perf_counter_upper_ns': mono_after,
        'monotonic_raw_ns': raw,
        'raw_perf_counter_lower_ns': raw_before,
        'raw_perf_counter_upper_ns': raw_after,
        'perf_counter_clock': time.get_clock_info('perf_counter').implementation,
    }
