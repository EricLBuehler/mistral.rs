"""Small Linux process/memory snapshots for the final serving run."""

import datetime
import os
from pathlib import Path
import time

WATCHED_COMMANDS = {
    "cargo",
    "cargo-clippy",
    "rustc",
    "clippy-driver",
    "rust-analyzer",
    "nvcc",
    "cicc",
    "ptxas",
    "cc1plus",
    "g++",
    "clang++",
    "nsys",
    "ncu",
    "llama-server",
    "llama-bench",
    "mistralrs",
}
WATCHED_PREFIXES = ("moe_dispatch", "dense_backend", "cutile_gguf", "grouped_mmq")
PROCESS_FIELDS = {"VmRSS", "VmSwap", "VmHWM", "RssAnon", "RssFile"}
MEMORY_FIELDS = {"MemAvailable", "SwapFree", "SwapTotal"}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stamp():
    return {"utc": utc(), "monotonic": time.monotonic()}


def process_record(pid):
    path = Path("/proc") / str(pid)
    stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
    status = {}
    for line in (path / "status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in PROCESS_FIELDS:
            status[key + "_KiB"] = int(value.split()[0])
    return {
        "pid": int(pid),
        "state": stat[0],
        "parent_pid": int(stat[1]),
        "starttime_ticks": int(stat[19]),
        "comm": (path / "comm").read_text().strip(),
        "command": (path / "cmdline")
        .read_bytes()
        .decode(errors="replace")
        .split("\0")[:-1],
        **status,
    }


def isolation_snapshot(owned_pids=()):
    watched = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            if comm in WATCHED_COMMANDS or comm.startswith(WATCHED_PREFIXES):
                watched.append(process_record(int(entry.name)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    unexpected = [
        item
        for item in watched
        if item["pid"] not in owned_pids and item["state"] not in ("T", "t", "Z", "X")
    ]
    return {**stamp(), "watched": watched, "unexpected_active": unexpected}


def memory_snapshot(model_pid=None):
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in MEMORY_FIELDS:
            memory[key + "_KiB"] = int(value.split()[0])
    counters = dict(
        line.split() for line in Path("/proc/vmstat").read_text().splitlines()
    )
    record = {
        **stamp(),
        "system": memory,
        "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
        "pswpin_pages": int(counters["pswpin"]),
        "pswpout_pages": int(counters["pswpout"]),
        "model_pid": model_pid,
    }
    if model_pid is not None:
        try:
            record["model"] = process_record(model_pid)
        except (FileNotFoundError, ProcessLookupError):
            record["model_exited"] = True
    return record
