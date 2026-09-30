# GPU L2 refill diagnostic

The authorized profile completed at 2026-09-30 17:14:18 UTC. The raw report and
all three projection records are preserved in `masked_mmq/`; `summary.json`
contains independently recomputed refill and selected-weight accounting.

| Projection | Selected weights, MB | L2 refill equivalents, MB | Refill / selected weights | Profiled refill-equivalent rate, GB/s |
|---|---:|---:|---:|---:|
| Gate | 196.301 | 204.533 | 1.0419 | 169.8 |
| Up | 196.301 | 204.604 | 1.0423 | 157.1 |
| Down | 218.112 | 218.931 | 1.0038 | 170.4 |
| Total | 610.714 | 628.069 | 1.0284 | 165.6 |

MB and GB are decimal. The total rate divides summed refill equivalents by summed
profiled projection durations. All fills appeared under the system-memory aperture;
the device-memory fill counters were zero. Each kernel used five replay passes.
The full native captured output still matched exactly, and boundary process checks
found only the stopped editor process.

For these projections, refills are only 2.84% above the selected quantized-weight
payload. The counters do not show large excess refill traffic over that payload.
They do not prove physical DRAM saturation or a full-model performance ceiling.

The default command profiles the frozen masked-MMQ replay binary with SHA256
`eaf7d867dadfd56433f0f8a093e0f6dd320b1bcfde76ee76c58e32027443f7f1`.
It selects `c8_target_verify_b8_q7_04.json`, the native 56-row input with 213
unique experts, closest to the median of the five actual C8 captures. This is the
same selection rule and sample as the earlier native-56 NCU profile.

The first 12 matching `mul_mat_q` launches are one reference forward and three
warmups, with gate, up, and down in each forward. The next three launches are
profiled. This selects the native shape before the replay starts derived shapes.
The profiler uses kernel replay, cache control `none`, and clock control `none`.
No profiler timing is an ordinary performance benchmark.

Read-only plan (no profiler process or GPU query):

```bash
python3 /home/ericbuehler/qwen4exp_work/moe_headroom_20260930/l2_refill/profile_l2_refill.py
```

After the coordinator releases the GPU slot and pauses background compilation:

```bash
python3 /home/ericbuehler/qwen4exp_work/moe_headroom_20260930/l2_refill/profile_l2_refill.py --released
```

The run creates `masked_mmq/`, refuses overwrite, verifies the frozen binary and
saved metric-query hashes, checks model/compiler process state before and after,
and preserves exact commands, sample metadata, NCU report, raw CSV, parsed counters,
and a SHA256 manifest. Existing `MISTRALRS_*` environment values are recorded and
removed before applying the two replay-specific variables. Large unchanged weight
files are symlinked to the original capture; their bytes are not recopied or
redundantly hashed.

For a new run, supply a fresh `--output-dir`; the recorded `masked_mmq/` is complete.
Recompute its accounting without GPU access:

```bash
python3 summarize_refills.py masked_mmq /tmp/l2-refill-summary.json
```

## Counter interpretation

The exact names below were found in the saved GB20B query from Nsight Compute
2026.2.1. The script checks their presence without repeating the 89 MB query.

- `lts__d_sectors_fill_sysmem.sum`: sectors filled into L2 from system memory.
- `lts__d_sectors_fill_device.sum`: sectors filled into L2 from the device-memory interface.
- `lts__t_sectors_op_read.sum`, `lts__t_sectors_op_read_lookup_hit.sum`, and
  `lts__t_sectors_op_read_lookup_miss.sum`: read demand, hits, and misses at L2.
- `lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum` and
  `lts__t_sectors_aperture_device_op_read_lookup_miss.sum`: read misses by aperture.
- `lts__t_sectors_op_write.sum`: L2 write requests, not external writebacks.
- `lts__t_sector_hit_rate.pct`: overall sector hit rate, not specifically read hit rate.
- `gpu__time_duration.sum`: profiled kernel duration.

The local NCU profiling guide defines a sector as 32 bytes. The report therefore
computes `32 * fill_sectors` and divides this by the profiled kernel duration for
an **L2 refill byte-equivalent rate**. It does not name this DRAM bandwidth.
Both apertures are retained because integrated-GPU physical memory does not imply
that all CUDA allocations appear under a particular aperture counter.

Refills are closer to external reads than total L2 requests, but they do not
establish LPDDR bytes, CPU traffic, DRAM bus utilization, or bandwidth headroom.
An intervening SoC cache, request merging, other access classes, and profiler
replay can separate refill counts from physical DRAM reads. This is one warmed
isolated layer; a full model changes cache reuse. No external-write counter is
collected. Counter replay can change cache state even with cache flushing disabled.

`--binary`, `--binary-sha256`, `--build-metadata`, `--test`, `--kernel-regex`,
`--launch-skip`, `--launch-count`, `--label`, and `--output-dir` permit a later
selected-scan comparison. Its kernel filter and warmup count must be audited against
the eventual scan source before running; the MMQ defaults must not be reused by
assumption. Profiles should use the same captured input and label distinct access
patterns. The scan is a synthetic access probe, not an equivalent FFN operation.

## Local discovery and unavailable interfaces

- Saved NCU query: `/home/ericbuehler/qwen4exp_work/real_routing_20260930/ncu_native56/query_metrics.log`,
  SHA256 `52dc30eb05091c32c404e7fe2a1ff61d3b898aee0b0d89f3110fd41acdc45d4b`.
  It contains no `dram__`, `dramc__`, or `mcc__` metric family. Relevant metric
  names and original line numbers are preserved in `plan.json`.
- Nsight Systems 2026.1.3 `profile --gpu-metrics-devices=help` identifies
  `Blackwell GB20B | NVIDIA GB10`. Its installed `GpuMetrics/gb20y.config`
  matches GB20B/GB20C and includes activity/clocks/occupancy, without DRAM metrics.
- `profile --soc-metrics-set=help` lists T234 and T264 sets. They are not a
  device-support probe. The installed documentation calls SoC metrics an Embedded
  Platforms Edition feature. `nsys status --environment` identifies this host as
  Linux SBSA; there is no exposed Tegra HWPM device, loaded HWPM driver, devfreq
  entry, or installed `tegrastats` executable. The ACPI `DRAM8901:00` device has
  no bound driver or counter attributes. No memory-controller PMU appears under
  `/sys/bus/event_source/devices`; SMMU counters are translation events.
- CUDA 13.2 CUPTI release notes describe `mcc__*` support for Orin+ mobile chips.
  This does not establish support for this GB20B/SBSA host. Switching to CUPTI
  alone does not supply a documented working counter path here. The Nsight
  Systems NVML plugin exposes power and temperature, not memory throughput.
- No SoC profile or direct CUPTI collection was attempted. The evidence supports
  "no currently exposed, verified LPDDR counter path", not a claim that all future
  driver/tool versions or privileged vendor interfaces can never expose one.

Primary local references: Nsight Systems 2026.1.3 `documentation/UserGuide/index.html`
lines 6750-6875; its `target-linux-sbsa-armv8/GpuMetrics/gb20y.config` and
`SocMetrics/t264.config`; CUDA 13.2 CUPTI `doc/html/release-notes/release-notes.html`
lines 824-826; Nsight Compute 2026.2.1 `docs/ProfilingGuide/index.html` line 1466
(sector definition) and lines 3279-3295 (L2 miss destinations).
