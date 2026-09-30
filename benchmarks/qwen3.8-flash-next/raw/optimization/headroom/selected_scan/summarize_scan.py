#!/usr/bin/env python3
"""Pair fixed-chunk selected scans with the bracketing native FFN replays."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import statistics

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
PRIMARY_CHUNK_KIB = 8
PROJECTIONS = ('gate', 'up', 'down', 'total')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def median_ffn(row, backend):
    return statistics.median(t['stream_elapsed_ms_per_layer'] for t in row[backend])


def main():
    assert load(WORK/'timed/metadata.json')['complete']
    scan = load(WORK/'timed/results.json')['results']
    phases = {}
    replay_diagnostics = {}
    for phase in ('ffn_before', 'ffn_after'):
        assert load(ROOT/phase/'metadata.json')['complete'], phase
        full_replay = load(ROOT/phase/'replay.json')['results']
        phases[phase] = {row['source_metadata']: row for row in full_replay if row['selection'] == 'native'}
        failures = [dict(source_metadata=row['source_metadata'], selection=row['selection'], rows=row['rows'], metrics=row['numerical']) for row in full_replay if not row['cross_kernel_guard_passed']]
        assert len(full_replay) == 71 and len(phases[phase]) == 26 and len(failures) == 6
        assert all(row['native_replay_guard_passed'] is True for row in phases[phase].values())
        replay_diagnostics[phase] = dict(total_cases=71,native_source_path_guards_passed=26,
                                        cross_gemv_mmq_diagnostic_failures=failures,
                                        meaning='Existing indexed-GEMV versus alternate-MMQ arithmetic differences, not failures of a new candidate or of the scan. Four are native M1/M7; two are derived M8. Current native default is GEMV at M1/M7 and MMQ at M42/M56.')
    records = []
    for case in scan:
        samples = defaultdict(list)
        for timing in case['timings']:
            samples[(timing['projection'], timing['chunk_kib'], timing['cache_mode'])].append(timing['event_ms_per_scan'])
        assert len(samples) == 32 and all(len(values) == 7 for values in samples.values())
        scans = [dict(projection=key[0],chunk_kib=key[1],cache_mode=key[2],median_ms=statistics.median(values),min_ms=min(values),max_ms=max(values),sample_stdev_ms=statistics.stdev(values)) for key, values in sorted(samples.items())]
        native = {phase: values[case['source_metadata']] for phase, values in phases.items()}
        for row in native.values():
            assert row['active_experts'] == case['selected_count'] and row['rows'] == case['rows']
            assert row['native_replay_guard_passed'] is True
        backend = 'grouped' if case['rows'] >= 32 else 'gemv'
        before = median_ffn(native['ffn_before'], backend)
        after = median_ffn(native['ffn_after'], backend)
        midpoint = (before+after)/2
        warm = statistics.median(samples[('total',PRIMARY_CHUNK_KIB,'warm')])
        flushed = statistics.median(samples[('total',PRIMARY_CHUNK_KIB,'eviction_attempt')])
        records.append(dict(source_metadata=case['source_metadata'],rows=case['rows'],selected_experts=case['selected_count'],
                            selected_logical_payload_bytes=case['selected_count']*2867200,
                            assignments=case['assignments'],assignments_per_selected_expert=case['assignments_per_selected_expert'],
                            scans=scans,default_backend=backend,ffn_before_ms=before,ffn_after_ms=after,
                            ffn_stage_midpoint_ms=midpoint,ffn_after_vs_before_percent=100*(after/before-1),
                            primary_warm_scan_ms=warm,primary_flush_control_ms=flushed,
                            paired_ffn_midpoint_over_warm_scan=midpoint/warm))
    summaries = []
    sensitivity = []
    for rows in (1,7,42,56):
        cases = [case for case in records if case['rows']==rows]
        summarize = lambda key: statistics.median(case[key] for case in cases)
        summaries.append(dict(rows=rows,captures=len(cases),selected_experts_median=summarize('selected_experts'),
                              selected_payload_mib_median=summarize('selected_logical_payload_bytes')/2**20,
                              warm_scan_ms_median=summarize('primary_warm_scan_ms'),flush_control_ms_median=summarize('primary_flush_control_ms'),
                              ffn_before_ms_median=summarize('ffn_before_ms'),ffn_after_ms_median=summarize('ffn_after_ms'),
                              ffn_stage_midpoint_ms_median=summarize('ffn_stage_midpoint_ms'),
                              paired_ffn_midpoint_over_warm_scan_median=summarize('paired_ffn_midpoint_over_warm_scan'),
                              ffn_after_vs_before_percent_min=min(case['ffn_after_vs_before_percent'] for case in cases),
                              ffn_after_vs_before_percent_max=max(case['ffn_after_vs_before_percent'] for case in cases)))
        for chunk in (8,32,64,256):
            for mode in ('warm','eviction_attempt'):
                values = [next(scan['median_ms'] for scan in case['scans'] if scan['projection']=='total' and scan['chunk_kib']==chunk and scan['cache_mode']==mode) for case in cases]
                sensitivity.append(dict(rows=rows,chunk_kib=chunk,cache_mode=mode,
                                        median_across_capture_ms=statistics.median(values),
                                        min_across_capture_ms=min(values),max_across_capture_ms=max(values)))
    evidence = [WORK/'timed/metadata.json',WORK/'timed/results.json',WORK/'validation/metadata.json',WORK/'memcheck.log',WORK/'sass_evidence.json',ROOT/'ffn_before/metadata.json',ROOT/'ffn_before/replay.json',ROOT/'ffn_after/metadata.json',ROOT/'ffn_after/replay.json']
    result = dict(primary_chunk_kib=PRIMARY_CHUNK_KIB,
                  aggregation='Median of seven CUDA-event means, ten repetitions per mean. Across-capture entries are medians. Paired FFN uses the midpoint of before/after per-capture medians; ratios are computed per capture before their median.',
                  interpretation='Diagnostic headroom control for exact selected compressed expert bytes. Not a physical lower bound, actual DRAM bandwidth, whole-model speedup prediction, or evidence that scan speed can be attained while computing the FFN.',
                  evidence={str(path):sha(path) for path in evidence},summary=summaries,
                  chunk_sensitivity=sensitivity,replay_diagnostics=replay_diagnostics,cases=records)
    (WORK/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Selected-weight scan control','',
             'The scan reads the exact compressed weights of every expert selected by each native layer-8 capture. It performs no model computation. Four XOR checksums per block keep all vector loads observable; CPU per-expert checksums matched for all 312 projection/chunk combinations. Focused sanitizer validation passed with zero errors.','',
             'The table uses the same 8 KiB chunk configuration for every capture. All four tested chunk sizes remain in the raw results. FFN values are the midpoint of the before/after per-capture medians, then the median across captures. The scan-to-FFN ratios are paired per capture.','',
             '| Native rows | Captures | Median selected experts | Selected MiB | Warm scan ms | >L2 flush control ms | FFN ms | Paired FFN/scan |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in summaries:
        lines.append(f"| {row['rows']} | {row['captures']} | {row['selected_experts_median']:g} | {row['selected_payload_mib_median']:.2f} | {row['warm_scan_ms_median']:.4f} | {row['flush_control_ms_median']:.4f} | {row['ffn_stage_midpoint_ms_median']:.4f} | {row['paired_ffn_midpoint_over_warm_scan_median']:.3f}x |")
    lines += ['',
              'The M56 bracketing FFN measurements differed by -1.37% to +0.63% per capture. M42 differed by -0.96% to +3.49%; the much shorter M1 cases varied more (-23.29% to +12.38%). Treat the small-case ratios cautiously.','',
              'Both 71-case FFN replays retain six pre-existing GEMV-versus-MMQ numerical diagnostic failures: four native M1/M7 cases and two derived M8 cases. All 26 native source-path guards pass in each replay. No alternate MMQ output is substituted for the current M1/M7 GEMV default in this comparison. The scan has no model output.','',
              'The device reported 24 MiB of L2. The separate 96 MiB flush writes finish before each scan event window; this is an eviction attempt, not proof of cold DRAM. Repeated same-layer scans and FFNs have different access patterns, instruction costs, launch counts, and cache histories. The checksum adds work. Logical payload divided by event time must not be labeled measured DRAM bandwidth.','',
              'The selected payload is 2,867,200 bytes per expert: Q4K gate and up each use 921,600 bytes, and Q4_1 down uses 1,024,000. Output-feature tiles partition these weight rows. For the captured native M42/M56 cases, the current MMQ query tile covers each active expert in one tile; reducing query width can introduce additional reads of that expert.','',
              'These fixed-Q7, single-layer diagnostic captures do not describe every layer or the current adaptive-depth serving distribution. The remaining scan/FFN gap is an empirical comparison, not a guaranteed optimization budget or a model-wide ceiling.','']
    lines += ['## Chunk sensitivity','',
              '| Native rows | Chunk KiB | Warm total ms | >L2 flush control total ms |',
              '| --- | --- | --- | --- |']
    for rows in (1,7,42,56):
        for chunk in (8,32,64,256):
            values = {row['cache_mode']:row['median_across_capture_ms'] for row in sensitivity if row['rows']==rows and row['chunk_kib']==chunk}
            lines.append(f"| {rows} | {chunk} | {values['warm']:.4f} | {values['eviction_attempt']:.4f} |")
    lines.append('')
    (WORK/'report.md').write_text('\n'.join(lines))
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
