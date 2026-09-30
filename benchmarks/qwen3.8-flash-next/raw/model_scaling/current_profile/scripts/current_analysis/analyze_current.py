#!/usr/bin/env python3
"""Analyze exported full-model captures only after their server has exited."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3

import counter_reader
import nsys_reader
from semantic_classifier import is_projection, qformat, quantify

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PHASES = ('profile_c1', 'profile_c8')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def same_process(identity):
    path = Path('/proc') / str(identity['pid']) / 'stat'
    if not path.exists():
        return False
    fields = path.read_text().rsplit(') ', 1)[1].split()
    return int(fields[19]) == identity['starttime_ticks'] and fields[0] not in ('Z', 'X')


def require_finished(run, analysis):
    metadata = load(run / 'metadata.json')
    if not metadata.get('complete') or not metadata.get('cleanup', {}).get('model_exited'):
        raise ValueError('Wait for completed captures and model shutdown')
    if same_process(metadata['model_process']):
        raise ValueError('Original model process remains alive')
    provenance = load(analysis / 'provenance.json')
    if not provenance.get('complete'):
        raise ValueError('Wait for completed exports and base analysis')
    if provenance['run_metadata_sha256'] != sha(run / 'metadata.json'):
        raise ValueError('Export provenance refers to different run metadata')
    return metadata, provenance


def clipped_ns(kernel, start, end):
    return max(0, min(end, kernel.end) - max(start, kernel.start))


def forward_segments(kernels, semantic):
    streams = collections.defaultdict(list)
    for index, kernel in enumerate(kernels):
        streams[(kernel.pid, kernel.context, kernel.stream)].append(index)
    segments = []
    for indices in streams.values():
        previous = 0
        for position, index in enumerate(indices):
            if semantic.get(index) != 'vocabulary_head_projection' or not is_projection(kernels[index]):
                continue
            members = indices[previous:position + 1]
            hc = sum('q4_hc_mix_kernel' in kernels[i].name for i in members)
            gdn = sum(any(part in kernels[i].name for part in (
                'gdn_speculative_recurrence_rmsnorm_gate', 'gated_delta_rule_recurrence_kernel'))
                for i in members)
            kind = 'target' if hc == 97 else 'draft' if hc == 3 and gdn == 0 else 'unknown'
            head = kernels[index]
            match = re.search(r'_plain_cuda(\d+)$', head.name)
            segments.append({'kind': kind, 'members': members, 'head_index': index,
                             'hc_mix_count': hc, 'gdn_recurrence_count': gdn,
                             'head_format': qformat(head),
                             'projected_vocabulary_rows': int(match[1]) if match else None,
                             'start_ns': kernels[members[0]].start, 'head_end_ns': head.end})
            previous = position + 1
    return segments


def summarize_window(kernels, semantic, segments, window):
    start, end = window['start_ns'], window['end_ns']
    categories = collections.defaultdict(collections.Counter)
    details = collections.defaultdict(collections.Counter)
    for index, kernel in enumerate(kernels):
        duration = clipped_ns(kernel, start, end)
        if not duration:
            continue
        label = semantic.get(index, kernel.category)
        categories[label]['count'] += 1
        categories[label]['duration_ns'] += duration
        categories[label]['graph_ns' if kernel.graph else 'eager_ns'] += duration
        if 'stream_k_fixup' in kernel.name:
            categories[label]['fixup_ns'] += duration
        if index in semantic:
            details[(label, kernel.name, kernel.grid, kernel.block, bool(kernel.graph))]['count'] += 1
            details[(label, kernel.name, kernel.grid, kernel.block, bool(kernel.graph))]['duration_ns'] += duration
    total = sum(row['duration_ns'] for row in categories.values())
    forwards = collections.defaultdict(collections.Counter)
    serialized = []
    for segment in segments:
        members = segment['members']
        duration = sum(clipped_ns(kernels[i], start, end) for i in members)
        if not duration:
            continue
        kind = segment['kind']
        head = kernels[segment['head_index']]
        head_ns = clipped_ns(head, start, end)
        graph_ns = sum(clipped_ns(kernels[i], start, end) for i in members if kernels[i].graph)
        row = forwards[kind]
        row['overlapping_intervals'] += 1
        row['kernel_time_ns'] += duration
        row['head_kernel_time_ns'] += head_ns
        row['graph_kernel_time_ns'] += graph_ns
        complete = segment['start_ns'] >= start and segment['head_end_ns'] <= end
        row['fully_contained_intervals'] += complete
        if head.start >= start and head.end <= end:
            projected = segment['projected_vocabulary_rows']
            if projected is not None:
                row['known_projected_vocabulary_rows'] += projected
            else:
                row['head_calls_with_unknown_row_count'] += 1
        serialized.append({key: value for key, value in segment.items() if key not in ('members', 'head_index')} |
                          {'clipped_kernel_time_ns': duration, 'clipped_head_kernel_time_ns': head_ns,
                           'fully_contained': complete})
    return {'start_ns': start, 'end_ns': end, 'duration_ns': end - start,
            'summed_kernel_time_ns': total,
            'categories': [{'category': key, **row, 'share_pct': row['duration_ns'] * 100 / total if total else None}
                           for key, row in sorted(categories.items(), key=lambda item: -item[1]['duration_ns'])],
            'forward_kind_totals': dict(forwards), 'forward_intervals': serialized,
            'projection_details': [{'category': key[0], 'name': key[1], 'grid': key[2],
                                    'block': key[3], 'graph': key[4], **row}
                                   for key, row in sorted(details.items(), key=lambda item: -item[1]['duration_ns'])]}


def normalize(envelope, interior, output_tokens, counts):
    if output_tokens <= 0:
        raise ValueError('No completed output tokens')
    if interior['start_ns'] < envelope['start_ns'] or interior['end_ns'] > envelope['end_ns']:
        raise ValueError('Alignment interior is not contained in its envelope')
    inner = {row['category']: row['duration_ns'] for row in interior['categories']}
    rows = []
    for row in envelope['categories']:
        duration = row['duration_ns']
        lower = inner.get(row['category'], 0)
        if lower > duration:
            raise ValueError('Interior category cost exceeds envelope')
        rows.append({'category': row['category'],
                     'envelope_ms_per_completed_output_token': duration / output_tokens / 1e6,
                     'alignment_boundary_kernel_time_difference_ns': duration - lower})
    drafts = envelope['forward_kind_totals'].get('draft', {})
    proposed = counts['proposed_draft_tokens']
    sequence_proposals = counts['sequence_proposals']
    def per_ms(numerator, denominator):
        return numerator / denominator / 1e6 if denominator else None
    return {
        'scope': 'Whole finite request-phase envelope, including prefill and startup/drain. Not pure decode or an unprofiled benchmark.',
        'completed_output_tokens': output_tokens, 'counts': counts,
        'summed_kernel_ms_per_completed_output_token': envelope['summed_kernel_time_ns'] / output_tokens / 1e6,
        'alignment_boundary_kernel_time_difference_ns': envelope['summed_kernel_time_ns'] - interior['summed_kernel_time_ns'],
        'categories': rows,
        'draft_shaped_kernel_ms_per_recorded_proposed_draft_token': per_ms(drafts.get('kernel_time_ns', 0), proposed),
        'draft_head_ms_per_recorded_proposed_draft_token': per_ms(drafts.get('head_kernel_time_ns', 0), proposed),
        'draft_shaped_kernel_ms_per_sequence_proposal': per_ms(drafts.get('kernel_time_ns', 0), sequence_proposals),
        'known_draft_head_projected_rows': drafts.get('known_projected_vocabulary_rows', 0),
        'unknown_draft_head_row_count_calls': drafts.get('head_calls_with_unknown_row_count', 0),
        'projected_row_count_minus_proposed_counter': drafts.get('known_projected_vocabulary_rows', 0) - proposed,
        'proposal_limits': [
            'Metrics bracket the whole request command with small idle setup margins; sequence_proposals count verified sequence rows, not batch steps.',
            'Topology-classified draft intervals may include prefill or replay work; mixed/unknown intervals remain unassigned.',
            'Draft vocabulary rows and proposed-token counters need not match because of prefill/replay, discarded lookahead, boundaries or incomplete trace events; the difference is reported, not assumed zero.',
            'Post-head sampling lies in the next topology interval; interval costs are broad stage accounting.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'current_capture')
    parser.add_argument('--analysis', type=Path, default=ROOT / 'analysis')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--snapshot-label', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'plan_only': True, 'snapshot_label': args.snapshot_label,
                          'run': str(args.run), 'analysis': str(args.analysis), 'output': str(args.output),
                          'gate': 'Run complete, original server gone, exporter complete and provenance matches'}, indent=2))
        return
    if args.output.exists():
        raise ValueError('Refusing to overwrite previous analysis')
    metadata, export = require_finished(args.run, args.analysis)
    mapping_sources = load(HERE / 'mapping_source_provenance.json')
    if metadata['checkpoint_metadata_sha256']['config.json'] != mapping_sources['checkpoint_config_sha256']:
        raise ValueError('Semantic mapping geometry is for a different checkpoint config')
    run_manifest, analysis_manifest = load(args.run / 'manifest.json'), load(args.analysis / 'manifest.json')
    inputs = []
    def verify(path, manifest, base):
        name = str(path.relative_to(base))
        row = manifest[name]
        expected = row['sha256'] if isinstance(row, dict) else row
        digest = sha(path)
        if digest != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        inputs.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    results = []
    for phase in PHASES:
        for name in (f'{phase}.requests.json', f'{phase}.metrics.delta.json',
                     f'{phase}.metrics.before.prom', f'{phase}.metrics.after.prom'):
            verify(args.run / name, run_manifest, args.run)
        for name in (f'{phase}.sqlite', f'{phase}.analysis.json', f'{phase}.alignment.json'):
            verify(args.analysis / name, analysis_manifest, args.analysis)
        requests = load(args.run / f'{phase}.requests.json')
        if not requests['complete'] or not requests['trial']['complete']:
            raise ValueError('Incomplete phase requests')
        trial = requests['trial']
        outputs = sum(request['response']['usage']['completion_tokens'] for request in trial['requests'])
        if outputs != trial['completed_output_tokens'] or any(request['error'] for request in trial['requests']):
            raise ValueError('Request output/error inconsistency')
        counters = load(args.run / f'{phase}.metrics.delta.json')
        recomputed = counter_reader.summarize(phase, counter_reader.counter_deltas(
            (args.run / f'{phase}.metrics.before.prom').read_text(),
            (args.run / f'{phase}.metrics.after.prom').read_text()))
        for key in ('counts', 'accepted_positions', 'graph_dispatches'):
            if counters[key] != recomputed[key]:
                raise ValueError(f'Saved counter mismatch: {phase} {key}')
        with sqlite3.connect(f'file:{(args.analysis / (phase + ".sqlite")).resolve()}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA cache_size=-4096')
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            kernels = nsys_reader.read_kernels(connection, tables, 0)
        if not kernels:
            raise ValueError('No recorded GPU kernels')
        grouped = nsys_reader.annotate_categories(kernels)
        semantic, reasons, hc, unmatched = quantify(kernels)
        segments = forward_segments(kernels, semantic)
        base = load(args.analysis / f'{phase}.analysis.json')
        alignment = load(args.analysis / f'{phase}.alignment.json')
        windows = {}
        for name, original in base['windows'].items():
            window = summarize_window(kernels, semantic, segments, original)
            if window['summed_kernel_time_ns'] != original['summed_kernel_time_ns']:
                raise ValueError('Semantic partition differs from base kernel sum')
            window['base_gpu_activity_busy_pct'] = original['gpu_activity_busy_pct']
            window['base_graph_kernel_time_ns'] = original['graph_kernel_time_ns']
            window['base_api_call_kinds'] = original['api_call_kinds']
            windows[name] = window
        for required in ('client_request_envelope', 'client_request_interior'):
            if required not in windows:
                raise ValueError(f'Missing exact-request alignment window: {required}')
        normalized = normalize(windows['client_request_envelope'], windows['client_request_interior'], outputs, counters['counts'])
        result = {'phase': phase, 'snapshot_label': args.snapshot_label,
                  'hc_pair_validation': hc, 'hc_unmatched_examples': unmatched,
                  'grouped_classification': grouped, 'windows': windows, 'normalization': normalized,
                  'request_summary': {'completed_output_tokens': outputs, 'requests': len(trial['requests']),
                                      'trial_wall_seconds': trial['wall_seconds'],
                                      'exact_request_window_perf_counter_ns': requests['request_window_perf_counter_ns']},
                  'phase_counters': counters, 'alignment': alignment, 'profiler_diagnostics': base['diagnostics'],
                  'limitations': ['Recovered kernel timeline; retain profiler diagnostics and collection warnings.',
                                  'Instrumented finite phase includes prefill and drain; do not treat profiled elapsed throughput as an official benchmark.',
                                  'Semantic mapping uses source topology, formats and quantizer dimensions; unmatched matrices and mixed forward intervals stay unknown.']}
        results.append(result)
        del kernels
    args.output.mkdir(parents=True)
    save(args.output / 'semantic_breakdown.json', results)
    save(args.output / 'compact_summary.json', [
        {'phase': result['phase'], 'snapshot_label': result['snapshot_label'],
         'normalization': result['normalization'],
         'hc_pair_validation': result['hc_pair_validation'],
         'request_window': {key: value for key, value in result['windows']['client_request_envelope'].items()
                            if key not in ('projection_details', 'forward_intervals')},
         'phase_counters': result['phase_counters'], 'limitations': result['limitations']}
        for result in results])
    sources = [Path(__file__), HERE / 'semantic_classifier.py', HERE / 'nsys_reader.py', HERE / 'counter_reader.py']
    save(args.output / 'provenance.json', {
        'complete': True, 'snapshot_label': args.snapshot_label, 'run_metadata': metadata,
        'run_metadata_sha256': sha(args.run / 'metadata.json'),
        'export_provenance_sha256': sha(args.analysis / 'provenance.json'),
        'inputs': inputs, 'semantic_mapping_sources': mapping_sources, 'scripts': [{'path': str(path), 'sha256': sha(path)} for path in sources],
    })
    save(args.output / 'manifest.json', {path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file()})
    print(json.dumps({'complete': True, 'snapshot_label': args.snapshot_label, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
