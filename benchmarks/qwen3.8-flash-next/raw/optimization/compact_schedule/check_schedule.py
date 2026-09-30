#!/usr/bin/env python3
"""CPU validation of compact job mapping and exact captured-route work counts."""
import hashlib
import json
import random
import statistics
from pathlib import Path

WORK = Path(__file__).resolve().parent
CAPTURES = WORK.parents[1] / 'real_routing_20260930/runtime/captures'
WIDTHS = (8, 16, 24, 32, 40, 48, 64)
EDGE_COUNTS = (0, 1, 15, 16, 17, 31, 32, 33, 55, 56, 63, 64)
GRID = 96


def prefix_counts(counts, width):
    prefix = [0]
    for rows in counts:
        prefix.append(prefix[-1] + (rows + width - 1) // width)
    return prefix


def decode(job, prefix, width):
    tile_count = prefix[-1]
    tile = job % tile_count
    low, high = 0, len(prefix) - 1
    while low < high:
        middle = low + (high - low) // 2
        if prefix[middle + 1] <= tile:
            low = middle + 1
        else:
            high = middle
    return job // tile_count, low, (tile - prefix[low]) * width


def validate(counts, width, row_tiles):
    prefix = prefix_counts(counts, width)
    expected = [(it, expert, column)
                for it in range(row_tiles)
                for expert, count in enumerate(counts)
                for column in range(0, count, width)]
    actual = [decode(job, prefix, width) for job in range(prefix[-1] * row_tiles)]
    assert expected == actual
    visited = [job for block in range(GRID)
               for job in range(block, len(expected), GRID)]
    assert sorted(visited) == list(range(len(expected)))
    for _, expert, column in actual:
        assert 0 <= column < counts[expert]
        assert min(width, counts[expert] - column) > 0
    return len(expected)


def main():
    randomizer = random.Random(71)
    cases = [list(EDGE_COUNTS), [0] * 1024, [0] * 1023 + [64],
             [64] + [0] * 1023, [64] * 1024]
    cases += [[randomizer.choice(EDGE_COUNTS) if randomizer.random() < 0.4 else 0
               for _ in range(512)] for _ in range(100)]
    checks = 0
    for counts in cases:
        for width in WIDTHS:
            for row_tiles in (1, 5, 20):
                validate(counts, width, row_tiles)
                checks += 1
    captured = []
    for path in sorted(CAPTURES.glob('*.json')):
        metadata = json.loads(path.read_text())
        if 'expert_occupancy' not in metadata:
            continue
        counts = metadata['expert_occupancy']
        rows = metadata['rows']
        item = dict(source=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    rows=rows, unique_experts=sum(c > 0 for c in counts), tile_counts={})
        for width in WIDTHS:
            validate(counts, width, 20)
            compact = prefix_counts(counts, width)[-1]
            rectangular = len(counts) * ((rows + width - 1) // width)
            item['tile_counts'][str(width)] = dict(compact=compact, rectangular=rectangular,
                                                   empty_fraction=1 - compact / rectangular)
        captured.append(item)
    summary = {}
    for rows in (42, 56):
        samples = [s for s in captured if s['rows'] == rows]
        summary[str(rows)] = {str(width): dict(
            samples=len(samples),
            compact_median=statistics.median(s['tile_counts'][str(width)]['compact'] for s in samples),
            rectangular=samples[0]['tile_counts'][str(width)]['rectangular'],
            empty_fraction_median=statistics.median(s['tile_counts'][str(width)]['empty_fraction'] for s in samples))
            for width in (16, 32, 64)}
    output = WORK / 'mapping_check.json'
    output.write_text(json.dumps(dict(complete=True, cpu_only=True, synthetic_checks=checks,
                                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     scope='Job mapping and empty tile visits, not GPU timing or memory traffic.',
                                     summary=summary, captures=captured), indent=2) + '\n')
    print(json.dumps(dict(synthetic_checks=checks, captures=len(captured), summary=summary), indent=2))


if __name__ == '__main__':
    main()
