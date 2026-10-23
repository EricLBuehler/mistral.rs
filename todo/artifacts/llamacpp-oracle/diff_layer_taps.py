"""Diff two per-layer dumps and report the first divergent layer.

Both files are TSV. The reference side is `layer<TAB>quantity<TAB>position<TAB>value`; ours
adds a fifth column, `hidden_index`, so that a hidden vector can be compared elementwise
rather than only through some aggregate of it.

A key (layer, quantity, position) may have one row - a scalar, e.g. a norm of the whole hidden
vector - or many, one per hidden component. Both are handled, and the report says which it
found, because "the first layer where the max componentwise difference exceeds X" and "the
first layer where the scalar differs" are different questions with different sharpness.

Usage:
    python3 diff_layer_taps.py ours.tsv theirs.tsv [threshold]
"""

import collections
import math
import sys


def load(path):
    """-> {(layer, quantity, position): {hidden_index_or_None: value}}"""
    rows = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                layer = int(parts[0])
                value = float(parts[-1])
            except ValueError:
                continue
            quantity = parts[1]
            position = int(parts[2])
            hidden = int(parts[3]) if len(parts) >= 5 and parts[3].lstrip("-").isdigit() else None
            rows[(layer, quantity, position)][hidden] = value
    return rows


def stats(a, b):
    """-> (max_abs, rms, cosine) over the shared keys of two dicts."""
    shared = set(a) & set(b)
    if not shared:
        return None
    num = den_a = den_b = 0.0
    max_abs = 0.0
    sq = 0.0
    for k in shared:
        d = a[k] - b[k]
        max_abs = max(max_abs, abs(d))
        sq += d * d
        num += a[k] * b[k]
        den_a += a[k] * a[k]
        den_b += b[k] * b[k]
    rms = math.sqrt(sq / len(shared))
    cos = num / math.sqrt(den_a * den_b) if den_a > 0 and den_b > 0 else float("nan")
    return max_abs, rms, cos, len(shared)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    ours = load(sys.argv[1])
    theirs = load(sys.argv[2])
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

    print(f"ours   : {len(ours)} keys")
    print(f"theirs : {len(theirs)} keys")

    quantities = sorted({k[1] for k in ours} | {k[1] for k in theirs})
    positions = sorted({k[2] for k in ours} | {k[2] for k in theirs})
    layers = sorted({k[0] for k in ours} & {k[0] for k in theirs})
    print(f"layers  : {len(layers)}  quantities: {quantities}  positions: {positions}\n")

    first = {}
    for q in quantities:
        for p in positions:
            print(f"=== {q}, position {p}")
            print(f"  {'layer':>5} {'max|d|':>12} {'rms|d|':>12} {'cos':>10} {'n':>8}  scale")
            for layer in layers:
                a = ours.get((layer, q, p))
                b = theirs.get((layer, q, p))
                if a is None or b is None:
                    continue
                st = stats(a, b)
                if st is None:
                    continue
                max_abs, rms, cos, n = st
                scale = max(
                    (abs(v) for v in list(a.values()) + list(b.values())), default=0.0
                )
                flag = ""
                if q not in first and max_abs > thresh:
                    first[q] = (layer, max_abs, cos, p)
                    flag = "  <== first above threshold"
                print(
                    f"  {layer:>5} {max_abs:>12.4e} {rms:>12.4e} {cos:>10.6f} {n:>8}  {scale:.4g}{flag}"
                )
            print()

    print("=== summary")
    if first:
        for q, (layer, max_abs, cos, p) in sorted(first.items()):
            print(
                f"  {q}: first layer above {thresh:g} is {layer} "
                f"(max|d|={max_abs:.4e}, cos={cos:.6f}, position {p})"
            )
    else:
        print(f"  nothing exceeded the threshold {thresh:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
