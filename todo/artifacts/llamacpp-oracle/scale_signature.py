"""Is the difference a per-row constant (the scale signature) or a per-element scatter?

peer-Ce's method, applied to our CPU dump against the device dump: for each row, fit the
constant k that best explains the difference, remove it, and see what is left. If removing k
leaves little, the difference is the scale; if it leaves as much, the difference is
per-element.

The fit is k = sum(ours) / sum(theirs) per row, which is what peer-Ce used.
"""

import collections
import math
import sys


def load(path):
    rows = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            try:
                layer = int(p[0])
                pos = int(p[2])
                hid = int(p[3])
                val = float(p[4])
            except ValueError:
                continue
            rows[(layer, p[1], pos)][hid] = val
    return rows


def report(ours, theirs, layer, quantity, pos):
    a = ours.get((layer, quantity, pos))
    b = theirs.get((layer, quantity, pos))
    if not a or not b:
        return None
    shared = sorted(set(a) & set(b))
    if len(shared) < 8:
        return None

    av = [a[i] for i in shared]
    bv = [b[i] for i in shared]

    # The scale signature: one constant per row, fitted by the sums.
    k = sum(av) / sum(bv) if sum(bv) != 0 else 1.0
    raw = [abs(x - y) for x, y in zip(av, bv)]
    after = [abs(x - k * y) for x, y in zip(av, bv)]

    scale = max(abs(v) for v in bv) or 1.0
    # The per-element alternative: a single global constant, which is what a pure scale
    # error would look like if it were one constant for the whole vector.
    return {
        "n": len(shared),
        "scale": scale,
        "k": k,
        "raw_max": max(raw),
        "raw_rms": math.sqrt(sum(x * x for x in raw) / len(raw)),
        "after_max": max(after),
        "after_rms": math.sqrt(sum(x * x for x in after) / len(after)),
        "rel_raw_rms": math.sqrt(sum(x * x for x in raw) / len(raw)) / scale,
        "rel_after_rms": math.sqrt(sum(x * x for x in after) / len(after)) / scale,
    }


def main():
    ours = load(sys.argv[1])
    theirs = load(sys.argv[2])
    quantities = sys.argv[3].split(",") if len(sys.argv) > 3 else ["attn_out_pre_norm", "attn_post_norm", "ffn_post_norm", "l_out"]
    layers = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1, 10, 25]
    pos = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    for quantity in quantities:
        print(f"=== {quantity}, position {pos}")
        print(f"  {'layer':>5} {'k':>12} {'raw rms':>12} {'after k':>12} {'raw rel':>12} {'after rel':>12}  verdict")
        for layer in layers:
            r = report(ours, theirs, layer, quantity, pos)
            if r is None:
                continue
            verdict = "SCALE" if r["rel_after_rms"] < 0.25 * r["rel_raw_rms"] else "scatter"
            print(
                f"  {layer:>5} {r['k']:>12.9f} {r['raw_rms']:>12.4e} {r['after_rms']:>12.4e} "
                f"{r['rel_raw_rms']:>12.4e} {r['rel_after_rms']:>12.4e}  {verdict}"
            )
        print()


if __name__ == "__main__":
    main()
