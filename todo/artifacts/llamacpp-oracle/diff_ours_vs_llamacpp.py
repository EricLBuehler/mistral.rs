"""Diff our mistral.rs dump against the llama.cpp one, with the quantity names mapped.

The two implementations expose the same quantities under different names:

  ours   l_out(L)        = the block output at layer L
  theirs residual_in(L+1) = the residual entering layer L+1, which is the block output at L
  ours   residual_in(L)  = the residual entering layer L
  theirs residual_in(L)  = the same thing

So ours.l_out(L) is theirs.residual_in(L+1) for L in 0..n_layer-1, and ours.l_out(n_layer-1)
is theirs.l_out(n_layer) - the final block output, which llama.cpp exposes separately.

This is the CPU-to-CPU comparison, so it separates implementations rather than devices.
"""

import collections
import math
import sys

OURS = "/tmp/opencode/kolibri1_layer_taps.tsv"
THEIRS = "/tmp/opencode/kolibri1-llamacpp-layerdump.tsv"


def load(path):
    rows = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                layer = int(parts[0])
                value = float(parts[4])
            except ValueError:
                continue
            rows[(layer, parts[1], int(parts[2]))][int(parts[3])] = value
    return rows


def ulp(x):
    if x == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)  # bf16: 8 mantissa bits


def main():
    ours = load(OURS)
    theirs = load(THEIRS)

    n_layer = max(k[0] for k in ours) + 1
    print(f"ours: {len(ours)} keys, theirs: {len(theirs)} keys, n_layer={n_layer}\n")

    # map ours -> theirs
    pairs = []
    for L in range(n_layer):
        # ours.l_out(L) is theirs.residual_in(L+1): the residual entering the next layer is
        # the previous block's output.
        if L + 1 <= n_layer:
            pairs.append((L, "l_out", "residual_in", L + 1, "residual_in"))
        # ours.residual_in(L) is the ffn_inp of the previous layer, NOT the layer's input
        # stream: our implementation carries the residual in two parts and the layer's true
        # input is xs + res. It therefore has no counterpart in the llama.cpp dump and is
        # excluded from the comparison.
        pairs.append((L, "residual_in", "SKIP", 0, "SKIP"))

    print(f"{'layer':>5} {'quantity':>14} {'pos':>4} {'max|d|':>12} {'rms|d|':>12} {'cos':>10} {'>2ulp':>8}")
    first = {}
    for (ol, oq, tq, tl, tq2) in pairs:
        for pos in (0, 5):
            if tq2 == "SKIP":
                continue
            a = ours.get((ol, oq, pos))
            b = theirs.get((tl, tq2, pos))
            if not a or not b:
                continue
            shared = set(a) & set(b)
            if not shared:
                continue
            num = den_a = den_b = 0.0
            max_abs = 0.0
            sq = 0.0
            over2 = 0
            for k in shared:
                d = a[k] - b[k]
                max_abs = max(max_abs, abs(d))
                sq += d * d
                num += a[k] * b[k]
                den_a += a[k] * a[k]
                den_b += b[k] * b[k]
                if abs(d) > 2 * ulp(max(abs(a[k]), abs(b[k]))):
                    over2 += 1
            rms = math.sqrt(sq / len(shared))
            cos = num / math.sqrt(den_a * den_b) if den_a > 0 and den_b > 0 else float("nan")
            label = f"{oq}"
            print(
                f"{ol:>5} {label:>14} {pos:>4} {max_abs:>12.4e} {rms:>12.4e} {cos:>10.6f} "
                f"{100.0*over2/len(shared):>7.2f}%"
            )
            key = oq
            if key not in first or max_abs > first[key][0]:
                first[key] = (max_abs, ol, pos, cos, 100.0 * over2 / len(shared))

    print()
    for q, (mx, L, pos, cos, o2) in sorted(first.items()):
        print(f"  {q}: worst is layer {L} position {pos}: max|d|={mx:.4e} cos={cos:.6f} (>2ulp: {o2:.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
