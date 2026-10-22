"""The device joins the comparison: a fourth implementation, not ground truth.

Two questions:
  1. Does the device track the transcription, or the CPU path? That decides whether the
     transcription is a valid proxy for it.
  2. The taps: how far is mistral.rs from the device, per quantity and per layer, next to
     llama.cpp and the peer's CPU path.

The device's l_out is assembled the same way the CPU path assembles it, so the three dumps
are comparable quantity by quantity.
"""

import collections
import json
import math
import sys

TOPK = 6


def load_rows(path):
    """-> {layer: {position: [ids in rank order]}}"""
    g = collections.defaultdict(lambda: collections.defaultdict(dict))
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 4:
            continue
        try:
            layer, pos, rank, eid = int(p[0]), int(p[1]), int(p[2]), int(p[3])
        except ValueError:
            continue
        g[layer][pos][rank] = eid
    return {l: {p: [d[r] for r in sorted(d)] for p, d in pp.items()} for l, pp in g.items()}


def load_peer_ids(path):
    """-> {layer: [ids]}, the comma-separated one-line-per-layer form."""
    out = {}
    for line in open(path):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        layer, ids = line.split("\t")
        out[int(layer)] = [int(x) for x in ids.split(",")]
    return out


def load_json_ref(path):
    d = json.load(open(path))
    return {l: {i: list(f[i * TOPK:(i + 1) * TOPK]) for i in range(len(f) // TOPK)}
            for l, f in enumerate(d["topk_ids_per_layer"])}


def load_json_ours(path):
    d = json.load(open(path))
    return {l: {p: list(v) for p, v in enumerate(pp)} for l, pp in enumerate(d["topk_ids_per_layer"])}


def agree(a, b, exact_set=False):
    tot = 0.0
    n = 0
    for l in sorted(set(a) & set(b)):
        for p in sorted(set(a[l]) & set(b[l])):
            sa, sb = a[l][p], b[l][p]
            if not sa or not sb:
                continue
            if exact_set:
                m = 1.0 if set(sa) == set(sb) else 0.0
            else:
                m = len(set(sa) & set(sb)) / len(sb)
            tot += m
            n += 1
    return (tot / n if n else float("nan")), n


def load_taps(path):
    rows = collections.defaultdict(dict)
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        try:
            layer, qty, pos, hid, val = int(p[0]), p[1], int(p[2]), int(p[3]), float(p[4])
        except ValueError:
            continue
        rows[(layer, qty, pos)][hid] = val
    return rows


def rms(v):
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else 0.0


def rel(a, b):
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan"), 0
    av = [a[i] for i in shared]
    bv = [b[i] for i in shared]
    d = [x - y for x, y in zip(av, bv)]
    r = rms(bv) or 1.0
    return rms(d) / r, len(shared)


def main():
    tt_ids = load_peer_ids(sys.argv[1])            # layer, comma-separated
    tt_rows = load_rows(sys.argv[2]) if len(sys.argv) > 2 else None
    peer_rows = load_rows(sys.argv[3])
    llama_rows = load_rows(sys.argv[4])
    ours_json = load_json_ours(sys.argv[5])
    ref_json = load_json_ref(sys.argv[6])

    tt5 = {k: {5: v} for k, v in tt_ids.items()}
    peer5 = {k: {5: v[5]} for k, v in peer_rows.items() if 5 in v}
    llama5 = {k: {5: v[5]} for k, v in llama_rows.items() if 5 in v}
    ours5 = {k: {5: v[5]} for k, v in ours_json.items() if 5 in v}
    ref5 = {k: {5: v[5]} for k, v in ref_json.items() if 5 in v}

    impls = [("device", tt5), ("peerCPU", peer5), ("llama", llama5), ("ours", ours5), ("ref", ref5)]
    print("=== top-6 SET agreement at position 5 (mean |intersection|/6)")
    print(f"  {'':>9} " + " ".join(f"{n:>9}" for n, _ in impls))
    for na, a in impls:
        print(f"  {na:>9} " + " ".join(f"{agree(a,b)[0]*100:>8.2f}%" for _, b in impls))
    print("\n=== exact-SET agreement at position 5 (the six ids identical, order-independent)")
    for na, a in impls:
        print(f"  {na:>9} " + " ".join(f"{agree(a,b,True)[0]*100:>8.2f}%" for _, b in impls))

    # The taps: ours against each reference, per quantity, at a few layers.
    tt_taps = load_taps(sys.argv[7])
    peer_taps = load_taps(sys.argv[8])
    llama_taps = load_taps(sys.argv[9])
    our_taps = load_taps(sys.argv[10])

    print("\n=== taps: relative rms of the difference against ours, per quantity")
    for qty in ("attn_out_pre_norm", "attn_post_norm", "ffn_post_norm", "l_out"):
        print(f"  {qty}")
        for layer in (0, 6, 12, 25, 49):
            k = (layer, qty, 0)
            if k not in our_taps:
                continue
            cells = []
            for name, d in (("device", tt_taps), ("peerCPU", peer_taps), ("llama", llama_taps)):
                if k in d:
                    r, n = rel(our_taps[k], d[k])
                    cells.append(f"{name} {r:.3e}")
                else:
                    cells.append(f"{name} -")
            print(f"    layer {layer:>2}: " + "   ".join(cells))


if __name__ == "__main__":
    main()
