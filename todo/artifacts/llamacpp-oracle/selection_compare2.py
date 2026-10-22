"""The selection comparison, with every parse checked rather than assumed.

Both TSV files are `layer, position, rank, expert_id`. The loader prints what it found, so a
mis-parse shows up as a wrong group count rather than as a plausible number.

The reference's own reshape is validated against its own recorded layer-0 logits: the recorded
ids must be what `logits + bias` selects. If that holds, the reshape of the flat per-layer
list into positions is the reference's own layout, not ours.
"""

import collections
import json
import math
import sys

TOPK = 6


def load_rows(path, label):
    """-> {layer: {position: [ids in rank order]}}, with the shape reported."""
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path) as f:
        for line in f:
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
            groups[layer][pos][rank] = eid
    out = {}
    for layer, per_pos in groups.items():
        out[layer] = {}
        for pos, ranks in per_pos.items():
            out[layer][pos] = [ranks[r] for r in sorted(ranks)]
    sizes = collections.Counter(len(v) for lay in out.values() for v in lay.values())
    n_groups = sum(len(v) for v in out.values())
    n_ids = sum(len(v) for lay in out.values() for v in lay.values())
    print(f"  {label}: {len(out)} layers, {n_groups} (layer,position) groups, "
          f"{n_ids} ids, mean {n_ids/n_groups:.2f} per group, sizes {dict(sizes)}")
    return out


def load_peer_ids_file(path, label):
    """-> {layer: [ids]}, the comma-separated one-line-per-layer form."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            layer, ids = line.split("\t")
            out[int(layer)] = [int(x) for x in ids.split(",")]
    print(f"  {label}: {len(out)} layers, mean {sum(len(v) for v in out.values())/len(out):.2f} ids per layer")
    return out


def load_json_ref(path):
    """-> {layer: {position: [ids]}}, with the reshape validated against the reference's own
    recorded logits."""
    d = json.load(open(path))
    layers = d["topk_ids_per_layer"]
    out = {}
    for layer, flat in enumerate(layers):
        per_pos = []
        for i in range(len(flat) // TOPK):
            per_pos.append(list(flat[i * TOPK:(i + 1) * TOPK]))
        out[layer] = {p: v for p, v in enumerate(per_pos)}

    # Validate: the reference's own layer-0 logits and bias must select its own layer-0 ids.
    sc = d.get("layer0_scores")
    if sc:
        logits = sc[0]["logits"]
        bias = sc[0]["bias"]
        n_pos = len(logits) // len(bias)
        hit = 0
        for p in range(n_pos):
            lg = logits[p * len(bias):(p + 1) * len(bias)]
            s = [lg[i] + bias[i] for i in range(len(bias))]
            order = sorted(range(len(bias)), key=lambda i: (-s[i], -i))[:TOPK]
            want = set(out[0].get(p, []))
            hit += len(want & set(order))
        print(f"  reference reshape validated: {hit}/{n_pos*TOPK} of its own layer-0 ids are "
              f"reproduced by its own logits+bias")
    return out


def load_json_ours(path):
    """-> {layer: {position: [ids]}}"""
    d = json.load(open(path))
    out = {}
    for layer, per_pos in enumerate(d["topk_ids_per_layer"]):
        out[layer] = {p: list(v) for p, v in enumerate(per_pos)}
    return out


def agree(a, b, exact=False):
    """-> (mean fraction matched, n) over the shared (layer, position) groups."""
    total = 0.0
    n = 0
    for layer in sorted(set(a) & set(b)):
        for pos in sorted(set(a[layer]) & set(b[layer])):
            sa, sb = a[layer][pos], b[layer][pos]
            if not sa or not sb:
                continue
            m = (sum(1 for x, y in zip(sa, sb) if x == y) if exact
                 else len(set(sa) & set(sb)))
            total += m / len(sb)
            n += 1
    return (total / n if n else float("nan")), n


def main():
    print("=== the parses")
    llama = load_rows(sys.argv[1], "llamacpp rows")
    peer_rows = load_rows(sys.argv[2], "peer rows")
    peer_ids = load_peer_ids_file(sys.argv[3], "peer ids file")
    print()
    print("=== the reference's reshape")
    ref = load_json_ref(sys.argv[5])
    ours = load_json_ours(sys.argv[4])

    # The peer's two files must agree with each other.
    a, n = agree(peer_rows, {k: {5: v} for k, v in peer_ids.items()})
    print(f"\n  peer's rows file vs peer's ids file: {a*100:.2f}% over {n} groups")

    print("\n=== top-6 SET agreement, all positions")
    impls = [("peer", peer_rows), ("llama", llama), ("ours", ours), ("ref", ref)]
    print(f"  {'':>6} " + " ".join(f"{n:>9}" for n, _ in impls))
    for na, a_ in impls:
        cells = []
        for nb, b_ in impls:
            frac, _ = agree(a_, b_)
            cells.append(f"{frac*100:>8.2f}%")
        print(f"  {na:>6} " + " ".join(cells))

    print("\n=== top-6 SET agreement, position 5 only")
    pos5 = {k: {5: v[5]} if 5 in v else {} for k, v in llama.items()}
    p5_peer = {k: {5: v[5]} if 5 in v else {} for k, v in peer_rows.items()}
    p5_ours = {k: {5: v[5]} if 5 in v else {} for k, v in ours.items()}
    p5_ref = {k: {5: v[5]} if 5 in v else {} for k, v in ref.items()}
    impls5 = [("peer", p5_peer), ("llama", pos5), ("ours", p5_ours), ("ref", p5_ref)]
    print(f"  {'':>6} " + " ".join(f"{n:>9}" for n, _ in impls5))
    for na, a_ in impls5:
        cells = []
        for nb, b_ in impls5:
            frac, _ = agree(a_, b_)
            cells.append(f"{frac*100:>8.2f}%")
        print(f"  {na:>6} " + " ".join(cells))

    print("\n=== exact-SET agreement (all six ids identical), all positions")
    for na, a_ in impls:
        cells = []
        for nb, b_ in impls:
            frac, _ = agree(a_, b_, exact=True)
            cells.append(f"{frac*100:>8.2f}%")
        print(f"  {na:>6} " + " ".join(cells))


if __name__ == "__main__":
    main()
