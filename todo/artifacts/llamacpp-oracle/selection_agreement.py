"""How often do the expert selections agree with the reference's, per layer?

The hidden-state distance does not predict the logit distance: llama.cpp sits closer to the
device's residual stream than mistral.rs does, and is 34x further away in logits. The
mediating variable is the router's top-k cut, which is a discrete decision worth about a nat
when it flips.

So this compares the per-layer top-k sets of each implementation against the reference's, for
the same golden prompt.

Usage: selection_agreement.py <llamacpp-router.tsv> [our_ids_0.json] [ref_ids_0.json]
"""

import collections
import json
import sys

TOPK = 6


def load_llamacpp(path):
    """-> {layer: {position: [ids in rank order]}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            try:
                layer = int(p[0])
                pos = int(p[1])
                rank = int(p[2])
                eid = int(p[3])
            except ValueError:
                continue
            out[layer][pos].append((rank, eid))
    for layer in out:
        for pos in out[layer]:
            out[layer][pos] = [e for _, e in sorted(out[layer][pos])]
    return out


def load_json_ref(path):
    """-> {layer: {position: [ids]}}"""
    d = json.load(open(path))
    layers = d["topk_ids_per_layer"]
    out = {}
    for layer, flat in enumerate(layers):
        # The reference dump is flat per layer: topk per token, tokens in order.
        per_pos = []
        for i in range(len(flat) // TOPK):
            per_pos.append(list(flat[i * TOPK:(i + 1) * TOPK]))
        out[layer] = {p: v for p, v in enumerate(per_pos)}
    return out


def load_json_ours(path):
    """-> {layer: {position: [ids]}}"""
    d = json.load(open(path))
    layers = d["topk_ids_per_layer"]
    out = {}
    for layer, per_pos in enumerate(layers):
        out[layer] = {p: list(v) for p, v in enumerate(per_pos)}
    return out


def agreement(a, b):
    """-> (mean |intersection| / topk over shared (layer, position), count)"""
    total = 0.0
    n = 0
    for layer in sorted(set(a) & set(b)):
        for pos in sorted(set(a[layer]) & set(b[layer])):
            sa = set(a[layer][pos])
            sb = set(b[layer][pos])
            if not sa or not sb:
                continue
            total += len(sa & sb) / len(sb)
            n += 1
    return (total / n if n else float("nan")), n


def main():
    llama = load_llamacpp(sys.argv[1])
    ref = load_json_ref(sys.argv[3] if len(sys.argv) > 3 else "/tmp/opencode/ref_ids_0.json")
    ours = load_json_ours(sys.argv[2]) if len(sys.argv) > 2 else None

    la, ln = agreement(llama, ref)
    print(f"llamacpp vs reference : {la*100:.2f}% of the top-k matched, over {ln} layer/positions")
    if ours:
        oa, on = agreement(ours, ref)
        print(f"mistral.rs vs reference: {oa*100:.2f}%, over {on} layer/positions")
        ll, _ = agreement(llama, ours)
        print(f"llamacpp vs mistral.rs: {ll*100:.2f}%")

    print(f"\n{'layer':>5} {'llama vs ref':>14} {'ours vs ref':>14} {'llama vs ours':>14}")
    for layer in sorted(set(llama) & set(ref)):
        la, _ = agreement({layer: llama[layer]}, {layer: ref[layer]})
        if ours and layer in ours:
            oa, _ = agreement({layer: ours[layer]}, {layer: ref[layer]})
            lo, _ = agreement({layer: llama[layer]}, {layer: ours[layer]})
            print(f"{layer:>5} {la*100:>13.1f}% {oa*100:>13.1f}% {lo*100:>13.1f}%")
        else:
            print(f"{layer:>5} {la*100:>13.1f}%")


if __name__ == "__main__":
    main()
