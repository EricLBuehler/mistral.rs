"""The expert-access distribution, from a real 1000-step decode trace.

The 32 GB plan is: keep the 4.85 GB of non-expert weights resident and cache as many of the
19200 (layer, expert) slots as fit in the rest. Per token the model touches 300 distinct
slots - 6 experts in each of 50 layers - so the hit rate is the whole design.

This uses the decode portion of a 1000-step chained generation: 1000 forwards x 50 layers x
1 position x 6 experts = 300000 selections, in the order the model actually touches them.
That order matters - a cache is not a bag of hot experts, it is a sequence.
"""

import collections
import json
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/our_chain_long.json"
TOPK = 6
N_LAYERS = 50

d = json.load(open(SRC))
tl = d["topk_ids_per_layer"]
# The first N_LAYERS entries are the prefill; the rest are one position per (forward, layer).
decode = tl[N_LAYERS:]
n_forwards = len(decode) // N_LAYERS
print(f"source: {SRC}")
print(f"  {n_forwards} decode forwards x {N_LAYERS} layers x {TOPK} experts "
      f"= {n_forwards*N_LAYERS*TOPK} selections")
print()

# --- per-layer candidate pool ---
counts = [collections.Counter() for _ in range(N_LAYERS)]
for f in range(n_forwards):
    for L in range(N_LAYERS):
        for e in decode[f * N_LAYERS + L][0]:
            counts[L][e] += 1

print("per-layer candidate pool over the trace:")
print(f"  {'layer':>5} {'distinct':>9} {'top-6':>8} {'top-20':>8} {'top-50':>8}  {'for 90%':>8}")
need90 = []
for L in range(N_LAYERS):
    c = counts[L]
    tot = sum(c.values())
    s = sorted(c.values(), reverse=True)
    share = lambda k: sum(s[:k]) / tot
    k90 = next((k for k in range(1, tot + 1) if share(k) >= 0.90), tot)
    need90.append(k90)
    if L < 5 or L in (10, 25, 40, 49):
        print(f"  {L:>5} {len(c):>9} {share(6):>7.1%} {share(20):>7.1%} {share(50):>7.1%}  {k90:>8}")
n90 = sorted(need90)
med = n90[len(n90) // 2]
print()
print(f"  experts per layer needed for 90% of that layer's selections:")
print(f"    min {n90[0]}, median {med}, max {n90[-1]}  (of {384} experts per layer)")
print(f"    a per-layer cache of the top-{med} is {med*N_LAYERS} slots "
      f"= {med*N_LAYERS*7.5/1024:.1f} GB bf16, and covers every expert the layer")
print(f"    ever selected in this trace for those layers")

# --- the access order, and what a cache does with it ---
seq = []
for f in range(n_forwards):
    for L in range(N_LAYERS):
        for e in decode[f * N_LAYERS + L][0]:
            seq.append((L, e))
print()
print(f"access sequence: {len(seq)} slot touches over {n_forwards} decode steps")

print()
print("flat LRU over all (layer, expert) slots, on the real order:")
print(f"  {'slots':>7} {'GB bf16':>9} {'GB fp8':>8} {'hit rate':>9}")
for cache in (300, 1000, 2000, 2887, 4000, 5775):
    res = collections.OrderedDict()
    hits = 0
    for s in seq:
        if s in res:
            hits += 1
            res.move_to_end(s)
        else:
            res[s] = None
            if len(res) > cache:
                res.popitem(last=False)
    print(f"  {cache:>7} {cache*7.5/1024:>9.1f} {cache*3.75/1024:>8.1f} {100.0*hits/len(seq):>8.1f}%")

print()
print("per-layer cache (each layer keeps its own hot set), on the real order:")
for per_layer in (8, 16, 24, 32, 48):
    res = [collections.OrderedDict() for _ in range(N_LAYERS)]
    hits = 0
    for (L, e) in seq:
        r = res[L]
        if e in r:
            hits += 1
            r.move_to_end(e)
        else:
            r[e] = None
            if len(r) > per_layer:
                r.popitem(last=False)
    slots = per_layer * N_LAYERS
    print(
        f"  {per_layer:>3}/layer -> {slots:>6} slots ({slots*7.5/1024:>5.1f} GB bf16, "
        f"{slots*3.75/1024:>4.1f} GB fp8)  hit rate {100.0*hits/len(seq):>5.1f}%"
    )

print()
print("static residency (each layer keeps its top-k experts, no eviction):")
for k in (8, 16, 24, 32, 48):
    resident = [set(e for e, _ in counts[L].most_common(k)) for L in range(N_LAYERS)]
    hits = sum(1 for (L, e) in seq if e in resident[L])
    slots = k * N_LAYERS
    print(
        f"  top-{k:>3}/layer -> {slots:>6} slots ({slots*7.5/1024:>5.1f} GB bf16, "
        f"{slots*3.75/1024:>4.1f} GB fp8)  hit rate {100.0*hits/len(seq):>5.1f}%"
    )
