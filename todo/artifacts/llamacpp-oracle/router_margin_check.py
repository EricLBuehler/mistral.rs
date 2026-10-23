"""Is the expert-selection disagreement a margin effect?

The router's top-k is decided on differences that todo/07 measured as ~0.03 - below bf16
precision. If that is what is happening, the rows where our expert set differs from the
reference's must be exactly the rows where the 6th and 7th selected experts are close
together, and the rows where they agree must be the ones with a comfortable margin.

The reference dump carries `layer0_scores` (the router logits and the bias), so the margin
can be computed from the reference's own numbers rather than assumed.
"""

import json

OURS = "/tmp/opencode/our_ids_0.json"
REF = "/tmp/opencode/ref_ids_0.json"

o = json.load(open(OURS))
r = json.load(open(REF))
print(f"ours: {o['prompt']!r}  n_layers={o['n_layers']}")
print(f"ref : {r['prompt']!r}  n_layers={r['n_layers']}")
print(f"same input_ids: {o['input_ids'] == r['input_ids']}")

TOPK = 6


def reshape(layers, flat):
    """The reference dump is flat per layer; ours is [token][id]."""
    if flat:
        return [
            [layers[L][i * TOPK : (i + 1) * TOPK] for i in range(len(layers[L]) // TOPK)]
            for L in range(len(layers))
        ]
    return layers


o_layers = reshape(o["topk_ids_per_layer"], False)
r_layers = reshape(r["topk_ids_per_layer"], True)
n = min(len(o_layers), len(r_layers))

# per-layer: do the two select the same expert set?
same_sets = 0
same_order = 0
per_layer = []
for L in range(n):
    a, b = o_layers[L], r_layers[L]
    # both are [token][top_k] lists
    tok_same_set = tok_same_order = 0
    total = 0
    for ta, tb in zip(a, b):
        total += 1
        if set(ta) == set(tb):
            tok_same_set += 1
        if ta == tb:
            tok_same_order += 1
    if tok_same_set == total and total:
        same_sets += 1
    if tok_same_order == total and total:
        same_order += 1
    per_layer.append((L, tok_same_set, tok_same_order, total))

print()
print(f"layers where every token selected the identical expert set: {same_sets}/{n}")
print(f"layers where every token selected the identical experts in order: {same_order}/{n}")
print()
print("  layer  same_set  same_order  of")
for L, ss, so, tot in per_layer[:12]:
    print(f"  {L:>5}  {ss:>8}  {so:>10}  {tot}")
print("  ...")
for L, ss, so, tot in per_layer[-4:]:
    print(f"  {L:>5}  {ss:>8}  {so:>10}  {tot}")

# layer 0 margins from the reference's own scores
ls = r.get("layer0_scores")
if not ls:
    print("\nno layer0_scores in the reference dump")
    raise SystemExit
if isinstance(ls, list):
    ls = ls[0]
logits_flat, bias = ls["logits"], ls["bias"]
NEXP = len(bias)
logits = [logits_flat[i * NEXP : (i + 1) * NEXP] for i in range(len(logits_flat) // NEXP)]
print()
print(f"layer0_scores: {len(logits)} positions, {NEXP} experts, bias {len(bias)}")

rows = []
for t, lg in enumerate(logits):
    bs = bias
    sel = sorted(((lg[e] + bs[e]), e) for e in range(len(lg)))
    sel.sort(key=lambda x: -x[0])
    # the selection is over (logit + bias); the 6th and 7th are the top-k boundary
    top6 = sel[5][0]
    top7 = sel[6][0]
    margin = top6 - top7
    ours_set = set(o_layers[0][t])
    ref_set = set(r_layers[0][t])
    rows.append((t, margin, ours_set == ref_set, sorted(ours_set), sorted(ref_set)))

rows.sort(key=lambda x: x[1])
print()
print("  pos  top6-top7 margin  same_expert_set")
for t, m, same, oo, rr in rows:
    print(f"  {t:>3}  {m:>17.6f}  {str(same):>5}   ours={oo} ref={rr}")

agree = [m for _, m, s, _, _ in rows if s]
dis = [m for _, m, s, _, _ in rows if not s]
print()
if agree and dis:
    print(f"rows where the expert set agrees:   {len(agree)}  margin min={min(agree):.6f} median={sorted(agree)[len(agree)//2]:.6f} max={max(agree):.6f}")
    print(f"rows where it disagrees:            {len(dis)}  margin min={min(dis):.6f} median={sorted(dis)[len(dis)//2]:.6f} max={max(dis):.6f}")
    print()
    if max(agree) < min(dis):
        print("SEPARATION: every agreeing row has a smaller margin than every disagreeing row.")
    else:
        print("overlap: the margins do not separate the two groups.")
else:
    print(f"agree={len(agree)} disagree={len(dis)}")
