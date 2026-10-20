"""Test whether the primary's logprob rows are offset by one position relative to its chain.

The primary's chain and the primary's own top-100 disagree at 139 of 202 positions, with
gaps up to 11 nats - far too large for a rounding difference, and far too consistent to be
random sampling. The pattern in prompt 1 (chain alternates 948/35423 while the rows
alternate the other way) says the rows are labelled one step off.

Then, with the offset resolved, compare the primary's distributions against llama.cpp's,
which is the comparison the whole logit-gap investigation was trying to make.
"""

import json

ROWS = "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_primary_topk_rows.json"
LLAMA = "/tmp/opencode/kolibri1_goldens_llamacpp.json"

r = json.load(open(ROWS))
l = json.load(open(LLAMA))
pr = r["prompts"]
if isinstance(pr, dict):
    pr = list(pr.values())
lp = l["prompts"]


def rows_of(e):
    rows = e["rows"]
    if isinstance(rows, dict):
        return {int(k): v for k, v in rows.items()}
    return {row["gen_index"]: row for row in rows}


print("=== does a one-position offset explain it? ===")
for shift in (0, 1, -1, 2):
    hits = tot = 0
    for e in pr:
        rr = rows_of(e)
        gen = e["generated_ids"]
        for k in range(len(gen)):
            rk = k + shift
            if rk not in rr:
                continue
            top = rr[rk]["top"]
            if not top:
                continue
            tot += 1
            if top[0][0] == gen[k]:
                hits += 1
    label = f"generated[k] == rows[k{shift:+d}].top1"
    print(f"  shift {shift:+d}: {hits}/{tot}  ({100.0*hits/max(tot,1):.1f}%)   {label}")

print()
print("=== with the offset resolved: primary vs llama.cpp, per position ===")
SHIFT = 1
tot = hits = 0
lp_diffs = []
for i, e in enumerate(pr):
    rr = rows_of(e)
    lpos = lp[i]["per_position_logprobs"]
    for k in range(len(lpos)):
        rk = k + SHIFT
        if rk not in rr:
            continue
        ptop = rr[rk]["top"]
        ltop = [(t, v) for t, v in lpos[k]["top"]]
        if not ptop or not ltop:
            continue
        tot += 1
        if ptop[0][0] == ltop[0][0]:
            hits += 1
        # logprob of the shared tokens
        pl = dict(ptop)
        ll = dict(ltop)
        for t in set(pl) & set(ll):
            if pl[t] is not None and ll[t] is not None:
                lp_diffs.append(abs(pl[t] - ll[t]))
    if i < 3:
        print(f"  prompt {i}: primary first-choice chain (offset applied):", end=" ")
        print([rr[k + SHIFT]["top"][0][0] for k in range(min(6, len(e["generated_ids"]))) if (k + SHIFT) in rr])

print()
print(f"  same argmax: {hits}/{tot} ({100.0*hits/max(tot,1):.1f}%)")
if lp_diffs:
    lp_diffs.sort()
    print(
        f"  |logprob difference| on shared tokens: n={len(lp_diffs)} "
        f"median={lp_diffs[len(lp_diffs)//2]:.4f} p90={lp_diffs[int(0.9*len(lp_diffs))]:.4f} max={lp_diffs[-1]:.4f}"
    )
