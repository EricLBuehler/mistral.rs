"""The real comparison: how far apart are the primary and llama.cpp, before the chains diverge?

Comparing distributions after a divergence is meaningless - once the chains part, every
later position is conditioned on a different context. So this walks each prompt along the
common prefix only, and asks two questions at each position:

  1. How far apart are the two distributions, in nats?
  2. When they disagree on the argmax, how big was the winner's margin?

A disagreement at a position whose margin is 0.002 nats is a coin flip, not a defect.
"""

import json

ROWS = "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_primary_topk_rows.json"
LLAMA = "/tmp/opencode/kolibri1_goldens_llamacpp.json"
TRANS = "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens.json"

r = json.load(open(ROWS))
l = json.load(open(LLAMA))
t = json.load(open(TRANS))
pr = r["prompts"]
if isinstance(pr, dict):
    pr = list(pr.values())
lp = l["prompts"]
tp = t["prompts"]
if isinstance(tp, dict):
    tp = list(tp.values())


def rows_of(e):
    rows = e["rows"]
    if isinstance(rows, dict):
        return {int(k): v for k, v in rows.items()}
    return {row["gen_index"]: row for row in rows}


SHIFT = 1
print(f"{'prompt':<40} {'pos':>3} {'primary':>9} {'llama':>9} {'same':>5} {'dlogprob(shared)':>17} {'margin':>8}")
all_d = []
margins_at_disagree = []
margins_at_agree = []
for i, e in enumerate(pr):
    rr = rows_of(e)
    gen_p = e["generated_ids"]
    gen_l = lp[i]["generated_ids"]
    # common prefix length of the two chains
    cp = 0
    for a, b in zip(gen_p, gen_l):
        if a != b:
            break
        cp += 1
    for k in range(cp):
        rk = k + SHIFT
        if rk not in rr:
            continue
        ptop = dict(rr[rk]["top"])
        lrow = lp[i]["per_position_logprobs"][k]
        ltop = dict((t_, v) for t_, v in lrow["top"])
        if not ptop or not ltop:
            continue
        pb, lb = ptop and list(ptop.items())[0], list(ltop.items())[0]
        shared = set(ptop) & set(ltop)
        d = max(abs(ptop[s] - ltop[s]) for s in shared) if shared else float("nan")
        all_d.append(d)
        marg = rr[rk]["top1_top2_margin"]
        same = pb[0] == lb[0]
        (margins_at_agree if same else margins_at_disagree).append(marg)
        if not same or k < 2:
            print(
                f"{e['prompt'][:38]:<40} {k:>3} {str(pb[0]):>9} {str(lb[0]):>9} "
                f"{'yes' if same else 'NO':>5} {d:>17.4f} {marg:>8.4f}"
            )

print()
all_d.sort()
print(f"positions compared (common prefix only): {len(all_d)}")
print(
    f"max |dlogprob| on shared tokens: median={all_d[len(all_d)//2]:.4f} "
    f"p90={all_d[int(0.9*len(all_d))]:.4f} max={all_d[-1]:.4f} nats"
)
print()
print("where the two agree on the argmax, the primary's own top1/top2 margin:")
margins_at_agree.sort()
print(
    f"  n={len(margins_at_agree)} median={margins_at_agree[len(margins_at_agree)//2]:.4f} "
    f"min={margins_at_agree[0]:.4f} max={margins_at_agree[-1]:.4f}"
)
print()
print("where they disagree on the argmax:")
margins_at_disagree.sort()
if margins_at_disagree:
    print(
        f"  n={len(margins_at_disagree)} median={margins_at_disagree[len(margins_at_disagree)//2]:.4f} "
        f"min={margins_at_disagree[0]:.4f} max={margins_at_disagree[-1]:.4f}"
    )
    print(f"  of those, margin < 0.01 nats: {sum(1 for m in margins_at_disagree if m < 0.01)}")
    print(f"  of those, margin < 0.10 nats: {sum(1 for m in margins_at_disagree if m < 0.10)}")
    print(f"  of those, margin < 0.50 nats: {sum(1 for m in margins_at_disagree if m < 0.50)}")
else:
    print("  none")
