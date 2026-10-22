"""Compare the Kolibri-1 oracles position by position, with the lag auto-detected.

The primary's rows were written one position ahead (rows[k+1] holds the distribution for
step k). That offset is detected here rather than assumed, so this works on the old file
and on a repaired one without changes.

Usage:
    python3 compare-kolibri1-oracles.py [primary_rows.json] [llamacpp_rows.json]

Both files are expected in the primary's topk-rows format: prompts[] with
{input_ids, generated_ids, rows:[{gen_index, token, top:[[id,logprob]], logprob_of_token,
top1_top2_margin, top6_top7_margin}]}.
"""

import json
import sys

HERE = "tests/vllm/models/"
PRIMARY = sys.argv[1] if len(sys.argv) > 1 else HERE + "kolibri1_primary_topk_rows.json"
LLAMA = sys.argv[2] if len(sys.argv) > 2 else HERE + "kolibri1_llamacpp_topk_rows.json"


def load(p):
    d = json.load(open(p))
    pr = d["prompts"]
    pr = list(pr.values()) if isinstance(pr, dict) else pr
    out = []
    for e in pr:
        rows = e["rows"]
        if isinstance(rows, dict):
            rows = {int(k): v for k, v in rows.items()}
        else:
            rows = {r["gen_index"]: r for r in rows}
        out.append((e, rows))
    return out


def detect_shift(primary, llama):
    """Which offset makes the primary's chain its own rows' argmax?"""
    best_rate, best_shift = -1.0, 0
    for shift in (0, 1, -1, 2, -2):
        hits = tot = 0
        for (pe, prows), (le, lrows) in zip(primary, llama):
            for k, tok in enumerate(pe["generated_ids"]):
                rk = k + shift
                if rk not in prows or rk not in lrows:
                    continue
                top = prows[rk]["top"]
                if not top:
                    continue
                tot += 1
                if top[0][0] == tok:
                    hits += 1
        rate = hits / tot if tot else 0.0
        print(f"  shift {shift:+d}: {hits}/{tot} = {rate*100:.1f}%")
        if rate > best_rate:
            best_rate, best_shift = rate, shift
    return best_shift, best_rate


def main():
    primary, llama = load(PRIMARY), load(LLAMA)
    shift, rate = detect_shift(primary, llama)
    print(f"primary: {PRIMARY}")
    print(f"llama  : {LLAMA}")
    print(f"detected primary row offset: {shift:+d}  (chain-matches-own-argmax {rate*100:.1f}%)")
    print()

    n = len(primary)
    ft = {"p vs l": 0, "p vs t": 0}
    seq = {"p vs l": 0, "p vs t": 0}
    d_all = []
    margins_dis = []
    for i, ((pe, prows), (le, lrows)) in enumerate(zip(primary, llama)):
        pg, lg = pe["generated_ids"], le["generated_ids"]
        if pg and lg:
            if pg[0] == lg[0]:
                ft["p vs l"] += 1
            if pg == lg:
                seq["p vs l"] += 1
        cp = 0
        for a, b in zip(pg, lg):
            if a != b:
                break
            cp += 1
        for k in range(cp):
            rk, lk = k + shift, k
            if rk not in prows or lk not in lrows:
                continue
            ptop = dict((int(t), float(v)) for t, v in prows[rk]["top"])
            ltop = dict((int(t), float(v)) for t, v in lrows[lk]["top"])
            if not ptop or not ltop:
                continue
            shared = set(ptop) & set(ltop)
            if shared:
                d_all.append(max(abs(ptop[s] - ltop[s]) for s in shared))
            if list(ptop)[0] != list(ltop)[0]:
                margins_dis.append(prows[rk].get("top1_top2_margin"))
    d_all.sort()
    print(f"first token agrees: {ft['p vs l']}/{n}    identical full sequence: {seq['p vs l']}/{n}")
    if d_all:
        print(
            f"max |dlogprob| on shared tokens, common prefix only (n={len(d_all)}): "
            f"median={d_all[len(d_all)//2]:.4f} p90={d_all[int(0.9*len(d_all))]:.4f} max={d_all[-1]:.4f} nats"
        )
    if margins_dis:
        margins_dis.sort()
        print(
            f"argmax disagreements on the common prefix: {len(margins_dis)}, "
            f"primary top1/top2 margin at those positions: "
            f"median={margins_dis[len(margins_dis)//2]:.4f} min={margins_dis[0]:.4f}"
        )
    else:
        print("argmax disagreements on the common prefix: 0")


if __name__ == "__main__":
    main()
