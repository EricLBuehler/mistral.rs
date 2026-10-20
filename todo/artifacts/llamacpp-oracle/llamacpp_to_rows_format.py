"""Convert the llama.cpp run into the primary's topk-rows format.

Same field names, same margin definitions, same row shape, so the two files can be diffed
position by position with one piece of code once the primary's writer off-by-one is fixed.

One deliberate difference: llama.cpp's rows are NOT lagged. Row k is the distribution for
step k, and its `token` is the token that step actually generated, so
`rows[k].token == rows[k].top[0]` holds for every row. That property is what the primary's
lagged file fails, and it is the cheapest way to tell the two apart.
"""

import json
import sys

SRC = "/tmp/opencode/kolibri1_goldens_llamacpp.json"
OUT_ROWS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/kolibri1_llamacpp_topk_rows.json"
OUT_GOLD = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/kolibri1_goldens_llamacpp.json"

d = json.load(open(SRC))

rows_out = {
    "model": d["model"],
    "topk": 20,
    "max_new_tokens": d["max_new_tokens"],
    "provenance": d["provenance"],
    "prompts": [],
}
gold_out = {
    "model": d["model"],
    "max_new_tokens": d["max_new_tokens"],
    "decode": d["decode"],
    "provenance": d["provenance"],
    "prompts": [],
}

for e in d["prompts"]:
    pos = e["per_position_logprobs"]
    rows = []
    for k, p in enumerate(pos):
        top = [[int(t), float(v)] for t, v in p["top"]]
        rows.append(
            {
                "gen_index": k,
                "token": int(p["id"]),
                "top": top,
                "logprob_of_token": float(p["logprob"]),
                "top1_top2_margin": top[0][1] - top[1][1],
                "top6_top7_margin": top[5][1] - top[6][1],
            }
        )
    rows_out["prompts"].append(
        {
            "prompt": e["prompt"],
            "input_ids": e["input_ids"],
            "generated_ids": e["generated_ids"],
            "rows": rows,
        }
    )
    gold = {
        "prompt": e["prompt"],
        "input_ids": e["input_ids"],
        "generated_ids": e["generated_ids"],
    }
    if "final_topk" in e:
        gold["final_topk"] = e["final_topk"]
    gold_out["prompts"].append(gold)

json.dump(rows_out, open(OUT_ROWS, "w"), indent=1)
json.dump(gold_out, open(OUT_GOLD, "w"), indent=1)

# self-check: the unlagged invariant
bad = 0
for e in rows_out["prompts"]:
    for r in e["rows"]:
        if r["token"] != r["top"][0][0]:
            bad += 1
print(f"wrote {OUT_ROWS} and {OUT_GOLD}")
print(f"rows where token != top1: {bad}  (expected 0 - llama.cpp rows are unlagged)")
print(f"rows per prompt: {[len(e['rows']) for e in rows_out['prompts']]}")
