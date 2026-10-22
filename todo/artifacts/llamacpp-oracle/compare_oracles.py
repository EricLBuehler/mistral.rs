"""Compare the three Kolibri-1 implementations properly.

The goldens store token ids; llama.cpp's OpenAI response gives decoded strings. Token 121598
is "epsilon" and 116026 is "zeta", so comparing ids to strings silently produces nonsense.
Everything is decoded to text here, using the checkpoint's own tokenizer, so the comparison
is apples to apples.
"""

import json
import sys

from transformers import AutoTokenizer

TRANS = "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens.json"
PRIMARY = "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens_primary.json"
LLAMA = "/tmp/opencode/kolibri1_goldens_llamacpp.json"
TOK = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16"

tok = AutoTokenizer.from_pretrained(TOK)


def load(p):
    d = json.load(open(p))
    pr = d["prompts"]
    if isinstance(pr, dict):
        pr = list(pr.values())
    return pr


def dec(ids):
    return [tok.decode([i]) for i in ids]


def main():
    t, p, l = load(TRANS), load(PRIMARY), load(LLAMA)
    n = len(t)
    print(f"prompts: transcription={len(t)} primary={len(p)} llama.cpp={len(l)}\n")

    print(f"{'prompt':<44} {'trans':>10} {'primary':>10} {'llama.cpp':>10}   agree")
    ft = {("t", "p"): 0, ("t", "l"): 0, ("p", "l"): 0}
    seq = {("t", "p"): 0, ("t", "l"): 0, ("p", "l"): 0}
    pref = {("t", "p"): [], ("t", "l"): [], ("p", "l"): []}
    for i in range(n):
        tg = dec(t[i]["generated_ids"])
        pg = dec(p[i]["generated_ids"])
        lg = dec(l[i]["generated_ids"])
        marks = []
        for (x, y), a, b in ((("t", "p"), tg, pg), (("t", "l"), tg, lg), (("p", "l"), pg, lg)):
            if (a[:1] == b[:1]):
                ft[(x, y)] += 1
                marks.append(f"{x}{y}:1st")
            if a == b:
                seq[(x, y)] += 1
                marks.append(f"{x}{y}:seq")
            m = 0
            for u, v in zip(a, b):
                if u != v:
                    break
                m += 1
            pref[(x, y)].append(m)
        print(
            f"{t[i]['prompt'][:42]:<44} {str(tg[:1]):>10} {str(pg[:1]):>10} {str(lg[:1]):>10}   "
            + ", ".join(marks)
        )

    print("\nfirst-token agreement:")
    for k in (("t", "p"), ("t", "l"), ("p", "l")):
        print(f"  {k[0]} vs {k[1]}: {ft[k]}/{n}")
    print("identical full sequences:")
    for k in (("t", "p"), ("t", "l"), ("p", "l")):
        print(f"  {k[0]} vs {k[1]}: {seq[k]}/{n}")
    print("common prefix length (of 32):")
    for k in (("t", "p"), ("t", "l"), ("p", "l")):
        print(f"  {k[0]} vs {k[1]}: {pref[k]}")

    # majority vote: which implementation does the third one back?
    print("\nmajority on first token (3-way, 2-of-3):")
    for i in range(n):
        votes = [dec(t[i]["generated_ids"])[0], dec(p[i]["generated_ids"])[0], dec(l[i]["generated_ids"])[0]]
        c = {}
        for v in votes:
            c[v] = c.get(v, 0) + 1
        win = max(c, key=c.get)
        src = [nm for nm, v in zip("tpl", votes) if v == win]
        print(f"  {t[i]['prompt'][:40]:<42} {str(win):>10}  <- {','.join(src)}")


if __name__ == "__main__":
    sys.exit(main())
