"""Compare a routed MoE LoRA adapter with its base model.

Start a supported MoE model with a preloaded adapter, for example:
`mistralrs serve -m <base-model> --lora domain=<moe-lora-repo>`.
Then run `python examples/server/moe_lora_smoke.py --adapter domain`.
"""

import argparse

from openai import OpenAI


parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="domain")
parser.add_argument("--base-model")
parser.add_argument("--base-url", default="http://localhost:1234/v1/")
parser.add_argument(
    "--prompt", default="Explain why the sky appears blue in two sentences."
)
args = parser.parse_args()

client = OpenAI(api_key="unused", base_url=args.base_url)
card_list = client.models.list().data
cards = {card.id: card for card in card_list}
adapter_matches = [
    card
    for card in card_list
    if (card.model_extra or {}).get("adapter_generation") is not None
    and (
        card.id == args.adapter or (card.model_extra or {}).get("root") == args.adapter
    )
]
if not adapter_matches:
    raise SystemExit(
        f"Adapter {args.adapter!r} is not loaded. Available IDs: {sorted(cards)}"
    )
if len(adapter_matches) > 1:
    matches = sorted(card.id for card in adapter_matches)
    raise SystemExit(
        f"Adapter alias {args.adapter!r} is ambiguous; use one of {matches}."
    )
adapter_card = adapter_matches[0]

metadata = adapter_card.model_extra or {}
expected_generation = metadata["adapter_generation"]
base_model = args.base_model or metadata.get("parent")
if not base_model:
    raise SystemExit(
        "The adapter model card has no parent; pass --base-model explicitly."
    )
if base_model not in cards:
    raise SystemExit(f"Base model {base_model!r} is not advertised by GET /v1/models.")

results = {}
for model in [base_model, adapter_card.id]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        max_tokens=128,
        temperature=0.0,
    )
    generation = (response.model_extra or {}).get("adapter_generation")
    content = response.choices[0].message.content or ""
    if not content.strip():
        raise SystemExit(f"Model {model!r} returned an empty response.")
    results[model] = (generation, content)
    print(f"\n[{model}; generation={generation}]\n{content}")

if results[base_model][0] is not None:
    raise SystemExit("The base-model request unexpectedly selected an adapter.")
if results[adapter_card.id][0] != expected_generation:
    raise SystemExit(
        "The adapter request did not resolve to the generation advertised by GET /v1/models."
    )
if results[base_model][1] == results[adapter_card.id][1]:
    print("\nWarning: base and adapter text are identical for this prompt.")
print("\nMoE LoRA routing checks passed.")
