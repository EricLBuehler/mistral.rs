"""Compare a base model and its preloaded LoRA adapters without mixing history.

Start the server with one or more aliases, for example:
`mistralrs serve -m <base-model> --lora code=<adapter-repo>`.
For a supported text MoE model, the same example works with a routed expert
adapter. The CLI's `--lora-modules` JSON accepts `"is_3d_lora_weight": true`
as a vLLM compatibility hint, but mistral.rs detects the layout from the files.
The example discovers valid model IDs and keeps a separate conversation for each.
"""

from openai import APIStatusError, OpenAI


client = OpenAI(api_key="unused", base_url="http://localhost:1234/v1/")
cards = {card.id: card for card in client.models.list().data}

print("Available base models and adapters:")
for model_id, card in cards.items():
    metadata = card.model_extra or {}
    parent = metadata.get("parent")
    generation = metadata.get("adapter_generation")
    if generation is None:
        print(f"  {model_id} (base model)")
    else:
        print(f"  {model_id} (adapter for {parent}, generation {generation})")

system_prompt = input("System prompt (optional) >>> ").strip()
histories = {}

while True:
    model_id = input("Model ID (or 'quit') >>> ").strip()
    if model_id == "quit":
        break
    if model_id not in cards:
        print("Unknown model ID. Choose one of the IDs listed above.")
        continue

    prompt = input("User >>> ").strip()
    history = histories.setdefault(model_id, [])
    if not history and system_prompt:
        history.append({"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=history,
            max_tokens=256,
            temperature=0.0,
        )
    except APIStatusError as error:
        history.pop()
        print(f"Request failed ({error.status_code}): {error.response.text}")
        continue

    response = completion.choices[0].message.content or ""
    generation = (completion.model_extra or {}).get("adapter_generation")
    print(
        f"Assistant [model={completion.model}, generation={generation}] >>> {response}"
    )
    history.append({"role": "assistant", "content": response})
