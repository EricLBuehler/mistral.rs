from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    AnyMoeConfig,
    AnyMoeExpertType,
)

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
    anymoe_config=AnyMoeConfig(
        hidden_size=4096,
        dataset_json="examples/amoe.json",
        prefix="model.layers",
        mlp="mlp",
        expert_type=AnyMoeExpertType.LoraAdapter(
            rank=64, alpha=16.0, target_modules=["gate_proj"]
        ),
        lr=1e-3,
        epochs=100,
        batch_size=4,
        model_ids=["typeof/zephyr-7b-beta-lora"],
        # For inference (use a pretrained gating layer) see `anymoe_inference.py`
        loss_csv_path="loss.csv",
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
