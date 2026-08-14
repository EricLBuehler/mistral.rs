"""Run Muse Glimmer with its BF16 DFlash speculative-decoding assistant.

PagedAttention is required. The assistant proposes up to 15 tokens per target verification pass.
"""

from mistralrs import (
    ChatCompletionRequest,
    MultimodalArchitecture,
    Runner,
    Which,
)

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="meta-models/Muse-Glimmer-30B",
        arch=MultimodalArchitecture.MuseGlimmer,
    ),
    paged_attn=True,
    dflash_model="meta-models/Muse-Glimmer-30B-assistant",
    dflash_n_predict=15,
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "Explain why speculative decoding preserves target-model output.",
            }
        ],
        max_tokens=256,
    )
)
print(response.choices[0].message.content)
