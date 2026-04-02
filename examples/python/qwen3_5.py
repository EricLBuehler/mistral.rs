from mistralrs import Runner, Which, ChatCompletionRequest, MultimodalArchitecture

# Dense variant
MODEL_ID = "Qwen/Qwen3.5-27B"

runner = Runner(
    which=Which.MultimodalPlain(
        model_id=MODEL_ID,
        arch=MultimodalArchitecture.Qwen3_5,
    ),
)

# For MoE variant, use:
# MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
# runner = Runner(
#     which=Which.MultimodalPlain(
#         model_id=MODEL_ID,
#         arch=MultimodalArchitecture.Qwen3_5Moe,
#     ),
# )

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.garden-treasures.com/cdn/shop/products/IMG_6245.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What type of flower is this? Give some fun facts.",
                    },
                ],
            }
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
