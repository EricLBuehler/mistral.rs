from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="google/gemma-3n-E4B-it",
        arch=VisionArchitecture.Gemma3n,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="ignore",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pink_flower.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Please describe this image.",
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
