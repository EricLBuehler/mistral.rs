from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="lamm-mit/Cephalo-Idefics-2-vision-8b-beta",
        arch=VisionArchitecture.Idefics2,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="idefics2",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image?",
                    },
                ],
            },
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
