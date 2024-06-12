from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture
import base64

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-3-vision-128k-instruct",
        tokenizer_json=None,
        repeat_last_n=64,
        arch=VisionArchitecture.Phi3V,
    ),
)

FILENAME = "picture.jpg"
with open(FILENAME, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="phi3v",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(encoded_string),
                        },
                    },
                    {
                        "type": "text",
                        "text": "<|image_1|>\nWhat is shown in this image?",
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
