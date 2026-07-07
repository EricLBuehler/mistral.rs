"""
PaddleOCR-VL document OCR with the Python SDK.
"""

from mistralrs import ChatCompletionRequest, MultimodalArchitecture, Runner, Which

runner = Runner(
    which=Which.MultimodalPlain(
        model_id="PaddlePaddle/PaddleOCR-VL-1.5",
        arch=MultimodalArchitecture.PaddleOcrVl,
    ),
)

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
                            "url": "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
                        },
                    },
                    {
                        "type": "text",
                        "text": "Extract all the text from this document.",
                    },
                ],
            }
        ],
        max_tokens=512,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
