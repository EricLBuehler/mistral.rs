"""
PaddleOCR-VL document OCR through the OpenAI-compatible HTTP API.

Start the server:
    mistralrs serve -p 1234 -m PaddlePaddle/PaddleOCR-VL-1.5
"""

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
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
        },
    ],
    max_tokens=512,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0.1,
)
print(completion.choices[0].message.content)
