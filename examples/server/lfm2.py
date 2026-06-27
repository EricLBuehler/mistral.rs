"""
LiquidAI LFM2.5 text generation through the OpenAI-compatible HTTP API.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM2.5-230M
"""

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Explain what graphene is in two short paragraphs.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0.1,
)
print(completion.choices[0].message.content)
