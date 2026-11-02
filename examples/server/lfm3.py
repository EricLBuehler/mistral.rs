"""
LiquidAI LFM3 text generation through the OpenAI-compatible HTTP API.

Start the server:
    mistralrs serve -p 1234 -m LiquidAI/LFM3-1.7B --isq q4k

Then run `python examples/server/lfm3.py`.
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
)
print(completion.choices[0].message.content)
