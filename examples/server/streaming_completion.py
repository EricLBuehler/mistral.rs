import openai
import sys

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"


response = openai.completions.create(
    model="mistral",
    prompt="My favorite theorem is",
    max_tokens=32,
    stream=True,
)
for chunk in response:
    delta = chunk.choices[0].text
    print(delta, end="")
    sys.stdout.flush()
