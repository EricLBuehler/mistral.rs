import openai
import sys

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

eos_toks = ["</s>", "<eos>", "<|endoftext|>"]

while True:
    prompt = input(">>> ")
    response = openai.completions.create(
        model="mistral",
        prompt=prompt,
        max_tokens=256,
        stream=True,
    )
    resp = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta not in eos_toks:
            print(delta, end="")
            sys.stdout.flush()
