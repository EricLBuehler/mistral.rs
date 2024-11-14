import sys
from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})


while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    resp = ""
    response = client.chat.completions.create(
        model="mistral",
        messages=messages,
        max_tokens=256,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        print(delta, end="")
        sys.stdout.flush()
        resp += delta
    messages.append({"role": "assistant", "content": resp})
    print()
