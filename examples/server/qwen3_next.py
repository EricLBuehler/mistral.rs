"""
Start the server:
    mistralrs serve --isq 4 -p 1234 -m Qwen/Qwen3-Coder-Next
"""

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})


while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    resp = completion.choices[0].message.content
    print(resp)
    messages.append({"role": "assistant", "content": resp})
