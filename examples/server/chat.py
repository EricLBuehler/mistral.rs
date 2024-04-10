import openai
import httpx

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"
openai.http_client = httpx.Client(event_hooks={"request": [print], "response": [print]})

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})

eos_toks = ["</s>", "<eos>", "<|endoftext|>"]

while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    completion = openai.chat.completions.create(
        model="mistral",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    resp = completion.choices[0].message.content
    for eos in eos_toks:
        if resp.endswith(eos):
            out = resp[: -len(eos)]
            print(out)
            break
    else:
        print(resp + "...")
    messages.append({"role": "assistant", "content": resp})
