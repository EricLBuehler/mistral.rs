import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})

eos_toks = ["</s>", "<eos>", "<|endoftext|>"]

while True:
    prompt = input(">>> ")
    messages.append({"role": "user", "content": prompt})
    resp = ""
    response = openai.chat.completions.create(
        model="mistral",
        messages=messages,
        max_tokens=256,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta not in eos_toks:
            print(delta, end="")
        resp += delta
    for eos in eos_toks:
        if resp.endswith(eos):
            print()
            break
    else:
        print("...")
    messages.append({"role": "assistant", "content": resp})
