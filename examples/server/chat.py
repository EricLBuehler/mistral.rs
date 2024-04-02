import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

messages = []
prompt = input("Enter system prompt >>> ")
if len(prompt) > 0:
    messages.append({"role": "system", "content": prompt})

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
    if resp.endswith("</s>"):
        out = resp[:-4]
        print(out)
    elif resp.endswith("<eos>"):
        out = resp[:-5]
        print(out)
    else:
        print(resp + "...")
    messages.append({"role": "assistant", "content": resp})
