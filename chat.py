import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

messages = []
prompt = input("Enter system prompt >>> ")
messages.append({"role":"system", "content":prompt})

while True:
    prompt = input(">>> ")
    messages.append({"role":"user", "content":prompt})
    completion = openai.chat.completions.create(
        model="mistral",
        messages=messages,
        max_tokens = 256,
    )
    resp = completion.choices[0].message.content
    out = resp.split("[/INST]")[-1].strip()
    if out.endswith("</s>"):
        out = out[:-4]
        print(out)
    else:
        print(out+"...")
    messages.append({"role":"assistant", "content":out})