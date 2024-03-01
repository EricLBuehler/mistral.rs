import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

messages = []
prompt = input("Enter system prompt >>>")
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
    print(resp.split("[/INST]")[-1])
    messages.append({"role":"assistant", "content":resp})