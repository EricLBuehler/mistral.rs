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
    resp = ""
    response = openai.chat.completions.create(
        model="mistral",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,  # top_k=32,
        temperature=0.1,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta != "</s>" and delta != "<eos>":
            print(delta, end="")
        resp += delta
    if not resp.endswith("</s>") or (not resp.endswith("<eos>")):
        print("...")
    else:
        print() 
    messages.append({"role": "assistant", "content": resp})
