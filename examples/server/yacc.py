from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

with open("examples/server/c.y", "r") as f:
    c_yacc = f.read()

completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Write the main function in C, returning 42. Answer with just the code, no explanation.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "yacc", "value": c_yacc}},
)

print(completion.choices[0].message.content)
