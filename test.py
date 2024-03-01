import openai

openai.api_key = "EMPTY"

openai.base_url = "http://localhost:1234/v1/"

completion = openai.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Explain how to best learn Rust.",
        },
    ],
    max_tokens = 5,
)
print(completion.choices[0].message.content)