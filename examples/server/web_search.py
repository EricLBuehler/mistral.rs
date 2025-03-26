from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = [
    {
        "role": "user",
        "content": "What is mistral.rs?",
    }
]

completion = client.chat.completions.create(
    model="llama-3.1", messages=messages, tool_choice="auto"
)

# print(completion.usage)
print(completion.choices[0].message)

tool_called = completion.choices[0].message.tool_calls[0].function
print(tool_called)