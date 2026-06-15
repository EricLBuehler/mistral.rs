"""
Chat Completions web search with the HTTP API.

Start the server:
    mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B

Then run this script:
    python examples/server/web_search.py
"""

from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

messages = [
    {
        "role": "user",
        "content": "Can you show me some code using mistral.rs for running Llama 3.2 Vision?",
    }
]

completion = client.chat.completions.create(
    model="default",
    messages=messages,
    tool_choice="auto",
    max_tokens=1024,
    web_search_options={},
)

# print(completion.usage)
print(completion.choices[0].message.content)

if completion.choices[0].message.tool_calls is not None:
    # Should never happen.
    tool_called = completion.choices[0].message.tool_calls[0].function
    print(tool_called)
