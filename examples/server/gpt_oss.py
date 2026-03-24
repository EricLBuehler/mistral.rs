"""
Example of using GPT-OSS model via the HTTP API.

Start the server first:
    mistralrs serve -p 1234 -m openai/gpt-oss-20b

GPT-OSS is a Mixture of Experts model with MXFP4 quantized experts
and custom attention with per-head sinks.
"""

from openai import OpenAI
import httpx
import textwrap
import json


def log_response(response: httpx.Response):
    request = response.request
    print(f"Request: {request.method} {request.url}")
    print("  Headers:")
    for key, value in request.headers.items():
        if key.lower() == "authorization":
            value = "[...]"
        if key.lower() == "cookie":
            value = value.split("=")[0] + "=..."
        print(f"    {key}: {value}")
    print("  Body:")
    try:
        request_body = json.loads(request.content)
        print(textwrap.indent(json.dumps(request_body, indent=2), "    "))
    except json.JSONDecodeError:
        print(textwrap.indent(request.content.decode(), "    "))
    print(f"Response: status_code={response.status_code}")
    print("  Headers:")
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            value = value.split("=")[0] + "=..."
        print(f"    {key}: {value}")


client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Enable this to log requests and responses
# client._client = httpx.Client(
#     event_hooks={"request": [print], "response": [log_response]}
# )

messages = [
    {
        "role": "user",
        "content": "Hello! What is the capital of France?",
    },
]

completion = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)

messages.append({"role": "assistant", "content": completion.choices[0].message.content})

# Follow-up question
messages.append(
    {
        "role": "user",
        "content": "What is its population?",
    }
)

completion = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)
