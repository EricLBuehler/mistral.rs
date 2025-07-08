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
        "content": "Hello! How many rs in strawberry?",
    },
]

# ------------------------------------------------------------------
# First question, thinking mode is enabled by default
# ------------------------------------------------------------------
completion = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=1024,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)

messages.append({"role": "assistant", "content": completion.choices[0].message.content})

messages = [
    {
        "role": "user",
        "content": "How many rs in blueberry? /no_think",
    },
]

# ------------------------------------------------------------------
# Second question, disable thinking mode with extra body or /no_think
# ------------------------------------------------------------------
completion = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=1024,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    # extra_body={
    #     "enable_thinking": False
    # }
)
resp = completion.choices[0].message.content
print(resp)


messages.append({"role": "assistant", "content": completion.choices[0].message.content})

messages = [
    {
        "role": "user",
        "content": "Are you sure? /think",
    },
]

# ------------------------------------------------------------------
# Third question, reenable thinking mode with extra body or /think
# ------------------------------------------------------------------
completion = client.chat.completions.create(
    model="default",
    messages=messages,
    max_tokens=1024,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    # extra_body={
    #     "enable_thinking": True
    # }
)
resp = completion.choices[0].message.content
print(resp)
