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

# # Enable this to log requests and responses
# client._client = httpx.Client(
#     event_hooks={"request": [print], "response": [log_response]}
# )


# first turn
resp1 = client.responses.create(
    model="default",
    input="Plan a weekend in Montreal")

print(resp1)

# follow‑up: no need to resend the first question
resp2 = client.responses.create(
    model="default",
    previous_response_id=resp1.id,
    input="Add a kid-friendly science museum, please")

print(resp2)