import requests
import httpx
import textwrap
import json
import base64


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


BASE_URL = "http://localhost:1234/v1"

# Enable this to log requests and responses
# openai.http_client = httpx.Client(
#     event_hooks={"request": [print], "response": [log_response]}
# )

FILENAME = "picture.jpg"
with open(FILENAME, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

headers = {
    "Content-Type": "application/json",
}

payload = {
    "model": "phi3v",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_string}",
                    },
                },
                {
                    "type": "text",
                    "text": "What is shown in this image? Write a detailed response analyzing the scene.",
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
print(response.json())
