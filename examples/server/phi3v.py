import openai
import httpx
import textwrap, json


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


openai.api_key = "EMPTY"
openai.base_url = "http://localhost:1234/v1/"

# Enable this to log requests and responses
# openai.http_client = httpx.Client(
#     event_hooks={"request": [print], "response": [log_response]}
# )

completion = openai.chat.completions.create(
    model="phi3v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "<|image_1|>\nWhat is shown in this image?",
                },
            ],
        },
    ],
    max_tokens=32,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
)
resp = completion.choices[0].message.content
print(resp)
