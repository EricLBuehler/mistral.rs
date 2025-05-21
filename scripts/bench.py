# cargo run --release --features metal '--' --port 1234 --isq 8 --paged-attn --max-seqs 1000 plain -m ../hf_models/llama3.2_3b --max-seq-len 131072      
# cargo run --release --features metal '--' --port 1234 --paged-attn --max-seqs 1000 plain -m mlx-community/Mistral-7B-Instruct-v0.3-4bit --max-seq-len 131072
# mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080 

import asyncio
from openai import AsyncOpenAI
import httpx
import textwrap
import json
import time


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


async def timed_chat(client: AsyncOpenAI, messages):
    """
    Send one chat completion request and return (completion, elapsed_seconds).
    """
    start = time.perf_counter()
    completion = await client.chat.completions.create(
        model="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    return completion, time.perf_counter() - start


# Use the async-capable client
client = AsyncOpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

# Enable this to log requests and responses
# client._client = httpx.AsyncClient(
#     event_hooks={"request": [print], "response": [log_response]}
# )


async def main() -> None:
    """
    Simple REPL that fires **10 concurrent** chat completion requests for every user prompt.
    """
    messages = []

    # Optional system prompt
    prompt = input("Enter system prompt >>> ")
    if prompt:
        messages.append({"role": "system", "content": prompt})

    while True:
        user_prompt = input(">>> ")
        messages.append({"role": "user", "content": user_prompt})

        # Create 10 concurrent requests
        tasks = [timed_chat(client, list(messages)) for _ in range(30)]

        # Wait for them all to finish
        results = await asyncio.gather(*tasks)

        elapsed_times = []
        t_s = []
        # Show the responses
        for i, (completion, elapsed) in enumerate(results, 1):
            usage = completion.usage
            print(f"{elapsed:.2f}s {usage.total_tokens / elapsed:.2f} T/s")
            t_s.append(usage.total_tokens / elapsed)
            elapsed_times.append(elapsed)

        print(
            f"Average: {sum(elapsed_times) / len(elapsed_times):.2f}s {sum(t_s) / len(t_s):.2f} T/s"
        )

        # Keep only the first assistant reply in the conversation history
        messages.append(
            {"role": "assistant", "content": results[0][0].choices[0].message.content}
        )


if __name__ == "__main__":
    asyncio.run(main())
