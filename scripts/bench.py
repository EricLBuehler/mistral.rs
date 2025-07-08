## Comparing Qwen 3 30B A3B
# cargo run --release --features metal --' --port 1234 --paged-attn --max-seqs 128 plain -m mlx-community/Qwen3-30B-A3B-4bit--max-seq-len 1024 --max-batch-size 128
# ./llama-server -m ../gguf_models/Qwen3-30B-A3B-Q4_K_M.gguf

## Comparing Llama 3.2 3b
# cargo run --release --features metal '--' --port 1234 --isq 8 --paged-attn --max-seqs 128 plain -m ../hf_models/llama3.2_3b --max-seq-len 1024 --max-batch-size 128
# ./llama-server -m ../gguf_models/Llama-3.2-3B-Instruct-Q8_0.gguf

## Comparing Mistral 7b
# cargo run --release --features metal '--' --port 1234 --paged-attn --max-seqs 128 plain -m mlx-community/Mistral-7B-Instruct-v0.3-4bit --max-seq-len 1024 --max-batch-size 128
# mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --port 8080

import asyncio
from openai import AsyncOpenAI
import httpx
import textwrap
import json
import time

NUM_USERS = 8
REQUESTS_PER_USER = 8
PORT = 1234


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


# Use the async-capable client
client = AsyncOpenAI(api_key="foobar", base_url=f"http://localhost:{PORT}/v1/")


async def timed_chat(client: AsyncOpenAI, messages):
    """
    Send one chat completion request and return (completion, elapsed_seconds, completion_tokens).
    """
    start = time.perf_counter()
    completion = await client.chat.completions.create(
        model="default",
        messages=messages,
        max_tokens=256,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    elapsed = time.perf_counter() - start
    # Safely get number of completion tokens, default to 0 if missing
    completion_tokens = getattr(completion.usage, "completion_tokens", 0)
    return completion, elapsed, completion_tokens


async def user_task(client: AsyncOpenAI, system_prompt: str, user_message: str):
    """
    Returns list of (completion, elapsed_seconds, completion_tokens).
    """
    results = []
    base_messages = []
    if system_prompt:
        base_messages.append({"role": "system", "content": system_prompt})

    for _ in range(REQUESTS_PER_USER):
        messages = base_messages + [{"role": "user", "content": user_message}]
        completion, elapsed, completion_tokens = await timed_chat(client, messages)
        results.append((completion, elapsed, completion_tokens))
    return results


async def main() -> None:
    """
    Computes and prints overall average request time, total requests, and average T/s.
    """
    system_prompt = None  # "You are a helpful assistant."
    user_message = "Say hello!"

    tasks = [user_task(client, system_prompt, user_message) for _ in range(NUM_USERS)]
    all_results_nested = await asyncio.gather(*tasks)
    all_results = [item for sublist in all_results_nested for item in sublist]

    total_requests = len(all_results)
    total_time = sum(elapsed for _, elapsed, _ in all_results)
    total_tokens = sum(tokens for _, _, tokens in all_results)
    avg_time = total_time / total_requests if total_requests else 0.0
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0

    print(f"Total requests: {total_requests}")
    print(f"Average request time: {avg_time:.2f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per second (T/s): {avg_tps:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
