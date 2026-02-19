#!/usr/bin/env python
"""
Example of using GPT-OSS model with mistral.rs

GPT-OSS is a Mixture of Experts model with MXFP4 quantized experts
and custom attention with per-head sinks.
"""

from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

# Create a GPT-OSS model runner
runner = Runner(
    which=Which.Plain(
        model_id="openai/gpt-oss-20b",  # Replace with actual model ID
        arch=Architecture.GptOss,
    ),
)

# Send a chat completion request
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="gpt_oss",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=256,
        temperature=0.7,
    )
)

# Print the response
print(res.choices[0].message.content)
print(f"\nUsage: {res.usage}")
