#!/usr/bin/env python
"""
Example of using SmolLM3 model with mistral.rs
"""

from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

# Create a SmolLM3 model runner
runner = Runner(
    which=Which.Plain(
        model_id="HuggingFaceTB/SmolLM3-3B",  # You can use any SmolLM3 model from HuggingFace
        arch=Architecture.SmolLm3,
    ),
)

# Send a chat completion request
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="smollm3",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=256,
        temperature=0.7,
    )
)

# Print the response
print(res.choices[0].message.content)
print(f"\nUsage: {res.usage}")
