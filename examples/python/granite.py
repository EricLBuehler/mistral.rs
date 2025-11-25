#!/usr/bin/env python
"""
Example of using IBM Granite 4.0 model with mistral.rs
"""

from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

# Create a Granite model runner
runner = Runner(
    which=Which.Plain(
        model_id="ibm-granite/granite-4.0-tiny-preview",
        arch=Architecture.GraniteMoeHybrid,
    ),
)

# Send a chat completion request
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="granite",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=256,
        temperature=0.7,
    )
)

# Print the response
print(res.choices[0].message.content)
print(f"\nUsage: {res.usage}")
