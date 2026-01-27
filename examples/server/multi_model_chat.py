"""
Example demonstrating multi-model usage with mistral.rs server HTTP API.

This example shows how to:
1. List available models from the server
2. Send requests to specific models
3. Compare responses from different models
4. Handle model selection in the API

Prerequisites:
- Start the server in multi-model mode:
  mistralrs serve -p 1234 --multi-model-config example-multi-model-config.json
"""

import requests
import json
from typing import List, Dict, Any, Optional

# Server configuration
SERVER_URL = "http://localhost:1234"


def list_models() -> List[Dict[str, Any]]:
    """List all available models from the server."""
    response = requests.get(f"{SERVER_URL}/v1/models")
    response.raise_for_status()
    models_data = response.json()
    return models_data.get("data", [])


def chat_with_model(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """Send a chat completion request to a specific model."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Add model_id if specified
    if model_id:
        payload["model"] = model_id

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def stream_chat_with_model(
    messages: List[Dict[str, str]],
    model_id: Optional[str] = None,
    temperature: float = 0.7,
):
    """Stream chat completion from a specific model."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    if model_id:
        payload["model"] = model_id

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if chunk["choices"][0]["delta"].get("content"):
                        yield chunk["choices"][0]["delta"]["content"]
                except json.JSONDecodeError:
                    continue


def compare_models_example():
    """Compare responses from different models to the same prompt."""
    print("=== Comparing Multiple Models ===\n")

    # List available models
    models = list_models()
    print(f"Available models: {len(models)}")
    for model in models:
        print(f"  - {model['id']}")
    print()

    if len(models) < 2:
        print(
            "Need at least 2 models loaded for comparison. Please start the server with multiple models."
        )
        return

    # Test prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What are the main benefits of using renewable energy?",
        },
    ]

    # Get responses from different models
    print("Getting responses from different models...\n")

    for model in models[:2]:  # Compare first two models
        model_id = model["id"]
        print(f"Model: {model_id}")
        print("-" * 50)

        response = chat_with_model(messages, model_id=model_id, temperature=0.7)
        content = response["choices"][0]["message"]["content"]
        print(content)
        print("\n")


def model_specific_tasks_example():
    """Demonstrate using different models for different types of tasks."""
    print("=== Model-Specific Tasks ===\n")

    models = list_models()
    if not models:
        print("No models available. Please start the server with models loaded.")
        return

    # Example 1: Creative writing with one model
    creative_messages = [
        {"role": "system", "content": "You are a creative writer."},
        {"role": "user", "content": "Write a haiku about artificial intelligence."},
    ]

    model_id = models[0]["id"]
    print(f"Creative task with {model_id}:")
    response = chat_with_model(creative_messages, model_id=model_id, temperature=0.9)
    print(response["choices"][0]["message"]["content"])
    print("\n")

    # Example 2: Technical explanation with another model (or same if only one)
    technical_messages = [
        {"role": "system", "content": "You are a technical expert."},
        {
            "role": "user",
            "content": "Explain how a transformer neural network works in simple terms.",
        },
    ]

    # Use second model if available, otherwise use first
    model_id = models[1]["id"] if len(models) > 1 else models[0]["id"]
    print(f"Technical task with {model_id}:")
    response = chat_with_model(technical_messages, model_id=model_id, temperature=0.3)
    print(response["choices"][0]["message"]["content"])
    print("\n")


def streaming_example():
    """Demonstrate streaming with specific models."""
    print("=== Streaming Example ===\n")

    models = list_models()
    if not models:
        print("No models available.")
        return

    messages = [
        {
            "role": "user",
            "content": "Tell me a short story about a robot learning to paint.",
        }
    ]

    model_id = models[0]["id"]
    print(f"Streaming from {model_id}:\n")

    for chunk in stream_chat_with_model(messages, model_id=model_id):
        print(chunk, end="", flush=True)
    print("\n")


def model_fallback_example():
    """Demonstrate fallback behavior when model is not specified."""
    print("=== Default Model Fallback ===\n")

    models = list_models()
    if not models:
        print("No models available.")
        return

    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    # Request without specifying model (uses default)
    print("Request without model_id (uses default):")
    response = chat_with_model(messages)
    print(f"Response: {response['choices'][0]['message']['content']}")
    print(f"Model used: {response.get('model', 'Not specified')}\n")

    # Request with specific model
    if models:
        model_id = models[0]["id"]
        print(f"Request with model_id='{model_id}':")
        response = chat_with_model(messages, model_id=model_id)
        print(f"Response: {response['choices'][0]['message']['content']}")
        print(f"Model used: {response.get('model', 'Not specified')}")


def interactive_model_selection():
    """Interactive example allowing user to choose models."""
    print("=== Interactive Model Selection ===\n")

    models = list_models()
    if not models:
        print("No models available.")
        return

    print("Available models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model['id']}")

    while True:
        print("\nEnter a message (or 'quit' to exit):")
        user_input = input("> ")

        if user_input.lower() == "quit":
            break

        print("\nSelect a model (enter number):")
        try:
            model_idx = int(input("> ")) - 1
            if 0 <= model_idx < len(models):
                model_id = models[model_idx]["id"]

                messages = [{"role": "user", "content": user_input}]
                response = chat_with_model(messages, model_id=model_id)

                print(f"\n{model_id}: {response['choices'][0]['message']['content']}")
            else:
                print("Invalid model selection.")
        except (ValueError, KeyError):
            print("Invalid input.")


if __name__ == "__main__":
    try:
        # Test server connection
        response = requests.get(f"{SERVER_URL}/v1/models", timeout=2)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        print("Error: Could not connect to server.")
        print("Please start the server with:")
        print(
            "  mistralrs serve -p 1234 --multi-model-config example-multi-model-config.json"
        )
        exit(1)

    # Run examples
    compare_models_example()
    print("\n" + "=" * 60 + "\n")

    model_specific_tasks_example()
    print("\n" + "=" * 60 + "\n")

    streaming_example()
    print("\n" + "=" * 60 + "\n")

    model_fallback_example()
    print("\n" + "=" * 60 + "\n")

    # Uncomment to run interactive mode
    # interactive_model_selection()
