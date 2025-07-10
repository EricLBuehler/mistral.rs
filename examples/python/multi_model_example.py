"""
Example demonstrating multi-model usage with mistral.rs Python bindings.

This example shows how to:
1. Load multiple models
2. List available models
3. Switch between models for different requests
4. Manage default model selection
"""

from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    MultiModelRunner,
)


# Example 1: Using MultiModelRunner wrapper for cleaner API
def example_multi_model_runner():
    """Demonstrate using the MultiModelRunner for managing multiple models."""

    # First, create a regular runner with one model
    runner = Runner(
        which=Which.Plain(
            model_id="microsoft/DialoGPT-small",
            arch=Architecture.Gpt2,
            num_device_layers=None,
        )
    )

    # Convert to MultiModelRunner for multi-model operations
    multi_runner = MultiModelRunner(runner)

    # List available models (should show just the initial model)
    print("Available models:", multi_runner.list_models())

    # Send a request to a specific model
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    request = ChatCompletionRequest(messages=messages)

    # Use the specific model ID from list_models()
    model_ids = multi_runner.list_models()
    if model_ids:
        response = multi_runner.send_chat_completion_request_to_model(
            request=request,
            model_id=model_ids[0],  # Use first available model
        )
        print(f"Response from {model_ids[0]}:", response.choices[0].message.content)

    # Check and set default model
    default = multi_runner.get_default_model_id()
    print(f"Default model: {default}")

    # Send request to default model (no model_id specified)
    response = multi_runner.send_chat_completion_request(request=request)
    print("Response from default model:", response.choices[0].message.content)


# Example 2: Using regular Runner with model_id parameter
def example_runner_with_model_id():
    """Demonstrate using regular Runner with model_id in requests."""

    # Create a runner (in a real multi-model setup, this would have multiple models loaded)
    runner = Runner(
        which=Which.Plain(
            model_id="microsoft/DialoGPT-small",
            arch=Architecture.Gpt2,
        )
    )

    # List available models
    model_ids = runner.list_models()
    print("Available models:", model_ids)

    # Get default model
    default_model = runner.get_default_model_id()
    print(f"Default model: {default_model}")

    # Send request with specific model_id
    messages = [{"role": "user", "content": "Tell me a joke"}]
    request = ChatCompletionRequest(messages=messages)

    if model_ids:
        # Request to specific model
        response = runner.send_chat_completion_request(
            request=request, model_id=model_ids[0]
        )
        print(f"Response from {model_ids[0]}:", response.choices[0].message.content)

        # Request without model_id (uses default)
        response = runner.send_chat_completion_request(request=request)
        print("Response from default model:", response.choices[0].message.content)


# Example 3: Working with different types of models
def example_mixed_model_types():
    """Example showing how different model types could be used in multi-model setup."""

    # In a real scenario, you would load multiple models through configuration
    # This example shows the API usage pattern

    # Create a text generation model
    text_runner = Runner(
        which=Which.Plain(
            model_id="gpt2",
            arch=Architecture.Gpt2,
        )
    )

    multi_runner = MultiModelRunner(text_runner)

    # Different types of requests to different models
    # Text generation request
    text_messages = [{"role": "user", "content": "Write a short poem about AI"}]
    text_request = ChatCompletionRequest(messages=text_messages)

    model_ids = multi_runner.list_models()
    if model_ids:
        response = multi_runner.send_chat_completion_request_to_model(
            request=text_request, model_id=model_ids[0]
        )
        print(f"Text model response:\n{response.choices[0].message.content}")

    # In a real multi-model setup with vision models loaded, you could do:
    # vision_messages = [{"role": "user", "content": [
    #     {"type": "text", "text": "What's in this image?"},
    #     {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}}
    # ]}]
    # vision_request = ChatCompletionRequest(messages=vision_messages)
    # response = multi_runner.send_chat_completion_request_to_model(
    #     request=vision_request,
    #     model_id="vision-model-id"
    # )


# Example 4: Model management operations
def example_model_management():
    """Demonstrate model management operations."""

    runner = Runner(
        which=Which.Plain(
            model_id="gpt2",
            arch=Architecture.Gpt2,
        )
    )

    # List models
    print("Initial models:", runner.list_models())

    # Get and set default model
    current_default = runner.get_default_model_id()
    print(f"Current default model: {current_default}")

    # In a multi-model setup, you could change the default
    model_ids = runner.list_models()
    if model_ids and len(model_ids) > 1:
        # Set a different model as default
        runner.set_default_model_id(model_ids[1])
        print(f"Changed default model to: {model_ids[1]}")

    # Remove a model (in multi-model setup)
    # Note: Be careful not to remove all models or the currently active one
    # if len(model_ids) > 1:
    #     runner.remove_model(model_ids[0])
    #     print(f"Removed model: {model_ids[0]}")
    #     print("Remaining models:", runner.list_models())


# Example 5: Streaming with specific models
def example_streaming_with_models():
    """Demonstrate streaming responses from specific models."""

    runner = Runner(
        which=Which.Plain(
            model_id="gpt2",
            arch=Architecture.Gpt2,
        )
    )

    multi_runner = MultiModelRunner(runner)

    messages = [{"role": "user", "content": "Tell me a long story"}]
    request = ChatCompletionRequest(messages=messages, stream=True)

    model_ids = multi_runner.list_models()
    if model_ids:
        # Stream from specific model
        stream = multi_runner.send_chat_completion_request_to_model(
            request=request, model_id=model_ids[0]
        )

        print(f"Streaming from {model_ids[0]}:")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line after streaming


if __name__ == "__main__":
    print("=== Multi-Model Example 1: MultiModelRunner ===")
    example_multi_model_runner()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 2: Runner with model_id ===")
    example_runner_with_model_id()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 3: Mixed Model Types ===")
    example_mixed_model_types()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 4: Model Management ===")
    example_model_management()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 5: Streaming ===")
    example_streaming_with_models()
