"""
Example demonstrating multi-model usage with mistral.rs Python bindings.

This example shows how to:
1. Load a model using Runner
2. List available models
3. Use model_id parameter to target specific models
4. Manage default model selection
5. Model loading/unloading operations

Models used:
- Vision: google/gemma-3-4b-it (VisionArchitecture.Gemma3)
- Text: Qwen/Qwen3-4B (Architecture.Qwen3)
"""

from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    VisionArchitecture,
)


# Example 1: Using Runner with model_id parameter for multi-model operations
def example_runner_with_model_id():
    """Demonstrate using Runner with model_id in requests."""

    # Create a runner with Gemma 3 4B vision model
    runner = Runner(
        which=Which.VisionPlain(
            model_id="google/gemma-3-4b-it",
            arch=VisionArchitecture.Gemma3,
        ),
        in_situ_quant="Q4K",
    )

    # List available models
    model_ids = runner.list_models()
    print("Available models:", model_ids)

    # Get default model
    default_model = runner.get_default_model_id()
    print(f"Default model: {default_model}")

    # Send request with specific model_id
    messages = [{"role": "user", "content": "Hello, how are you?"}]
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


# Example 2: Model management operations
def example_model_management():
    """Demonstrate model management operations."""

    runner = Runner(
        which=Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=Architecture.Qwen3,
        ),
        in_situ_quant="Q4K",
    )

    # List models with their status
    print("Initial models with status:", runner.list_models_with_status())

    # Get default model
    current_default = runner.get_default_model_id()
    print(f"Current default model: {current_default}")

    # Check if a model is loaded
    model_ids = runner.list_models()
    if model_ids:
        is_loaded = runner.is_model_loaded(model_ids[0])
        print(f"Is {model_ids[0]} loaded? {is_loaded}")

    # In a multi-model setup, you could change the default
    if model_ids and len(model_ids) > 1:
        runner.set_default_model_id(model_ids[1])
        print(f"Changed default model to: {model_ids[1]}")


# Example 3: Model unloading and reloading
def example_unload_reload():
    """Demonstrate model unloading and reloading."""

    runner = Runner(
        which=Which.VisionPlain(
            model_id="google/gemma-3-4b-it",
            arch=VisionArchitecture.Gemma3,
        ),
        in_situ_quant="Q4K",
    )

    model_ids = runner.list_models()
    if not model_ids:
        print("No models loaded")
        return

    model_id = model_ids[0]
    print(f"Initial status: {runner.list_models_with_status()}")

    # Unload the model to free memory
    # Note: This preserves the model configuration for later reload
    print(f"Unloading model: {model_id}")
    runner.unload_model(model_id)

    # Check status after unload
    print(f"Status after unload: {runner.list_models_with_status()}")
    print(f"Is {model_id} loaded? {runner.is_model_loaded(model_id)}")

    # Reload the model when needed
    print(f"Reloading model: {model_id}")
    runner.reload_model(model_id)

    # Check status after reload
    print(f"Status after reload: {runner.list_models_with_status()}")


# Example 4: Streaming with specific models
def example_streaming_with_models():
    """Demonstrate streaming responses from specific models."""

    runner = Runner(
        which=Which.Plain(
            model_id="Qwen/Qwen3-4B",
            arch=Architecture.Qwen3,
        ),
        in_situ_quant="Q4K",
    )

    messages = [{"role": "user", "content": "Tell me a short story"}]
    request = ChatCompletionRequest(messages=messages, stream=True)

    model_ids = runner.list_models()
    if model_ids:
        # Stream from specific model
        stream = runner.send_chat_completion_request(
            request=request, model_id=model_ids[0]
        )

        print(f"Streaming from {model_ids[0]}:")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line after streaming


# Example 5: Multi-model setup with vision and text models
def example_multi_model_setup():
    """
    Example showing a real multi-model setup with vision and text models.

    This example loads:
    - Vision model: google/gemma-3-4b-it
    - Text model: Qwen/Qwen3-4B
    """
    # Load a vision model first
    runner = Runner(
        which=Which.VisionPlain(
            model_id="google/gemma-3-4b-it",
            arch=VisionArchitecture.Gemma3,
        ),
        in_situ_quant="Q4K",
    )

    print("Initial models:", runner.list_models())

    # Add a text model dynamically (if add_model is available)
    # runner.add_model(
    #     model_id="qwen",
    #     which=Which.Plain(
    #         model_id="Qwen/Qwen3-4B",
    #         arch=Architecture.Qwen3,
    #     ),
    #     in_situ_quant="Q4K",
    # )
    # print("After add_model:", runner.list_models())

    # Send a request to gemma
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    request = ChatCompletionRequest(messages=messages)
    response = runner.send_chat_completion_request(request)
    print(f"Gemma response: {response.choices[0].message.content}")


if __name__ == "__main__":
    print("=== Multi-Model Example 1: Runner with model_id ===")
    example_runner_with_model_id()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 2: Model Management ===")
    example_model_management()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 3: Unload/Reload ===")
    example_unload_reload()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 4: Streaming ===")
    example_streaming_with_models()
    print("\n" + "=" * 50 + "\n")

    print("=== Multi-Model Example 5: Multi-Model Setup ===")
    example_multi_model_setup()
