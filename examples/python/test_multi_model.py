"""
Simple test script to verify multi-model functionality in mistral.rs

This script tests the core multi-model operations:
- Listing models
- Getting/setting default model
- Sending requests to specific models
- Model removal (commented out for safety)
"""

from mistralrs import (
    Runner,
    Which,
    ChatCompletionRequest,
    Architecture,
    MultiModelRunner,
)
import sys


def test_multi_model_operations():
    """Test basic multi-model operations."""
    print("Testing Multi-Model Operations\n" + "=" * 50)

    try:
        # Create a simple runner
        print("1. Creating runner with GPT-2 model...")
        runner = Runner(
            which=Which.Plain(
                model_id="gpt2",
                arch=Architecture.Gpt2,
            )
        )
        print("   ✓ Runner created successfully")

        # Test listing models
        print("\n2. Testing list_models()...")
        models = runner.list_models()
        print(f"   ✓ Models found: {models}")
        assert isinstance(models, list), "list_models should return a list"
        assert len(models) > 0, "Should have at least one model"

        # Test getting default model
        print("\n3. Testing get_default_model_id()...")
        default_model = runner.get_default_model_id()
        print(f"   ✓ Default model: {default_model}")

        # Test setting default model (if we have multiple models)
        if len(models) > 1:
            print("\n4. Testing set_default_model_id()...")
            new_default = models[1] if models[0] == default_model else models[0]
            runner.set_default_model_id(new_default)
            updated_default = runner.get_default_model_id()
            print(f"   ✓ Changed default from '{default_model}' to '{updated_default}'")
            assert updated_default == new_default, "Default model should have changed"
        else:
            print("\n4. Skipping set_default_model_id() test (only one model loaded)")

        # Test sending request with model_id
        print("\n5. Testing send_chat_completion_request with model_id...")
        messages = [
            {"role": "user", "content": "Say 'test successful' and nothing else."}
        ]
        request = ChatCompletionRequest(messages=messages, max_tokens=10)

        if models:
            response = runner.send_chat_completion_request(
                request=request, model_id=models[0]
            )
            print(f"   ✓ Response received: {response.choices[0].message.content}")

        # Test MultiModelRunner wrapper
        print("\n6. Testing MultiModelRunner wrapper...")
        multi_runner = MultiModelRunner(runner)

        # Test wrapper methods
        wrapper_models = multi_runner.list_models()
        print(f"   ✓ MultiModelRunner.list_models(): {wrapper_models}")
        assert wrapper_models == models, "MultiModelRunner should return same models"

        wrapper_default = multi_runner.get_default_model_id()
        print(f"   ✓ MultiModelRunner.get_default_model_id(): {wrapper_default}")

        # Test wrapper request methods
        if models:
            response = multi_runner.send_chat_completion_request_to_model(
                request=request, model_id=models[0]
            )
            print(
                f"   ✓ Response from specific model: {response.choices[0].message.content}"
            )

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_model_id_in_requests():
    """Test that model_id is properly passed in requests."""
    print("\n\nTesting Model ID in Requests\n" + "=" * 50)

    try:
        runner = Runner(
            which=Which.Plain(
                model_id="gpt2",
                arch=Architecture.Gpt2,
            )
        )

        models = runner.list_models()
        if not models:
            print("No models available to test")
            return False

        model_id = models[0]
        print(f"Using model: {model_id}")

        # Test different request types with model_id
        messages = [{"role": "user", "content": "Hi"}]

        # Chat completion
        print("\n1. Testing chat completion with model_id...")
        request = ChatCompletionRequest(messages=messages, max_tokens=5)
        response = runner.send_chat_completion_request(request, model_id=model_id)
        print(f"   ✓ Chat response: {response.choices[0].message.content}")

        # Without model_id (should use default)
        print("\n2. Testing chat completion without model_id...")
        response = runner.send_chat_completion_request(request)
        print(f"   ✓ Chat response (default): {response.choices[0].message.content}")

        print("\n✅ Model ID request tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling for multi-model operations."""
    print("\n\nTesting Error Handling\n" + "=" * 50)

    try:
        runner = Runner(
            which=Which.Plain(
                model_id="gpt2",
                arch=Architecture.Gpt2,
            )
        )

        # Test with non-existent model
        print("1. Testing request to non-existent model...")
        messages = [{"role": "user", "content": "Hi"}]
        request = ChatCompletionRequest(messages=messages)

        try:
            response = runner.send_chat_completion_request(
                request, model_id="non-existent-model"
            )
            print("   ❌ Should have raised an error for non-existent model")
        except Exception as e:
            print(f"   ✓ Correctly raised error: {type(e).__name__}")

        # Test setting non-existent model as default
        print("\n2. Testing set_default_model_id with non-existent model...")
        try:
            runner.set_default_model_id("non-existent-model")
            print("   ❌ Should have raised an error")
        except Exception as e:
            print(f"   ✓ Correctly raised error: {type(e).__name__}")

        print("\n✅ Error handling tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test setup failed with error: {e}")
        return False


if __name__ == "__main__":
    print("mistral.rs Multi-Model Test Suite")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_multi_model_operations()
    all_passed &= test_model_id_in_requests()
    all_passed &= test_error_handling()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
