# Gemma 4 Prompt Formatting

Starting with Gemma 4, we introduce a new control tokens. For Gemma 3 and below, see the [previous document](/gemma/docs/core/prompt-structure).

Below, we specify the control tokens used by Gemma 4 and their use cases. Note that the control tokens are reserved in and specific to our tokenizer.

* Token to indicate a system instruction: `system`
* Token to indicate a user turn: `user`
* Token to indicate a model turn: `model`
* Token to indicate the beginning of dialogue turn: `<|turn>`
* Token to indicate the end of dialogue turn: `<turn|>`

Here's an example dialogue:

```none
<|turn>system
You are a helpful assistant.<turn|>
<|turn>user
Hello.<turn|>
```

## Multi-modalities {:#multimodal}

| Multimodal Token | Purpose |
| :--- | :--- |
| `<\|image>` <br> `<image\|>` | Indicate an image embeddings |
| `<\|audio>` <br> `<audio\|>` | Indicate an audio embeddings |
| `<\|image\|>` <br> `<\|audio\|>` | A special placeholder tokens |

We use 2 special placeholder tokens (`<|image|>` and `<|audio|>` to specify where image / audio tokens should be inserted. After tokenization, those tokens will be replaced by the the actual soft embeddings inside the model.

Here's an example dialogue:

```Python
prompt = """<|turn>user
Describe this image: <|image|>

And translate those audio:

a. <|audio|>
b. <|audio|><turn|>
<|turn>model"""
```

## Agentic and Reasoning Control Tokens {:#agentic-tokens}

To support agentic workflows, Gemma uses specialized control tokens that delineate internal reasoning (thinking) from external actions (function calling) to support agentic workflows. These tokens allow the model to process complex logic before providing a final response or interacting with outside tools.

### Function Calling

Gemma 4 is trained on six special tokens to manage the "tool use" lifecycle.

| Token Pair | Purpose |
| :--- | :--- |
| `<\|tool>` <br> `<tool\|>` | Define a tool |
| `<\|tool_call>` <br> `<tool_call\|>` | Indicates a model's request to use a tool. |
| `<\|tool_response>` <br> `<tool_response\|>` | Provides a tool's result back to the model. |

> NOTE: `<|tool_response>` is an additional stop sequence for the inference engine.

**Delimiter for String Values: `<|"|>`**

A single token, `<|"|>`, is used as a delimiter for **all string values**
within the structured data blocks.

*   **Purpose:** This token ensures that any special characters (like `{`, `}`,
    `,`, or quotes) inside a string are treated as literal text and not as part
    of the data structure's syntax.
*   **Usage:** All string literals in your function declarations, calls, and
    responses must be enclosed, like: `key:<|"|>string value<|"|>`.

### Thinking Mode

To activate thinking, put the control token `<|think|>` in the system instruction.

| Control Token | Purpose |
| :--- | :--- |
| `<\|think\|>` | Activate thinking mode |
| `<\|channel>` <br> `<channel\|>` | Indicate a model's internal process. |

> NOTE: `<|channel>` is always followed by the word "thought" in thinking mode.

Here's an example dialogue:

```none
<|turn>system
<|think|><turn|>
<|turn>user
What is the water formula?<turn|>
<|turn>model
<|channel>thought
...
<channel|>The most common interpretation of "the water formula" refers...<turn|>
```

### Reasoning and Function Calling Example

In an agentic turn, the model may "think" privately before deciding to call a function. The following example demonstrates a model using a weather tool:

```none
<|turn>system
<|think|>You are a helpful assistant.<|tool>declaration:get_current_temperature{...}<tool|><turn|>
<|turn>user
What's the temperature in London?<turn|>
<|turn>model
<|channel>thought
...
<channel|><|tool_call>call:get_current_temperature{location:<|"|>London<|"|>}<tool_call|><|tool_response>
```

Your application should parse the model's response to extract function name and argments, and append `tool_calls` and `tool_responses` with the `assistant` role.

```none
<|turn>model
<|tool_call>call:get_current_weather{location:<|"|>London<|"|>}<tool_call|><|tool_response>response:get_current_weather{temperature:15,weather:<|"|>sunny<|"|>}<tool_response|>
```

Finally, Gemma reads the tool response and reply to the user.

```none
The temperature in London is 15 degrees and it is sunny.<turn|>
```

And the following is the full chat history of this example.

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "What's the temperature in London?"
  },
  {
    "role": "assistant",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "location": "London"
          }
        }
      }
    ],
    "tool_responses": [
      {
        "name": "get_current_weather",
        "response": {
          "temperature": 15,
          "weather": "sunny"
        }
      }
    ],
    "content": "The temperature in London is 15 degrees and it is sunny."
  }
]
```

### Integration Notes

*   **Internal State:** The `<|channel>` and `<channel|>` tokens are typically
    used for Chain-of-Thought (CoT) processing. In many implementations, this
    content is hidden from the end-user.
*   **Tool Loop:** The `tool_call` and `tool_response` tokens facilitate a
    "handshake" between the model and the application environment. The
    application intercepts the `tool_call`, executes the code, and feeds the
    result back to the model within the `tool_response` tokens.
*   **Model Behavior:** Large models (gemma-4-26B-A4B-it, gemma-4-31B-it) may
    generate a though channel even when thinking is explicitly turned off.
    Consider adding an empty thinking token to stabilize model behavior in these
    cases.