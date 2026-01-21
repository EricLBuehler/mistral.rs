# mistral.rs

`mistralrs` is a Python package which provides an easy to use API for `mistral.rs`. 

## Example
More examples can be found [here](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)!

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    in_situ_quant="4",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

### Multimodal (audio + image) example

`mistralrs` also supports multimodal vision models that can reason over both
images *and* audio clips via the same OpenAI-style `audio_url` / `image_url`
format. The example below queries the Phi-4-Multimodal model with a single
image and an audio recording – notice how the text prompt references them via
`<|audio_1|>` and `<|image_1|>` tokens (indexing starts at 1):

```python
from mistralrs import Runner, Which, ChatCompletionRequest, VisionArchitecture

runner = Runner(
    which=Which.VisionPlain(
        model_id="microsoft/Phi-4-multimodal-instruct",
        arch=VisionArchitecture.Phi4MM,
    ),
)

IMAGE_URL = "https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg"
AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg"

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": AUDIO_URL}},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {
                        "type": "text",
                        "text": "<|audio_1|><|image_1|> Describe in detail what is happening, referencing both what you hear and what you see.",
                    },
                ],
            }
        ],
        max_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )
)

print(response.choices[0].message.content)
```

See [`examples/python/phi4mm_audio.py`](../examples/python/phi4mm_audio.py) for a ready-to-run version.

Please find [API docs here](API.md) and the type stubs [here](mistralrs.pyi), which are another great form of documentation.

We also provide [a cookbook here](../examples/python/cookbook.ipynb)!

## Multi-model and Model Management

For serving multiple models and dynamic model management (unloading/reloading), see the [multi-model documentation](API.md#multi-model-support).
