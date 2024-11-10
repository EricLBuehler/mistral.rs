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
    in_situ_quant="Q4K",
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
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

Please find [API docs here](API.md) and the type stubs [here](mistralrs.pyi), which are another great form of documentation.

We also provide [a cookbook here](../examples/python/cookbook.ipynb)!
