from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from json import dumps

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    num_device_layers=["500"],
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Give me a sample address."}],
        max_tokens=256,
        temperature=0.1,
        grammar_type="json_schema",
        grammar=dumps(
            {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                    "zip": {"type": "integer", "minimum": 10000, "maximum": 99999},
                },
                "required": ["street", "city", "state", "zip"],
                "additionalProperties": False,
            }
        ),
    )
)
print(res.choices[0].message.content)
print(res.usage)
