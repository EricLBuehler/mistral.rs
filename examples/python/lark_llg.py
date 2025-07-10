from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from json import dumps

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    num_device_layers=["500"],
)

# see https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md for docs on syntax

top_lark = r"""
start: "Reasoning: " /.+/ "\nJSON: " answer
answer: %json {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No"]}
    },
    "required": ["answer"],
    "additionalProperties": false
}
"""


res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "If all dogs are mammals, and all mammals are animals, are dogs animals?",
            }
        ],
        max_tokens=100,
        temperature=0.1,
        grammar_type="lark",
        grammar=top_lark,
    )
)
print(res.choices[0].message.content)
print(res.usage)
