from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from json import dumps

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    num_device_layers=["500"],
)

# In fact, JSON object can be also defined in the grammar itself, see
# lark_llg.py and https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md#inline-json-schemas

# @myobj will reference the JSON schema defined below (see grammars = [ ... ])
top_lark = r"""
start: "Reasoning: " /.+/ "\nJSON: " @myobj
"""

answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["Yes", "No"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

grammars = [
    {"lark_grammar": top_lark},
    {"name": "myobj", "json_schema": answer_schema},
]

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {
                "role": "user",
                "content": "If all dogs are mammals, and all mammals are animals, are dogs animals?",
            }
        ],
        max_tokens=30,
        temperature=0.1,
        grammar_type="llguidance",
        grammar=dumps({"grammars": grammars}),
    )
)
print(res.choices[0].message.content)
print(res.usage)
