from mistralrs import Runner, Which, ChatCompletionRequest, Architecture
from json import dumps

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    num_device_layers=["500"],
)

# see lark_llg.py for a better way of dealing with JSON and Lark together

json_lark = r"""
start: object # we only want objects

value: object
     | array
     | STRING
     | NUMBER
     | "true"
     | "false"
     | "null"

object: "{" [pair ("," pair)*] "}"
pair: STRING ":" value

array: "[" [value ("," value)*] "]"

STRING: /"(\\.|[^"\\])*"/
NUMBER: /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/

%import common.WS
%ignore WS
"""

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Give me a sample address."}],
        max_tokens=30,
        temperature=0.1,
        grammar_type="lark",
        grammar=json_lark,
    )
)
print(res.choices[0].message.content)
print(res.usage)
