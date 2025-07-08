from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

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

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "If all dogs are mammals, and all mammals are animals, are dogs animals?",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={
        "grammar": {
            "type": "llguidance",
            "value": {"grammars": grammars},
        }
    },
)

print(completion.choices[0].message.content)
