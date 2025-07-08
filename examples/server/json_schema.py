from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

addr_schema = {
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

completion = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": "Gimme a sample address.",
        }
    ],
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "My Schema", "schema": addr_schema},
    },
)

print(completion.choices[0].message.content)
