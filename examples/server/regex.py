from openai import OpenAI

client = OpenAI(api_key="foobar", base_url="http://localhost:1234/v1/")

BULLET_LIST_REGEX = "(- [^\n]*\n)+(- [^\n]*)(\n\n)?"

completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Write a list of jokes. Return a markdown list where each item is a joke.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "regex", "value": BULLET_LIST_REGEX}},
)

print(completion.choices[0].message.content)

print("---")

# The following does token healing. Prompting the model to continue after a space usually breaks
# the text because the model wants to start the new token with a space. By setting the a space after
# "Sure!" we guarantee a space after "Sure!" but we haven't forced which token that starts with space should be used yet.

completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "regex", "value": "Sure! (?s:.)*"}},
)

print(completion.choices[0].message.content)
