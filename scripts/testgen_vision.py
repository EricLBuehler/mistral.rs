import transformers

processor = transformers.AutoProcessor.from_pretrained(
    "microsoft/Phi-3.5-vision-instruct", trust_remote_code=True
)

res = processor.tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant"}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Hello, please describe the above.",
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "This is me, who are you"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "   I am an assistant   "}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Another question, what is this?"},
            ],
        },
    ],
    add_generation_prompt=True,
    tokenize=False,
)
print(res.replace("\n", "\\n"))
