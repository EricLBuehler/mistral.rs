import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3"
)

res = tokenizer.apply_chat_template(
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Who are you"},
        {"role": "assistant", "content": "   I am an assistant   "},
        {"role": "user", "content": "Another question"},
    ],
    add_generation_prompt=True,
    tokenize=False,
)
print(res.replace("\n", "\\n"))
