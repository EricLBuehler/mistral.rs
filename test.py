from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
print(tok.apply_chat_template([{"role":"user","content":"Hi"}], tokenize=False, add_generation_prompt=True).replace("\n","\\n"))