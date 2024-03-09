from transformers import AutoTokenizer

model = input("Enter model ID: ")
tok = AutoTokenizer.from_pretrained(model)
tok.save_pretrained(".")
