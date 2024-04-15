from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.mistral_rs import MistralRS

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = MistralRS(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    arch="x-lora-mistral",
    xlora_order_file="ordering.json",  # Copy your ordering file to where this file will be run from
    xlora_model_id="lamm-mit/x-lora",
    max_new_tokens=4096,
    context_window=1000,
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("Please summarize the above.")
print(response)
