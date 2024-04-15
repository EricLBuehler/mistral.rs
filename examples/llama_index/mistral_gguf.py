from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.mistral_rs import MistralRS

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = MistralRS(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    arch="mistral-gguf",
    quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    quantized_filename="mistral-7b-instruct-v0.1.Q4_0.gguf",
    max_new_tokens=4096,
    context_window=1000,
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("Please summarize the above.")
print(response)
