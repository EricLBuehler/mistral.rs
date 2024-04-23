from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.mistral_rs import MistralRS
from mistralrs import Which

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

Settings.llm = MistralRS(
    which=Which.XLoraGGUF(
        tok_model_id=None,  # Automatically determine from ordering file
        quantized_model_id="TheBloke/zephyr-7B-beta-GGUF",
        quantized_filename="zephyr-7b-beta.Q4_0.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
        xlora_model_id="lamm-mit/x-lora",
        order="../orderings/xlora-paper-ordering.json",
        tgt_non_granular_index=None,
    ),
    max_new_tokens=4096,
    context_window=1024 * 5,
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("How do I pronounce graphene?")
print(response)
