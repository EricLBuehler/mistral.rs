from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.mistral_rs import MistralRS
from mistralrs import Which
import sys

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

Settings.llm = MistralRS(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        tokenizer_json=None,
        repeat_last_n=64,
    ),
    max_new_tokens=4096,
    context_window=1024 * 5,
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What are non-granular scalings?")
for text in response.response_gen:
    print(text, end="")
    sys.stdout.flush()
