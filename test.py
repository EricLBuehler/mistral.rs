import numpy as np
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "What is graphene?"
embeddings = model.encode_query(query)
# (768,)
np.save("embeddings.npy", embeddings)
