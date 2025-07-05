from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and split the document
with open("info.txt", "r", encoding="utf-8") as f:
    content = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_text(content)

# Load small embedding model
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# Embed chunks
embeddings = model.encode(docs, convert_to_numpy=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and docs
faiss.write_index(index, "faiss_index.index")
with open("docs.pkl", "wb") as f:
    pickle.dump(docs, f)
