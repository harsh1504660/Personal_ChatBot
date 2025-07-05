from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
import os

# Constants
HUGGINGFACE_TOKEN = os.environ.get("")  # You must set this!
LLAMA_ENDPOINT = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8b-instruct"

# Load model, index, and docs
embed_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
index = faiss.read_index("faiss_index.index")
with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str

# LLaMA Query
def query_llama(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(LLAMA_ENDPOINT, headers=headers, json=payload)
    return response.json()[0]["generated_text"] if response.ok else "LLM error"

# Endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    query_embedding = embed_model.encode([req.user_input])
    D, I = index.search(query_embedding, k=3)
    retrieved = [docs[i] for i in I[0] if i < len(docs)]
    context = "\n\n".join(retrieved)

    prompt = f"""You are Harsh Joshi's AI assistant. Use the context below to answer the question.

Context:
{context}

Question: {req.user_input}
Answer as Harsh in first person:"""

    answer = query_llama(prompt)
    return {"response": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,)