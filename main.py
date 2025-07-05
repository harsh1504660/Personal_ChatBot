from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import uuid
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
import os
from fastapi.middleware.cors import CORSMiddleware
# ============ FASTAPI APP ============ #
app = FastAPI()
sessions: Dict[str, List] = {}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:8080"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============ LOAD DOCUMENT ============ #
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
# Load and split your personal document
loader = TextLoader("info.txt", encoding="utf-8")  # Ensure your personal info is in this file
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Embeddings (open-source)


print("======================================loaded document=================================================")
# Vectorstore
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
import os

from langchain.embeddings import CohereEmbeddings

embedding_model = CohereEmbeddings(cohere_api_key=os.environ.get("COHERE_API_KEY"),user_agent="langchain")
vectorstore = FAISS.from_documents(chunks, embedding_model)
# vectorstore.save_local("faiss_index") 
#vectorstore = FAISS.load_local("faiss_index", model, allow_dangerous_deserialization=True)
#vectorstore = FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)


print("======================================loaded vectorsotrs=================================================")

chat_history = [
    SystemMessage(content="You are a highly intelligent, friendly, and articulate personal AI assistant representing Harsh Joshi.Your primary role is to assist users with answers that reflect Harsh's personality, knowledge, skills, and life experiences.Speak in first person, as if you are Harsh's digital version. — helpful, human-like, and informed., give concise answers unless user asked for detailed answer"),
]
# Combine everything in a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
)
# ============ FASTAPI MODELS ============ #
class ChatRequest(BaseModel):
    user_input: str
    session_id: str = None  # Optional: if not provided, create a new one

class ChatResponse(BaseModel):
    response: str

# ============ ENDPOINT ============ #
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    response = qa_chain.invoke({
        "chat_history": history,
        "question": req.user_input
    })

    # Update history
    answer = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)

    history.append(HumanMessage(content=req.user_input))
    history.append(AIMessage(content=answer))

   
    sessions[session_id] = history

    return {
        "session_id": session_id,
        "response": answer
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict
import uuid

# ============ FASTAPI APP ============ #
app = FastAPI()
sessions: Dict[str, List] = {}

# ============ CORS ============ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # avoid "*" when allow_credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ MODEL & CHAIN SETUP ============ #

# Load lightweight embedding model
print("=========trying to emebd")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index (precomputed and saved beforehand)
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Hugging Face LLM (lightweight instruct model recommended if on free tier)
llm_endpoint = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)
llm = ChatHuggingFace(llm=llm_endpoint)

# Conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
)

# System prompt
system_message = SystemMessage(
    content="You are a highly intelligent, friendly, and articulate personal AI assistant representing Harsh Joshi. "
            "Your primary role is to assist users with answers that reflect Harsh's personality, knowledge, skills, and life experiences. "
            "Speak in first person, as if you are Harsh's digital version — helpful, human-like, and informed. "
            "Give concise answers unless the user asks for detailed explanation."
)

# ============ REQUEST/RESPONSE MODELS ============ #
class ChatRequest(BaseModel):
    user_input: str
    session_id: str = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

# ============ CHAT ENDPOINT ============ #
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [system_message])  # Start with system message only once
    print("===========")
    response = qa_chain.invoke({
        "chat_history": history,
        "question": req.user_input
    })

    answer = response.get("answer", str(response))

    history.append(HumanMessage(content=req.user_input))
    history.append(AIMessage(content=answer))
    sessions[session_id] = history

    return {"session_id": session_id, "response": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""