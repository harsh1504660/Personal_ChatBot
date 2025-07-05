from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import uuid
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
import os
from fastapi.middleware.cors import CORSMiddleware
# ============ FASTAPI APP ============ #
app = FastAPI()
sessions: Dict[str, List] = {}



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("======================================loaded document=================================================")
# Vectorstore

# vectorstore = FAISS.from_documents(chunks, embedding_model)
# vectorstore.save_local("faiss_index") 
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)


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
    uvicorn.run(app,)