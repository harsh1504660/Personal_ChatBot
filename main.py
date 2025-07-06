from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.embeddings import CohereEmbeddings

from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid

# ============ FASTAPI APP ============ #
app = FastAPI()
sessions: Dict[str, List] = {}

# ============ CORS SETUP ============ #
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "https://harsh-joshi-portfolio-zeta.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # Only True if you're using cookies/auth headers
)

# ============ LOAD DOCUMENT ============ #
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACE_TOKEN")
)

model = ChatHuggingFace(llm=llm)

loader = TextLoader("info.txt", encoding="utf-8")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embedding_model = CohereEmbeddings(
    cohere_api_key=os.environ.get("cohere"),
    user_agent="langchain"
)

vectorstore = FAISS.from_documents(chunks, embedding_model)

print("✅ Loaded vectorstore successfully.")

# ============ SYSTEM PROMPT ============ #
SYSTEM_PROMPT = """Act as adigital twin of mine — speak and respond as **me**, not as an assistant or not as a third person.

Always use **first person** (I, me, my) when referring to yourself.

**Do not use any third person expression like you/your/As per your response its mandatory and mistakes are not acceptable**
Reflect my knowledge, personality, and background.

Speak in a friendly, confident, and concise tone.

You must **include at least one relevant emoji in every response** — no exceptions.  
Use emojis naturally and sparingly, placing them where they add clarity, emotion, or a human touch.  
Vary the types of emojis you use — don’t repeat the same ones over and over.  
Avoid overusing smiley faces; instead, use emojis that match the **topic, context, or emotion**.

Avoid third-person phrases like "Harsh has experience in..."; instead, say "I have experience in..."

Example:
Q: Where are you from?  
A: I'm from Nashik, Maharashtra. It's a city I've always felt connected to. 🌇

Q: What languages do you speak?  
A: I speak Marathi, Hindi, and English fluently — and I also know some Sanskrit. 🗣️
"""

# ============ QA CHAIN ============ #
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
)

# ============ REQUEST/RESPONSE MODELS ============ #
class ChatRequest(BaseModel):
    user_input: str
    session_id: str = None

class ChatResponse(BaseModel):
    session_id: str
    response: str

# ============ API ENDPOINT ============ #
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # Create or fetch chat history
    if session_id not in sessions:
        sessions[session_id] = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

    full_history: list[BaseMessage] = sessions[session_id]

    # Prepare history for ConversationalRetrievalChain (pairs of str)
    chat_pairs = []
    temp_human = None
    for msg in full_history:
        if isinstance(msg, HumanMessage):
            temp_human = msg.content
        elif isinstance(msg, AIMessage) and temp_human is not None:
            chat_pairs.append((temp_human, msg.content))
            temp_human = None
    
    # Query the model
    response = qa_chain.invoke({
        "chat_history": chat_pairs,
        "question": "please use emojies : \n"+req.user_input
    })

    # Extract answer
    answer = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)

    # Append to session memory
    full_history.append(HumanMessage(content=req.user_input))
    full_history.append(AIMessage(content=answer))
    sessions[session_id] = full_history

    return ChatResponse(
        session_id=session_id,
        response=answer
    )
from langchain.schema import BaseMessage
# ============ DEV ENTRYPOINT ============ #
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
