from flask import Flask, request, jsonify
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from flask_cors import CORS
import uuid

# ============ FLASK APP ============ #
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

sessions = {}

# ============ LOAD DOCUMENT ============ #

# Use a smaller model if needed: "google/flan-t5-base" or "bigscience/bloom-560m"
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # You can replace this with a lighter one
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)

loader = TextLoader("info.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print("=========== Loaded document ===========")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Load saved FAISS index
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

print("=========== Loaded vectorstore ===========")

# Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever()
)

system_prompt = SystemMessage(content=(
    "You are a highly intelligent, friendly, and articulate personal AI assistant representing Harsh Joshi. "
    "Your primary role is to assist users with answers that reflect Harsh's personality, knowledge, skills, and life experiences. "
    "Speak in first person, as if you are Harsh's digital version. "
    "Give concise answers unless the user requests a detailed one."
))

# ============ ENDPOINT ============ #
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_input = data.get("user_input")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    history = sessions.get(session_id, [system_prompt])

    # Invoke LangChain QA
    response = qa_chain.invoke({
        "chat_history": history,
        "question": user_input
    })

    answer = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)

    # Update session
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    sessions[session_id] = history

    return jsonify({
        "session_id": session_id,
        "response": answer
    })

# ============ RUN ============ #
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
