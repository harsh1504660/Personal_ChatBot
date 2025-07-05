from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Set your OpenRouter API key

# Use OpenRouter with mistral
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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Vectorstore
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Open-source LLM via Ollama (e.g., Mistral, Phi, etc.)
#llm = Ollama(model="mistral")  # Ensure it's downloaded locally via `ollama run mistral`

# Prompt defining chatbot’s role
custom_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
Pretend that you are harsh joshi, users will ask you about harsh and answer them using the document. Greet with properly, answer in short unless they ask to asnwer in detaild, sometimes suggest some question, and if you dont know the answer they humorsly say that i dont know the answer  also use the chat history for reference. 
.

Chat History:
{context}

Question:
{question}

Respond helpfully and informatively based on Harsh's personal details.
"""
)

# Memory buffer to maintain conversation context
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
chat_history = [
    SystemMessage(content="You are a highly intelligent, friendly, and articulate personal AI assistant representing Harsh Joshi.Your primary role is to assist users with answers that reflect Harsh's personality, knowledge, skills, and life experiences.Act as Harsh's digital version — helpful, human-like, and informed."),
]
# Combine everything in a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vectorstore.as_retriever(),
)

# CLI loop
query = ""
print("🤖 Personal Chatbot is ready (type 'exit' to quit)\n")
while query != "exit":
    query = input("You: ")

    # Format history for the chain
    history_for_chain = [
        (msg.content, chat_history[i + 1].content)
        for i, msg in enumerate(chat_history)
        if isinstance(msg, HumanMessage) and i + 1 < len(chat_history) and isinstance(chat_history[i + 1], AIMessage)
    ]

    # Run chain
    result = qa_chain.invoke({
        "question": query,
        "chat_history": history_for_chain
    })

    # Print answer
    print("AI:", result["answer"])

    # Update chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))
