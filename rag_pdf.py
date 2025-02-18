## RAG model : LLAMA3.2 LLM model used to chat with pdf and user history stored in Json File.

import os
import json
import torch
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress TensorFlow logs/
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


SESSION_FILE = "user_sessions.json"


def save_sessions():
    with open(SESSION_FILE, "w") as f:
        json.dump(user_sessions, f)
    print("User sessions saved.")


def load_sessions():
    global user_sessions
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r") as f:
            user_sessions = json.load(f)
        print("User sessions loaded.")
    else:
        user_sessions = {}

load_sessions()


device = "cuda" if torch.cuda.is_available() else "cpu"


try:
    llm = OllamaLLM(
        model="llama3.2",  
        temperature=0.5,  
        device=device  # Use GPU for LLM if available
    )
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    exit()


def load_pdfs_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                loader = PDFPlumberLoader(pdf_path)
                documents.extend(loader.load())
                print(f"Loaded PDF: {file_name}")
            except Exception as e:
                print(f"Error loading PDF {file_name}: {e}")
    return documents


pdf_folder_path = r"C:/Github/RAG-Based-PDF-Chatbot\pdfs"  ## folder path of pdfs


documents = load_pdfs_from_folder(pdf_folder_path)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
texts = text_splitter.split_documents(documents)


print(f"Total number of chunks created: {len(texts)}\n")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": device}  # Use GPU if available
)


db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  
    retriever=retriever
)


def get_response(user_id, query):
    try:
        if user_id not in user_sessions:
            user_sessions[user_id] = {"history": []}

        user_history = user_sessions[user_id]["history"]

        retrieved_docs = retriever.invoke(query)

        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        if not retrieved_texts:
            return "I can only answer questions based on the uploaded documents. No relevant information was found for your query."

        
        filtered_texts = [text for text in retrieved_texts if any(word in text.lower() for word in query.lower().split())]

        if not filtered_texts:
            return "I can only answer questions based on the uploaded HR documents. No relevant information was found for your query."

        limited_texts = filtered_texts[:4]  

        document_context = "\n\n".join(limited_texts)

        history_context = "\n".join(
            [f"User: {item['query']}\nAssistant: {item['response']}" for item in user_history[-5:]]
        )

        
        if history_context:
            prompt = f"You are an assistant for answering the questions asked by user, from the provided documents. STRICTLY Use the given documents to answer the question. Use atmost five sentences  and keep the answer concise. If asked a question outside the stored data or out of context say you don't know.Some questions may be asked related to the context of previous questions, so in such cases remember the user history.\n\nConversation history:\n{history_context}\n\nDocuments:\n{document_context}\n\nCurrent query:\nUser: {query}\nAssistant:"
        else:
            prompt = f"You are an assistant for question-answering tasks. STRICTLY Use the given documents to answer the question. Use atmost five sentences and keep the answer concise. If asked a question outside the stored data or out of context say you don't know.\n\nDocuments:\n{document_context}\n\nUser: {query}\nAssistant:"

        result = qa.invoke({"query": prompt})

        response = result["result"]

        user_history.append({"query": query, "response": response})

        if len(user_history) > 5:
            user_history.pop(0)  

        return response
    except Exception as e:
        return f"Error during QA: {e}"


# Main Loop for user queries
if __name__ == "__main__":
    while True:
        user_id = input("Enter your user ID (or 'exit' to terminate): ")
        if user_id.lower() == 'exit':
            save_sessions()  
            print("Program terminated. Goodbye!")
            break
        
        while True:
            query = input(f"Enter your question, {user_id} (or 'quit' to switch users): ")
            if query.lower() == 'quit':  
                print(f"Session ended for user {user_id}.\n")
                break
            response = get_response(user_id, query)
            print("\nResponse:", response, "\n")