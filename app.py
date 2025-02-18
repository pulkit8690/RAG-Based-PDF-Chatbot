from flask import Flask, render_template, request, jsonify
import os
import json
import shutil
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
USER_SESSION_FILE = "user_sessions.json"

# Load LLM model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    llm = OllamaLLM(model="llama3.2", temperature=0.5, device=device)
except Exception as e:
    print(f"Error initializing Ollama: {e}")
    exit()

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2",
    model_kwargs={"device": device}
)

retriever = None  # Global retriever variable

# Load user session history from file
def load_user_sessions():
    if os.path.exists(USER_SESSION_FILE):
        with open(USER_SESSION_FILE, "r") as file:
            return json.load(file)
    return {}

# Save user session history to file
def save_user_sessions(user_sessions):
    with open(USER_SESSION_FILE, "w") as file:
        json.dump(user_sessions, file, indent=4)

user_sessions = load_user_sessions()

def process_uploaded_pdfs():
    """Loads and processes all uploaded PDFs to create a retriever."""
    global retriever
    documents = []
    for file_name in os.listdir(UPLOAD_FOLDER):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(UPLOAD_FOLDER, file_name)
            try:
                loader = PDFPlumberLoader(pdf_path)
                documents.extend(loader.load())
                print(f"Loaded PDF: {file_name}")
            except Exception as e:
                print(f"Error loading PDF {file_name}: {e}")

    if not documents:
        retriever = None  # Reset retriever if no documents are found
        return None  

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    if not texts:
        retriever = None
        return None  

    print(f"Total number of chunks created: {len(texts)}")

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    # Remove PDFs after processing
    delete_uploaded_pdfs()

    return retriever

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles PDF uploads and updates retriever immediately."""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({"error": "No selected files"}), 400

    uploaded_files = []
    for file in files:
        if file.filename == '':
            continue
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_files.append(file_path)
    
    if not uploaded_files:
        return jsonify({"error": "No valid PDF files uploaded"}), 400

    process_uploaded_pdfs()
    return jsonify({"message": "Files uploaded successfully and processed. PDFs deleted after processing."})

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests using the latest uploaded PDFs with user history context."""
    global retriever
    data = request.get_json()
    user_id = data.get('user_id', '').strip()
    query = data.get('query', '').strip()
    
    if not user_id:
        return jsonify({"error": "User ID cannot be empty"}), 400

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    if retriever is None:
        return jsonify({"error": "No uploaded PDFs found. Please upload PDFs first."}), 400

    retrieved_docs = retriever.invoke(query)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    if not retrieved_texts:
        return jsonify({"error": "No relevant information found in the uploaded PDFs."}), 400

    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": []}

    user_history = user_sessions[user_id]["history"]

    history_context = "\n".join(
        [f"User: {item['query']}\nAssistant: {item['response']}" for item in user_history[-5:]]
    )

    document_context = "\n\n".join(retrieved_texts[:4])

    if history_context:
        prompt = f"""You are an AI assistant that can answer questions using provided documents and general knowledge. 
If the user asks about something covered in the documents, STRICTLY use the given documents to answer concisely in at most five sentences.
If the question is general and does not require document references, provide a natural response based on general knowledge.
If the question is outside the scope of the documents (like famous personalities or general facts), clearly state "The uploaded documents do not contain this information."

Conversation history:
{history_context}

Documents:
{document_context}

Current query:
User: {query}
Assistant:"""
    else:
        prompt = f"""You are an AI assistant that can answer questions using provided documents and general knowledge. 
If the user asks about something covered in the documents, STRICTLY use the given documents to answer concisely in at most five sentences.
If the question is general and does not require document references, provide a natural response based on general knowledge.
If the question is outside the scope of the documents (like famous personalities or general facts), clearly state "The uploaded documents do not contain this information."

Documents:
{document_context}

User: {query}
Assistant:"""

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa.invoke({"query": prompt})["result"]

    user_history.append({"query": query, "response": response})
    if len(user_history) > 5:
        user_history.pop(0)
    save_user_sessions(user_sessions)

    return jsonify({"query": query, "response": response})

def delete_uploaded_pdfs():
    """Deletes all uploaded PDFs without affecting chat history."""
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/reset', methods=['POST'])
def reset():
    """Resets only the uploaded PDFs, keeping chat history intact."""
    delete_uploaded_pdfs()
    return jsonify({"message": "All uploaded PDFs have been deleted successfully."})

if __name__ == '__main__':
    app.run(debug=True)
