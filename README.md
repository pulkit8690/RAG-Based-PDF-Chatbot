# 📝 RAG-Based PDF Chatbot Using LLAMA3.2

This project implements a **Retrieval-Augmented Generation (RAG)** model using the **LLAMA3.2** large language model (LLM) to enable conversational interactions with PDFs. It maintains user chat history in a JSON file for personalized responses. The system retrieves relevant document chunks using **FAISS** vector search and **HuggingFace embeddings**.

---

## 🚀 Features
✅ **Chat with PDFs** – The chatbot retrieves answers from uploaded PDFs using RAG.  
✅ **Persistent User History** – Stores chat history in a JSON file to maintain session continuity.  
✅ **Efficient Search with FAISS** – Uses **FAISS (Facebook AI Similarity Search)** for document retrieval.  
✅ **LLAMA3.2 for Response Generation** – Utilizes **Ollama LLM** for generating responses.  
✅ **GPU Acceleration** – Automatically detects and uses GPU if available for faster processing.  

---

## 📂 Project Structure

```
📂 RAG-Based-PDF-Chatbot/
│-- 📄 main.py                    # Main script for chatbot interaction
│-- 📂 pdf/                       # Folder to store PDFs
│-- 📂 embeddings/                # FAISS vector storage
│-- 📄 user_sessions.json         # User chat history JSON
│-- 📄 requirements.txt           # Required dependencies
│-- 📄 README.md                  # Project documentation
```

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/pulkit8690/RAG-Based-PDF-Chatbot.git
cd RAG-Based-PDF-Chatbot
```

### 2️⃣ Install Dependencies
Make sure you have **Python 3.8+** installed. Then, install the required dependencies:

```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up the LLAMA3.2 Model
Ensure **Ollama** is installed and configure the model:

```python
from ollama import OllamaLLM
import torch

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.5,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

---

## 💬 Usage

### Running the Chatbot
```sh
python main.py
```

### Interactive Chat
```sh
Enter your user ID (or 'exit' to terminate): user123
Enter your question, user123 (or 'quit' to switch users): What is the document about?
Response: The document discusses...
```

---

## 📌 Notes
- If **no relevant information** is found in the uploaded PDFs, the chatbot will inform the user accordingly.
- The chatbot **only answers based on the uploaded documents**; it does not generate responses beyond the given context.
- User history is stored in **`user_sessions.json`** for context-aware conversations.

---

## 🤖 Tech Stack

| Component      | Technology Used |
|---------------|----------------|
| **LLM**       | LLAMA3.2 via Ollama |
| **Embeddings** | sentence-transformers/all-MiniLM-L12-v2 |
| **Retrieval**  | FAISS Vector Database |
| **PDF Processing** | pdfplumber |
| **Backend**    | Python, LangChain |

---

## 🛠 Future Enhancements
- ✅ Implement **multi-document retrieval** for broader context.
- ✅ Add **web-based UI** using **Streamlit or Gradio**.
- ✅ Support **additional document formats (Word, TXT, etc.)**.
- ✅ Improve **response ranking with hybrid search** (BM25 + FAISS).

---

## 📜 License
This project is **open-source** and available under the [MIT License](LICENSE).

---

### 🌟 Contributions Welcome!
Feel free to **fork, contribute, or suggest improvements** via pull requests!

🚀 **Happy Coding!**
```
