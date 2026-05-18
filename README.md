---
title: AI Support Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🤖 AI Support Assistant (RAG)

## 📌 Project Overview

AI Support Assistant is a Retrieval-Augmented Generation (RAG) chatbot that answers user questions using uploaded PDF documents.

The system uses:
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM
- Streamlit

If the AI cannot find an answer in the documents, the user can create a support ticket.

---

# 🚀 Features

✅ Multi-document PDF support

✅ Semantic search using vector embeddings

✅ Source citation with page numbers

✅ Chat history

✅ Hallucination prevention

✅ Support ticket creation workflow

✅ Persistent ticket storage using JSON

✅ Streamlit web interface

---

# 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| LangChain | RAG pipeline |
| ChromaDB | Vector database |
| HuggingFace Embeddings | Semantic embeddings |
| Groq API | Large Language Model |
| Streamlit | Web interface |
| PyPDF | PDF processing |

---

# 📂 Project Structure

```bash
ai-support-rag/
│
├── data/
│   ├── 2022-toyota-hilux.pdf
│   ├── T-MMS-25Crown.pdf
│   └── TLCL-25.12.pdf
│
├── vectordb/
│
├── src/
│   ├── ingest.py
│   └── chatbot.py
│
├── app.py
├── tickets.json
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-support-rag.git
cd ai-support-rag
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Create Environment File

Create `.env`

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 4. Add PDF Files

Place PDFs inside:

```bash
data/
```

---

## 5. Create Vector Database

```bash
python src/ingest.py
```

---

## 6. Run Streamlit App

```bash
streamlit run app.py
```

---

# 💡 Example Questions

```text
How do I maintain Toyota Hilux?
```

```text
How do I use Toyota multimedia system?
```

```text
What does the ls command do?
```

---

# 🎫 Support Ticket Workflow

If the AI cannot find an answer in the uploaded documents:

1. The assistant notifies the user.
2. A support ticket form appears.
3. User enters:
   - Name
   - Email
   - Issue description
4. Ticket is saved into:

```bash
tickets.json
```

---

# 📸 Screenshots

Add screenshots of:
- Main chatbot UI
- Source citations
- Support ticket form
- Successful ticket creation

---

# 🔍 RAG Workflow

1. PDFs are loaded using PyPDFLoader.
2. Documents are split into chunks.
3. HuggingFace embeddings are generated.
4. Chunks are stored in ChromaDB.
5. User question is embedded.
6. Relevant chunks are retrieved.
7. Groq LLM generates grounded response.
8. Sources and page numbers are displayed.

---

# 🌐 Deployment

This project can be deployed on:
- HuggingFace Spaces
- Streamlit Community Cloud

---

# 👨‍💻 Author

Alpha

---

# 📄 License

Educational project for academic purposes.

