from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DB_PATH = "vectordb"

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

print("Chatbot Ready! Type 'exit' to quit.\n")

while True:
    question = input("Ask Question: ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
You are a helpful support assistant.

Answer ONLY using the provided context.

If answer is not found, say:
"I could not find this information in the documents."

Context:
{context}

Question:
{question}

Provide:
- clear answer
- source filename
- page number
"""

    response = llm.invoke(prompt)

    print("\nANSWER:\n")
    print(response.content)

    print("\n" + "="*50 + "\n")