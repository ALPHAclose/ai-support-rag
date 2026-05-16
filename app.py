import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Support Assistant")

st.markdown("Ask questions from uploaded documents")

DB_PATH = "vectordb"

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load database
db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM
@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant"
    )

    return retriever, llm


retriever, llm = load_rag()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask your question")

if question:

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve docs
    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
You are a helpful AI support assistant.

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

    answer = response.content

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)