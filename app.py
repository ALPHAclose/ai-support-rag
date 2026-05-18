import streamlit as st
import json

import os

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Support Assistant")
st.markdown("Ask questions from uploaded documents")

with st.sidebar:

    st.header("🏢 Company Info")

    st.write("**Company:** AI Support Assistant")
    st.write("**Support Email:** support@aiassistant.com")
    st.write("**Phone:** +998 90 123 45 67")
    st.write("**Working Hours:** 9 AM - 6 PM")

    st.divider()

    st.write("### 📚 Loaded Documents")
    st.write("- Toyota Hilux Manual")
    st.write("- Toyota Multimedia System")
    st.write("- Linux Command Guide")

DB_PATH = "vectordb"


# -----------------------------------
# Session State
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_ticket_form" not in st.session_state:
    st.session_state.show_ticket_form = False


# -----------------------------------
# Create Ticket Function
# -----------------------------------
def create_ticket(name, email, issue):

    ticket = {
        "name": name,
        "email": email,
        "issue": issue
    }

    with open("tickets.json", "r") as file:
        tickets = json.load(file)

    tickets.append(ticket)

    with open("tickets.json", "w") as file:
        json.dump(tickets, file, indent=4)

    return "✅ Support ticket created successfully!"


# -----------------------------------
# Load RAG
# -----------------------------------
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


# -----------------------------------
# Display Chat History
# -----------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------------
# Ticket Counter
# -----------------------------------
try:
    with open("tickets.json", "r") as file:
        tickets = json.load(file)

    st.info(f"🎫 Total Support Tickets: {len(tickets)}")

except:
    st.info("🎫 Total Support Tickets: 0")

# -----------------------------------
# User Input
# -----------------------------------
question = st.chat_input("Ask your question")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)

    context = "\n\n".join([
        f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""
You are a helpful AI support assistant.

Answer ONLY using the provided context.

If the answer is not found in the documents, say:
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

    with st.spinner("Thinking..."):

        response = llm.invoke(prompt)

        answer = response.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    no_answer = "could not find" in answer.lower()

    if no_answer:
        st.session_state.show_ticket_form = True
    else:
        st.session_state.show_ticket_form = False

    st.rerun()


# -----------------------------------
# Show Ticket Form
# -----------------------------------
if st.session_state.show_ticket_form:

    st.warning("Would you like to create a support ticket?")

    with st.form("ticket_form"):

        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        issue = st.text_area("Describe your issue")

        submitted = st.form_submit_button("Create Ticket")

        if submitted:

            if name and email and issue:

                result = create_ticket(name, email, issue)

                st.success(result)

                st.session_state.show_ticket_form = False

            else:

                st.error("Please fill all fields.")