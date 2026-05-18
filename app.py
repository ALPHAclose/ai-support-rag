import streamlit as st
import json
import os

from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #050816 0%,
        #0b1120 50%,
        #111827 100%
    );
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Main Title */
.main-title {
    font-size: 56px;
    font-weight: 800;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Ticket Box */
.ticket-box {
    background: linear-gradient(
        135deg,
        rgba(59,130,246,0.18),
        rgba(168,85,247,0.18)
    );
    padding: 18px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
    font-size: 18px;
    font-weight: 600;
    color: white;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 20px;
    padding: 12px;
    margin-bottom: 15px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 14px;
    background: linear-gradient(90deg,#3b82f6,#8b5cf6);
    color: white;
    border: none;
    padding: 12px;
    font-weight: 700;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# HEADER
# -----------------------------------
st.markdown(
    '<div class="main-title">🤖 AI Support Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Enterprise RAG-powered support assistant with semantic document retrieval</div>',
    unsafe_allow_html=True
)


# -----------------------------------
# SIDEBAR
# -----------------------------------
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
# SESSION STATE
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_ticket_form" not in st.session_state:
    st.session_state.show_ticket_form = False


# -----------------------------------
# CREATE TICKET
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
# LOAD RAG
# -----------------------------------
@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector DB if missing
    if not os.path.exists(DB_PATH):

        documents = []

        for file in os.listdir("data"):

            if file.endswith(".pdf"):

                loader = PyPDFLoader(f"data/{file}")

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = file

                documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )

    else:

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
# DISPLAY CHAT HISTORY
# -----------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------------
# TICKET COUNTER
# -----------------------------------
try:

    with open("tickets.json", "r") as file:
        tickets = json.load(file)

    st.markdown(
        f'''
        <div class="ticket-box">
            🎫 Total Support Tickets: {len(tickets)}
        </div>
        ''',
        unsafe_allow_html=True
    )

except:

    st.markdown(
        '''
        <div class="ticket-box">
            🎫 Total Support Tickets: 0
        </div>
        ''',
        unsafe_allow_html=True
    )


# -----------------------------------
# USER INPUT
# -----------------------------------
question = st.chat_input("Ask your question")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

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
# SHOW TICKET FORM
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