from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

DATA_PATH = "data"
DB_PATH = "vectordb"

documents = []

# Load PDFs
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(DATA_PATH, file)

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Add filename metadata
        for doc in docs:
            doc.metadata["source"] = file

        documents.extend(docs)

print(f"Loaded {len(documents)} pages")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Free HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("Vector database created successfully!")