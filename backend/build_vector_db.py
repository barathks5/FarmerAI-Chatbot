import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading documents...")

documents = []
for file in os.listdir("data/documents"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data/documents", file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local("models/vector_db")

print("✅ Vector database created successfully!")
