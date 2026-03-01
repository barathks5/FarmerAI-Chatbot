import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DATA_PATH = "data/documents"
VECTOR_PATH = "models/vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("\n📄 Loading documents...\n")

# -------------------------------------------------
# CHECK FOLDER EXISTS
# -------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Folder not found: {DATA_PATH}")

documents = []

# -------------------------------------------------
# LOAD PDFs WITH METADATA
# -------------------------------------------------
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        file_path = os.path.join(DATA_PATH, file)

        print(f"🔹 Reading: {file}")

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Add source metadata
        for page in pages:
            page.metadata["source"] = file

        documents.extend(pages)

if not documents:
    raise ValueError("❌ No PDF files found in data/documents")

print(f"\n✅ Loaded {len(documents)} pages")

# -------------------------------------------------
# SPLIT INTO CHUNKS
# -------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"✂️ Created {len(chunks)} chunks")

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------
print("\n🧠 Generating embeddings...\n")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL
)

# -------------------------------------------------
# DELETE OLD VECTOR DB (OPTIONAL RESET)
# -------------------------------------------------
if os.path.exists(VECTOR_PATH):
    print("♻️ Removing old vector database...")
    shutil.rmtree(VECTOR_PATH)

# -------------------------------------------------
# CREATE FAISS VECTOR STORE
# -------------------------------------------------
vector_db = FAISS.from_documents(chunks, embeddings)

vector_db.save_local(VECTOR_PATH)

print("\n✅ Vector database created successfully!")
print(f"📦 Saved at: {VECTOR_PATH}")