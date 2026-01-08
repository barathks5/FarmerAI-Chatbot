import torch
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------------
# DEVICE CONFIGURATION (CPU / GPU)
# -------------------------------
DEVICE = 0 if torch.cuda.is_available() else -1
print("Using GPU" if DEVICE == 0 else "Using CPU")

# -------------------------------
# LOAD EMBEDDINGS MODEL
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# LOAD VECTOR DATABASE
# -------------------------------
db = FAISS.load_local(
    "models/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------------
# LOAD LLM (GPU AWARE)
# -------------------------------
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=DEVICE
)

# -------------------------------
# CORE RAG FUNCTION
# -------------------------------
def ask_question(query: str, language: str = "English") -> str:
    """
    Takes a farmer query and returns a clean,
    document-grounded agricultural advisory answer.
    Language parameter is kept for multilingual support.
    """

    # (For now, we process in English directly)
    query_en = query

    # Retrieve relevant documents
    docs = retriever.invoke(query_en)

    if not docs:
        return "Information not available in the knowledge base."

    # Build context
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are an agricultural advisory assistant for small farmers.

Give a clear, step-by-step answer using ONLY the context.
Use simple language.
Format the answer as bullet points or numbered steps.
Include:
1. Fertilizer name
2. Quantity
3. Time of application (days after sowing)
4. Any important precautions

Do NOT repeat the question or context.

Context:
{context}

Answer:
"""


    response = llm(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.3
    )

    answer = response[0]["generated_text"]
    answer = answer.split("Answer:")[-1].strip()

    return answer



# -------------------------------
# TERMINAL TEST MODE
# -------------------------------
if __name__ == "__main__":
    print("\n🌾 Farmer Advisory RAG Chatbot (CLI Mode)")
    print("Type 'exit' to stop.\n")

    while True:
        user_query = input("Farmer: ")
        if user_query.lower() == "exit":
            break

        reply = ask_question(user_query)
        print("AI:", reply, "\n")
