"""
Farmer Advisory RAG System
LLM: Qwen2.5-3B-Instruct
Purpose: Step-by-step, farmer-friendly agricultural guidance
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------------------------
# OPTIONAL TRANSLATION (SAFE FALLBACK)
# -------------------------------------------------
try:
    from backend.translator import translate_to_english, translate_from_english
except Exception:
    def translate_to_english(text, lang): return text
    def translate_from_english(text, lang): return text

# -------------------------------------------------
# DEVICE CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# -------------------------------------------------
# EMBEDDINGS (SEMANTIC SEARCH)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# LOAD VECTOR DATABASE
# -------------------------------------------------
db = FAISS.load_local(
    "models/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------
# LOAD QWEN MODEL (OPTIMIZED SETTINGS)
# -------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("⏳ Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=260,      # ⚡ Faster generation
    temperature=0.2,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False   # ⭐ Prevent prompt echo
)

print("✅ Model Ready")

# -------------------------------------------------
# CORE RAG FUNCTION
# -------------------------------------------------
def ask_question(query: str, language: str = "English") -> str:

    # ---- Translate Query ----
    query_en = translate_to_english(query, language)

    # ---- Retrieve Docs ----
    docs = retriever.invoke(query_en)

    if not docs:
        return translate_from_english(
            "Information not available in the knowledge base.",
            language
        )

    context = "\n".join(doc.page_content for doc in docs)

    # ---- CLEAN, STRONG PROMPT ----
    prompt = f"""
You are an experienced agricultural extension officer.

Follow these STRICT rules:
- Use ONLY the provided context.
- Explain simply for farmers.
- Give clear step-by-step instructions.
- Do NOT repeat the context.
- Avoid technical jargon.

FORMAT:
CROP:
STEP 1:
STEP 2:
STEP 3:
PRECAUTIONS:

CONTEXT:
{context}

QUESTION:
{query_en}

ANSWER:
"""

    # ---- GENERATE ----
    result = llm(prompt)[0]["generated_text"].strip()

    # ---- QUALITY CHECK ----
    if len(result) < 150 or "STEP" not in result:
        return translate_from_english(
            "⚠️ Unable to generate a reliable advisory from the available documents.",
            language
        )

    return translate_from_english(result, language)

# -------------------------------------------------
# CLI TEST MODE
# -------------------------------------------------
if __name__ == "__main__":
    print("\n🌾 Farmer Advisory RAG Chatbot (CLI Mode)")
    print("Type 'exit' to stop\n")

    while True:
        q = input("Farmer: ")
        if q.lower() == "exit":
            break

        print("\nAI Advisory:\n")
        print(ask_question(q))
        print("\n" + "-" * 60)
