"""
Farmer Advisory RAG System
LLM: Qwen2.5-3B-Instruct (Balanced Detailed Mode)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------
# OPTIONAL TRANSLATION
# -------------------------------------------------
try:
    from backend.translator import translate_to_english, translate_from_english
except Exception:
    def translate_to_english(x, l): return x
    def translate_from_english(x, l): return x

# -------------------------------------------------
# DEVICE CONFIG
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {DEVICE}")

# -------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# VECTOR DATABASE
# -------------------------------------------------
db = FAISS.load_local(
    "models/vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

# 🔥 Increase retrieval for richer context
retriever = db.as_retriever(search_kwargs={"k": 4})

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print("⏳ Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)

model.eval()

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=650,           # 🔥 increased for detailed output
    temperature=0.4,              # balanced creativity
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print("✅ Model Ready")

# -------------------------------------------------
# CORE RAG FUNCTION
# -------------------------------------------------
def ask_question(query: str, language: str = "English") -> str:

    query_en = translate_to_english(query, language)

    docs = retriever.invoke(query_en)

    if not docs:
        return translate_from_english(
            "Information not available in the knowledge base.",
            language
        )

    # 🔥 Allow larger context (but safe)
    context = "\n\n".join(d.page_content[:1000] for d in docs)

    # ---- CLEAN, STRONG PROMPT ----
    prompt = f"""
You are a senior agricultural extension officer.

Use ONLY the provided context.
Provide detailed, step-by-step instructions for farmers.
Each step must include:
- Quantity (if available)
- Timing
- Application method
- Field condition guidance

STRICT FORMAT:

CROP:
STEP 1: (Detailed explanation)
STEP 2: (Detailed explanation)
STEP 3: (Detailed explanation)
STEP 4:
STEP 5:
PRECAUTIONS: (Minimum 4 detailed precaution points)

CONTEXT:
{context}

Question:
{query_en}

Answer:
"""

    with torch.no_grad():
        result = llm(prompt)[0]["generated_text"]

    answer = result.split("Answer:")[-1].strip()

    # Better validation
    if answer.count("STEP") < 3 or len(answer) < 150:
        return translate_from_english(
            "⚠️ The system could not generate a sufficiently detailed advisory.",
            language
        )

    return translate_from_english(answer, language)


# -------------------------------------------------
# CLI MODE
# -------------------------------------------------
if __name__ == "__main__":
    print("\n🌾 Farmer Advisory RAG Chatbot")
    print("Type 'exit' to stop\n")

    while True:
        q = input("Farmer: ")
        if q.lower() == "exit":
            break
        print("\nAI Advisory:\n")
        print(ask_question(q))
        print("\n" + "-" * 60)
