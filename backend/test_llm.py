from transformers import pipeline
import torch

print("Loading model...")

chatbot = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0 if torch.cuda.is_available() else -1
)

print("Model loaded successfully!")

response = chatbot(
    "Explain crop rotation in simple words.",
    max_new_tokens=100
)

print("\nModel response:\n")
print(response[0]["generated_text"])
