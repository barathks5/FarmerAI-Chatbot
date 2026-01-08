import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_chat import ask_question

st.set_page_config(page_title="Farmer Advisory AI", layout="centered")

st.title("🌾 Farmer Advisory Chatbot")
st.write("Ask agriculture-related questions and get verified guidance.")

# Language selector
language = st.selectbox(
    "Select Language",
    ["English", "Tamil", "Hindi"]
)

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_input:
        answer = ask_question(user_input, language)
        st.session_state.chat.append(("Farmer", user_input))
        st.session_state.chat.append(("AI", answer))

# Display chat
for sender, message in st.session_state.chat:
    if sender == "Farmer":
        st.markdown(f"**🧑 Farmer:** {message}")
    else:
        st.markdown(f"**🤖 AI:** {message}")
