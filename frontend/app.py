import streamlit as st
import sys, os
from streamlit_mic_recorder import mic_recorder

# -------------------------------------------------
# PROJECT PATH SETUP
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from backend.rag_chat import ask_question
from backend.speech_to_text import speech_to_text

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Farmer Advisory AI",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Farmer Advisory Chatbot")
st.caption("Step-by-step | Multilingual | Voice Enabled")

# -------------------------------------------------
# LANGUAGE SELECTION
# -------------------------------------------------
language = st.selectbox(
    "Select Language",
    ["English", "Tamil", "Hindi"]
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# -------------------------------------------------
# TEXT INPUT
# -------------------------------------------------
st.session_state.user_input = st.text_input(
    "Enter your question",
    value=st.session_state.user_input
)

# -------------------------------------------------
# 🎤 LIVE MIC BUTTON (Tap → Speak → Stop)
# -------------------------------------------------
audio = mic_recorder(
    start_prompt="🎙️ Tap to Talk",
    stop_prompt="⏹️ Stop",
    just_once=True,
    use_container_width=True,
)

if audio:

    voice_path = os.path.join(PROJECT_ROOT, "voice.wav")

    with open(voice_path, "wb") as f:
        f.write(audio["bytes"])

    try:
        text = speech_to_text(voice_path)
        st.session_state.user_input = text
        st.success(f"You said: {text}")
    except Exception as e:
        st.error(f"Speech recognition failed: {e}")

# -------------------------------------------------
# ASK BUTTON
# -------------------------------------------------
if st.button("Ask", use_container_width=True):

    if st.session_state.user_input:

        with st.spinner("🌾 Generating farmer-friendly advisory..."):

            answer = ask_question(
                st.session_state.user_input,
                language
            )

        st.session_state.chat.append(("Farmer", st.session_state.user_input))
        st.session_state.chat.append(("AI", answer))

        st.session_state.user_input = ""

# -------------------------------------------------
# CHAT DISPLAY (ChatGPT Style)
# -------------------------------------------------
st.divider()

for role, msg in st.session_state.chat:

    if role == "Farmer":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
