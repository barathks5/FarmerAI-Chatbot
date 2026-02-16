import streamlit as st
import sys, os
import sounddevice as sd
from scipy.io.wavfile import write

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from backend.rag_chat import ask_question
from backend.speech_to_text import speech_to_text

st.set_page_config("Farmer Advisory AI", "🌾", "centered")

st.title("🌾 Farmer Advisory Chatbot")
st.caption("Step-by-step | Multilingual | Voice Enabled")

language = st.selectbox("Select Language", ["English", "Tamil", "Hindi"])

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Enter your question")

if st.button("🎙️ Speak"):
    st.info("Recording for 5 seconds...")
    fs = 44100
    recording = sd.rec(int(5 * fs), samplerate=fs, channels=1)
    sd.wait()
    voice_path = os.path.join(PROJECT_ROOT, "voice.wav")
    write(voice_path, fs, recording)
    user_input = speech_to_text(voice_path)
    st.success(f"You said: {user_input}")

if st.button("Ask", use_container_width=True):
    if user_input:
        with st.spinner("Generating detailed advisory..."):
            answer = ask_question(user_input, language)

        st.session_state.chat.append(("Farmer", user_input))
        st.session_state.chat.append(("AI", answer))

st.divider()

for role, msg in st.session_state.chat:
    if role == "Farmer":
        st.markdown(f"### 🧑 Farmer\n{msg}")
    else:
        st.markdown(f"### 🤖 AI Advisory\n{msg}")
