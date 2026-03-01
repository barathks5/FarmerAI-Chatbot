from faster_whisper import WhisperModel

# -------------------------------------------------
# LOAD MODEL ONLY ONCE (VERY IMPORTANT)
# -------------------------------------------------
print("🎙️ Loading Speech Model...")

model = WhisperModel(
    "small",           # fast + accurate
    device="cuda",     # uses GPU automatically
    compute_type="float16"
)

print("✅ Speech Model Ready")


# -------------------------------------------------
# SPEECH TO TEXT FUNCTION
# -------------------------------------------------
def speech_to_text(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    text = ""

    for seg in segments:
        text += seg.text

    return text.strip()