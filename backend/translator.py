from deep_translator import GoogleTranslator

def translate_to_english(text: str, language: str) -> str:
    if language == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

def translate_from_english(text: str, language: str) -> str:
    if language == "English":
        return text

    target_map = {
        "Tamil": "ta",
        "Hindi": "hi"
    }

    try:
        return GoogleTranslator(
            source="en",
            target=target_map[language]
        ).translate(text)
    except Exception:
        return text
