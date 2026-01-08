from transformers import pipeline

translator_to_en = pipeline(
    "translation",
    model="ai4bharat/indictrans2-indic-en"
)

translator_from_en = pipeline(
    "translation",
    model="ai4bharat/indictrans2-en-indic"
)

LANG_MAP = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva"
}

def to_english(text, lang):
    if lang == "English":
        return text
    return translator_to_en(
        text,
        src_lang=LANG_MAP[lang],
        tgt_lang="eng_Latn"
    )[0]["translation_text"]

def from_english(text, lang):
    if lang == "English":
        return text
    return translator_from_en(
        text,
        src_lang="eng_Latn",
        tgt_lang=LANG_MAP[lang]
    )[0]["translation_text"]
