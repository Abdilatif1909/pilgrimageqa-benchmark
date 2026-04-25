from typing import Optional
from langdetect import detect

try:
    from transformers import MarianMTModel, MarianTokenizer
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False


class Translator:
    def __init__(self):
        self.cache = {}

    def _marian_model_name(self, src: str, tgt: str) -> str:
        return f"Helsinki-NLP/opus-mt-{src}-{tgt}"

    def detect_lang(self, text: str) -> str:
        try:
            lang = detect(text)
            if lang not in ["uz", "ru", "en"]:
                return "uz"
            return lang
        except Exception:
            return "uz"

    def _load_model(self, src: str, tgt: str):
        key = (src, tgt)

        if key in self.cache:
            return self.cache[key]

        model_name = self._marian_model_name(src, tgt)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        self.cache[key] = (tokenizer, model)
        return tokenizer, model

    def _hf_translate(self, text: str, src: str, tgt: str):
        tokenizer, model = self._load_model(src, tgt)
        batch = tokenizer([text], return_tensors="pt", truncation=True)
        gen = model.generate(**batch)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return out

    def _dictionary_fallback(self, text: str, src: str, tgt: str):
        mini_map = {
            "qayerda": "где",
            "qanday": "как",
            "yaqin": "рядом",
            "namoz": "молитва",
            "hojatxona": "туалет",
            "dorixona": "аптека",
            "mehmonxona": "гостиница",
            "transport": "транспорт",
            "visa": "виза",
            "bola": "ребенок"
        }

        out = text
        if src == "uz" and tgt == "ru":
            for k, v in mini_map.items():
                out = out.replace(k, v)
            return out

        if src == "ru" and tgt == "uz":
            rev = {v: k for k, v in mini_map.items()}
            for k, v in rev.items():
                out = out.replace(k, v)
            return out

        return text

    def translate_text(self, text: str, src: Optional[str] = None, tgt: str = "ru") -> str:
        if not text:
            return text

        if src is None:
            src = self.detect_lang(text)

        if src == tgt:
            return text

        if HAS_TRANSFORMERS:
            try:
                return self._hf_translate(text, src, tgt)
            except Exception:
                pass

        return self._dictionary_fallback(text, src, tgt)