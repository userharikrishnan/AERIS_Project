import re


class Intent:
    def __init__(self, type: str, entities: dict):
        self.type = type
        self.entities = entities


class NLPProcessor:
    def extract_intent(self, text: str) -> Intent:
        raw = text
        text = text.lower().strip()

        # 👁 Vision intents
        if any(kw in text for kw in ["what's on my screen", "what is on my screen", "see screen"]):
            return Intent("VISION_QUERY", {})

        if "active window" in text:
            return Intent("ACTIVE_WINDOW", {})

        if "list windows" in text or "open windows" in text:
            return Intent("LIST_WINDOWS", {})

        if "read screen" in text or "read what's on screen" in text:
            return Intent("READ_SCREEN", {})

        # 🧭 Existing intents (unchanged)
        if text.startswith("open"):
            app = text.replace("open", "").strip()
            return Intent("OPEN_APP", {"app": app})

        if text.startswith("search"):
            query = text.replace("search", "").strip()
            return Intent("SEARCH", {"query": query})

        return Intent("CHAT", {"text": raw})
