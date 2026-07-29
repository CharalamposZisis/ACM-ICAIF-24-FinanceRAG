import re
from typing import List
import unicodedata
from load_data import load_data

class TextPreprocessor:
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }

    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to consistent form."""
        return unicodedata.normalize('NFKC', text)

    def redact_pii(self, text: str) -> str:
        """Detect and redact PII patterns."""
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', text)
        return text

    def remove_boilerplate(self, text: str, patterns: List[str] = None) -> str:
        """Remove known boilerplate text patterns."""
        default_patterns = [
            r'Page \d+ of \d+',
            r'Copyright \d{4}.*?(?=\n|$)',
            r'All rights reserved\.?',
        ]
        patterns = patterns or default_patterns
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text

    def process(self, text: str, redact_pii: bool = True) -> str:
        """Run full preprocessing pipeline."""
        text = self.normalize_unicode(text)
        text = self.remove_boilerplate(text)
        text = self.normalize_whitespace(text)
        if redact_pii:
            text = self.redact_pii(text)
        return text


if __name__ == "__main__":
    docs = load_data()
    preprocessor = TextPreprocessor()

    processed_docs = []

    for doc in docs:
        processed_doc = doc.copy()

        if "text" in processed_doc:
            processed_doc["text"] = preprocessor.process(processed_doc["text"])

        processed_docs.append(processed_doc)

    print(f"Processed {len(processed_docs)} documents")
    print(processed_doc)