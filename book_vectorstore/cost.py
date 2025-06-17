# cost.py

# --- 1) Instala dependencias:
# pip install ebooklib beautifulsoup4 tiktoken

import os
from ebooklib import epub
from bs4 import BeautifulSoup
import tiktoken

# Configuración
EMBEDDING_MODEL = "text-embedding-ada-002"
PRICE_PER_1K_TOKENS = 0.0001  # USD por 1k tokens

def epub_to_text(epub_path: str) -> str:
    """
    Extrae y concatena el texto de todos los capítulos de un EPUB.
    """
    book = epub.read_epub(epub_path)
    parts = []
    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            parts.append(soup.get_text(separator="\n"))
    return "\n".join(parts)

def estimate_embedding_cost(epub_paths, chunk_size=1000, chunk_overlap=200):
    """
    Estima tokens y coste para embeddings usando la misma lógica de chunks.
    """
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    total_tokens = 0
    step = chunk_size - chunk_overlap

    for path in epub_paths:
        text = epub_to_text(path)
        words = text.split()
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            total_tokens += len(enc.encode(chunk))

    cost_usd = (total_tokens / 1000) * PRICE_PER_1K_TOKENS
    print(f"Total aproximado de tokens: {total_tokens:,}")
    print(f"Costo estimado (@${PRICE_PER_1K_TOKENS}/1k tokens): ${cost_usd:.4f}")


if __name__ == "__main__":
    epub_files = [
        "Nancy Harhut - Using Behavioral Science in Marketing_ Drive Customer Action and Loyalty by Prompting Instinctive Responses-Kogan Page (2022).epub",
        "Brendan Kane - One Million Followers, Updated Edition-BenBella Books (2020).epub",
    ]
    estimate_embedding_cost(epub_files)
