# src/build_vectorstore.py

"""
Script para construir y guardar tu FAISS vectorstore a partir de EPUBs.
Úsalo solo una vez (o cuando agregues más libros).
"""

import os
import pickle
from dotenv import load_dotenv
from ebooklib import epub
from ebooklib.epub import EpubHtml
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# 1) Carga API key desde .env o entorno
load_dotenv()  # Busca un archivo .env en cwd
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Falta la variable OPENAI_API_KEY en el entorno o en .env")

def epub_to_document(epub_path: str) -> Document:
    """
    Convierte un archivo .epub a un objeto langchain.schema.Document.
    """
    book = epub.read_epub(epub_path)
    full_text = []
    for item in book.get_items():
        if isinstance(item, EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            full_text.append(soup.get_text(separator="\n"))
    return Document(
        page_content="\n".join(full_text),
        metadata={"source": os.path.basename(epub_path)}
    )

def build_vectorstore(epub_paths: list[str],
                      output_dir: str = "vectorstores/books_faiss"):
    """
    Lee cada .epub de la lista, los fragmenta, genera embeddings y guarda
    el FAISS index + pickle en `output_dir`.
    """
    # 2) EPUB → Document
    docs = [epub_to_document(p) for p in epub_paths]

    # 3) Fragmentación
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 4) Embeddings y FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5) Guardado local
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f"✅ Vectorstore guardado en '{output_dir}'")

if __name__ == "__main__":
    # Ejemplo: ajusta las rutas a tus archivos .epub
    epub_files = [
        "assets/Nancy Harhut - Using Behavioral Science in Marketing.epub",
        "assets/Brendan Kane - One Million Followers.epub",
    ]
    build_vectorstore(epub_files)
