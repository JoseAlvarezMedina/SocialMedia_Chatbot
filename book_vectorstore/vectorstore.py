# vectorstore.py

# --- 1) Instala dependencias:
# pip install ebooklib beautifulsoup4 langchain faiss-cpu openai python-dotenv

import os
from dotenv import load_dotenv
from ebooklib import epub
from ebooklib.epub import EpubHtml
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# --- 2) Carga API key desde .env o entorno
load_dotenv()  # busca un archivo .env en cwd
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Falta la variable OPENAI_API_KEY en el entorno o en .env")

# --- 3) EPUB → Document ---
def epub_to_document(epub_path: str) -> Document:
    book = epub.read_epub(epub_path)
    full_text = []
    for item in book.get_items():
        if isinstance(item, EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            full_text.append(soup.get_text(separator="\n"))
    return Document(page_content="\n".join(full_text),
                    metadata={"source": os.path.basename(epub_path)})

# --- 4) Construir y guardar vectorstore ---
def build_vectorstore(epub_paths, output_dir="vectorstores/books_faiss"):
    docs = [epub_to_document(p) for p in epub_paths]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f"Vectorstore guardado en '{output_dir}'")


if __name__ == "__main__":
    # Rutas a tus dos libros .epub
    epub_files = [
        "Nancy Harhut - Using Behavioral Science in Marketing_ Drive Customer Action and Loyalty by Prompting Instinctive Responses-Kogan Page (2022).epub",
        "Brendan Kane - One Million Followers, Updated Edition-BenBella Books (2020).epub",
    ]
    build_vectorstore(epub_files)
