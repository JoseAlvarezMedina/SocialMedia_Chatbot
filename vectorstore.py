"""
vectorstore.py

Carga y devuelve una instancia de FAISS VectorStore a partir de archivos
preconstruidos: un índice FAISS y un pickle con el docstore y el mapeo
index_to_docstore_id.
"""

# ───────────────────────────────────────────────────────────────────────────────
# 1) Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import pickle

import faiss
from langchain.vectorstores import FAISS


# ───────────────────────────────────────────────────────────────────────────────
# 2) Función principal
# ───────────────────────────────────────────────────────────────────────────────
def initialize_vectorstore(
    embedding_model,
    vs_subpath: str = "book_vectorstore/vectorstores/books_faiss"
) -> FAISS:
    """
    Inicializa y retorna un FAISS vectorstore listo para búsquedas.

    Args:
      embedding_model: objeto con método `.embed_query` para computar embeddings.
      vs_subpath (str): ruta relativa al directorio que contiene
                        'index.faiss' e 'index.pkl'.

    Returns:
      FAISS: instancia de la vectorstore configurada.

    Raises:
      ValueError: si `embedding_model` es None o no tiene `embed_query`.
      FileNotFoundError: si faltan los archivos 'index.pkl' o 'index.faiss'.
    """
    # 2.1) Construir ruta absoluta al directorio de vectorstore
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vs_dir = os.path.join(base_dir, vs_subpath)

    # 2.2) Cargar docstore e index_to_docstore_id desde pickle
    pkl_path = os.path.join(vs_dir, "index.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"No se encontró 'index.pkl' en {pkl_path}")
    with open(pkl_path, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    # 2.3) Leer y cargar el índice FAISS
    faiss_path = os.path.join(vs_dir, "index.faiss")
    if not os.path.isfile(faiss_path):
        raise FileNotFoundError(f"No se encontró 'index.faiss' en {faiss_path}")
    index = faiss.read_index(faiss_path)

    # 2.4) Validar el modelo de embeddings
    if embedding_model is None or not hasattr(embedding_model, "embed_query"):
        raise ValueError(
            "Se requiere un 'embedding_model' válido con el método 'embed_query'."
        )

    # 2.5) Crear y devolver la vectorstore FAISS
    #     Firma: FAISS(embedding_function, index, docstore, index_to_docstore_id)
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore
