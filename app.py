"""
app.py

Interfaz principal de Streamlit para EstrategIA MKT:
- Configura la página y carga estilos.
- Gestiona la UI de chat con RAG, carga de PDF/CSV y generación rápida de contenido.
"""

# ───────────────────────────────────────────────────────────────────────────────
# 1) Imports
# ───────────────────────────────────────────────────────────────────────────────
# Librerías estándar
import os
import json

# Librerías de terceros
import streamlit as st
import PyPDF2
import pandas as pd
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Módulos propios
from callbacks import StreamHandler
from utils import (
    enable_chat_history,
    display_msg,
    configure_llm,
    configure_embedding_model,
    get_openai_api_key,
    load_user_profile
)
from vectorstore import initialize_vectorstore


# ───────────────────────────────────────────────────────────────────────────────
# 2) Configuración de página y estilos
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EstrategIA MKT",
    page_icon="🤖",
    layout="wide",
)

def load_css(path: str):
    """
    Carga un fichero CSS y lo inyecta en la app Streamlit.
    """
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_css(os.path.join(BASE_DIR, "assets", "style.css"))


# ───────────────────────────────────────────────────────────────────────────────
# 3) Función principal
# ───────────────────────────────────────────────────────────────────────────────
@enable_chat_history
def main():
    """
    Renderiza la aplicación, maneja el flujo de chat, 
    la generación rápida de contenido y la lógica RAG/PDF/CSV.
    """
    # 3.1) Header con logo y título
    st.markdown("## 🤖 EstrategIA MKT")

    # 3.2) Configurar LLM, embeddings y cliente OpenAI
    llm      = configure_llm()
    embedder = configure_embedding_model()
    client   = OpenAI(api_key=get_openai_api_key())

    # 3.3) Cargar perfil de usuario
    profile = load_user_profile()

    # 3.4) Definir prompt templates y LLMChains para generación rápida
    script_sys = SystemMessagePromptTemplate.from_template(
        "Eres un guionista experto en marketing digital para PYMEs.\n"
        "Estructura del guion de video:\n"
        "1. Intro/Hook\n"
        "2. Problema\n"
        "3. Pasos con ejemplos\n"
        "4. Conclusión y CTA\n"
        "Perfil: {nombreNegocio}, producto estrella: {productoEstrella}, público: {publicoObjetivo}."
    )
    calendar_sys = SystemMessagePromptTemplate.from_template(
        "Eres un planificador de contenido para PYMEs.\n"
        "Calendario de 7 días con:\n"
        "- Fecha (YYYY-MM-DD)\n- Plataforma\n- Tipo de contenido\n- CTA\n- Hora\n"
        "Perfil: {nombreNegocio}, contenido: {tipoContenidoMarca}, frecuencia: {frecuenciaPublicacion}."
    )
    ideas_sys = SystemMessagePromptTemplate.from_template(
        "Eres un creativo digital para PYMEs.\n"
        "Sugiere 5 ideas de publicaciones:\n"
        "- How-to\n- Listas\n- Preguntas abiertas\n- UGC\n- Citas motivacionales\n"
        "Perfil: {nombreNegocio}, público: {publicoObjetivo}."
    )
    human = HumanMessagePromptTemplate.from_template("{input}")

    script_chain   = LLMChain(llm=llm,
                              prompt=ChatPromptTemplate.from_messages([script_sys, human]).partial(**profile),
                              verbose=False)
    calendar_chain = LLMChain(llm=llm,
                              prompt=ChatPromptTemplate.from_messages([calendar_sys, human]).partial(**profile),
                              verbose=False)
    ideas_chain    = LLMChain(llm=llm,
                              prompt=ChatPromptTemplate.from_messages([ideas_sys, human]).partial(**profile),
                              verbose=False)

    # 3.5) Sidebar: perfil
    with st.sidebar.expander("Perfil", expanded=False):
        labels = {
            "nombreNegocio":      "Nombre de Negocio",
            "tipoProducto":       "Tipo de Producto",
            "productoEstrella":   "Producto Estrella",
            "personalidad":       "Personalidad",
            "identidadVisual":    "Identidad Visual",
            "tipoContenidoMarca": "Tipo de Contenido",
            "publicoObjetivo":    "Público Objetivo",
            "redMasVentas":       "Canal Principal",
            "metodoVenta":        "Método de Venta",
        }
        for key, val in profile.items():
            st.write(f"**{labels.get(key, key)}:** {val}")

    # 3.6) Sidebar: generación rápida
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Generar Contenido")
    if st.sidebar.button("📝 Guion"):
        st.session_state.mode = "guion"
    if st.sidebar.button("📅 Calendario"):
        st.session_state.mode = "calendario"
    if st.sidebar.button("💡 Ideas"):
        st.session_state.mode = "ideas"

    # 3.7) Sidebar: carga de documentos opcionales
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Documentos (opcional)")
    pdf_file    = st.sidebar.file_uploader("📄 Subir PDF", type="pdf")
    include_pdf = st.sidebar.checkbox("Incluir PDF en contexto")
    csv_file    = st.sidebar.file_uploader("📑 Subir CSV", type="csv")
    include_csv = st.sidebar.checkbox("Incluir CSV en contexto")

    # 3.8) Inicializar vectorstore para RAG
    vectorstore = initialize_vectorstore(embedding_model=embedder)

    # 3.9) Mostrar historial existente sin duplicar
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 3.10) Entrada de usuario y lógica de respuesta
    prompt_label = "🔊 Escribe tu pregunta"
    if st.session_state.get("mode"):
        prompt_label += f" para generar {st.session_state.mode}"
    user_input = st.chat_input(f"{prompt_label}:")

    if user_input:
        display_msg(user_input, author="user")
        handler = StreamHandler(st.empty())

        # 3.10.1) Generación rápida vía prompt templates
        mode = st.session_state.get("mode")
        if mode:
            if mode == "guion":
                resp = script_chain.run(input=user_input)
            elif mode == "calendario":
                resp = calendar_chain.run(input=user_input)
            else:
                resp = ideas_chain.run(input=user_input)
            display_msg(resp, author="assistant")
            del st.session_state.mode

        # 3.10.2) Flujo normal: RAG + PDF + CSV
        else:
            docs = vectorstore.similarity_search(user_input, k=4)
            rag_ctx = "\n\n".join(d.page_content for d in docs)

            pdf_ctx = ""
            if include_pdf and pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                pdf_ctx = "PDF:\n" + "\n".join(p.extract_text() or "" for p in reader.pages)

            csv_ctx = ""
            if include_csv and csv_file:
                df = pd.read_csv(csv_file)
                cols   = ", ".join(df.columns)
                sample = df.head(5).to_csv(index=False)
                csv_ctx = f"CSV columnas: {cols}\nEjemplo filas:\n{sample}"

            all_ctx = "\n\n".join(c for c in (pdf_ctx, csv_ctx, rag_ctx) if c)
            system_msg = {
                "role": "system",
                "content": (
                    "Eres un asistente de marketing digital para PYMEs.\n"
                    f"Perfil completo:\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
                )
            }
            user_msg = {
                "role": "user",
                "content": f"Contexto:\n{all_ctx}\n\nPregunta: {user_input}"
            }

            full_resp = ""
            stream = client.chat.completions.create(
                model=llm.model_name,
                messages=[system_msg, user_msg],
                stream=True
            )
            for chunk in stream:
                tok = chunk.choices[0].delta.content or ""
                if tok:
                    handler.on_llm_new_token(tok)
                    full_resp += tok

            st.session_state.messages.append({"role":"assistant","content":full_resp})


if __name__ == "__main__":
    main()
