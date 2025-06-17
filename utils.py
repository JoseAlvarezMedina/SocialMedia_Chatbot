# utils.py

import os
import openai
import json
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

logger = get_logger('Langchain-Chatbot')

def enable_chat_history(func):
    """
    Decorator que:
      1) Inyecta CSS personalizado (burbujas y botones) una vez.
      2) Limpia historial al cambiar de página.
      3) Inserta saludo dinámico si no hay mensajes.
      4) Renderiza TODO el historial antes de ejecutar la página.
    """
    def wrapper(*args, **kwargs):
        # 1) CSS global (solo la primera vez)
        if "css_injected" not in st.session_state:
            st.markdown("""
                <style>
                  /* Burbujas de chat full-width */
                  [data-testid="stChatMessage"] {
                    width: 100% !important;
                    max-width: none !important;
                  }
                  /* Botones full-width con colores de marca */
                  div.stButton > button {
                    width: 100% !important;
                    background-color: #c29a7c !important;  /* secondary */
                    color: #321a2a !important;              /* accent */
                    border: none !important;
                  }
                </style>
            """, unsafe_allow_html=True)
            st.session_state.css_injected = True

        # 2) Reset si cambio de página
        current = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state.current_page = current
        elif st.session_state.current_page != current:
            st.cache_resource.clear()
            st.session_state.current_page = current
            st.session_state.messages = []

        # 3) Saludo dinámico si está vacío
        if not st.session_state.messages:
            profile = load_user_profile()
            audience = profile.get("publicoObjetivo", "tu público objetivo")
            intro = (
                f"¡Hola! ¿En qué puedo ayudarte hoy? "
                f"Te ofrezco estrategias digitales para {audience}."
            )
            st.session_state.messages = [{"role": "assistant", "content": intro}]

        # 4) Renderizar historial completo
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Ejecutar la página decorada
        return func(*args, **kwargs)

    return wrapper

def display_msg(msg, author):
    """
    Añade un mensaje al historial, lo muestra y hace scroll al final.
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
    st.markdown(
        '<div id="end"></div>'
        '<script>document.getElementById("end").scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True
    )

def message_func(content: str, is_user: bool = False, is_df: bool = False, model: str = ""):
    """
    Renderiza un mensaje tipo burbuja con colores de marca:
    - Usuario → derecha con primary (#947158)
    - Asistente → izquierda con accent (#321a2a)
    Texto en neutral (#F3F4F6)
    """
    align = "right" if is_user else "left"
    bg = "#947158" if is_user else "#321a2a"
    color = "#F3F4F6"
    st.markdown(
        f"""
        <div style="
          text-align: {align};
          background-color: {bg};
          color: {color};
          border-radius: 10px;
          padding: 8px 12px;
          margin: 4px 0;
          display: inline-block;
          max-width: 80%;
          font-size: 14px;
        ">
          {content}
        </div>
        """,
        unsafe_allow_html=True
    )

class StreamlitUICallbackHandler:
    """
    Handler de ejemplo para streaming callbacks.
    """
    def __init__(self):
        pass

def choose_custom_openai_key():
    openai_api_key = st.sidebar.text_input(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="SELECTED_OPENAI_API_KEY"
    )
    if not openai_api_key:
        st.error("Please add your OpenAI API key to continue.")
        st.info("Obtain your key from: https://platform.openai.com/account/api-keys")
        st.stop()
    model = "gpt-4o-mini"
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        available = [
            {"id": m.id, "created": datetime.fromtimestamp(m.created)}
            for m in client.models.list() if str(m.id).startswith("gpt")
        ]
        available = sorted(available, key=lambda x: x["created"])
        models = [m["id"] for m in available]
        model = st.sidebar.selectbox("Model", options=models, key="SELECTED_OPENAI_MODEL")
    except Exception as e:
        st.error(str(e))
        st.stop()
    return model, openai_api_key

def configure_llm():
    return ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

@st.cache_resource
def configure_embedding_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def sync_st_session():
    for k, v in st.session_state.items():
        st.session_state[k] = v

def get_current_user():
    """
    Extrae el email del último registro en data/surveys.json.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "surveys.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            surveys = json.load(f)
    except:
        return ""
    return surveys[-1].get("user", "") if surveys else ""

def load_user_profile():
    """
    Lee la última encuesta de data/surveys.json (sin la clave 'user').
    """
    user = get_current_user()
    if not user:
        return {}
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "surveys.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            surveys = json.load(f)
    except:
        return {}
    for entry in reversed(surveys):
        if entry.get("user") == user:
            return {k: v for k, v in entry.items() if k != "user"}
    return {}
