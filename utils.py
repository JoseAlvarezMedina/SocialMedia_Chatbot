"""
utils.py

Funciones auxiliares para:
  1) Configuración de LLM y selección de modelo.
  2) Configuración de embeddings.
  3) Gestión de historial de chat y UI helpers.
  4) Carga de usuario y perfil desde JSON.

No modifica la lógica principal de la aplicación, solo ofrece utilidades reutilizables.
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import json
import streamlit as st
from datetime import datetime
from streamlit.logger import get_logger

from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

logger = get_logger('Langchain-Chatbot')


# ───────────────────────────────────────────────────────────────────────────────
# 1) Lectura de API Key y configuración del LLM
# ───────────────────────────────────────────────────────────────────────────────
def get_openai_api_key() -> str:
    """
    Lee la OpenAI API Key desde .streamlit/secrets.toml.
    Si no existe, muestra un error y detiene la app.
    """
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        st.error("🔑 No se encontró la OpenAI API Key en .streamlit/secrets.toml")
        st.stop()


def choose_model(default: str = "gpt-3.5-turbo") -> str:
    """
    Muestra un selectbox en la sidebar para elegir el modelo LLM,
    limitando a opciones de coste moderado.
    """
    models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o-mini"]
    # Garantiza que el default esté al inicio
    if default not in models:
        models.insert(0, default)
    idx = models.index(default)
    return st.sidebar.selectbox("Modelo LLM", models, index=idx)


def configure_llm() -> ChatOpenAI:
    """
    Crea y devuelve un cliente ChatOpenAI configurado para streaming,
    usando la API Key y el modelo seleccionado en choose_model().
    """
    api_key = get_openai_api_key()
    model_name = choose_model()
    return ChatOpenAI(
        model_name=model_name,
        temperature=0,
        streaming=True,
        openai_api_key=api_key
    )


# ───────────────────────────────────────────────────────────────────────────────
# 2) Configuración de embeddings
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def configure_embedding_model() -> OpenAIEmbeddings:
    """
    Instancia OpenAIEmbeddings cached para evitar recargas repetidas.
    """
    api_key = get_openai_api_key()
    return OpenAIEmbeddings(openai_api_key=api_key)


# ───────────────────────────────────────────────────────────────────────────────
# 3) Historial de chat y helpers de UI
# ───────────────────────────────────────────────────────────────────────────────
def enable_chat_history(func):
    """
    Decorator que:
      1) Inyecta CSS global para el chat (una sola vez).
      2) Resetea el historial al cambiar de página.
      3) Inserta un saludo inicial basado en el perfil.
      4) Llama a la función decorada.
    """
    def wrapper(*args, **kwargs):
        # 1) Inyectar CSS si es la primera carga
        if "css_injected" not in st.session_state:
            st.markdown("""
                <style>
                  [data-testid="stChatMessage"] { width:100% !important; }
                  div.stButton > button {
                    width:100% !important;
                    background-color:#c29a7c !important;
                    color:#321a2a !important;
                    border:none !important;
                  }
                </style>
            """, unsafe_allow_html=True)
            st.session_state.css_injected = True

        # 2) Reset historial al navegar entre páginas
        current = func.__name__
        if st.session_state.get("current_page") != current:
            st.session_state.current_page = current
            st.session_state.messages = []

        # 3) Saludo inicial si no hay mensajes
        if not st.session_state.get("messages"):
            profile = load_user_profile()
            audiencia = profile.get("publicoObjetivo", "tu público objetivo")
            saludo = f"¡Hola! Soy tu asistente de marketing digital para {audiencia}."
            st.session_state.messages = [{"role": "assistant", "content": saludo}]

        # 4) Ejecutar función original
        return func(*args, **kwargs)

    return wrapper


def display_msg(msg: str, author: str):
    """
    Agrega un mensaje a session_state y lo muestra con st.chat_message(),
    luego hace scroll al final.
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)
    st.markdown(
        '<div id="end"></div>'
        '<script>document.getElementById("end")'
        '.scrollIntoView({behavior:"smooth"});</script>',
        unsafe_allow_html=True
    )


def message_func(content: str, is_user: bool = False):
    """
    Muestra un único mensaje con st.chat_message(), sin duplicar el historial.
    """
    role = "user" if is_user else "assistant"
    st.chat_message(role).write(content)


def sync_st_session():
    """
    No-op: mantenido para compatibilidad, evita reasignar session_state.
    """
    pass


# ───────────────────────────────────────────────────────────────────────────────
# 4) Carga de usuario y perfil desde JSON
# ───────────────────────────────────────────────────────────────────────────────
def get_current_user() -> str:
    """
    Retorna el campo 'user' de la última entrada en data/surveys.json,
    o cadena vacía si no existe.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "surveys.json")
    try:
        surveys = json.load(open(path, "r", encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return ""
    if not isinstance(surveys, list) or not surveys:
        return ""
    return surveys[-1].get("user", "")


def load_user_profile() -> dict:
    """
    Devuelve la última encuesta de data/surveys.json (sin la clave 'user'),
    o un diccionario vacío si no existe.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "surveys.json")
    try:
        surveys = json.load(open(path, "r", encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(surveys, list) or not surveys:
        return {}
    last = surveys[-1].copy()
    last.pop("user", None)
    return last
