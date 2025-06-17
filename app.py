# app.py

import os
import tempfile
import sqlite3

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

from utils import get_current_user, load_user_profile, message_func

# --- 1) Page config & hide menu/header ---
st.set_page_config(page_title="Tu colaborador de marketing", page_icon="🤖")
st.markdown("""
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- 2) Global CSS for palette, backgrounds & components ---
st.markdown("""
<style>
  /* App background white */
  [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
  }
  /* Sidebar light neutral */
  [data-testid="stSidebar"] {
    background-color: #c29a7c !important;
  }
  /* Buttons primary */
  div.stButton > button {
    background-color: #947158 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0.25rem !important;
  }
  /* Radio buttons labels */
  .stRadio label, .stRadio div {
    color: #321a2a !important;
  }
  /* Hide default chat bubbles */
  [data-testid="stChatMessage"] {
    display: none !important;
  }
  /* File uploader container */
  .stFileUploader > div {
    background-color: #F3F4F6 !important;
    border: 1px dashed #c29a7c !important;
    color: #321a2a !important;
  }
  .stFileUploader label, .stFileUploader .uploading-text {
    color: #321a2a !important;
  }
  /* Text input */
  input[type="text"], textarea {
    background-color: #FFFFFF !important;
    color: #321a2a !important;
  }
  /* Chat input box */
  .stForm input {
    background-color: #FFFFFF !important;
    color: #321a2a !important;
  }
</style>
""", unsafe_allow_html=True)

# --- 3) Header with logo and friendly title ---
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
col_logo, col_title = st.columns([1, 4], gap="small")
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=140)
with col_title:
    st.markdown("<h1 style='margin:0; line-height:80px; color:#947158;'>Robotitus</h1>", unsafe_allow_html=True)

# --- 4) Avatar ---
user = get_current_user()
if user:
    initial = user[0].upper()
    st.markdown(f"""
        <style>
          .avatar {{
            position: absolute; top: 1rem; right: 1rem;
            width: 2.5rem; height: 2.5rem;
            background: #c29a7c; color: #321a2a;
            border-radius: 50%; display: flex;
            align-items: center; justify-content: center;
            font-weight: bold; cursor: pointer; z-index:100;
          }}
        </style>
        <div class="avatar" onclick="window.location.reload()">{initial}</div>
    """, unsafe_allow_html=True)

# --- 5) Session state init ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# --- 6) Initial greeting ---
profile = load_user_profile()
if not st.session_state.greeted:
    audience = profile.get("publicoObjetivo", "tu público objetivo")
    intro = (
        f"¡Hola! ¿En qué puedo ayudarte hoy? "
        f"Estrategias de marketing digital para {audience}."
    )
    st.session_state.messages.append({"role": "assistant", "content": intro})
    st.session_state.greeted = True

# --- 7) Render chat history ---
for msg in st.session_state.messages:
    is_user = (msg["role"] == "user")
    bg = "#c29a7c" if is_user else "#947158"
    color = "#321a2a" if is_user else "#FFFFFF"
    align = "right" if is_user else "left"
    st.markdown(f"""
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
          {msg["content"]}
        </div>
    """, unsafe_allow_html=True)

# --- 8) Sidebar: Perfil ---
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False
if st.sidebar.button("Perfil"):
    st.session_state.show_profile = not st.session_state.show_profile
if st.session_state.show_profile:
    st.sidebar.markdown("### 🔍 Tu perfil")
    labels = {
        "nombreNegocio":"Negocio", "tipoProducto":"Productos/Servicios",
        "productoEstrella":"Producto estrella","personalidad":"Personalidad",
        "identidadVisual":"Identidad visual","tipoContenidoMarca":"Tipo contenido",
        "publicoObjetivo":"Público objetivo","redMasVentas":"Red principal",
        "metodoVenta":"Método de venta","formatoContenido":"Formato contenido",
        "frecuenciaPublicacion":"Frecuencia","contenidoPromocional":"Promocional",
        "retoVentas":"Reto ventas","objetivoIAgora":"Objetivo IAgora",
        "objetivoConcreto":"Objetivo concreto"
    }
    for k, v in profile.items():
        st.sidebar.write(f"**{labels.get(k,k)}:** {v}")

# --- 9) Sidebar: PDF loader ---
st.sidebar.markdown("### 📄 Carga un PDF (opcional)")
uploaded_pdf = st.sidebar.file_uploader("", type=["pdf"])
document_chain = None
if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(
        PyPDFLoader(pdf_path).load()
    )
    vs = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]))
    document_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, streaming=True, api_key=st.secrets["OPENAI_API_KEY"]),
        chain_type="stuff", retriever=vs.as_retriever()
    )

# --- 10) Sidebar: CSV & SQL Agent ---
st.sidebar.markdown("### 📊 Análisis CSV (opcional)")
uploaded_csv = st.sidebar.file_uploader("", type=["csv"])
if uploaded_csv:
    df_csv = pd.read_csv(uploaded_csv).loc[:, lambda df: ~df.columns.str.contains("^Unnamed")]
    df_csv.columns = [c.strip().lower().replace(" ", "_") for c in df_csv.columns]
    with st.sidebar.expander("Vista previa y columnas"):
        st.sidebar.write(df_csv.head())
        st.sidebar.write("Columnas:", ", ".join(df_csv.columns))

    if "csv_agent" not in st.session_state:
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        conn = sqlite3.connect(tmp_db.name)
        df_csv.to_sql("posts", conn, index=False, if_exists="replace")
        conn.close()
        engine = create_engine(f"sqlite:///{tmp_db.name}")
        sql_db = SQLDatabase(engine, sample_rows_in_table_info=5)
        st.session_state.csv_agent = create_sql_agent(
            llm=ChatOpenAI(temperature=0, streaming=False, api_key=st.secrets["OPENAI_API_KEY"]),
            db=sql_db, top_k=5, verbose=False, agent_type="openai-tools",
            handle_parsing_errors=True, handle_sql_errors=True
        )

    query = st.sidebar.text_input("Pregunta sobre tu CSV:")
    if st.sidebar.button("▶️") and query:
        st.session_state.messages.append({"role":"user","content":query})
        st.markdown(f"""
            <div style="
              text-align: right; background-color: #c29a7c; color:#321a2a;
              border-radius:10px; padding:8px; max-width:80%;
            ">{query}</div>
        """, unsafe_allow_html=True)
        with st.chat_message("assistant"):
            cb = StreamlitCallbackHandler(st.container())
            res = st.session_state.csv_agent.invoke({"input": query}, {"callbacks":[cb]})
            ans = res["output"]
        st.session_state.messages.append({"role":"assistant","content":ans})
        st.markdown(f"""
            <div style="
              text-align: left; background-color: #947158; color:#FFFFFF;
              border-radius:10px; padding:8px; max-width:80%;
            ">{ans}</div>
        """, unsafe_allow_html=True)

# --- 11) Mode selector + scroll ---
col1, col2 = st.columns([5,1])
with col1:
    modo = st.radio("", ["Chat libre","Generar Guion","Generar Calendario"], horizontal=True)
with col2:
    if st.button("🔽"):
        st.markdown(
            '<div id="end"></div>'
            '<script>document.getElementById("end").scrollIntoView({behavior:"smooth"});</script>',
            unsafe_allow_html=True
        )

# --- 12) Configure LLM & prompt with full profile context ---
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(temperature=0, streaming=True, api_key=openai_api_key)

system_template = """
Eres un asistente de marketing digital para la empresa {nombreNegocio}.
Esta empresa ofrece {tipoProducto} y su producto estrella es {productoEstrella}.
La personalidad de la marca es {personalidad} y su identidad visual es {identidadVisual}.
Produce {tipoContenidoMarca} en {redMasVentas}, en formato {formatoContenido}, con frecuencia {frecuenciaPublicacion}.
Público objetivo: {publicoObjetivo}, método de venta: {metodoVenta}, objetivo concreto: {objetivoConcreto}.
Siempre responde basándote en este contexto.
""".strip()

system_msg = SystemMessagePromptTemplate.from_template(system_template)
human_msg = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([
    system_msg,
    MessagesPlaceholder(variable_name="history"),
    human_msg
]).partial(**profile)

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(
        llm=llm, memory=st.session_state.memory, prompt=chat_prompt
    )

# --- 13) Main input form ---
with st.form("input_form", clear_on_submit=True):
    c1, c2 = st.columns([0.9,0.1])
    user_input = c1.text_input("", placeholder="Escribe aquí…", label_visibility="collapsed")
    send = c2.form_submit_button("➤")
    if send and user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        st.markdown(f"""
            <div style="
              text-align: right; background-color: #c29a7c; color:#321a2a;
              border-radius:10px; padding:8px; max-width:80%;
            ">{user_input}</div>
        """, unsafe_allow_html=True)

        if modo == "Chat libre":
            resp = st.session_state.conversation_chain.predict(input=user_input)
            if document_chain:
                resp += "\n\n📄 " + document_chain.run(user_input)
        elif modo == "Generar Guion":
            prompt = "Genera un guion estructurado que incluya:\n1. Guía visual\n2. Duración\n3. Narración\n4. Hashtags\n5. CTA\n\n" + user_input
            resp = st.session_state.conversation_chain.predict(input=prompt)
        else:
            prompt = "Crea un calendario de contenido, incluye la hora de publicacion, tipo de contenido, plataforma:\n\n" + user_input
            resp = st.session_state.conversation_chain.predict(input=prompt)

        st.session_state.messages.append({"role":"assistant","content":resp})
        st.markdown(f"""
            <div style="
              text-align: left; background-color: #947158; color:#FFFFFF;
              border-radius:10px; padding:8px; max-width:80%;
            ">{resp}</div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<script>window.scrollTo(0,document.body.scrollHeight);</script>',
            unsafe_allow_html=True
        )

# --- 14) Final anchor ---
st.markdown('<div id="end"></div>', unsafe_allow_html=True)
