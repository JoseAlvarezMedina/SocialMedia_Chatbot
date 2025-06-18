```markdown
# EstrategIA MKT

**Un asistente de marketing digital para PYMEs, potenciado por IA, RAG y Streamlit.**

<p align="center">
  <img src="assets/logo.png" width="150" alt="EstrategIA MKT logo">
</p>

---

## 📖 Descripción

**EstrategIA MKT** es un chatbot web que ayuda a pequeñas y medianas empresas a planificar y optimizar su estrategia en redes sociales. Utiliza:

- **Streamlit** para la interfaz web interactiva.
- **OpenAI GPT-3.5-turbo** y **GPT-4o-mini** como motor de lenguaje.
- **LangChain** para orquestar conversaciones y plantillas de prompts.
- **Retrieval-Augmented Generation (RAG)** sobre FAISS para respuestas basadas en documentos.
- **Carga opcional de PDF y CSV** para incorporar datos propios en el contexto de la respuesta.

El resultado es un asistente capaz de:
- Generar guiones de video.
- Crear calendarios de contenido.
- Sugerir ideas de publicaciones.
- Responder preguntas basadas en documentación interna de la PYME (PDF/CSV).

---

## 🚀 Características principales

1. **Generación Rápida de Contenido**  
   Tres modos predefinidos (Guion, Calendario, Ideas) usando **prompt templates** para respuestas estructuradas.  
2. **RAG con FAISS**  
   Indexa tus documentos (epub, PDF, CSV) en FAISS y recupera los más relevantes para enriquecer las respuestas.  
3. **Streaming de respuesta**  
   Visualiza cada token en tiempo real con handlers de LangChain.  
4. **Carga de documentos**  
   Incorpora PDFs y CSVs como contexto adicional para respuestas más precisas.  
5. **Customización de modelo**  
   Elige entre `gpt-3.5-turbo`, `gpt-3.5-turbo-16k` o `gpt-4o-mini` directamente en la barra lateral.

---

## 🎯 Arquitectura Técnica

```

User  ↔ Streamlit UI
│
├─ Prompt Templates (LangChain)
│    ├─ Guion
│    ├─ Calendario
│    └─ Ideas
│
├─ RAG Pipeline (vectorstore.py)
│    ├─ FAISS Index (index.faiss + index.pkl)
│    └─ OpenAIEmbeddings → embed\_query()
│
└─ OpenAI API
├─ ChatCompletion.stream()
└─ ResponseHandler (callbacks.py)

````

---

## 💾 Requisitos

- Python 3.9+  
- Virtualenv o Conda  
- Cuenta de OpenAI con API Key válida  

### Dependencias principales

```text
streamlit>=1.27.0
openai>=0.27.0
langchain-openai>=0.0.1
faiss-cpu>=1.7.4
python-dotenv
PyPDF2
pandas
````

Instálalas con:

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuración

1. **Clonar repositorio**

   ```bash
   git clone https://github.com/JoseAlvarezMedina/SocialMedia_Chatbot.git
   cd SocialMedia_Chatbot
   ```

2. **Ignorar tus credenciales**
   Crea un archivo `.gitignore` en la raíz con:

   ```
   .streamlit/secrets.toml
   .venv/
   __pycache__/
   ```

3. **Agregar tu API Key**
   En `.streamlit/secrets.toml`:

   ```toml
   [openai]
   api_key = "sk-..."
   ```

4. **Preconstruir tu vectorstore** (solo la primera vez)

   ```bash
   python vectorstore.py \
     "ruta/a/tu/Libro1.epub" \
     "ruta/a/tu/Libro2.epub"
   ```

   Esto creará la carpeta `book_vectorstore/vectorstores/books_faiss` con el índice FAISS.

---

## ▶️ Ejecución

```bash
streamlit run app.py
```

Luego abre tu navegador en `http://localhost:8501`.

---

## 📚 Casos de uso

* **Agencias de marketing**: planificación de campañas semanales.
* **Tiendas de moda**: generación de ideas de publicaciones y guiones de reels.
* **Restaurantes**: calendario de contenido combinando menús y reseñas.
* **Educadores**: respuestas basadas en material PDF o CSV de cursos.

---

## 🤝 Contribuciones

1. Haz un **fork** del repositorio.
2. Crea una **rama** nueva (`git checkout -b feature/nombre`).
3. Realiza **commits** claros y concisos.
4. Envía un **pull request** describiendo tu mejora.

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**.

```
```
