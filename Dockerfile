# Dockerfile para desplegar EstrategIA MKT en Hugging Face Spaces (Streamlit)

FROM python:3.10-slim

# 1) Definir directorio de trabajo
WORKDIR /app

# 2) Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copiar el código fuente
COPY . .

# 4) Exponer el puerto de Streamlit
EXPOSE 8501

# 5) Variables de entorno recomendadas para Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 6) Comando por defecto
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
