"""
callbacks.py

Módulo que define handlers de callback para integrar LangChain con Streamlit.
Permite mostrar respuestas en streaming token a token y otros hooks de UI personalizados.
"""

# ───────────────────────────────────────────────────────────────────────────────
# Imports
# ───────────────────────────────────────────────────────────────────────────────
from langchain_core.callbacks import BaseCallbackHandler


# ───────────────────────────────────────────────────────────────────────────────
# StreamHandler
# ───────────────────────────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    """
    Callback handler que muestra cada token generado por el LLM
    en tiempo real dentro de un contenedor de Streamlit.
    """
    def __init__(self, container, initial_text: str = ""):
        """
        Args:
            container: objeto Streamlit que debe tener método .markdown()
                       para renderizar texto actualizado.
            initial_text (str): texto inicial antes de iniciar el streaming.
        """
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        """
        Método invocado por LangChain con cada nuevo token.
        Acumula el token y re-renderiza el contenedor completo.

        Args:
            token (str): fragmento de texto generado por el LLM.
        """
        self.text += token
        # Reemplaza el contenido previo por la nueva cadena completa
        self.container.markdown(self.text)


# ───────────────────────────────────────────────────────────────────────────────
# StreamlitUICallbackHandler
# ───────────────────────────────────────────────────────────────────────────────
class StreamlitUICallbackHandler(BaseCallbackHandler):
    """
    Ejemplo de handler adicional para personalizaciones de UI en Streamlit.
    Actualmente no implementa lógica; se puede extender según necesidades.
    """
    def __init__(self):
        super().__init__()
