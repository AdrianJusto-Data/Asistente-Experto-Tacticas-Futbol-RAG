# src/embeddings.py
"""
Wrapper de embeddings de Gemini.
Expone get_embeddings() como singleton reutilizable (notebook + Streamlit).
"""

from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import MODEL_EMBEDDING, GOOGLE_API_KEY


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Devuelve un singleton del wrapper de embeddings de Gemini.
    - Reutiliza el cliente HTTP entre llamadas (importante en reindexado).
    - Compatible con @st.cache_resource en Streamlit (no colisiona).
    """
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY no está definida. Revisa api.env y que "
            "src/config.py la haya cargado correctamente."
        )
    return GoogleGenerativeAIEmbeddings(
        model=MODEL_EMBEDDING,
        google_api_key=GOOGLE_API_KEY,
    )


def test_embeddings(texto: str = "prueba de humo") -> dict:
    """
    Prueba de humo: embeddea un texto y devuelve dimensión + primeros valores.
    Pensada para validar que la key y el modelo funcionan antes de reindexar.
    """
    vec = get_embeddings().embed_query(texto)
    return {
        "texto":      texto,
        "dim":        len(vec),
        "primeros_3": vec[:3],
    }