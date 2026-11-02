# src/config.py
"""
Configuración central del proyecto.
Carga api.env y expone constantes de rutas, modelos y parámetros RAG.
Única fuente de verdad: si un valor aparece en más de un sitio, vive aquí.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# --- Raíz del proyecto (robusta a cwd, válida en notebook y Streamlit Cloud) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Carga de variables de entorno ---
ENV_PATH = PROJECT_ROOT / "api.env"
load_dotenv(dotenv_path=ENV_PATH)

# Claves: se aceptan ambas por compatibilidad con el notebook y con langchain
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY

# Puente a os.environ: langchain-google-genai lee GOOGLE_API_KEY del entorno
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Rutas de datos ---
DATA_DIR      = PROJECT_ROOT / "data"
IMAGENES_DIR  = DATA_DIR / "imagenes"
PERSIST_DIR   = PROJECT_ROOT / "chroma_db_v4"

# --- Chroma ---
COLLECTION_NAME = "tacticas_futbol"

# --- Modelos ---
MODEL_EMBEDDING     = "gemini-embedding-2"
MODEL_LLM           = "gemini-3.5-flash-lite"
MODEL_LLM_FALLBACK  = "gemini-3.8-flash"

# --- Parámetros RAG ---
N_DOCS_RETRIEVAL       = 4
BATCH_TOKENS_EMBEDDING = 25_000   # margen bajo 30k TPM (free tier)
PAUSA_FALLBACK_SEG     = int(os.getenv("PAUSA_FALLBACK_SEG", "15"))


def validar_config() -> dict:
    """Comprueba presencia de claves y rutas clave. Devuelve dict de estado."""
    return {
        "ENV_PATH_existe":      ENV_PATH.exists(),
        "GEMINI_API_KEY":       bool(GEMINI_API_KEY),
        "GOOGLE_API_KEY":       bool(GOOGLE_API_KEY),
        "GOOGLE_API_KEY_en_env":bool(os.getenv("GOOGLE_API_KEY")),
        "DATA_DIR_existe":      DATA_DIR.exists(),
        "IMAGENES_DIR_existe":  IMAGENES_DIR.exists(),
        "PERSIST_DIR_existe":   PERSIST_DIR.exists(),
    }