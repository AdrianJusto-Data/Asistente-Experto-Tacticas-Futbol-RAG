# src/corpus.py
"""
Carga del corpus híbrido desde data/*.md.
cargar_todo() es el daily driver (notebook + Streamlit + reindexado).
doc_a_md / pdf_a_markdown se añaden en un bloque posterior (reconstrucción).
"""

from pathlib import Path
from collections import Counter
import yaml
from langchain_core.documents import Document

from src.config import DATA_DIR


def cargar_md(path: Path, data_dir: Path = DATA_DIR) -> Document:
    """
    Lee un .md con front matter YAML.
    Enriquece metadata con:
      - ruta   = path relativo a data_dir (str)
      - source = nombre del archivo (str)
    Lanza ValueError si falta el front matter (--- al inicio).
    """
    texto = path.read_text(encoding="utf-8")
    if not texto.startswith("---"):
        raise ValueError(f"{path}: falta front matter YAML (debe empezar por '---')")

    partes = texto.split("---", 2)
    if len(partes) < 3:
        raise ValueError(f"{path}: front matter mal formado (no cierra con '---')")

    _, fm_raw, body = partes
    metadata = yaml.safe_load(fm_raw) or {}

    metadata["ruta"] = str(path.relative_to(data_dir))
    metadata["source"] = path.name

    return Document(page_content=body.strip(), metadata=metadata)


def cargar_todo(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Carga todos los .md bajo data/, excluyendo data/imagenes/.
    Devuelve lista ordenada por ruta. Esperado: 123 docs.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_DIR no existe: {data_dir}")

    paths = sorted(data_dir.rglob("*.md"))
    docs = [
        cargar_md(p, data_dir)
        for p in paths
        if "imagenes" not in p.relative_to(data_dir).parts
    ]
    return docs


def resumen_por_fuente(docs: list[Document]) -> dict[str, int]:
    """Conteo por 'fuente' en metadata. Útil para verificar el split del corpus."""
    return dict(Counter(d.metadata.get("fuente", "desconocido") for d in docs))



# ---------------------------------------------------------------------------
# Búsqueda literal en el libro (diagnóstico / auditoría, no runtime del agente)
# ---------------------------------------------------------------------------
def _normalizar(texto: str) -> str:
    """Lowercase + quita acentos. Para búsquedas insensibles a 'fútbol'/'futbol'."""
    import unicodedata
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return texto


def buscar_en_libro(
    termino: str,
    docs: list[Document] | None = None,
    contexto_chars: int = 120,
) -> list[dict]:
    """
    Búsqueda literal (case+acentos-insensible) sobre los fragmentos del libro.
    Devuelve lista de hits con ruta, paginas y snippet.
    """
    if docs is None:
        docs = cargar_todo()

    termino_norm = _normalizar(termino)
    hits = []
    for d in docs:
        if d.metadata.get("fuente") != "libro":
            continue
        contenido = d.page_content
        contenido_norm = _normalizar(contenido)
        idx = contenido_norm.find(termino_norm)
        if idx == -1:
            continue

        ocurrencias = []
        start = 0
        while True:
            j = contenido_norm.find(termino_norm, start)
            if j == -1:
                break
            ocurrencias.append(j)
            start = j + 1

        s_start = max(0, idx - contexto_chars // 2)
        s_end = min(len(contenido), idx + len(termino) + contexto_chars // 2)

        hits.append({
            "ruta": d.metadata.get("ruta"),
            "paginas": (d.metadata.get("pagina_inicio"), d.metadata.get("pagina_fin")),
            "ocurrencias": len(ocurrencias),
            "snippet": "…" + contenido[s_start:s_end].replace("\n", " ") + "…",
        })

    return hits