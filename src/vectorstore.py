# src/vectorstore.py
"""
Chroma + retrieval + reindexado con control de TPM.

Tres modos de vectorstore:
- get_vectorstore(): conexión al Chroma persistente (notebook local).
- build_chroma(docs): Chroma efímero en RAM (Streamlit Cloud).
- reindexar(docs): reconstruye la colección persistente respetando 30k TPM.
"""

import time
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import (
    PERSIST_DIR,
    COLLECTION_NAME,
    N_DOCS_RETRIEVAL,
    BATCH_TOKENS_EMBEDDING,
)
from src.embeddings import get_embeddings


# ---------------------------------------------------------------------------
# Cliente chromadb (singleton por ruta; evita colisiones de PersistentClient)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_client(path: str):
    """Un solo cliente por ruta en el proceso (WinError 32 a raya)."""
    return chromadb.PersistentClient(path=path)


# ---------------------------------------------------------------------------
# Conexiones
# ---------------------------------------------------------------------------
def get_vectorstore() -> Chroma:
    """Chroma persistente en PERSIST_DIR (modo local/notebook)."""
    return Chroma(
        client=_get_client(str(PERSIST_DIR)),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def build_chroma(
    docs: list[Document],
    collection_name: str = COLLECTION_NAME,
    pausa_entre_lotes_seg: int = 60,
) -> Chroma:
    """
    Chroma efímero en RAM a partir de docs (modo Streamlit Cloud).
    Respeta TPM igual que reindexar(): lotes + sleep.
    Envolver en @st.cache_resource — paga el coste una sola vez por app.
    """
    client = chromadb.EphemeralClient()
    vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )
    _add_batches(vs, docs, pausa_entre_lotes_seg, verbose=False)
    return vs


# ---------------------------------------------------------------------------
# Reindexado con control de TPM
# ---------------------------------------------------------------------------
def _estimar_tokens(texto: str) -> int:
    """Aproximación conservadora: 1 token ≈ 4 caracteres."""
    return max(1, len(texto) // 4)


def _agrupar_por_tokens(
    docs: list[Document], max_tokens: int
) -> list[list[Document]]:
    """Agrupa docs en lotes cuyo total estimado no supere max_tokens."""
    batches, actual, tokens_actual = [], [], 0
    for doc in docs:
        t = _estimar_tokens(doc.page_content)
        if actual and tokens_actual + t > max_tokens:
            batches.append(actual)
            actual, tokens_actual = [], 0
        actual.append(doc)
        tokens_actual += t
    if actual:
        batches.append(actual)
    return batches


def _add_con_reintento(
    vs: Chroma,
    batch: list[Document],
    max_intentos: int = 3,
    pausa_inicial: int = 30,
) -> None:
    """Añade un lote; reintenta con backoff si huele a rate limit."""
    for intento in range(1, max_intentos + 1):
        try:
            vs.add_documents(batch)
            return
        except Exception as e:
            msg = str(e).lower()
            es_rate = "429" in str(e) or "rate" in msg or "quota" in msg
            if es_rate and intento < max_intentos:
                time.sleep(pausa_inicial * intento)
            else:
                raise


def _add_batches(
    vs: Chroma,
    docs: list[Document],
    pausa_seg: int,
    verbose: bool = False,
) -> None:
    """Lógica compartida por reindexar() y build_chroma()."""
    batches = _agrupar_por_tokens(docs, BATCH_TOKENS_EMBEDDING)
    if verbose:
        print(
            f"[vs] {len(docs)} docs en {len(batches)} lotes "
            f"(~{BATCH_TOKENS_EMBEDDING} tokens/lote)"
        )
    for i, batch in enumerate(batches, 1):
        _add_con_reintento(vs, batch)
        if verbose:
            print(f"[vs] lote {i}/{len(batches)} OK → total={vs._collection.count()}")
        if i < len(batches):
            time.sleep(pausa_seg)


def reindexar(
    docs: list[Document],
    collection_name: str = COLLECTION_NAME,
    persist_dir: str | None = None,
    pausa_entre_lotes_seg: int = 60,
    verbose: bool = True,
) -> Chroma:
    """
    Reconstruye la colección persistente: borra y repuebla con docs.
    DESTRUCTIVO por diseño. pausa_entre_lotes_seg=60 cubre 30k TPM.
    """
    path = persist_dir or str(PERSIST_DIR)
    client = _get_client(path)

    try:
        client.delete_collection(collection_name)
        if verbose:
            print(f"[vs] colección '{collection_name}' borrada")
    except Exception:
        if verbose:
            print(f"[vs] colección '{collection_name}' no existía")

    vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )
    _add_batches(vs, docs, pausa_entre_lotes_seg, verbose=verbose)

    if verbose:
        print(f"[vs] FINAL: {vs._collection.count()} docs en '{collection_name}'")
    return vs


# ---------------------------------------------------------------------------
# Retrieval (helper delgado)
# ---------------------------------------------------------------------------
def busqueda_mmr(
    vs: Chroma, query: str, k: int = N_DOCS_RETRIEVAL
) -> list[Document]:
    """MMR sobre el vectorstore. Firma estable para notebook y graph."""
    return vs.max_marginal_relevance_search(query, k=k)