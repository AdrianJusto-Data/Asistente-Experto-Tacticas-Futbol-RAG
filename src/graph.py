# src/graph.py
"""
Grafo del agente RAG: reformular → retrieve → generate.
build_graph(vs) es la factory; get_default_graph() cubre el notebook.
"""

import time
from functools import lru_cache
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.config import (
    MODEL_LLM, MODEL_LLM_FALLBACK, PAUSA_FALLBACK_SEG, N_DOCS_RETRIEVAL
)

# ---------------------------------------------------------------------------
# LLMs (singleton de módulo)
# ---------------------------------------------------------------------------
llm_principal = ChatGoogleGenerativeAI(
    model=MODEL_LLM,
    temperature=0.2,
    max_output_tokens=800,
    max_retries=2,
)

llm_respaldo = ChatGoogleGenerativeAI(
    model=MODEL_LLM_FALLBACK,
    temperature=0.2,
    max_output_tokens=800,
)

def invocar_con_fallback(mensajes, pausa_seg: int = PAUSA_FALLBACK_SEG):
    """Invoca el LLM con fallback y pausa configurable (free tier)."""
    try:
        respuesta = llm_principal.invoke(mensajes)
    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
            print(f"⚠️ Principal saturado. Esperando {pausa_seg}s y usando respaldo...")
            time.sleep(pausa_seg)
            respuesta = llm_respaldo.invoke(mensajes)
        else:
            raise
    time.sleep(pausa_seg)  # respeta el RPM del free tier
    return respuesta

# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[Document]
    query: str

# ---------------------------------------------------------------------------
# Helper normalizador de content
# ---------------------------------------------------------------------------
def extraer_texto(mensaje) -> str:
    c = mensaje.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in c
        )
    return str(c)

# ---------------------------------------------------------------------------
# SYSTEM_PROMPT_V2 (internamente v4) — FUENTE DE VERDAD + EXPLICACIÓN DIDÁCTICA
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_V2 = """
# ROL
Eres un historiador y analista táctico de fútbol de élite. Dominas tanto la evolución
estratégica clásica (desde los años 40) como los sistemas híbridos modernos del fútbol actual.
Tu registro es técnico, analítico y divulgativo: usas terminología futbolística precisa
pero accesible, con tono educativo.

# FUENTE DE VERDAD (CRÍTICO)
Cada documento recuperado tiene DOS partes:
1. Una METADATA estructurada (Época, Formación base, Entrenador, etc.).
2. Un CONTENIDO narrativo (texto del libro o de las fichas).

AMBAS partes son válidas. La METADATA es la fuente de verdad para datos estructurados.
El CONTENIDO es la fuente de verdad para explicaciones, conceptos y narrativa histórica.

# REGLA DE ORO: DOS TIPOS DE CONSULTAS
Debes identificar qué tipo de pregunta te están haciendo y adaptar tu respuesta:

## TIPO 1: CONSULTA DE FICHA (Táctica o Formación del catálogo)
*Ocurre cuando* la pregunta menciona o sugiere una entidad con ficha en el corpus (ej. "Catenaccio", "4-4-2", "Milan de Sacchi").
*Cómo responder*: Usa EXACTAMENTE el formato estructurado (Entidad principal, Tipo, Época, etc.).
*Restricción*: Los campos de la ficha deben salir SOLO de la METADATA o de las fichas curadas.

## TIPO 2: CONSULTA CONCEPTUAL / NARRATIVA (Conceptos del libro)
*Ocurre cuando* la pregunta busca explicar un término, un concepto histórico o una evolución (ej. "¿qué es el tercer defensa?", "¿qué es la pirámide invertida?", "háblame de la evolución del fútbol soviético").
*Cómo responder*: NO uses el formato de ficha. Redacta una respuesta narrativa y didáctica (2-3 párrafos) usando el CONTENIDO de los fragmentos del libro (`libro_fragmento`) recuperados. Cita la época o el autor si el libro lo menciona.

# REGLAS DE RESPUESTA
1. **Fidelidad en Fichas**: En el Tipo 1, no inventes datos estructurados.
2. **Explicación Didáctica**: En el Tipo 2, tienes permiso para hilar los fragmentos del libro y usar tu conocimiento general como historiador para explicar el concepto. Si el libro habla de un concepto pero no lo define del todo, puedes completarlo con conocimiento táctico verídico.
3. **Manejo de la Ignorancia**: SOLO si el contexto recuperado NO CONTIENE NADA relacionado con la pregunta (ni ficha, ni fragmento relevante), responde: "Lo siento, esa información no se encuentra en mi base de datos táctica actual."

# FORMATO DE RESPUESTA
- **Si es Tipo 1 (Ficha)**, usa esta plantilla Markdown:
  - **Entidad principal**: [Nombre de la táctica o formación]
  - **Tipo**: [tactica | formacion]
  - **Época**: [...]
  - **Descripción**: [1-2 frases]
  - **Fortalezas**: [...]
  - **Debilidades**: [...]
  - **Entrenador / Equipo icónico**: [...]
  - **Formación base** (solo si es una táctica): [...]
  - **Imagen de referencia**: [aplica la sección IMAGEN]
  - **Explicación Didáctica Ampliada**: [1 párrafo usando tu conocimiento general]

- **Si es Tipo 2 (Conceptual)**, responde con texto libre:
  - Un título en negrita con el concepto.
  - 2-3 párrafos narrativos explicando el concepto, su origen y contexto.
  - Si el contexto recuperado incluye una imagen, añade: **Imagen de referencia**: [aplica la sección IMAGEN].

# IMAGEN
- Usa EXACTAMENTE el valor del campo `Imagen` de la METADATA (si existe).
- Si no hay campo `Imagen` o está vacío, no escribas nada sobre la imagen.
"""

# ---------------------------------------------------------------------------
# Prompt del reformulador — copia íntegra
# ---------------------------------------------------------------------------
PROMPT_REFORMULADOR = """Eres un reescritor de consultas para un sistema RAG de táctica futbolística.

Recibes la conversación previa y la última pregunta del usuario. Tu tarea:
- Si la última pregunta se entiende POR SÍ SOLA (menciona entidad concreta: "Catenaccio", "4-4-2", "Gegenpressing"...), devuélvela SIN CAMBIOS.
- Si contiene referencias anafóricas ("esa táctica", "su entrenador", "lo", "eso", "este sistema"...), REESCRÍBELA como consulta autónoma usando la entidad de la conversación previa.

Devuelve SOLO la consulta reescrita, sin comillas, sin explicaciones."""

# ---------------------------------------------------------------------------
# Etiquetas legibles de metadata
# ---------------------------------------------------------------------------
ETIQUETAS_META = {
    "tipo":                 "Tipo",
    "tactica":              "Táctica",
    "formacion":            "Formación",
    "era":                  "Época",
    "epoca_origen":         "Época de origen",
    "formacion_base":       "Formación base",
    "entrenador_principal": "Entrenador principal",
    "maximo_exponente":     "Máximo exponente",
    "tipo_lineas":          "Líneas",
    "equipos_iconicos":     "Equipos icónicos",
    "imagen_url":           "Imagen",
}

# ---------------------------------------------------------------------------
# Nodos
# ---------------------------------------------------------------------------
def nodo_reformulador(state: AgentState) -> dict:
    mensajes = state["messages"]
    humanos = [m for m in mensajes if isinstance(m, HumanMessage)]

    if len(humanos) <= 1:
        query = humanos[0].content if humanos else ""
        print("🧭 [reformulador] primer turno → query original")
        return {"query": query}

    historial_txt = "\n".join(
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: "
        f"{m.content if isinstance(m.content, str) else str(m.content)[:200]}"
        for m in mensajes[:-1]
    )
    ultima = humanos[-1].content

    msgs_llm = [
        SystemMessage(content=PROMPT_REFORMULADOR),
        HumanMessage(
            content=f"HISTORIAL:\n{historial_txt}\n\nÚLTIMA PREGUNTA:\n{ultima}\n\nConsulta autónoma:"
        ),
    ]
    respuesta = llm_principal.invoke(msgs_llm)
    time.sleep(PAUSA_FALLBACK_SEG)

    query_raw = extraer_texto(respuesta)
    query = query_raw.strip().strip('"').strip("'")
    print(f"🧭 [reformulador] '{ultima}' → '{query}'")
    return {"query": query}

def nodo_generacion(state: AgentState) -> dict:
    documentos = state.get("context", [])
    ultimo_mensaje = state["messages"][-1]
    pregunta = (
        ultimo_mensaje.content
        if hasattr(ultimo_mensaje, "content")
        else str(ultimo_mensaje)
    )

    if not documentos:
        contexto_str = "(sin documentos recuperados)"
    else:
        bloques = []
        for i, doc in enumerate(documentos, start=1):
            md = doc.metadata
            tipo = md.get("tipo", "?")
            nombre = (
                md.get("tactica")
                or md.get("formacion")
                or md.get("titulo")
                or md.get("source")
                or "?"
            )
            lineas = [f"[DOC {i} · tipo={tipo} · entidad={nombre}]"]

            lineas.append("METADATA:")
            for clave, etiqueta in ETIQUETAS_META.items():
                valor = md.get(clave)
                if valor not in (None, "", "No disponible en el registro"):
                    lineas.append(f"  - {etiqueta}: {valor}")

            lineas.append(f"CONTENIDO: {doc.page_content}")
            bloques.append("\n".join(lineas))
        contexto_str = "\n\n".join(bloques)

    mensajes = [
        SystemMessage(content=SYSTEM_PROMPT_V2),
        HumanMessage(
            content=(
                f"CONTEXTO RECUPERADO:\n{contexto_str}\n\n"
                f"PREGUNTA DEL USUARIO:\n{pregunta}"
            )
        ),
    ]

    print(
        f"🧠 [generation] prompt construido · {len(documentos)} docs · "
        f"contexto={len(contexto_str)} chars"
    )

    respuesta = invocar_con_fallback(mensajes)
    return {"messages": [respuesta]}

# ---------------------------------------------------------------------------
# build_graph(vs) — factory con closure
# ---------------------------------------------------------------------------
def build_graph(vs, model_name=MODEL_LLM):
    # Re-creamos los LLMs locales para este grafo específico
    llm_principal = ChatGoogleGenerativeAI(model=model_name, temperature=0.2, max_output_tokens=800, max_retries=2)
    llm_respaldo = ChatGoogleGenerativeAI(model=MODEL_LLM_FALLBACK, temperature=0.2, max_output_tokens=800)
    
    # Actualizamos las referencias globales para que invocar_con_fallback use el modelo correcto
    globals()["llm_principal"] = llm_principal
    globals()["llm_respaldo"] = llm_respaldo

    def nodo_retrieval(state: AgentState) -> dict:
        pregunta = state.get("query") or state["messages"][-1].content
        documentos = vs.max_marginal_relevance_search(
            query=pregunta,
            k=6,
            fetch_k=20,
        )
        print(f"🔍 [retrieval] Pregunta: {pregunta[:80]}...")
        print(f"   → {len(documentos)} docs recuperados:")
        for d in documentos:
            nombre = (
                d.metadata.get("tactica")
                or d.metadata.get("formacion")
                or d.metadata.get("titulo")
                or d.metadata.get("source")
                or "?"
            )
            tipo = d.metadata.get("tipo", "?")
            print(f"     · [{tipo}] {nombre}")
        return {"context": documentos}

    builder = StateGraph(AgentState)
    builder.add_node("reformular", nodo_reformulador)
    builder.add_node("retrieve", nodo_retrieval)
    builder.add_node("generate", nodo_generacion)

    builder.add_edge(START, "reformular")
    builder.add_edge("reformular", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=MemorySaver())

@lru_cache(maxsize=1)
def get_default_graph():
    from src.vectorstore import get_vectorstore
    return build_graph(get_vectorstore())