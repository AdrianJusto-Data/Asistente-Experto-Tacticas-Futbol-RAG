# app.py
"""
Asistente Experto en Tácticas de Fútbol — Streamlit app.
"""
import os
# ⚠️ Debe ir ANTES de importar src: pausas a 0 para Streamlit
os.environ.setdefault("PAUSA_FALLBACK_SEG", "0")

import time
import re
import uuid
import zipfile
import streamlit as st
from langchain_core.messages import HumanMessage

from src.corpus import cargar_todo, resumen_por_fuente, buscar_en_libro
from src.vectorstore import build_chroma, get_vectorstore
from src.graph import build_graph, extraer_texto, SYSTEM_PROMPT_V2
from src.config import DATA_DIR, PERSIST_DIR

# ---------------------------------------------------------------------------
# Configuración de la página y CSS Personalizado
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Asistente Tácticas",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #1e1e24;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _nuevo_thread_id() -> str:
    return f"st_{int(time.time())}"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = _nuevo_thread_id()
if "mensajes_ui" not in st.session_state:
    st.session_state.mensajes_ui = []
if "prompt_generado" not in st.session_state:
    st.session_state.prompt_generado = None

def set_prompt(texto):
    st.session_state.prompt_generado = texto

# ---------------------------------------------------------------------------
# Sidebar y Selector de Modelo
# ---------------------------------------------------------------------------
with st.sidebar:
    icono_path = DATA_DIR / "imagenes" / "Icono.jpg"
    if icono_path.exists():
        _, col_icon, _ = st.columns([1, 2, 1])
        with col_icon:
            st.image(str(icono_path), use_container_width=True)
    
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>Asistente Táctico</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: gray;'>RAG con Gemini + LangGraph</p>", unsafe_allow_html=True)

    if st.button("🔄 Nueva sesión", use_container_width=True):
        st.session_state.thread_id = _nuevo_thread_id()
        st.session_state.mensajes_ui = []
        st.rerun()

    st.divider()
    
    # NUEVO: Selector de modelo
    st.markdown("### 🧠 Motor de IA")
    modelo_sel = st.selectbox(
        "Selecciona el modelo:",
        ["gemini-3.5-flash-lite", "gemini-3.8-flash"],
        help="Lite es más rápido, Flash es más analítico."
    )

# ---------------------------------------------------------------------------
# Backend cacheado (Depende del modelo elegido)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _init_backend(model_name: str):
    docs = cargar_todo()

    # ── NUEVO: si no hay chroma persistente, intentar seed empaquetado ──
    # En local: chroma_db_v4/ ya existe → se salta el seed.
    # En Cloud: chroma_db_v4/ no existe → descomprime chroma_db_v4.zip (0 llamadas a Gemini).
    if not (PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())):
        seed_zip = PERSIST_DIR.parent / f"{PERSIST_DIR.name}.zip"
        if seed_zip.exists():
            with st.spinner("📦 Descomprimiendo vectorstore precalculado…"):
                with zipfile.ZipFile(seed_zip, "r") as zf:
                    zf.extractall(".")
            st.toast("✅ Vectorstore listo (seed empaquetado)", icon="📦")

    if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
        vs = get_vectorstore()
    else:
        # Fallback: si no había seed, reindexar (consume cuota — solo local sin zip)
        vs = build_chroma(docs, pausa_entre_lotes_seg=0)

    graph = build_graph(vs, model_name=model_name)

    tacticas = sorted(list(set(
        d.metadata.get("tactica") for d in docs
        if d.metadata.get("tactica") and d.metadata.get("fuente") == "curado"
    )))

    # Lista de imágenes para el catálogo (ignorando el icono y aceptando .jpg/.png)
    imagenes = sorted([
        p for p in (DATA_DIR / "imagenes").glob("*.*")
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"] and "icono" not in p.name.lower()
    ])

    return graph, len(docs), resumen_por_fuente(docs), tacticas, imagenes

with st.spinner("⏳ Inicializando base de conocimiento…"):
    graph, n_docs, resumen, tacticas, imagenes = _init_backend(modelo_sel)

# ---------------------------------------------------------------------------
# Helpers visuales
# ---------------------------------------------------------------------------
def _imagenes_locales(docs, texto_respuesta=""):
    imagenes_res = []
    for d in docs:
        url = d.metadata.get("imagen_url") or ""
        if url and not url.startswith("http"):
            p = DATA_DIR / url
            if p.exists() and p not in imagenes_res:
                imagenes_res.append(p)
                
    texto_lower = texto_respuesta.lower()
    todas = list((DATA_DIR / "imagenes").glob("*.png"))
    for img_path in todas:
        nombre_limpio = img_path.stem.replace("_", " ").lower()
        if nombre_limpio in texto_lower and img_path not in imagenes_res:
            imagenes_res.append(img_path)
    return imagenes_res

def _limpiar_linea_imagen(texto: str, hay_imagen_local: bool) -> str:
    if not hay_imagen_local:
        return texto
    return re.sub(r"(- \*\*Imagen de referencia\*\*:).*", r"\1 _(ver imagen adjunta)_", texto)

def _limpiar_texto_prompt(texto: str) -> str:
    texto = re.sub(r"- \*\*Formación base\*\* \(solo si es una táctica\):\s*", "", texto)
    texto = re.sub(r"- \*\*Formación base\*\* \(solo si Tipo=\"tactica\"\):\s*", "", texto)
    return texto

def _render_tarjeta_doc(doc_meta: dict, index: int):
    tipo = doc_meta.get("tipo", "?")
    entidad = doc_meta.get("tactica") or doc_meta.get("formacion") or doc_meta.get("titulo") or doc_meta.get("source", "?")
    ruta = doc_meta.get("ruta") or doc_meta.get("source", "")
    era = doc_meta.get("era", "N/A")
    bg_color = "#1e293b" if tipo == "libro_fragmento" else "#064e3b"
    border_color = "#3b82f6" if tipo == "libro_fragmento" else "#10b981"
    html = f"""
    <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; padding: 10px 15px; border-radius: 5px; margin-bottom: 8px; font-size: 0.9rem;">
        <strong style="color: #fff;">[{index}] {entidad}</strong> 
        <span style="color: #94a3b8; font-size: 0.8rem;">({tipo} | {era})</span><br>
        <span style="color: #cbd5e1; font-size: 0.8rem;">📄 {ruta}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def _render_mensaje(msg: dict) -> None:
    avatar = "👤" if msg["role"] == "user" else "⚽"
    with st.chat_message(msg["role"], avatar=avatar):
        imagenes = msg.get("imagenes_urls", [])
        if imagenes:
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                if len(imagenes) == 1:
                    st.image(str(imagenes[0]), width="stretch")
                    img_actual = imagenes[0]
                else:
                    msg_id = msg.get("id", str(uuid.uuid4()))
                    idx_key = f"img_idx_{msg_id}"
                    if idx_key not in st.session_state:
                        st.session_state[idx_key] = 0
                    idx = st.session_state[idx_key]
                    img_actual = imagenes[idx]
                    st.image(str(img_actual), width="stretch")
                    c1, c2, c3 = st.columns([1, 2, 1])
                    if c1.button("⬅️", key=f"prev_{msg_id}", use_container_width=True):
                        st.session_state[idx_key] = (idx - 1) % len(imagenes)
                        st.rerun()
                    c2.markdown(f"<div style='text-align: center; font-size: 0.9rem; padding-top: 5px;'>{idx+1} / {len(imagenes)}</div>", unsafe_allow_html=True)
                    if c3.button("➡️", key=f"next_{msg_id}", use_container_width=True):
                        st.session_state[idx_key] = (idx + 1) % len(imagenes)
                        st.rerun()
                with open(img_actual, "rb") as f:
                    st.download_button("📥 Descargar", data=f, file_name=os.path.basename(img_actual), mime="image/png", key=f"dl_{msg.get('id', str(uuid.uuid4()))}", use_container_width=True)
            with col_txt:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])

        if msg.get("query_reformulada") and msg["query_reformulada"] != msg.get("prompt_original"):
            st.caption(f"🧠 *El asistente entendió:* «{msg['query_reformulada']}»")

        if msg["role"] == "assistant" and not msg.get("error"):
            col_fb, col_time = st.columns([1, 4])
            with col_fb:
                st.feedback("thumbs", key=f"fb_{msg.get('id', str(uuid.uuid4()))}")
            with col_time:
                if "tiempo" in msg:
                    st.caption(f"⏱️ Generado en {msg['tiempo']:.2f}s")

        if msg.get("docs"):
            with st.expander("🔍 Documentos recuperados"):
                for i, m in enumerate(msg["docs"], 1):
                    _render_tarjeta_doc(m, i)

# ---------------------------------------------------------------------------
# Interfaz principal: Pestañas (Tabs)
# ---------------------------------------------------------------------------
st.title("⚽ Asistente Experto en Tácticas de Fútbol")
tab_chat, tab_catalogo = st.tabs(["💬 Chat Inteligente", "🖼️ Catálogo Visual"])

# --- PESTAÑA 1: CHAT ---
with tab_chat:
    for msg in st.session_state.mensajes_ui:
        _render_mensaje(msg)

    if not st.session_state.mensajes_ui:
        st.markdown("#### 💡 Prueba a preguntar:")
        cols = st.columns(3)
        ejemplos = ["¿Qué es el Catenaccio?", "¿Cómo jugaba el Milan de Sacchi?", "Explícame la Zona Mista"]
        for i, ej in enumerate(ejemplos):
            cols[i].button(ej, on_click=set_prompt, args=(ej,), use_container_width=True)

    prompt_input = st.chat_input("Pregunta sobre táctica…")
    prompt = prompt_input
    if not prompt and st.session_state.prompt_generado:
        prompt = st.session_state.prompt_generado
        st.session_state.prompt_generado = None

    if prompt:
        msg_user = {"role": "user", "content": prompt}
        st.session_state.mensajes_ui.append(msg_user)
        _render_mensaje(msg_user)

        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        t0 = time.time()
        
        with st.status("Procesando consulta...", expanded=True) as status:
            st.write("🧭 Reformulando consulta...")
            status.update(label="Buscando en documentos...", state="running")
            try:
                salida = graph.invoke({"messages": [HumanMessage(content=prompt)], "context": [], "query": ""}, config=config)
                st.write("🔍 Recuperando contexto...")
                docs = salida.get("context", [])
                st.write("🧠 Redactando respuesta...")
                respuesta_raw = extraer_texto(salida["messages"][-1])
                imagenes_msg = _imagenes_locales(docs, respuesta_raw)
                respuesta = _limpiar_linea_imagen(respuesta_raw, len(imagenes_msg) > 0)
                respuesta = _limpiar_texto_prompt(respuesta)
                
                status.update(label="¡Listo!", state="complete", expanded=False)
                t1 = time.time()
                msg_bot = {
                    "role": "assistant", "content": respuesta, "imagenes_urls": imagenes_msg,
                    "docs": [d.metadata for d in docs], "query_reformulada": salida.get("query", ""),
                    "prompt_original": prompt, "id": str(uuid.uuid4()), "tiempo": t1 - t0
                }
            except Exception as e:
                status.update(label="Error", state="error", expanded=True)
                msg_bot = {"role": "assistant", "content": f"❌ Error: `{type(e).__name__}`", "error": True}

        st.session_state.mensajes_ui.append(msg_bot)
        _render_mensaje(msg_bot)

# --- PESTAÑA 2: CATÁLOGO VISUAL ---
with tab_catalogo:
    st.header("Galería de Tácticas")
    st.caption("Explora visualmente las formaciones y tácticas. Haz clic en 'Preguntar' para que el asistente te explique los detalles.")
    
    # Cuadrícula de 3 columnas
    cols = st.columns(3)
    for i, img_path in enumerate(imagenes):
        with cols[i % 3]:
            st.image(str(img_path), use_container_width=True)
            nombre_limpio = img_path.stem.replace("_", " ").title()
            st.markdown(f"**{nombre_limpio}**")
            if st.button(f"Preguntar por {nombre_limpio}", key=f"btn_cat_{i}", use_container_width=True):
                set_prompt(f"¿Qué es {nombre_limpio}?")
                st.rerun()

# ---------------------------------------------------------------------------
# Widgets informativos del Sidebar (debajo de todo)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.divider()
    with st.expander("🎯 Búsqueda Guiada", expanded=False):
        tactica_sel = st.selectbox("Por Táctica", ["Ninguna"] + tacticas)
        if tactica_sel != "Ninguna" and st.button(f"Explicar {tactica_sel}", use_container_width=True):
            set_prompt(f"¿Qué es el {tactica_sel}?"); st.rerun()
            
        formacion_sel = st.selectbox("Por Formación", ["4-4-2", "4-3-3", "5-3-2", "4-2-3-1", "3-5-2", "4-4-1-1"])
        if st.button(f"Buscar con {formacion_sel}", use_container_width=True):
            set_prompt(f"¿Qué tácticas utilizaron la formación {formacion_sel}?"); st.rerun()

        epoca_sel = st.selectbox("Por Época", ["1920s", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"])
        if st.button(f"Explorar los {epoca_sel}", use_container_width=True):
            set_prompt(f"¿Qué tácticas destacaron en los {epoca_sel}?"); st.rerun()

    with st.expander("📚 Explorar el libro (Búsqueda Literal)"):
        busqueda = st.text_input("Palabra clave:", placeholder="ej. tercer defensa")
        if busqueda:
            hits = buscar_en_libro(busqueda)
            if hits:
                st.success(f"{len(hits)} fragmentos")
                for h in hits[:2]:
                    st.markdown(f"**{h['ruta']}** (págs. {h['paginas'][0]}-{h['paginas'][1]})\n_{h['snippet'][:60]}..._")
            else:
                st.warning("Sin resultados exactos.")

    with st.expander("📊 Arquitectura del Agente"):
        try:
            png = graph.get_graph().draw_mermaid_png()
            st.image(png)
        except:
            st.warning("No se pudo generar el diagrama.")

    with st.expander("📝 System Prompt"):
        st.code(SYSTEM_PROMPT_V2, language="markdown")

    if st.session_state.mensajes_ui:
        st.divider()
        chat_md = "# Conversación Asistente Táctico\n\n"
        for m in st.session_state.mensajes_ui:
            chat_md += f"### {'Usuario' if m['role']=='user' else 'Asistente'}\n{m['content']}\n\n"
        st.download_button("📥 Descargar Chat (.md)", data=chat_md, file_name="conversacion.md", mime="text/markdown", use_container_width=True)

    st.divider()
    st.caption(f"**Corpus:** {n_docs} docs | **Sesión:** `{st.session_state.thread_id}`")