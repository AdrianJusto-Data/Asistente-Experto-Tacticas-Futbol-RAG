# src/chat.py
"""
Chat interactivo en terminal con memoria multi-turno.
Uso: notebook (Jupyter) o ejecución directa desde VS Code.
En Streamlit no se usa — app.py hará st.chat_input + graph.invoke() directamente.
"""

import time
from langchain_core.messages import HumanMessage

from src.graph import extraer_texto


def chat_interactivo(graph, thread_id_inicial: str | None = None) -> str:
    """
    Bucle de chat con memoria. Comandos: /salir, /reset, /historial.
    Devuelve el thread_id final (útil para tests o para reanudar la sesión).
    """
    thread_id = thread_id_inicial or f"sesion_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print(f"🤖 Asistente de Tácticas · sesión: {thread_id}")
    print("Comandos: /salir · /reset · /historial")
    print("=" * 60)

    while True:
        try:
            pregunta = input("\n👤 Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Saliendo…")
            break

        if not pregunta:
            continue

        # --- Comandos ---
        if pregunta.lower() == "/salir":
            print("👋 Hasta luego.")
            break

        if pregunta.lower() == "/reset":
            thread_id = f"sesion_{int(time.time())}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"🔄 Nueva sesión: {thread_id}")
            continue

        if pregunta.lower() == "/historial":
            state = graph.get_state(config)
            msgs = state.values.get("messages", [])
            if not msgs:
                print("📭 Sin mensajes en esta sesión.")
            else:
                for i, m in enumerate(msgs, 1):
                    rol = "👤" if isinstance(m, HumanMessage) else "🤖"
                    texto = extraer_texto(m)[:120].replace("\n", " ")
                    print(f"  {i}. {rol} {texto}…")
            continue

        # --- Invocación normal ---
        try:
            salida = graph.invoke(
                {"messages": [HumanMessage(content=pregunta)],
                 "context": [], "query": ""},
                config=config,
            )
            respuesta = extraer_texto(salida["messages"][-1])
            print(f"\n🤖 {respuesta}")
        except Exception as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")

    return thread_id