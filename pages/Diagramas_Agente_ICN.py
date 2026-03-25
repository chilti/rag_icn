import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Diagramas del Agente RAG ICN", page_icon="📊")

st.title("📊 Diagramas del Agente RAG - Instituto de Ciencias Nucleares (ICN)")
st.write("Esta sección muestra la arquitectura y el flujo de funcionamiento del Agente RAG diseñado para el ICN.")

st.write("El agente tiene acceso a la producción científica del Instituto de Ciencias Nucleares de la UNAM (ICN) mediante varias herramientas: 1) recuperación de registros de artículos con Título+Resumen+Autores+Keywords, 2) Base de datos vectorial con el texto completo de alrededor de 1200 artículos (vectorizados con ptext-embedding-nomic-embed-text-v1.5 y un \"chunking\" de tamaño fijo) 3) OpenAlex mediante API, 4) búsqueda en la web con DuckDuckGo y 5) búsqueda en la wikipedia. El agente utiliza el LLM para determinar qué herramienta (s) utilizar. En este prototipo la conversación se mantiene mientras no se actualice la página, aun no se implementa la gestión de memoria. SI LA CONVERSACIÓN SE VUELVE MUY LARGA, REINICIE EL CHAT.")

# Función para renderizar diagramas Mermaid
def mermaid_chart(code: str, height: int = 400):
    components.html(
        f"""
        <div class="mermaid">
        {code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height,
        scrolling=True,
    )

# -------------------------
# Diagrama 1: Arquitectura
# -------------------------
st.header("Arquitectura General")
mermaid_chart("""
flowchart TD
    U[Usuario en Streamlit] -->|Pregunta| S[Interfaz Streamlit]
    S --> A[Agente RAG ICN]
    A --> H1["ChromaDB\\n(ICN títulos+resúmenes+autores+keywords)"]
    A --> H2["ChromaDB\\n(ICN texto completo ~1200 artículos)"]
    A --> H3["OpenAlex API\\n(DOI, Autores, Trabajos)"]
    A --> H4["DuckDuckGo\\nWeb Search"]
    A --> H5[Wikipedia]
    H1 --> A
    H2 --> A
    H3 --> A
    H4 --> A
    H5 --> A
    A --> L["LLM local\\nLM Studio"]
    L --> S
    S --> U
""", height=450)

# -------------------------
# Diagrama 2: Secuencia
# -------------------------
st.header("Diagrama de Secuencia")
mermaid_chart("""
sequenceDiagram
    actor U as Usuario
    participant S as Streamlit
    participant A as Agente ICN
    participant L as LLM
    participant C1 as ChromaDB (títulos+resúmenes)
    participant C2 as ChromaDB (texto completo)
    participant O as OpenAlex
    participant D as DuckDuckGo
    participant W as Wikipedia

    U->>S: Pregunta
    S->>A: Envía mensaje
    A->>L: Analiza intención
    L-->>A: Plan de acción
    A->>C1: Consulta títulos+resúmenes (prioridad)
    C1-->>A: Devuelve resultados
    A->>C2: (Opcional) Busca en texto completo
    C2-->>A: Devuelve fragmentos
    A->>O: (Opcional) Consulta por DOI/autor
    O-->>A: Devuelve metadatos
    A->>D: (Opcional) Busca info actualizada
    D-->>A: Devuelve snippets
    A->>W: (Opcional) Busca contexto enciclopédico
    W-->>A: Devuelve resumen
    A->>L: Integra evidencia
    L-->>A: Redacta respuesta en español
    A->>S: Envía respuesta
    S->>U: Muestra resultado
""", height=650)

# -------------------------
# Diagrama 3: Pipeline
# -------------------------
st.header("Pipeline paso a paso")
mermaid_chart("""
flowchart LR
    A[Usuario pregunta en Streamlit] 
    --> B[Agente RAG ICN recibe entrada]

    B --> C[LLM local analiza intención]
    C --> D[Decide herramienta con ReAct]

    D -->|Prioridad| E["ChromaDB\\n(ICN títulos+resúmenes+autores+keywords)"]
    D -->|Opcional| F["ChromaDB\\n(ICN texto completo ~1200 artículos)"]
    D -->|Opcional| G["OpenAlex API\\n(DOI, Autores, Trabajos)"]
    D -->|Opcional| H[DuckDuckGo Web Search]
    D -->|Opcional| I[Wikipedia]

    E --> J[Agente integra resultados]
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K[LLM traduce y redacta en español]
    K --> L[Streamlit muestra respuesta al usuario]
""", height=450)

# -------------------------
# Explicación textual
# -------------------------
st.markdown(
    """
    ### 🔎 Explicación
    1. El usuario hace una pregunta en **Streamlit**.  
    2. El **Agente RAG ICN** pasa la consulta al **LLM local**.  
    3. El LLM decide con **ReAct** qué herramienta usar.  
    4. Se consulta **ChromaDB títulos+resúmenes (prioritaria)** y, de ser necesario, la base de **texto completo**.  
    5. También puede consultar **OpenAlex**, **DuckDuckGo** o **Wikipedia**.  
    6. El agente integra toda la evidencia.  
    7. El LLM redacta y traduce la respuesta al español.  
    8. La respuesta final aparece en **Streamlit**.  
    """
)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9f9f9;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #555;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        📊  Agente RAG del Instituto de Ciencias Nucleares (ICN) | Desarrollado por José Luis Jiménez Andrade.
            <br>
            Diseño: Humberto Carrillo Calvet, Ricardo Arencibia Jorge
    </div>
    """,
    unsafe_allow_html=True
)
