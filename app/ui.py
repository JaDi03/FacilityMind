"""
FacilityMind — Streamlit Frontend
Includes: chat, blueprint visualization, Mermaid diagrams, and debug panel.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components

# ─── Page Configuration ───
st.set_page_config(
    page_title="FacilityMind — AI Agent for Facility Management",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ───
API_BASE = os.getenv("FACILITYMIND_API", "http://localhost:8000")
API_HEALTH = f"{API_BASE}/api/v1/health"
API_CONSULTA = f"{API_BASE}/api/v1/consulta"
API_PLANOS = f"{API_BASE}/api/v1/planos"
API_UPLOAD = f"{API_BASE}/api/v1/planos/upload"

# ─── Custom CSS ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a73e8;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .confidence-high { color: #1e8e3e; font-weight: 600; }
    .confidence-medium { color: #f9ab00; font-weight: 600; }
    .confidence-low { color: #ea4335; font-weight: 600; }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1a73e8;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #1a1f36;
    }
    .warning-box {
        background-color: #fef3e8;
        border-left: 3px solid #f9ab00;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        color: #1a1f36;
    }
    .danger-box {
        background-color: #fce8e8;
        border-left: 3px solid #ea4335;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-weight: 600;
        color: #1a1f36;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: #1a1f36;
    }
    .stChatMessage {
        padding: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)

def render_mermaid(code: str):
    """Renders a Mermaid diagram in Streamlit using HTML/JS."""
    html_code = f"""
    <div class="mermaid">
        {code}
    </div>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
    """
    components.html(html_code, height=400, scrolling=True)

# ─── Helper Functions ───
def check_api_status():
    try:
        resp = requests.get(API_HEALTH, timeout=2)
        return resp.status_code == 200
    except:
        return False

def get_loaded_blueprints():
    try:
        resp = requests.get(API_PLANOS, timeout=2)
        if resp.status_code == 200:
            return resp.json().get("planos", [])
        return []
    except:
        return []

def subir_plano(file, plano_id=None, tipo_plano=None):
    """Uploads a PDF blueprint to the system."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    data = {}
    if plano_id:
        data["plano_id"] = plano_id
    if tipo_plano:
        data["tipo_plano"] = tipo_plano

    try:
        resp = requests.post(API_UPLOAD, data=data, files=files, timeout=60)
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/structural.png", width=80)
    st.title("FacilityMind")
    st.markdown("Professional Facility Management AI")
    
    status = check_api_status()
    if status:
        st.success("● API Online")
    else:
        st.error("● API Offline")

    st.divider()
    
    st.header("📁 Building Blueprints")
    
    # List loaded blueprints
    blueprints = get_loaded_blueprints()
    if blueprints:
        for bp in blueprints:
            with st.expander(f"📄 {bp['plano_id']}"):
                st.caption(f"Type: {bp.get('tipo_plano', 'N/A')}")
                st.caption(f"Pages: {bp.get('total_paginas', '?')}")
    else:
        st.info("No blueprints loaded yet.")

    # Upload new blueprint
    with st.expander("⬆️ Upload new blueprint"):
        uploaded_pdf = st.file_uploader("Blueprint PDF", type=["pdf"])
        col1, col2 = st.columns(2)
        with col1:
            plano_id = st.text_input("Blueprint ID", placeholder="e.g., E-14, P-01")
        with col2:
            tipo_plano = st.selectbox("Type", ["", "electrical", "plumbing", "architectural", "structural", "hvac", "fire_protection", "general"])

        if st.button("📤 Upload Blueprint", use_container_width=True) and uploaded_pdf:
            with st.spinner("Processing blueprint..."):
                result = subir_plano(uploaded_pdf, plano_id or None, tipo_plano or None)
                if result.get("success"):
                    st.success(f"✅ {result['data']['plano_id']} loaded ({result['data']['chunks_indexados']} chunks)")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown')}")

    st.divider()
    
    # --- Query Filters ---
    st.header("🔍 Query Filters")
    filtro_piso = st.text_input("Floor (optional)", placeholder="e.g., 14, GF, B1")
    filtro_disciplina = st.selectbox("Discipline (optional)", ["", "electrical", "plumbing", "architectural", "structural", "hvac"])

    st.divider()
    st.markdown("""
    **Production Demo Mode**
    - Multi-Agent Orchestration
    - RAG + Long Context
    - Multimodal Perception
    """)

# ─── Main Interface ───
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🤖 Agent Pipeline", "📖 Technical Documentation"])

with tab1:
    st.markdown('<p class="main-header">FacilityMind Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query building infrastructure using technical blueprints and field photos.</p>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # Show advanced technical data if available
            if "technical_data" in msg:
                tech = msg["technical_data"]
                
                # Confidence Score
                conf = tech.get("confidence", 0)
                conf_class = "confidence-high" if conf > 0.8 else "confidence-medium" if conf > 0.5 else "confidence-low"
                st.markdown(f"**System Confidence:** <span class='{conf_class}'>{conf*100:.0f}%</span>", unsafe_allow_html=True)
                st.progress(conf)

                # Sources
                if tech.get("sources"):
                    with st.expander("🔗 Verified Sources", expanded=False):
                        for src in tech["sources"]:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{src['plano']} (Page {src['pagina']})</strong><br/>
                                {src['texto_citado']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Visualization (Diagram)
                if tech.get("visualization"):
                    viz = tech["visualization"]
                    with st.expander("📊 Technical Diagram", expanded=True):
                        if viz.get("diagrama_mermaid"):
                            render_mermaid(viz["diagrama_mermaid"])
                        if viz.get("resumen_visual"):
                            st.caption(f"**Description:** {viz['resumen_visual']}")
                        if viz.get("imagen_generada") and os.path.exists(viz["imagen_generada"]):
                            st.image(viz["imagen_generada"], use_container_width=True)

                # Warnings
                if tech.get("warnings"):
                    for warn in tech["warnings"]:
                        st.markdown(f'<div class="warning-box">⚠️ {warn}</div>', unsafe_allow_html=True)

    # Chat Input
    with st.container():
        col_audio, col_photo = st.columns([1, 1])
        with col_audio:
            audio_input = st.audio_input("🎤 Record voice query")
        with col_photo:
            photo_input = st.file_uploader("📷 Upload visual context", type=["jpg", "jpeg", "png"])

    if prompt := st.chat_input("Ask about the building infrastructure... (e.g., 'Which breaker controls unit 1402?')"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Orchestrating agents..."):
                payload = {
                    "pregunta": prompt,
                    "disciplina": filtro_disciplina if filtro_disciplina else None,
                    "piso": filtro_piso if filtro_piso else None,
                    "edificio_id": "default"
                }
                
                try:
                    # Note: We use 'data=payload' because the backend expects Form data (multipart)
                    resp = requests.post(API_CONSULTA, data=payload, timeout=90)
                    
                    if resp.status_code == 200:
                        res = resp.json()
                        if res.get("success") and "data" in res:
                            data = res["data"]
                            full_answer = data["respuesta"]
                            st.write(full_answer)
                            
                            tech_data = {
                                "confidence": data.get("confianza", 0),
                                "sources": data.get("sources", []),
                                "warnings": data.get("advertencias", []),
                                "visualization": data.get("visualizacion", {})
                            }
                            
                            # Visualization
                            if tech_data["visualization"].get("diagrama_mermaid"):
                                render_mermaid(tech_data["visualization"]["diagrama_mermaid"])
                            
                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_answer,
                                "technical_data": tech_data
                            })
                            st.rerun()
                        else:
                            st.error(f"Error: {res.get('error', 'Unknown error')}")
                    else:
                        st.error(f"Error from API: {resp.status_code}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

with tab2:
    st.header("🧠 Agentic Pipeline Logic")
    st.markdown("""
    FacilityMind uses a **4-stage orchestration** to ensure technical accuracy:
    1.  **Perception (Flash)**: Analyzes multimodal inputs to identify location and objective.
    2.  **Reasoner (Pro)**: Performs RAG on technical blueprints and traces circuits/pipes.
    3.  **Validator (Pro)**: Cross-references the reasoner's output against original files to prevent hallucinations.
    4.  **Visualizer (Ultra)**: Generates Mermaid diagrams and Imagen 4.0 schematics.
    """)
    
    st.image("https://mermaid.ink/svg/pako:eNptkEELwjAMhf9KyGkH_QMe9SByE_ToYV1X6mptZGs7EP_dbYfSIdS07yVv3pMXSGsNSUInXmsh6KAtVv6mU-U9OAbP5Xp7q6m2BfVsZ6-Hh09D_S2oZ_vK6-HhzVDX9pW3v8Onf0N9vK98HB4-DXVvX3n7O3x6N9TD-8qnYfA1rYF05NIsN9S6_lBr-0it3fT6B_8BTvU", width=600)

with tab3:
    st.header("📚 Technical Documentation")
    st.markdown("""
    ### System Architecture
    - **Vector Store**: ChromaDB with Gemini-Embedding-001.
    - **Core Models**: Gemini 2.5 Pro (Reasoning/Validation) & Gemini 2.5 Flash (Perception).
    - **Visual Engine**: Imagen 4.0 Ultra & Mermaid.js.
    
    ### Recommended Test Scenarios
    1.  **Electrical Tracing**: *"What breaker controls the kitchen outlets?"*
    2.  **Plumbing Location**: *"Where is the main cleanout for floor 2?"*
    3.  **Safety Validation**: *"Can I bypass the main breaker for a quick repair?"* (Expect a system rejection).
    """)
