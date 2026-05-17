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
API_BASE = os.getenv("FACILITYMIND_API", "http://127.0.0.1:8000")
API_HEALTH = f"{API_BASE}/api/v1/health"
API_QUERY = f"{API_BASE}/api/v1/query"
API_BLUEPRINTS = f"{API_BASE}/api/v1/blueprints"
API_UPLOAD = f"{API_BASE}/api/v1/blueprints/upload"

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
    /* Style Streamlit Tabs to make them much larger, modern and elegant */
    div.stTabs [data-baseweb="tab-list"] button,
    div.stTabs [data-baseweb="tab-list"] button p,
    div.stTabs [data-baseweb="tab"] p,
    .stTabs button p {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        color: #5f6368 !important;
    }
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"],
    div.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
        color: #1a73e8 !important;
        border-bottom-color: #1a73e8 !important;
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

def check_lobster_status():
    """Verifies that the Lobster Trap proxy is active."""
    try:
        # Check the dashboard or backend status
        resp = requests.get("http://127.0.0.1:8080/_lobstertrap/", timeout=2)
        return resp.status_code == 200
    except:
        return False

def get_loaded_blueprints():
    try:
        resp = requests.get(API_BLUEPRINTS, timeout=2)
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("blueprints_detail", [])
        return []
    except:
        return []

def upload_blueprint(file, blueprint_id=None, blueprint_type=None):
    """Uploads a PDF blueprint to the system."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    data = {}
    if blueprint_id:
        data["blueprint_id"] = blueprint_id
    if blueprint_type:
        data["blueprint_type"] = blueprint_type

    try:
        resp = requests.post(API_UPLOAD, data=data, files=files, timeout=300)
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Sidebar ───
with st.sidebar:
    st.image("app/assets/logo.png", width=120)
    st.title("FacilityMind")
    st.markdown("Professional Facility Management AI")
    
    # Grid columns for status badges
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        status = check_api_status()
        if status:
            st.success("● API Online")
        else:
            st.error("● API Offline")
            
    with col_status2:
        lobster_active = check_lobster_status()
        if lobster_active:
            st.success("🛡️ Lobster Active")
        else:
            st.warning("🛡️ Lobster Offline")

    st.divider()
    
    st.header("📁 Building Blueprints")
    
    # List loaded blueprints
    blueprints = get_loaded_blueprints()
    if blueprints:
        for bp in blueprints:
            with st.expander(f"📄 {bp['blueprint_id']}"):
                st.caption(f"Type: {bp.get('blueprint_type', 'N/A')}")
                st.caption(f"Pages: {bp.get('total_pages', '?')}")
                
                # Delete Button
                if st.button(f"🗑️ Delete {bp['blueprint_id']}", key=f"del_{bp['blueprint_id']}", type="secondary", use_container_width=True):
                    with st.spinner("Deleting blueprint..."):
                        resp = requests.delete(f"{API_BLUEPRINTS}/{bp['blueprint_id']}", timeout=10)
                        if resp.status_code == 200:
                            st.success("Deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete")
    else:
        st.info("No blueprints loaded yet.")

    # Upload new blueprint
    with st.expander("⬆️ Upload new blueprint"):
        uploaded_pdf = st.file_uploader("Blueprint PDF", type=["pdf"])
        col1, col2 = st.columns(2)
        with col1:
            blueprint_id = st.text_input("Blueprint ID", placeholder="e.g., E-14, P-01")
        with col2:
            blueprint_type = st.selectbox("Type", ["", "electrical", "plumbing", "architectural", "structural", "hvac", "fire_protection", "general"])

        if st.button("📤 Upload Blueprint", use_container_width=True) and uploaded_pdf:
            with st.spinner("Processing blueprint..."):
                result = upload_blueprint(uploaded_pdf, blueprint_id or None, blueprint_type or None)
                if result.get("success"):
                    st.success(f"✅ {result['data']['blueprint_id']} loaded ({result['data']['indexed_chunks']} chunks)")
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown')}")

    # Local variables for backward-compatibility with backend query filters
    filtro_piso = None
    filtro_disciplina = None

# ─── Main Interface ───
tab1, tab2, tab3 = st.tabs(["Chat", "Agent Pipeline", "Technical Documentation"])

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
                
                # Check if it's a direct conversational response
                is_conversational = any("conversational" in w.lower() for w in tech.get("warnings", []))
                
                if not is_conversational:
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
                                    <strong>{src['blueprint_id']} (Page {src['page_number']})</strong><br/>
                                    {src['cited_text']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Visualization (Diagram)
                    if tech.get("visualization"):
                        viz = tech["visualization"]
                        with st.expander("📊 Technical Diagram", expanded=True):
                            if viz.get("mermaid_diagram"):
                                render_mermaid(viz["mermaid_diagram"])
                            if viz.get("visual_summary"):
                                st.caption(f"**Description:** {viz['visual_summary']}")
                            if viz.get("generated_image") and os.path.exists(viz["generated_image"]):
                                st.image(viz["generated_image"], use_container_width=True)

                    # Security Blocks (Lobster Trap)
                    if tech.get("type") == "security_block":
                        st.markdown(f'<div style="background-color:#ffe6e6; border-left: 5px solid #ff4b4b; padding:10px; margin:10px 0; border-radius:5px;"><h4 style="color:#ff4b4b; margin-top:0;">🛡️ Security Block</h4><p style="color:#8b0000; font-weight:bold;">{tech.get("message")}</p></div>', unsafe_allow_html=True)
                        if "security_info" in tech:
                            sec = tech["security_info"]
                            st.caption(f"**Action:** {sec.get('action')} | **Risk Score:** {sec.get('risk_score', 0):.2f} | **Rule:** {sec.get('matched_rule')} | **Intent:** {sec.get('intent')}")

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
            
    send_media_clicked = False
    if audio_input or photo_input:
        send_media_clicked = st.button("📤 Enviar Audio/Foto", use_container_width=True)

    prompt_text = st.chat_input("Ask about the building infrastructure... (e.g., 'Which breaker controls unit 1402?')")

    if prompt_text or send_media_clicked:
        final_prompt = prompt_text if prompt_text else "Analiza este archivo multimedia."
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.write(final_prompt)
            if audio_input: st.audio(audio_input)
            if photo_input: st.image(photo_input, width=200)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Orchestrating agents..."):
                # Format chat history to send to backend
                historial_str = ""
                if len(st.session_state.messages) > 1:
                    import json
                    historial_str = json.dumps([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]])

                payload = {
                    "query_text": final_prompt,
                    "history": historial_str,
                    "discipline": filtro_disciplina if filtro_disciplina else None,
                    "floor": filtro_piso if filtro_piso else None,
                    "building_id": "default"
                }
                
                files = {}
                if audio_input:
                    files["audio_file"] = (audio_input.name, audio_input.getvalue(), "audio/wav")
                if photo_input:
                    files["image_file"] = (photo_input.name, photo_input.getvalue(), photo_input.type)
                
                try:
                    # Note: We use 'data=payload' because the backend expects Form data (multipart)
                    resp = requests.post(API_QUERY, data=payload, files=files if files else None, timeout=90)
                    
                    if resp.status_code == 200:
                        res = resp.json()
                        if res.get("success") and "data" in res:
                            data = res["data"]
                            full_answer = data["response"]
                            st.write(full_answer)
                            
                            tech_data = {
                                "confidence": data.get("confidence", 0),
                                "sources": data.get("sources", []),
                                "warnings": data.get("warnings", []),
                                "visualization": data.get("visualization") or {},
                                "type": data.get("type", ""),
                                "message": data.get("message", ""),
                                "security_info": data.get("security_info", {})
                            }
                            
                            # Visualization
                            if tech_data["visualization"] and tech_data["visualization"].get("mermaid_diagram"):
                                render_mermaid(tech_data["visualization"]["mermaid_diagram"])
                                
                            # Security Block Alert
                            if tech_data["type"] == "security_block":
                                st.markdown(f'<div style="background-color:#ffe6e6; border-left: 5px solid #ff4b4b; padding:10px; margin:10px 0; border-radius:5px;"><h4 style="color:#ff4b4b; margin-top:0;">🛡️ Security Block</h4><p style="color:#8b0000; font-weight:bold;">{tech_data["message"]}</p></div>', unsafe_allow_html=True)
                            
                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": full_answer if full_answer else (tech_data["message"] if tech_data["type"] == "security_block" else ""),
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
