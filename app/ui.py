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

# ─── Local Wallet Cache Helper ───
CACHE_FILE = "data/active_wallet.json"

def load_cached_wallet():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_cached_wallet(wallet_id, address):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"wallet_id": wallet_id, "address": address}, f)
    except Exception:
        pass

# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = []

cached_wallet = load_cached_wallet()
if "client_wallet_id" not in st.session_state:
    st.session_state.client_wallet_id = cached_wallet["wallet_id"] if cached_wallet else None
if "client_wallet_address" not in st.session_state:
    st.session_state.client_wallet_address = cached_wallet["address"] if cached_wallet else None
if "client_wallet_balance" not in st.session_state:
    st.session_state.client_wallet_balance = 0.0
if "client_gateway_balance" not in st.session_state:
    st.session_state.client_gateway_balance = 0.0
    # Try to load cached wallet's actual balances on startup
    if st.session_state.client_wallet_id:
        try:
            resp = requests.get(f"{API_BASE}/api/v1/billing/balance/{st.session_state.client_wallet_id}", timeout=10)
            if resp.status_code == 200 and resp.json().get("success"):
                st.session_state.client_wallet_balance = resp.json()["data"]["balance"]
                st.session_state.client_gateway_balance = resp.json()["data"].get("gateway_balance", 0.0)
        except Exception:
            pass

# ─── Sidebar ───
with st.sidebar:
    # Use columns to center the logo
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.image("app/assets/logo.png", width="stretch")
        
    st.markdown("<h1 style='text-align: center; margin-top: -15px;'>FacilityMind</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Professional Facility Management AI</p>", unsafe_allow_html=True)
    
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
    
    st.markdown("<h3 style='margin-bottom:0;'>🪙 Web3 Billing & Wallet</h3>", unsafe_allow_html=True)
    st.caption("Self-sufficiency and pay-as-you-go usage under Circle's x402 protocol.")
    
    if not st.session_state.client_wallet_id:
        if st.button("⚡ Generate Query Wallet", type="primary", width="stretch"):
            with st.spinner("Creating real programmable wallet on Circle..."):
                try:
                    resp = requests.post(f"{API_BASE}/api/v1/billing/create-wallet", timeout=30)
                    if resp.status_code == 200 and resp.json().get("success"):
                        wdata = resp.json()["data"]
                        st.session_state.client_wallet_id = wdata["wallet_id"]
                        st.session_state.client_wallet_address = wdata["address"]
                        save_cached_wallet(wdata["wallet_id"], wdata["address"])
                        st.success("Wallet Created Successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to create wallet: {resp.json().get('error', 'Unknown')}")
                except Exception as e:
                    st.error(f"Network error: {e}")
    else:
        # Mostrar detalles de la Wallet
        st.markdown(f"""
        <div style="background-color:#0e1117; padding:10px; border-radius:5px; border:1px solid #30363d; margin-bottom:10px;">
            <span style="font-size:10px; color:gray; font-weight:bold;">PAYMENT ADDRESS (ARC TESTNET)</span><br/>
            <code style="font-size:11px; color:#58a6ff; word-break:break-all;">{st.session_state.client_wallet_address}</code>
        </div>
        """, unsafe_allow_html=True)
        
        # Consultar saldos
        col_bal1, col_bal2 = st.columns(2)
        with col_bal1:
            st.markdown(f"**EOA Balance:**<br/>`{st.session_state.client_wallet_balance:.4f} USDC`", unsafe_allow_html=True)
        with col_bal2:
            st.markdown(f"**Gateway Balance:**<br/>`{st.session_state.client_gateway_balance:.4f} USDC`", unsafe_allow_html=True)
            
        col_ref, col_faucet = st.columns([1.2, 2])
        with col_ref:
            if st.button("🔄 Refresh", key="ref_bal", use_container_width=True):
                with st.spinner(""):
                    try:
                        resp = requests.get(f"{API_BASE}/api/v1/billing/balance/{st.session_state.client_wallet_id}", timeout=10)
                        if resp.status_code == 200 and resp.json().get("success"):
                            st.session_state.client_wallet_balance = resp.json()["data"]["balance"]
                            st.session_state.client_gateway_balance = resp.json()["data"].get("gateway_balance", 0.0)
                            st.rerun()
                    except Exception as e:
                        st.error(f"{e}")
        with col_faucet:
            st.markdown(f"""
            <a href="https://faucet.circle.com" target="_blank" style="text-decoration:none;">
                <button style="width:100%; border:1px solid #30363d; background-color:#161b22; color:#58a6ff; padding:5px; border-radius:5px; cursor:pointer; height:38px; font-size:12px; font-weight:bold;">
                    🚰 Circle Faucet
                </button>
            </a>
            """, unsafe_allow_html=True)
            
        st.markdown("<h4 style='margin-top:15px; margin-bottom:5px; font-size:13px; color:#58a6ff; font-weight:bold;'>⚡ Deposit to Gateway (Gas-Free)</h4>", unsafe_allow_html=True)
        st.caption("USDC inside the Gateway enables zero-gas EIP-3009 batched settlements.")
        
        dep_amount = st.number_input("USDC to Deposit", min_value=0.1, max_value=100.0, value=1.0, step=0.5, key="dep_amount_val", label_visibility="collapsed")
        if st.button("Deposit into Gateway", key="exec_deposit", type="primary", use_container_width=True):
            if dep_amount > st.session_state.client_wallet_balance:
                st.error("Insufficient EOA Balance.")
            else:
                with st.spinner("Executing secure onchain deposit... (15-20s)"):
                    try:
                        resp = requests.post(f"{API_BASE}/api/v1/billing/gateway-deposit", json={
                            "wallet_id": st.session_state.client_wallet_id,
                            "amount": dep_amount
                        }, timeout=45)
                        if resp.status_code == 200 and resp.json().get("success"):
                            st.success(f"Deposited {dep_amount} USDC!")
                            # Refresh balances
                            resp_bal = requests.get(f"{API_BASE}/api/v1/billing/balance/{st.session_state.client_wallet_id}", timeout=10)
                            if resp_bal.status_code == 200 and resp_bal.json().get("success"):
                                st.session_state.client_wallet_balance = resp_bal.json()["data"]["balance"]
                                st.session_state.client_gateway_balance = resp_bal.json()["data"].get("gateway_balance", 0.0)
                            st.rerun()
                        else:
                            st.error(f"Failed: {resp.json().get('error', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Network error: {e}")

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
                if st.button(f"🗑️ Delete {bp['blueprint_id']}", key=f"del_{bp['blueprint_id']}", type="secondary", width="stretch"):
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

        if st.button("📤 Upload Blueprint", width="stretch") and uploaded_pdf:
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
                                st.image(viz["generated_image"], width="stretch")
                                
                    # Billing Receipt (History)
                    if tech.get("billing"):
                        bill = tech["billing"]
                        st.markdown(f'''
                        <div style="background-color:#f8f9fa; border-left: 5px solid #005571; padding:10px; margin:10px 0; border-radius:5px;">
                            <strong style="color:#005571; font-size:15px;">🪙 Nanopayment Processed (Web3 {bill['network']})</strong><br/>
                            <span style="font-size:14px; color:#333;">Automatic offchain charge of <code>${bill['amount_charged']} {bill['currency']}</code> via EIP-3009.</span><br/>
                            <span style="color:gray; font-size:12px;">Auth: <code>{bill.get('eip3009_auth', '...')}</code> | Batched Settlement: {bill.get('settlement_status', 'PENDING')}</span>
                        </div>
                        ''', unsafe_allow_html=True)

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
        send_media_clicked = st.button("📤 Send Audio/Photo", width="stretch")

    prompt_text = st.chat_input("Ask about the building infrastructure... (e.g., 'Which breaker controls unit 1402?')")

    if prompt_text or send_media_clicked:
        final_prompt = prompt_text if prompt_text else "Analyze this media file."
        
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
                    "building_id": "default",
                    "client_wallet_id": st.session_state.client_wallet_id if st.session_state.client_wallet_id else None
                }
                
                files = {}
                if audio_input:
                    files["audio_file"] = (audio_input.name, audio_input.getvalue(), "audio/wav")
                if photo_input:
                    files["image_file"] = (photo_input.name, photo_input.getvalue(), photo_input.type)
                
                try:
                    # Note: We use 'data=payload' because the backend expects Form data (multipart)
                    resp = requests.post(API_QUERY, data=payload, files=files if files else None, timeout=180)
                    
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
                                "security_info": data.get("security_info", {}),
                                "billing": data.get("billing")
                            }
                            
                            # Visualization
                            if tech_data["visualization"] and tech_data["visualization"].get("mermaid_diagram"):
                                render_mermaid(tech_data["visualization"]["mermaid_diagram"])
                                
                            # Security Block Alert
                            if tech_data["type"] == "security_block":
                                st.markdown(f'<div style="background-color:#ffe6e6; border-left: 5px solid #ff4b4b; padding:10px; margin:10px 0; border-radius:5px;"><h4 style="color:#ff4b4b; margin-top:0;">🛡️ Security Block</h4><p style="color:#8b0000; font-weight:bold;">{tech_data["message"]}</p></div>', unsafe_allow_html=True)
                                
                            # Billing Receipt
                            if tech_data.get("billing"):
                                bill = tech_data["billing"]
                                st.markdown(f'''
                                <div style="background-color:#f8f9fa; border-left: 5px solid #005571; padding:10px; margin:10px 0; border-radius:5px;">
                                    <strong style="color:#005571; font-size:15px;">🪙 Nanopayment Processed (Web3 {bill['network']})</strong><br/>
                                    <span style="font-size:14px; color:#333;">Automatic offchain charge of <code>${bill['amount_charged']} {bill['currency']}</code> via EIP-3009.</span><br/>
                                    <span style="color:gray; font-size:12px;">Auth: <code>{bill.get('eip3009_auth', '...')}</code> | Batched Settlement: {bill.get('settlement_status', 'PENDING')}</span>
                                </div>
                                ''', unsafe_allow_html=True)
                            
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
