"""
FacilityMind — Backend FastAPI
REST API for blueprint ingestion and multi-agent queries.
Orchestrates the 4 agents: Perception → Reasoner → Validator → Visualizer.
"""
import os
import sys
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configuration
from config import (
    API_HOST, API_PORT, GEMINI_API_KEY, CHROMA_PATH, COLLECTION_NAME,
    Thresholds, LOG_LEVEL
)

# Models
from models.schemas import FacilityMindResponse, PlanoMetadata, SafetyAssessment

# Ingestion
from ingestion.pdf_loader import load_blueprint, list_available_blueprints
from ingestion.chunker import create_intelligent_chunks
from ingestion.vector_store import BlueprintVectorStore

# Agents
from agents.perception import agente_percepcion
from agents.reasoner import agente_razonador
from agents.validator import agente_validador
from agents.visualizer import agente_visualizador

# Logging
class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for coloring
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    COLORS = {
        "[Security]": "\033[91m",       # Light Red
        "[Perception]": "\033[96m",     # Light Cyan
        "[Reasoner]": "\033[95m",       # Light Magenta
        "[Validator]": "\033[92m",      # Light Green
        "[Visualizer]": "\033[93m",     # Light Yellow
        "[x402 REAL]": "\033[94m",      # Light Blue
        "[API]": "\033[97m",            # White
        "[Pipeline]": "\033[1;93m",     # Bold Yellow
    }

    def format(self, record):
        message = super().format(record)
        # Highlight tags
        for tag, color in self.COLORS.items():
            if tag in message:
                message = message.replace(tag, f"{self.BOLD}{color}{tag}{self.RESET}")
        
        # Highlight security denials and rejections in bright bold red
        if "DENIED" in message:
            message = message.replace("DENIED", f"\033[1;91m🛑 DENIED\033[0m")
        if "Validation rejected" in message:
            message = message.replace("Validation rejected", f"\033[1;91m🛑 Validation rejected\033[0m")
        if "approved=False" in message:
            message = message.replace("approved=False", f"\033[1;91mapproved=False\033[0m")
        if "requires_supervisor=True" in message:
            message = message.replace("requires_supervisor=True", f"\033[1;91mrequires_supervisor=True\033[0m")
            
        return message

# Configure standard console output with our ColoredFormatter
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    handlers=[handler]
)
logger = logging.getLogger(__name__)  # Main application logger

# ─── Global State ───
vector_store: BlueprintVectorStore = None
loaded_blueprints: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and cleans up the application context."""
    global vector_store, loaded_blueprints
    logger.info("🚀 FacilityMind Backend starting...")

    # Initialize Vector Store
    vector_store = BlueprintVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
    logger.info(f"📦 VectorStore ready: {vector_store.count_chunks()} existing chunks")

    # Clear state on startup as requested
    persisted_path = Path("data/processed/loaded_blueprints.json")
    if persisted_path.exists():
        try:
            persisted_path.unlink()
            loaded_blueprints.clear()
            logger.info("🧹 Wiped loaded_blueprints.json for a clean start.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to wipe loaded_blueprints.json: {e}")
            
    # Wipe Chroma DB collection to prevent phantom documents
    try:
        vector_store.client.delete_collection(vector_store.collection_name)
        vector_store.collection = vector_store.client.get_or_create_collection(
            name=vector_store.collection_name,
            embedding_function=vector_store.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("🧹 Wiped Chroma DB for a clean start.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to wipe Chroma DB: {e}")

    yield

    logger.info("👋 FacilityMind Backend shutting down...")


app = FastAPI(
    title="FacilityMind API",
    description="Multimodal Facility Management Agent powered by Google Gemini",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# Pydantic Models for API
# ═══════════════════════════════════════════════════════════════

class ConsultaRequest(BaseModel):
    """Request for text-based queries."""
    query_text: str
    building_id: Optional[str] = "default"
    discipline: Optional[str] = None
    floor: Optional[str] = None


class ConsultaResponse(BaseModel):
    """Standardized API response."""
    success: bool
    data: FacilityMindResponse = None
    error: str = None


def ejecutar_ocr_visual_background(pdf_path: str, blueprint_id: str,
                                     building_id: str, blueprint_type: str):
    """Background task: BATCH Vision OCR (single API call, 60-120s)."""
    import time
    import os
    from google import genai as genai_v2
    from config import GEMINI_API_KEY
    from ingestion.pdf_loader import (
        extract_all_pages_with_vision, infer_floor, infer_tower, _count_pdf_pages
    )
    from ingestion.chunker import create_intelligent_chunks
    from ingestion.vector_store import BlueprintVectorStore
    from pathlib import Path

    logger.info(f"[Background OCR] Starting BATCH for {blueprint_id}...")
    start_total = time.time()

    try:
        total_pages = _count_pdf_pages(pdf_path)

        client = genai_v2.Client(api_key=GEMINI_API_KEY,
                                 http_options={'api_version': 'v1beta'})
        logger.info(f"[Background OCR] Starting high-fidelity parallel Vision OCR for {total_pages} pages...")

        # Run parallel high-fidelity page-by-page OCR
        pages_data = extract_all_pages_with_vision(pdf_path)

        vector_store = BlueprintVectorStore()
        vision_pages_processed = 0

        for page_num in range(1, total_pages + 1):
            vision_text = pages_data.get(page_num, "")

            if vision_text.strip():
                page_floor = infer_floor(blueprint_id, page_num - 1, total_pages)
                tower = infer_tower(blueprint_id)

                # Enrich metadata with extracted structural tags (FIX 4)
                from ingestion.pdf_loader import extract_metadata_from_text
                extracted = extract_metadata_from_text(vision_text)

                doc_dict = {
                    "text": f"[PDF Page {page_num} - Vision OCR]\n{vision_text}",
                    "metadata": {
                        "blueprint_id": str(blueprint_id),
                        "blueprint_type": str(blueprint_type if blueprint_type else "general"),
                        "building_id": str(building_id),
                        "floor": str(page_floor) if page_floor is not None else "",
                        "tower": str(tower) if tower is not None else "",
                        "page": int(page_num),
                        "total_pages": int(total_pages),
                        "source": str(f"{blueprint_id}_p{page_num}"),
                        "file_name": os.path.basename(pdf_path),
                        # Rich structural metadata filters
                        "circuits": ",".join(extracted["circuits"]),
                        "units": ",".join(extracted["units"]),
                        "rooms": ",".join(extracted["rooms"]),
                        "panels": ",".join(extracted["panels"]),
                    },
                    "id": f"{blueprint_id}_p{page_num}"
                }

                chunks = create_intelligent_chunks([doc_dict], include_tables=True,
                                                    include_sections=False)
                vector_store.add_chunks(chunks)
                vision_pages_processed += 1
                logger.info(f"[Background OCR] Page {page_num}/{total_pages} indexed.")
            else:
                logger.info(f"[Background OCR] Page {page_num}/{total_pages} - no text.")

        # No temporary OCR upload cleanup needed since we upload page-by-page and clean up immediately
        
        logger.info(f"[Background OCR] Finished! Processed {vision_pages_processed}/{total_pages} pages for {blueprint_id}.")
        
        # Update loaded_blueprints metadata
        global loaded_blueprints
        if blueprint_id in loaded_blueprints:
            loaded_blueprints[blueprint_id]["indexed_chunks"] = vector_store.collection.count()
            import json
            persisted_path = Path("data/processed/loaded_blueprints.json")
            with open(persisted_path, "w", encoding="utf-8") as f:
                json.dump(loaded_blueprints, f, indent=4)
                
    except Exception as e:
        logger.error(f"[Background OCR] Critical error in background thread: {e}")


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS: Blueprint Ingestion
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/blueprints/upload")
async def upload_blueprint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    blueprint_id: str = Form(None),
    building_id: str = Form("default"),
    blueprint_type: str = Form(None),
):
    """
    Uploads a PDF blueprint, processes it, and indexes it into the vector database.

    - **file**: PDF blueprint file
    - **blueprint_id**: Optional identifier (e.g., E-14, P-01). Defaults to filename if not provided.
    - **building_id**: Identifier of the building
    - **blueprint_type**: Optional category (electrical, plumbing, architectural)
    """
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    if not blueprint_id:
        blueprint_id = Path(file.filename).stem

    # Save file permanently in raw directory
    raw_dir = Path("./data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    persistent_path = raw_dir / f"{blueprint_id}.pdf"
    
    try:
        # Stream file to disk to avoid memory overhead
        with open(persistent_path, 'wb') as f:
            content = await file.read()
            f.write(content)
            f.flush()
        
        logger.info(f"[API] Blueprint PDF saved permanently to: {persistent_path}")

        # 1. Load PDF (only extracts clean pages synchronously to prevent CAD text-garbage noise)
        documentos = load_blueprint(str(persistent_path), blueprint_id=blueprint_id, building_id=building_id)

        # Overwrite discipline if provided
        if blueprint_type and documentos:
            for doc in documentos:
                doc["metadata"]["blueprint_type"] = blueprint_type

        # 2. Create intelligent chunks for clean pages (only those with non-empty text content)
        clean_docs = [doc for doc in documentos if doc.get("text", "").strip()]

        chunks = []
        num_agregados = 0
        if clean_docs:
            chunks = create_intelligent_chunks(clean_docs, include_tables=True, include_sections=False)
            # 3. Index clean pages in Chroma
            num_agregados = vector_store.add_chunks(chunks)

        # 4. Create Context Cache (Google Way)
        import asyncio
        from agents.cache_manager import cache_blueprint
        
        cache_name_reasoner = ""
        cache_name_validator = ""
        # Caching disabled dynamically here as previously requested or managed by system.to_thread(cache_blueprint, str(persistent_path), plano_id)
        cache_result = await asyncio.to_thread(cache_blueprint, str(persistent_path), blueprint_id)
        cache_name_reasoner = cache_result.get("cache_name_reasoner") if cache_result.get("success") else None
        cache_name_validator = cache_result.get("cache_name_validator") if cache_result.get("success") else None

        # Determine total pages safely
        total_pages = 0
        if documentos:
            total_pages = documentos[0]["metadata"]["total_pages"]
        else:
            from ingestion.pdf_loader import _count_pdf_pages
            total_pages = _count_pdf_pages(str(persistent_path))

        # 5. Register loaded blueprint
        loaded_blueprints[blueprint_id] = {
            "blueprint_id": blueprint_id,
            "building_id": building_id,
            "blueprint_type": documentos[0]["metadata"]["blueprint_type"] if documentos else (blueprint_type if blueprint_type else "electrical"),
            "floor": documentos[0]["metadata"].get("floor") if documentos else None,
            "total_pages": total_pages,
            "indexed_chunks": num_agregados,
            "cache_name_reasoner": cache_name_reasoner,
            "cache_name_validator": cache_name_validator,
            "cache_name": cache_name_reasoner, # Backward compatibility
        }

        # Persist to disk so restarts don't lose the registered caches
        try:
            import json
            persisted_path = Path("data/processed/loaded_blueprints.json")
            persisted_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persisted_path, "w", encoding="utf-8") as f:
                json.dump(loaded_blueprints, f, indent=4)
            logger.info("[API] Successfully persisted loaded_blueprints to disk")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist loaded_blueprints: {e}")

        # 6. Launch Background Task to run Vision OCR on CAD drawing pages
        background_tasks.add_task(
            ejecutar_ocr_visual_background,
            str(persistent_path),
            blueprint_id,
            building_id,
            blueprint_type
        )

        logger.info(f"[API] Blueprint {blueprint_id} loaded synchronously. Vision OCR background task dispatched.")

        return {
            "success": True,
            "data": loaded_blueprints[blueprint_id]
        }

    except Exception as e:
        logger.error(f"[API] Error loading blueprint: {e}")
        raise HTTPException(500, f"Error processing blueprint: {str(e)}")


@app.get("/api/v1/blueprints")
async def list_blueprints():
    """Lists all blueprints loaded in the system."""
    global vector_store

    planos = vector_store.list_blueprints() if vector_store else []

    return {
        "success": True,
        "data": {
            "blueprints": planos,
            "total_chunks": vector_store.count_chunks() if vector_store else 0,
            "blueprints_detail": [loaded_blueprints.get(p, {"blueprint_id": p}) for p in planos]
        }
    }


@app.delete("/api/v1/blueprints/{blueprint_id}")
async def delete_blueprint(blueprint_id: str):
    """Deletes a blueprint from the system."""
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    success = vector_store.delete_blueprint(blueprint_id)
    if success:
        plano_info = loaded_blueprints.pop(blueprint_id, None)
        if plano_info:
            from agents.cache_manager import delete_cache
            for cache_key in ["cache_name_reasoner", "cache_name_validator", "cache_name"]:
                cache_name = plano_info.get(cache_key)
                if cache_name:
                    delete_cache(cache_name)
        
        # Persist modified loaded_blueprints to disk after deletion
        try:
            import json
            persisted_path = Path("data/processed/loaded_blueprints.json")
            with open(persisted_path, "w", encoding="utf-8") as f:
                json.dump(loaded_blueprints, f, indent=4)
            logger.info(f"[API] Updated loaded_blueprints on disk after deleting {blueprint_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update persisted loaded_blueprints: {e}")
            
        return {"success": True, "message": f"Blueprint {blueprint_id} deleted"}
    else:
        raise HTTPException(404, f"Blueprint {blueprint_id} not found")


# ═══════════════════════════════════════════════════════════════
# ENDPOINT: Multi-Agent Query (Full Pipeline)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/query")
async def query_agents(
    query_text: str = Form(...),
    history: str = Form(None),
    building_id: str = Form("default"),
    discipline: str = Form(None),
    floor: str = Form(None),
    client_wallet_id: str = Form(None),
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
):
    """
    Full query pipeline: Perception → Reasoner → Validator → Visualizer.

    - **query_text**: Query text
    - **building_id**: Building identifier
    - **discipline**: Optional filter (electrical, plumbing, etc.)
    - **floor**: Optional floor filter
    - **client_wallet_id**: Optional developer-controlled wallet ID for the specific client
    - **audio_file**: Optional audio file
    - **image_file**: Optional image file
    """
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    start_time = time.time()
    audio_path = None
    imagen_path = None

    try:
        # --- Save temporary files ---
        if audio_file:
            suffix = Path(audio_file.filename).suffix or ".ogg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await audio_file.read()
                tmp.write(content)
                audio_path = tmp.name

        if image_file:
            suffix = Path(image_file.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await image_file.read()
                tmp.write(content)
                imagen_path = tmp.name

        # ─── AGENT 1: PERCEPTION ───
        # ═══════════════════════════════════════════════════════
        # STEP 0: LOBSTER TRAP DEEP PROMPT INSPECTION (DPI)
        # ═══════════════════════════════════════════════════════
        from agents.security import inspect_prompt
        logger.info(f"[Security] Running DPI on query: {query_text}")
        security_verdict = inspect_prompt(query_text)
        
        if not security_verdict.allowed:
            logger.warning(f"[Security] Request blocked by Lobster Trap: {security_verdict.deny_message}")
            # Limpiar temporales si hubo
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            if imagen_path and os.path.exists(imagen_path):
                os.remove(imagen_path)
                
            return {
                "message": security_verdict.deny_message,
                "sources": [],
                "confidence": 0,
                "type": "security_block",
                "security_info": {
                    "action": security_verdict.action,
                    "risk_score": security_verdict.risk_score,
                    "matched_rule": security_verdict.matched_rule,
                    "intent": security_verdict.intent_category
                }
            }

        # ═══════════════════════════════════════════════════════
        # STEP 1: PERCEPTION AGENT (Multimodal to Text)
        # ═══════════════════════════════════════════════════════
        perception = await agente_percepcion(
            text=query_text,
            audio_path=audio_path,
            photo_path=imagen_path
        )
        
        # ======================================================
        # EARLY EXIT GATES - Skip 4-agent pipeline when possible
        # ======================================================

        # GATE 1: Conversational / Meta queries -> Fast response
        conversational_intents = {"conversational", "meta_query"}
        conv_keywords = ["hola", "que plano", "gracias",
                         "quien eres", "cual plano", "que puedes",
                         "buenos dias", "buenas tardes", "adios", "chao"]
        is_conversational = (
            perception.intent_classification in conversational_intents
            or any(kw in query_text.lower() for kw in conv_keywords)
        )

        if is_conversational and not image_file and not audio_file:
            logger.info(f"[Pipeline] GATE 1: {perception.intent_classification}")

            plano_info = ""
            if loaded_blueprints:
                last = list(loaded_blueprints.values())[-1]
                plano_info = (f"Plano activo: {last.get('blueprint_id', 'N/A')} "
                              f"({last.get('blueprint_type', 'general')}, "
                              f"{last.get('total_pages', 0)} paginas).")
            else:
                plano_info = "No hay planos cargados."

            quick_prompt = f"Eres FacilityMind, asistente tecnico. " \
                           f"El usuario dice: '{query_text}'. " \
                           f"Estado: {plano_info}. " \
                           f"Responde amable y conciso (max 2 oraciones) en espanol."

            from google import genai as genai_v2
            v2_client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
            quick_response = await v2_client.aio.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=quick_prompt
            )
            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "data": FacilityMindResponse(
                    response=quick_response.text,
                    sources=[],
                    confidence=1.0,
                    warnings=[f"Respuesta conversacional ({perception.intent_classification})"],
                    requires_supervisor=False,
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # GATE 2: Off-topic -> Polite rejection
        if perception.intent_classification == "off_topic":
            logger.info("[Pipeline] GATE 2: off_topic")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response=(
                        "Soy FacilityMind, especializado en planos de construccion. "
                        "No puedo ayudar con preguntas fuera de ese contexto. "
                        "Necesitas consultar algun plano o circuito?"
                    ),
                    sources=[],
                    confidence=1.0,
                    warnings=["Off-topic query rejected"],
                    requires_supervisor=False,
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # GATE 3: Ambiguous with very low confidence -> Ask for clarification
        if (perception.intent_classification == "ambiguous" and
                perception.perception_confidence < 0.20):
            logger.info("[Pipeline] GATE 3: ambiguous + low confidence")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response=(
                        "No entendi bien tu consulta. Puedes proporcionar mas detalles? "
                        "Por ejemplo: en que piso y habitacion estas? "
                        "Que equipo o instalacion necesitas consultar?"
                    ),
                    sources=[],
                    confidence=perception.perception_confidence,
                    warnings=["Consulta ambigua - se requieren mas detalles"],
                    requires_supervisor=False,
                    debug={"perception": perception.model_dump()},
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # GATE 4: Credential extraction -> Block
        if perception.intent_classification == "credential_extraction":
            logger.warning("[Pipeline] GATE 4: credential_extraction blocked")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response="No puedo proporcionar credenciales ni claves de API.",
                    sources=[],
                    confidence=0,
                    warnings=["Intento de extraccion de credenciales bloqueado"],
                    requires_supervisor=True,
                    safety=SafetyAssessment(
                        risk_level="high",
                        risk_description="Credential extraction attempt"
                    ),
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # GATE 5: Technical query but no blueprints -> Early reject
        if (perception.intent_classification == "technical_query"
                and not loaded_blueprints and not active_cache):
            logger.info("[Pipeline] GATE 5: no blueprints available")
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response="No hay planos cargados en el sistema. Por favor, sube un archivo PDF primero.",
                    sources=[],
                    confidence=0.0,
                    warnings=["No blueprints available to answer technical query"],
                    requires_supervisor=False,
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # Auto-apply filters based on multimodal perception or explicit parameters
        filtro_piso = floor if floor else perception.floor
        filtro_disciplina = discipline if discipline else perception.discipline

        logger.info(f"Filters applied -> Floor: {filtro_piso}, Discipline: {filtro_disciplina}")

        # Retrieve active context cache for native long-context
        active_cache_reasoner = None
        active_cache_validator = None
        if loaded_blueprints:
            last_plano = list(loaded_blueprints.values())[-1]
            active_cache_reasoner = last_plano.get("cache_name_reasoner") or last_plano.get("cache_name")
            active_cache_validator = last_plano.get("cache_name_validator") or last_plano.get("cache_name")

        # ─── STEP 2: REASONER AGENT ───
        logger.info(f"[Pipeline] === STEP 2: Reasoner ===")
        reasoner = await agente_razonador(
            perception=perception,
            vector_store=vector_store,
            plano_completo_path=None,  # Auto-detect
            building_id=building_id,
            active_cache=active_cache_reasoner,
            historial=history
        )

        # ─── STEP 3: VALIDATOR AGENT ───
        logger.info(f"[Pipeline] === STEP 3: Validator ===")
        pdf_path_on_disk = f"data/raw/{last_plano['blueprint_id']}.pdf" if loaded_blueprints else None
        validator = await agente_validador(
            reasoner=reasoner,
            plano_completo_path=pdf_path_on_disk,
            contexto_rag=reasoner.rag_context,
            active_cache=active_cache_validator
        )

        # ─── STEP 4: VISUALIZER AGENT ───
        logger.info(f"[Pipeline] === STEP 4: Visualizer ===")
        visualizacion = await agente_visualizador(
            reasoner=reasoner,
            validator=validator,
            perception_data=perception.model_dump()
        )

        # ─── BUILD FINAL RESPONSE ───
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # --- Nanopayments Web3 Billing (ARC Testnet / x402 Protocol) ---
        from app.nanopayments import X402NanopaymentGateway
        # Estimate characters for input (query + rag + base prompt)
        input_chars = len(query_text) + (len(reasoner.rag_context) if hasattr(reasoner, 'rag_context') else 0) + 2000
        # Estimate characters for output (final response)
        output_chars = len(validator.final_response or reasoner.candidate_response) + 500
        
        billing_receipt = X402NanopaymentGateway.process_payment(input_chars, output_chars, client_wallet_id=client_wallet_id)

        response = FacilityMindResponse(
            response=validator.final_response or reasoner.candidate_response,
            sources=validator.verified_sources if validator.verified_sources else reasoner.sources,
            traced_circuit=reasoner.traced_circuit,
            confidence=validator.final_confidence,
            warnings=validator.warnings + reasoner.technical_warnings,
            requires_supervisor=validator.requires_supervisor,
            visualization=visualizacion,
            safety=validator.safety,
            billing=billing_receipt,
            debug={
                "perception": perception.model_dump(),
                "reasoner": reasoner.model_dump(),
                "validator": validator.model_dump(),
            },
            processing_time_ms=elapsed_ms
        )

        logger.info(f"[Pipeline] Complete in {elapsed_ms}ms | conf={response.confidence:.2f} | supervisor={response.requires_supervisor}")

        return {
            "success": True,
            "data": response.model_dump()
        }

    except Exception as e:
        logger.error(f"[Pipeline] Error: {e}", exc_info=True)
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "error": str(e),
            "data": FacilityMindResponse(
                response=f"System error: {str(e)[:200]}. Please try again.",
                confidence=0.0,
                warnings=["System error"],
                requires_supervisor=True,
                debug={"error": str(e)},
                processing_time_ms=elapsed_ms
            ).model_dump()
        }

    finally:
        # Clean up temporary files
        for p in [audio_path, imagen_path]:
            if p and os.path.exists(p):
                os.remove(p)


# ═══════════════════════════════════════════════════════════════
# ENDPOINT: Health Check
# ═══════════════════════════════════════════════════════════════

@app.get("/api/v1/health")
async def health():
    """Verifies that the system is functioning correctly."""
    global vector_store

    status = {
        "status": "ok",
        "gemini_api": bool(GEMINI_API_KEY),
        "vector_store": vector_store is not None,
        "chunks_indexados": vector_store.count_chunks() if vector_store else 0,
        "version": "1.0.0",
    }

    return {"success": True, "data": status}


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS: Circle Web3 Onboarding (x402 real)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/billing/create-wallet")
async def api_create_wallet():
    """Creates a real developer-controlled wallet for a client in the background."""
    from app.nanopayments import X402NanopaymentGateway
    try:
        wallet_info = X402NanopaymentGateway.create_client_wallet()
        return {"success": True, "data": wallet_info}
    except Exception as e:
        logger.error(f"[API Billing] Error creating client wallet: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/billing/balance/{wallet_id}")
async def api_get_balance(wallet_id: str):
    """Fetches real-time USDC blockchain token balance and Gateway contract balance for a specific wallet."""
    from app.nanopayments import X402NanopaymentGateway
    try:
        balance = X402NanopaymentGateway.get_wallet_balance(wallet_id)
        address = X402NanopaymentGateway.get_wallet_address(wallet_id)
        gateway_balance = X402NanopaymentGateway.get_gateway_balance(address) if address else 0.0
        return {
            "success": True, 
            "data": {
                "balance": balance,
                "address": address,
                "gateway_balance": gateway_balance
            }
        }
    except Exception as e:
        logger.error(f"[API Billing] Error getting balance: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/v1/billing/gateway-deposit")
async def api_gateway_deposit(payload: dict):
    """Triggers an onchain deposit of USDC into the Gateway Wallet contract for gasless nanopayments."""
    from app.nanopayments import X402NanopaymentGateway
    try:
        wallet_id = payload.get("wallet_id")
        amount = float(payload.get("amount", 0.0))
        if not wallet_id or amount <= 0:
            return {"success": False, "error": "Invalid wallet ID or deposit amount."}
        
        result = X402NanopaymentGateway.deposit_to_gateway(wallet_id, amount)
        return {"success": True, "data": result}
    except Exception as e:
        logger.error(f"[API Billing] Error executing Gateway deposit: {e}")
        return {"success": False, "error": str(e)}



# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting FacilityMind API at http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
