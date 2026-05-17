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
from models.schemas import FacilityMindResponse, PlanoMetadata

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
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
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

    # Load persisted loaded_blueprints if exists
    persisted_path = Path("data/processed/loaded_blueprints.json")
    if persisted_path.exists():
        try:
            import json
            with open(persisted_path, "r", encoding="utf-8") as f:
                loaded_blueprints.update(json.load(f))
            logger.info(f"📂 Loaded {len(loaded_blueprints)} persisted blueprints from disk")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load persisted blueprints: {e}")

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


def ejecutar_ocr_visual_background(pdf_path: str, blueprint_id: str, building_id: str, blueprint_type: str):
    """
    Background task to run Vision OCR on ALL pages of a blueprint PDF and index them into Chroma DB.
    Uses Gemini's native PDF processing — no PyMuPDF dependency.
    """
    import time
    from google import genai as genai_v2
    from config import GEMINI_API_KEY
    from ingestion.pdf_loader import extract_text_with_vision, infer_floor, infer_tower, _count_pdf_pages
    from ingestion.chunker import create_intelligent_chunks
    from ingestion.vector_store import BlueprintVectorStore
    
    logger.info(f"[Background OCR] Starting Vision OCR for {blueprint_id} (All pages)...")
    try:
        total_pages = _count_pdf_pages(pdf_path)
        
        # Upload PDF to Gemini ONCE, reuse for all page extractions
        client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        uploaded_file = client.files.upload(file=pdf_path, config={'display_name': f'{blueprint_id}_ocr'})
        
        file_info = client.files.get(name=uploaded_file.name)
        while file_info.state.name == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=uploaded_file.name)
        
        if file_info.state.name == "FAILED":
            raise Exception("Google Gemini failed to process the PDF document.")
        
        logger.info(f"[Background OCR] PDF uploaded to Gemini. Processing {total_pages} pages...")
        
        vector_store = BlueprintVectorStore()
        vision_pages_processed = 0
        
        for page_num in range(1, total_pages + 1):
            logger.info(f"[Background OCR] Processing Page {page_num}/{total_pages} with Vision OCR...")
            
            # Extract structured text using Gemini's native PDF understanding
            vision_text = extract_text_with_vision(pdf_path, page_num, uploaded_file=uploaded_file)
            
            if vision_text.strip():
                page_floor = infer_floor(blueprint_id, page_num - 1, total_pages)
                tower = infer_tower(blueprint_id)
                
                doc_dict = {
                    "text": f"[PDF Page {page_num} — Vision OCR]\n{vision_text}",
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
                    },
                    "id": f"{blueprint_id}_p{page_num}"
                }
                
                # Chunk and index immediately
                chunks = create_intelligent_chunks([doc_dict], include_tables=True, include_sections=False)
                vector_store.add_chunks(chunks)
                vision_pages_processed += 1
                logger.info(f"[Background OCR] Page {page_num}/{total_pages} indexed into Chroma.")
            
            # Rate limit guard (avoid 429 Resource Exhausted)
            time.sleep(1.5)
        
        # Clean up the temporary OCR upload
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
        
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
    audio_file: UploadFile = File(None),
    image_file: UploadFile = File(None),
):
    """
    Full query pipeline: Perception → Reasoner → Validator → Visualizer.

    - **query_text**: Query text
    - **building_id**: Building identifier
    - **discipline**: Optional filter (electrical, plumbing, etc.)
    - **floor**: Optional floor filter
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
        # PASO 0: LOBSTER TRAP DEEP PROMPT INSPECTION (DPI)
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
        # PASO 1: AGENTE DE PERCEPCIÓN (Multimodal a Texto)
        # ═══════════════════════════════════════════════════════
        perception = await agente_percepcion(
            text=query_text,
            audio_path=audio_path,
            photo_path=imagen_path
        )
        
        # ═══════════════════════════════════════════════════════
        # SEMANTIC ROUTING (Conversational vs Technical)
        # ═══════════════════════════════════════════════════════
        conversational_keywords = ["hola", "qué plano", "que plano", "gracias", "quién eres", "quien eres", "cuál plano", "cual plano"]
        is_conversational = any(kw in query_text.lower() for kw in conversational_keywords)
        
        if is_conversational and not image_file and not audio_file:
            logger.info("[Pipeline] Query routed to Conversational Agent (skipping technical pipeline)")
            import google.generativeai as genai
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            
            chat_context = f"Chat History:\n{history}\n\n" if history else ""
            plano_info = "No blueprints loaded."
            if loaded_blueprints:
                plano_info = f"Active blueprint: ID {list(loaded_blueprints.values())[-1].get('blueprint_id', 'Desconocido')}."
                
            prompt = f"You are the FacilityMind Orchestrator. {chat_context} The user says: '{query_text}'. Respond in a friendly, concise, and helpful way (max 1 paragraph) in the user's language (Spanish). Current system state: {plano_info}."
            
            quick_response = await model.generate_content_async(prompt)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response=quick_response.text,
                    sources=[],
                    confidence=1.0,
                    warnings=["Direct conversational response (no technical analysis)"],
                    requires_supervisor=False,
                    processing_time_ms=elapsed_ms
                ).model_dump()
            }

        # Abort if perception confidence is too low ONLY when multimodal media is uploaded.
        # If it's a pure text query, we let it pass to the Reasoner (Gemini Pro) to answer building-wide or abstract queries.
        if (audio_file or image_file) and perception.perception_confidence < Thresholds.PERCEPTION_MIN:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    response="I couldn't fully understand your query. Please provide a clearer photo or more details about the location and your requirement.",
                    sources=[],
                    confidence=perception.perception_confidence,
                    warnings=["Low confidence in multimodal perception"],
                    requires_supervisor=True,
                    debug={"perception": perception.model_dump()},
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
        validator = await agente_validador(
            reasoner=reasoner,
            plano_completo_path=None,
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

        response = FacilityMindResponse(
            response=validator.final_response or reasoner.candidate_response,
            sources=validator.verified_sources if validator.verified_sources else reasoner.sources,
            traced_circuit=reasoner.traced_circuit,
            confidence=validator.final_confidence,
            warnings=validator.warnings + reasoner.technical_warnings,
            requires_supervisor=validator.requires_supervisor,
            visualization=visualizacion,
            safety=validator.safety,
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
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting FacilityMind API at http://{API_HOST}:{API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
