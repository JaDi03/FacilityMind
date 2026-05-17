
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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
from ingestion.pdf_loader import cargar_plano, listar_planos_disponibles
from ingestion.chunker import crear_chunks_inteligentes
from ingestion.vector_store import PlanoVectorStore

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
vector_store: PlanoVectorStore = None
planos_cargados: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and cleans up the application context."""
    global vector_store, planos_cargados
    logger.info("🚀 FacilityMind Backend starting...")

    # Initialize Vector Store
    vector_store = PlanoVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
    logger.info(f"📦 VectorStore ready: {vector_store.contar_chunks()} existing chunks")

    # Load persisted planos_cargados if exists
    persisted_path = Path("data/processed/planos_cargados.json")
    if persisted_path.exists():
        try:
            import json
            with open(persisted_path, "r", encoding="utf-8") as f:
                planos_cargados.update(json.load(f))
            logger.info(f"📂 Loaded {len(planos_cargados)} persisted blueprints from disk")
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
    pregunta: str
    edificio_id: Optional[str] = "default"
    disciplina: Optional[str] = None
    piso: Optional[str] = None


class ConsultaResponse(BaseModel):
    """Standardized API response."""
    success: bool
    data: FacilityMindResponse = None
    error: str = None


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS: Blueprint Ingestion
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/planos/upload")
async def upload_plano(
    file: UploadFile = File(...),
    plano_id: str = Form(None),
    edificio_id: str = Form("default"),
    tipo_plano: str = Form(None),
):
    """
    Uploads a PDF blueprint, processes it, and indexes it into the vector database.

    - **file**: PDF blueprint file
    - **plano_id**: Optional identifier (e.g., E-14, P-01). Defaults to filename if not provided.
    - **edificio_id**: Identifier of the building
    - **tipo_plano**: Optional category (electrical, plumbing, architectural)
    """
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    if not plano_id:
        plano_id = Path(file.filename).stem

    # Save file temporarily in a local directory to avoid Windows permission issues
    tmp_dir = Path("./data/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = Path(file.filename).suffix or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(tmp_dir))
    
    try:
        with os.fdopen(fd, 'wb') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
        
        # 1. Load PDF
        documentos = cargar_plano(tmp_path, plano_id=plano_id, edificio_id=edificio_id)

        if not documentos:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Could not extract text from PDF. Is it a scanned PDF without OCR?"}
            )

        # Overwrite discipline if provided
        if tipo_plano:
            for doc in documentos:
                doc["metadata"]["tipo_plano"] = tipo_plano

        # 2. Create intelligent chunks (pages + tables)
        chunks = crear_chunks_inteligentes(documentos, incluir_tablas=True, incluir_secciones=False)

        # 3. Index in Chroma
        num_agregados = vector_store.add_chunks(chunks)

        # 4. Create Context Cache (Google Way)
        import asyncio
        from agents.cache_manager import cache_blueprint
        
        cache_result = await asyncio.to_thread(cache_blueprint, tmp_path, plano_id)
        cache_name = cache_result.get("cache_name") if cache_result.get("success") else None

        # 5. Register loaded blueprint
        planos_cargados[plano_id] = {
            "plano_id": plano_id,
            "edificio_id": edificio_id,
            "tipo_plano": documentos[0]["metadata"]["tipo_plano"] if documentos else tipo_plano,
            "piso": documentos[0]["metadata"].get("piso") if documentos else None,
            "total_paginas": documentos[0]["metadata"]["total_paginas"] if documentos else 0,
            "chunks_indexados": num_agregados,
            "cache_name": cache_name,
        }

        # Persist to disk so restarts don't lose the registered caches
        try:
            import json
            persisted_path = Path("data/processed/planos_cargados.json")
            persisted_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persisted_path, "w", encoding="utf-8") as f:
                json.dump(planos_cargados, f, indent=4)
            logger.info("[API] Successfully persisted planos_cargados to disk")
        except Exception as e:
            logger.warning(f"⚠️ Failed to persist planos_cargados: {e}")

        logger.info(f"[API] Blueprint {plano_id} loaded: {num_agregados} chunks")

        return {
            "success": True,
            "data": planos_cargados[plano_id]
        }

    except Exception as e:
        logger.error(f"[API] Error loading blueprint: {e}")
        raise HTTPException(500, f"Error processing blueprint: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/v1/planos")
async def listar_planos():
    """Lists all blueprints loaded in the system."""
    global vector_store

    planos = vector_store.listar_planos() if vector_store else []

    return {
        "success": True,
        "data": {
            "planos": planos,
            "total_chunks": vector_store.contar_chunks() if vector_store else 0,
            "planos_detalle": [planos_cargados.get(p, {"plano_id": p}) for p in planos]
        }
    }


@app.delete("/api/v1/planos/{plano_id}")
async def eliminar_plano(plano_id: str):
    """Deletes a blueprint from the system."""
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    success = vector_store.delete_plano(plano_id)
    if success:
        planos_cargados.pop(plano_id, None)
        
        # Persist modified planos_cargados to disk after deletion
        try:
            import json
            persisted_path = Path("data/processed/planos_cargados.json")
            with open(persisted_path, "w", encoding="utf-8") as f:
                json.dump(planos_cargados, f, indent=4)
            logger.info(f"[API] Updated planos_cargados on disk after deleting {plano_id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update persisted planos_cargados: {e}")
            
        return {"success": True, "message": f"Blueprint {plano_id} deleted"}
    else:
        raise HTTPException(404, f"Blueprint {plano_id} not found")


# ═══════════════════════════════════════════════════════════════
# ENDPOINT: Multi-Agent Query (Full Pipeline)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/v1/consulta")
async def consulta(
    pregunta: str = Form(...),
    historial: str = Form(None),
    edificio_id: str = Form("default"),
    disciplina: str = Form(None),
    piso: str = Form(None),
    audio: UploadFile = File(None),
    imagen: UploadFile = File(None),
):
    """
    Full query pipeline: Perception → Reasoner → Validator → Visualizer.

    - **pregunta**: Query text
    - **edificio_id**: Building identifier
    - **disciplina**: Optional filter (electrical, plumbing, etc.)
    - **piso**: Optional floor filter
    - **audio**: Optional audio file
    - **imagen**: Optional image file
    """
    global vector_store

    if not vector_store:
        raise HTTPException(500, "VectorStore not initialized")

    start_time = time.time()
    audio_path = None
    imagen_path = None

    try:
        # --- Save temporary files ---
        if audio:
            suffix = Path(audio.filename).suffix or ".ogg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await audio.read())
                audio_path = tmp.name

        if imagen:
            suffix = Path(imagen.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await imagen.read())
                imagen_path = tmp.name

        # ═══════════════════════════════════════════════════════
        # SEMANTIC ROUTING (Conversational vs Technical)
        # ═══════════════════════════════════════════════════════
        conversational_keywords = ["hola", "qué plano", "que plano", "gracias", "quién eres", "quien eres", "cuál plano", "cual plano"]
        is_conversational = any(kw in pregunta.lower() for kw in conversational_keywords)
        
        if is_conversational and not imagen and not audio:
            logger.info("[Pipeline] Query routed to Conversational Agent (skipping technical pipeline)")
            import google.generativeai as genai
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            
            chat_context = f"Chat History:\n{historial}\n\n" if historial else ""
            plano_info = "No blueprints loaded."
            if planos_cargados:
                plano_info = f"Active blueprint: ID {list(planos_cargados.values())[-1].get('plano_id', 'Desconocido')}."
                
            prompt = f"You are the FacilityMind Orchestrator. {chat_context} The user says: '{pregunta}'. Respond in a friendly, concise, and helpful way (max 1 paragraph) in the user's language (Spanish). Current system state: {plano_info}."
            
            quick_response = await model.generate_content_async(prompt)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            return {
                "success": True,
                "data": FacilityMindResponse(
                    respuesta=quick_response.text,
                    sources=[],
                    confianza=1.0,
                    advertencias=["Direct conversational response (no technical analysis)"],
                    requiere_supervisor=False,
                    tiempo_procesamiento_ms=elapsed_ms
                ).model_dump()
            }

        # ═══════════════════════════════════════════════════════
        # STEP 1: PERCEPTION AGENT
        # ═══════════════════════════════════════════════════════
        logger.info(f"[Pipeline] === STEP 1: Perception ===")
        perception = await agente_percepcion(
            audio_path=audio_path,
            photo_path=imagen_path,
            text=pregunta
        )

        # Abort if perception confidence is too low
        if perception.confianza_percepcion < Thresholds.PERCEPTION_MIN:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "success": True,
                "data": FacilityMindResponse(
                    respuesta="I couldn't fully understand your query. Please provide a clearer photo or more details about the location and your requirement.",
                    sources=[],
                    confianza=perception.confianza_percepcion,
                    advertencias=["Low confidence in multimodal perception"],
                    requiere_supervisor=True,
                    debug={"percepcion": perception.model_dump()},
                    tiempo_procesamiento_ms=elapsed_ms
                ).model_dump()
            }

        # Overwrite with explicit filters if provided by the user
        if disciplina:
            perception.disciplina = disciplina
        if piso:
            perception.piso = piso

        # Retrieve active context cache for native long-context
        active_cache = None
        if planos_cargados:
            last_plano = list(planos_cargados.values())[-1]
            active_cache = last_plano.get("cache_name")

        # ═══════════════════════════════════════════════════════
        # STEP 2: REASONER AGENT
        # ═══════════════════════════════════════════════════════
        logger.info(f"[Pipeline] === STEP 2: Reasoner ===")
        reasoner = await agente_razonador(
            perception=perception,
            vector_store=vector_store,
            plano_completo_path=None,  # Auto-detect
            building_id=edificio_id,
            active_cache=active_cache,
            historial=historial
        )

        # ═══════════════════════════════════════════════════════
        # STEP 3: VALIDATOR AGENT
        # ═══════════════════════════════════════════════════════
        logger.info(f"[Pipeline] === STEP 3: Validator ===")
        validator = await agente_validador(
            reasoner=reasoner,
            plano_completo_path=None,
            active_cache=active_cache
        )

        # ═══════════════════════════════════════════════════════
        # STEP 4: VISUALIZER AGENT
        # ═══════════════════════════════════════════════════════
        logger.info(f"[Pipeline] === STEP 4: Visualizer ===")
        visualizacion = await agente_visualizador(
            reasoner=reasoner,
            validator=validator,
            perception_data=perception.model_dump()
        )

        # ═══════════════════════════════════════════════════════
        # BUILD FINAL RESPONSE
        # ═══════════════════════════════════════════════════════
        elapsed_ms = int((time.time() - start_time) * 1000)

        response = FacilityMindResponse(
            respuesta=validator.respuesta_final or reasoner.respuesta_candidata,
            sources=validator.sources_verificadas if validator.sources_verificadas else reasoner.sources,
            circuito_trazado=reasoner.circuito_trazado,
            confianza=validator.confianza_final,
            advertencias=validator.advertencias + reasoner.advertencias_tecnicas,
            requiere_supervisor=validator.requiere_supervisor,
            visualizacion=visualizacion,
            safety=validator.safety,
            debug={
                "percepcion": perception.model_dump(),
                "razonador": reasoner.model_dump(),
                "validador": validator.model_dump(),
            },
            tiempo_procesamiento_ms=elapsed_ms
        )

        logger.info(f"[Pipeline] Complete in {elapsed_ms}ms | conf={response.confianza:.2f} | supervisor={response.requiere_supervisor}")

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
                respuesta=f"System error: {str(e)[:200]}. Please try again.",
                confianza=0.0,
                advertencias=["System error"],
                requiere_supervisor=True,
                debug={"error": str(e)},
                tiempo_procesamiento_ms=elapsed_ms
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
        "chunks_indexados": vector_store.contar_chunks() if vector_store else 0,
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
