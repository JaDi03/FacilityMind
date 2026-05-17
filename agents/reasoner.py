"""
FacilityMind — Agent 2: Technical Reasoner
Consults building blueprints via RAG + Long Context using Gemini 2.5 Pro.
Generates technical responses with exact citations and circuit tracing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GeminiModels, MAX_CONTEXT_CHARS
from models.schemas import PerceptionOutput, ReasonerOutput, SourceCitation, TracedCircuit
from ingestion.pdf_loader import load_blueprint_as_text

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reasoner.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

MODEL = GeminiModels.REASONER


async def agente_razonador(
    perception: PerceptionOutput,
    vector_store,
    plano_completo_path: Optional[str] = None,
    building_id: str = "default",
    active_cache: Optional[str] = None,
    historial: Optional[str] = None
) -> ReasonerOutput:
    """
    Technical Reasoner Agent: Consults blueprints and generates a candidate technical response.

    Args:
        perception: Output from the Perception Agent.
        vector_store: Instance of BlueprintVectorStore for RAG retrieval.
        plano_completo_path: Path to the full PDF (for long-context analysis).
        building_id: Identifier for the building.

    Returns:
        ReasonerOutput containing the candidate response and citations.
    """
    logger.info(f"[Reasoner] Starting analysis | floor={perception.floor}, room={perception.room}, object={perception.detected_object}")

    # --- 1. RAG Retrieval via Chroma (Fallback) ---
    query = _construir_query(perception)
    logger.info(f"[Reasoner] Query: {query[:100]}...")
    
    contexto_rag = ""
    if not active_cache:
        # Apply filters to enhance precision
        discipline = perception.discipline if perception.discipline else None
        floor = perception.floor if perception.floor else None
    
        try:
            results = vector_store.query_with_discipline_filter(
                query_text=query,
                discipline=discipline,
                floor=floor,
                n_results=10
            )
            contexto_rag = _formatear_rag(results)
            
            # Fallback: Si el filtro fue muy estricto y no encontró nada, buscar en todos los planos
            if len(contexto_rag.strip()) == 0 and (discipline or floor):
                logger.info("[Reasoner] Filtered query returned 0 results. Falling back to unfiltered search across ALL blueprints.")
                results = vector_store.query(query, n_results=10)
                contexto_rag = _formatear_rag(results)
                
        except Exception as e:
            logger.warning(f"[Reasoner] Error during filtered query, falling back to unfiltered: {e}")
            results = vector_store.query(query, n_results=10)
            contexto_rag = _formatear_rag(results)
    
        logger.info(f"[Reasoner] RAG context retrieved: {len(contexto_rag)} characters")

    # --- 2. Long-Context Blueprint Analysis ---
    plano_texto = ""
    if plano_completo_path and os.path.exists(plano_completo_path):
        plano_texto = load_blueprint_as_text(plano_completo_path, max_chars=MAX_CONTEXT_CHARS)
        logger.info(f"[Reasoner] Full blueprint loaded: {len(plano_texto)} characters")
    else:
        # Attempt to automatically locate the relevant blueprint
        plano_texto = _buscar_plano_automatico(perception, building_id)
        
    # If using cache, we don't need to load the full text manually
    if active_cache:
        plano_texto = ""

    # --- SAFETY CHECK: Refuse to answer if no data is available ---
    # If there's no cache, no RAG data, and no full blueprint text, 
    # the model would hallucinate a complete answer. Abort early.
    if not active_cache and len(contexto_rag.strip()) == 0 and len(plano_texto.strip()) == 0:
        logger.warning("[Reasoner] ABORT: No cache, no RAG data, no blueprint text. Refusing to fabricate an answer.")
        return ReasonerOutput(
            candidate_response=(
                "No tengo acceso a los planos en este momento. "
                "Por favor, sube el archivo PDF del plano en la barra lateral de Streamlit "
                "y vuelve a realizar tu consulta."
            ),
            sources=[],
            initial_confidence=0.05,
            technical_warnings=["No blueprint data available — response would be fabricated"]
        )

    # --- 3. Prompt Construction & Gemini Pro Execution ---
    prompt = _construir_prompt(perception, contexto_rag, plano_texto, historial)

    try:
        import asyncio
        from google import genai as genai_v2
        from google.genai import types
        
        v2_client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        
        response = None
        if active_cache:
            # --- Native Long-Context via Gemini Cache ---
            logger.info(f"[Reasoner] Active Context Cache found ({active_cache}). Using native full-blueprint analysis.")
            
            try:
                response = await asyncio.to_thread(
                    v2_client.models.generate_content,
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=active_cache,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=ReasonerOutput
                    )
                )
            except Exception as e:
                logger.warning(f"[Reasoner] Cache request failed ({e}). Falling back to traditional RAG.")
                active_cache = None
                
        if not active_cache:
            # --- Traditional RAG Execution ---
            response = await asyncio.to_thread(
                v2_client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ReasonerOutput,
                    system_instruction=SYSTEM_PROMPT
                )
            )
            
        raw = response.text.replace("```json", "").replace("```", "").strip()

        logger.info(f"[Reasoner] Raw response snippet: {raw[:400]}...")

        # Parse JSON output
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[Reasoner] Malformed JSON received, applying fallback reasoning")
            data = _fallback_reasoning(perception, contexto_rag)

        # Normalize and validate data structures
        if "sources" in data and isinstance(data["sources"], list):
            data["sources"] = [SourceCitation(**s) for s in data["sources"]]
        else:
            data["sources"] = []

        if "traced_circuit" in data and data["traced_circuit"]:
            if isinstance(data["traced_circuit"], dict):
                data["traced_circuit"] = TracedCircuit(**data["traced_circuit"])
            elif isinstance(data["traced_circuit"], list):
                data["traced_circuit"] = TracedCircuit(
                    nodes=data["traced_circuit"],
                    description="Traced circuit path",
                    type=perception.discipline or "general"
                )

        if "initial_confidence" not in data:
            data["initial_confidence"] = 0.5

        # Enforce required field 'candidate_response'
        if "candidate_response" not in data:
            if "respuesta_candidata" in data:
                data["candidate_response"] = data["respuesta_candidata"]
            elif "respuesta" in data:
                data["candidate_response"] = data["respuesta"]
            elif "response" in data:
                data["candidate_response"] = data["response"]
            elif "answer" in data:
                data["candidate_response"] = data["answer"]
            else:
                data["candidate_response"] = str(data)

        # Inject RAG context so Validator can use it as fallback
        data["rag_context"] = contexto_rag
        
        output = ReasonerOutput(**data)
        logger.info(f"[Reasoner] Analysis complete | sources={len(output.sources)}, confidence={output.initial_confidence:.2f}")
        return output

    except Exception as e:
        logger.error(f"[Reasoner] Processing error: {e}")
        return ReasonerOutput(
            candidate_response=f"Reasoner Agent error: {str(e)[:200]}. Please try again with a more specific query.",
            sources=[],
            initial_confidence=0.1,
            technical_warnings=["Technical processing error in Reasoner Agent"],
            rag_context=contexto_rag
        )


def _construir_query(perception: PerceptionOutput) -> str:
    """Constructs an optimized query for RAG retrieval."""
    partes = []
    if perception.query_objective:
        partes.append(perception.query_objective)
    if perception.floor:
        partes.append(f"floor {perception.floor}")
    if perception.room:
        partes.append(f"room {perception.room}")
    if perception.detected_object:
        partes.append(perception.detected_object)
    if perception.discipline:
        partes.append(perception.discipline)

    query = " ".join(partes) if partes else perception.query_objective or "general inquiry"
    return query


def _formatear_rag(results: dict) -> str:
    """Formats RAG retrieval results for inclusion in the prompt."""
    if not results or not results.get("documents"):
        return "(No relevant document fragments found in the database)"

    contexto = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        plano_id = meta.get("blueprint_id", "unknown")
        pagina = meta.get("page", "?")
        tipo = meta.get("blueprint_type", "")
        contexto += f"\n--- SOURCE: {plano_id} (Page {pagina}, {tipo}) ---\n{doc}\n"

    return contexto


def _buscar_plano_automatico(perception: PerceptionOutput, building_id: str) -> str:
    """Attempts to automatically locate and load the full relevant blueprint."""
    from config import PDF_RAW_PATH

    plano_texto = ""
    candidatos = []

    # Match by floor and discipline patterns
    if perception.floor and perception.discipline:
        prefijo = {
            "electrical": "E-",
            "plumbing": "P-",
            "architectural": "A-",
            "structural": "S-",
            "hvac": "M-",
        }.get(perception.discipline, "")

        if prefijo:
            piso_str = str(perception.floor).upper()
            patron = f"{prefijo}*{piso_str}*"
            candidatos = list(PDF_RAW_PATH.glob(patron))

    # Fallback: search for any blueprint associated with the building ID
    if not candidatos:
        candidatos = list(PDF_RAW_PATH.glob(f"*{building_id}*.pdf"))

    if candidatos:
        plano_texto = load_blueprint_as_text(str(candidatos[0]), max_chars=MAX_CONTEXT_CHARS)
        logger.info(f"[Reasoner] Automatically detected blueprint: {candidatos[0].name}")

    return plano_texto


def _construir_prompt(perception: PerceptionOutput, contexto_rag: str, plano_texto: str, historial: Optional[str] = None) -> str:
    """Constructs the final prompt for the Reasoner Agent."""

    historial_text = ""
    if historial:
        historial_text = f"--- CHAT HISTORY ---\n{historial}\n--------------------\nUse this history to understand the context of the user's current query.\n\n"

    prompt = f"""{historial_text}FIELD TECHNICIAN DATA (from Perception Agent):
- Floor: {perception.floor or "Not detected"}
- Tower: {perception.tower or "Not detected"}
- Room/Unit: {perception.room or "Not detected"}
- Identified Object: {perception.detected_object or "Not detected"}
- Query Objective: {perception.query_objective}
- Discipline: {perception.discipline or "Not determined"}
- Urgency Notes: {perception.urgency_notes or "None"}
- Detected Language: {perception.detected_language or "en"}

RELEVANT BLUEPRINT FRAGMENTS (RAG):
{contexto_rag}
"""

    if plano_texto:
        prompt += f"""
FULL BLUEPRINT TEXT (Long context for cross-verification):
{plano_texto}
"""
    else:
        prompt += """
FULL BLUEPRINT TEXT: Not provided. Rely solely on RAG fragments.
"""

    prompt += """
Generate the candidate technical response in the specified strict JSON format.
CRITICAL: Do NOT fabricate data. ALWAYS cite sources. Trace the complete circuit/path if applicable.
"""

    return prompt


def _fallback_reasoning(perception: PerceptionOutput, contexto_rag: str) -> dict:
    """Fallback logic when JSON parsing fails."""
    return {
        "candidate_response": (
            f"Based on the available blueprints for {perception.room or 'the location'} "
            f"on floor {perception.floor or 'unspecified'}: "
            f"Found references related to '{perception.query_objective}'. "
            f"However, manual review is required to confirm exact technical specifications "
            f"(e.g., breaker numbers, wire gauges)."
        ),
        "sources": [],
        "traced_circuit": None,
        "recommended_steps": ["Verify information on the physical panel", "Consult with a technical supervisor"],
        "mentioned_materials": [],
        "cited_regulation": None,
        "initial_confidence": 0.3,
        "technical_warnings": ["Response generated in fallback mode — requires manual verification"]
    }
