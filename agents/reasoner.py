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


from config import GEMINI_API_KEY, GeminiModels, MAX_CONTEXT_CHARS
from models.schemas import PerceptionOutput, ReasonerOutput, SourceCitation, TracedCircuit
from ingestion.pdf_loader import load_blueprint_as_text

logger = logging.getLogger(__name__)

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
    # ALWAYS retrieve RAG context to provide precise textual guidance and focus for Gemini
    if True:
        import re
        
        # Precise technical extraction of circuits and units from all perception outputs
        texts_to_check = [
            perception.query_objective,
            perception.room or "",
            perception.detected_object or "",
            perception.raw_transcription or ""
        ]
        combined_text = " ".join([t for t in texts_to_check if t])
        
        # Regex for circuit patterns (matches "A-9", "A 10", "A9", etc. with high precision)
        circuit_match = re.search(r'\b([A-Z])\s*[-–—]?\s*(\d+)\b', combined_text, re.IGNORECASE)
        circuit = None
        if circuit_match:
            candidate_circuit = f"{circuit_match.group(1).upper()}-{circuit_match.group(2)}"
            # Avoid matching sheet IDs or blueprint IDs (like E-14, P-2) as circuits.
            is_sheet_id = False
            
            # Check 1: Is it a registered blueprint ID in our vector store?
            try:
                registered_blueprints = vector_store.list_blueprints()
                if candidate_circuit in registered_blueprints or candidate_circuit.replace("-", "") in registered_blueprints:
                    is_sheet_id = True
                    logger.info(f"[Reasoner] Rejected candidate circuit '{candidate_circuit}' because it matches a registered blueprint ID.")
            except Exception as e:
                logger.warning(f"[Reasoner] Could not query registered blueprints for circuit filtering: {e}")
            
            # Check 2: Does it match sheet indicators in the text?
            if not is_sheet_id and candidate_circuit.startswith(("E-", "P-", "A-", "M-", "S-")):
                context_lower = combined_text.lower()
                for indicator in ["blueprint", "plano", "sheet", "hoja", "document"]:
                    if f"{indicator} {candidate_circuit.lower()}" in context_lower or f"{indicator} {circuit_match.group(1).lower()}-{circuit_match.group(2)}" in context_lower:
                        is_sheet_id = True
                        break
            
            if not is_sheet_id:
                circuit = candidate_circuit
        
        # Regex for unit patterns (matches "UNIT B", "UNIDAD B", "UNITB")
        unit_match = re.search(r'\b(?:UNIT|UNIDAD)\s*([A-G])\b', combined_text, re.IGNORECASE)
        unit = f"UNIT {unit_match.group(1).upper()}" if unit_match else None

        logger.info(f"[Reasoner] Extracted filters from text: unit={unit}, circuit={circuit}")
        
        results = None
        # Try structured metadata index query first if any key filter is extracted
        if circuit or unit:
            try:
                logger.info(f"[Reasoner] Running precise metadata search for unit={unit}, circuit={circuit}")
                results = vector_store.query_by_metadata(unit=unit, circuit=circuit)
                if results and results.get("documents") and len(results["documents"][0]) > 0:
                    contexto_rag = _formatear_rag(results)
                    logger.info(f"[Reasoner] Precise metadata search succeeded. Found {len(results['documents'][0])} chunks.")
            except Exception as e:
                logger.warning(f"[Reasoner] Error during precise metadata search: {e}")

        # Fallback: Use standard semantic embedding RAG query if no metadata match was found
        if not contexto_rag:
            logger.info("[Reasoner] Falling back to semantic embedding search.")
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
                
                # Filter strict fallback
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

    # --- SAFETY CHECK: Refuse to answer if no verifiable data is available ---
    # We abort ONLY if there is no cache, no RAG data, and no raw blueprint text.
    # If there is an active_cache, Gemini has access to the full PDF and CAN answer natively.
    rag_is_empty = len(contexto_rag.strip()) == 0
    blueprint_is_empty = len(plano_texto.strip()) == 0

    if not active_cache and rag_is_empty and blueprint_is_empty:
        logger.warning("[Reasoner] ABORT: No active cache, no RAG data, and no blueprint text.")
        return ReasonerOutput(
            candidate_response=(
                "No tengo informacion verificable para responder esta consulta. "
                "El sistema no encontro datos relevantes en los planos cargados. "
                "Por favor verifica que: (1) hay planos cargados, "
                "(2) la consulta esta relacionada con los planos disponibles."
            ),
            sources=[],
            initial_confidence=0.05,
            technical_warnings=["No blueprint data available - response deliberately withheld"],
            rag_context=contexto_rag
        )

    # Additional safety: if RAG is empty but we have cache, log it and rely on native PDF cache (FIX 7)
    if rag_is_empty and active_cache:
        logger.info("[Reasoner] RAG empty but cache active - relying on native PDF analysis")

    # --- 3. Prompt Construction & Gemini Pro Execution ---
    prompt = _construir_prompt(perception, contexto_rag, plano_texto, historial, has_cache=bool(active_cache))

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
                # CRITICAL FIX: Reload the text we cleared earlier!
                if not plano_texto:
                    if plano_completo_path and os.path.exists(plano_completo_path):
                        plano_texto = load_blueprint_as_text(plano_completo_path)
                    else:
                        plano_texto = _buscar_plano_automatico(perception, building_id)
                # Rebuild prompt with the newly loaded text and no cache
                prompt = _construir_prompt(perception, contexto_rag, plano_texto, historial, has_cache=False)
                
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

    query_en = " ".join(partes) if partes else perception.query_objective or "general inquiry"
    
    # Multilingual search: Append original transcription if it is not in English
    if perception.raw_transcription and perception.detected_language != "en":
        return f"{query_en} | {perception.raw_transcription}"
    return query_en


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


def _construir_prompt(perception: PerceptionOutput, contexto_rag: str, plano_texto: str, historial: Optional[str] = None, has_cache: bool = False) -> str:
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
    elif has_cache:
        prompt += """
FULL BLUEPRINT PDF: Provided directly via the context cache. You have full high-fidelity visual and text access to the entire 32-page blueprint PDF. Trace physical lines, curves, and dashed loops to verify connections between rooms, panels, and circuits.
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
