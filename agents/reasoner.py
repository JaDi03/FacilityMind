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
from ingestion.pdf_loader import cargar_plano_como_texto

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
    active_cache: Optional[str] = None
) -> ReasonerOutput:
    """
    Technical Reasoner Agent: Consults blueprints and generates a candidate technical response.

    Args:
        perception: Output from the Perception Agent.
        vector_store: Instance of PlanoVectorStore for RAG retrieval.
        plano_completo_path: Path to the full PDF (for long-context analysis).
        building_id: Identifier for the building.

    Returns:
        ReasonerOutput containing the candidate response and citations.
    """
    logger.info(f"[Reasoner] Starting analysis | floor={perception.piso}, room={perception.habitacion}, object={perception.objeto_detectado}")

    # --- 1. RAG Retrieval via Chroma (Fallback) ---
    query = _construir_query(perception)
    logger.info(f"[Reasoner] Query: {query[:100]}...")
    
    contexto_rag = ""
    if not active_cache:
        # Apply filters to enhance precision
        disciplina = perception.disciplina if perception.disciplina else None
        piso = perception.piso if perception.piso else None
    
        try:
            results = vector_store.query_con_filtro_disciplina(
                query_text=query,
                disciplina=disciplina,
                piso=piso,
                n_results=10
            )
            contexto_rag = _formatear_rag(results)
            
            # Fallback: Si el filtro fue muy estricto y no encontró nada, buscar en todos los planos
            if len(contexto_rag.strip()) == 0 and (disciplina or piso):
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
        plano_texto = cargar_plano_como_texto(plano_completo_path, max_chars=MAX_CONTEXT_CHARS)
        logger.info(f"[Reasoner] Full blueprint loaded: {len(plano_texto)} characters")
    else:
        # Attempt to automatically locate the relevant blueprint
        plano_texto = _buscar_plano_automatico(perception, building_id)
        
    # If using cache, we don't need to load the full text manually
    if active_cache:
        plano_texto = ""

    # --- 3. Prompt Construction & Gemini Pro Execution ---
    prompt = _construir_prompt(perception, contexto_rag, plano_texto)

    try:
        import asyncio
        from google import genai as genai_v2
        from google.genai import types
        
        v2_client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        
        if active_cache:
            # --- Native Long-Context via Gemini Cache ---
            logger.info(f"[Reasoner] Active Context Cache found ({active_cache}). Using native full-blueprint analysis.")
            
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
        else:
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

        if "circuito_trazado" in data and data["circuito_trazado"]:
            if isinstance(data["circuito_trazado"], dict):
                data["circuito_trazado"] = TracedCircuit(**data["circuito_trazado"])
            elif isinstance(data["circuito_trazado"], list):
                data["circuito_trazado"] = TracedCircuit(
                    nodos=data["circuito_trazado"],
                    descripcion="Traced circuit path",
                    tipo=perception.disciplina or "general"
                )

        if "confianza_inicial" not in data:
            data["confianza_inicial"] = 0.5

        # Enforce required field 'respuesta_candidata'
        if "respuesta_candidata" not in data:
            if "respuesta" in data:
                data["respuesta_candidata"] = data["respuesta"]
            elif "response" in data:
                data["respuesta_candidata"] = data["response"]
            elif "answer" in data:
                data["respuesta_candidata"] = data["answer"]
            else:
                data["respuesta_candidata"] = str(data)

        output = ReasonerOutput(**data)
        logger.info(f"[Reasoner] Analysis complete | sources={len(output.sources)}, confidence={output.confianza_inicial:.2f}")
        return output

    except Exception as e:
        logger.error(f"[Reasoner] Processing error: {e}")
        return ReasonerOutput(
            respuesta_candidata=f"Reasoner Agent error: {str(e)[:200]}. Please try again with a more specific query.",
            sources=[],
            confianza_inicial=0.1,
            advertencias_tecnicas=["Technical processing error in Reasoner Agent"]
        )


def _construir_query(perception: PerceptionOutput) -> str:
    """Constructs an optimized query for RAG retrieval."""
    partes = []
    if perception.objetivo_consulta:
        partes.append(perception.objetivo_consulta)
    if perception.piso:
        partes.append(f"floor {perception.piso}")
    if perception.habitacion:
        partes.append(f"room {perception.habitacion}")
    if perception.objeto_detectado:
        partes.append(perception.objeto_detectado)
    if perception.disciplina:
        partes.append(perception.disciplina)

    query = " ".join(partes) if partes else perception.objetivo_consulta or "general inquiry"
    return query


def _formatear_rag(results: dict) -> str:
    """Formats RAG retrieval results for inclusion in the prompt."""
    if not results or not results.get("documents"):
        return "(No relevant document fragments found in the database)"

    contexto = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results.get("metadatas") else {}
        plano_id = meta.get("plano_id", "unknown")
        pagina = meta.get("pagina", "?")
        tipo = meta.get("tipo_plano", "")
        contexto += f"\n--- SOURCE: {plano_id} (Page {pagina}, {tipo}) ---\n{doc}\n"

    return contexto


def _buscar_plano_automatico(perception: PerceptionOutput, building_id: str) -> str:
    """Attempts to automatically locate and load the full relevant blueprint."""
    from config import PDF_RAW_PATH

    plano_texto = ""
    candidatos = []

    # Match by floor and discipline patterns
    if perception.piso and perception.disciplina:
        prefijo = {
            "electrical": "E-",
            "plumbing": "P-",
            "architectural": "A-",
            "structural": "S-",
            "hvac": "M-",
        }.get(perception.disciplina, "")

        if prefijo:
            piso_str = str(perception.piso).upper()
            patron = f"{prefijo}*{piso_str}*"
            candidatos = list(PDF_RAW_PATH.glob(patron))

    # Fallback: search for any blueprint associated with the building ID
    if not candidatos:
        candidatos = list(PDF_RAW_PATH.glob(f"*{building_id}*.pdf"))

    if candidatos:
        plano_texto = cargar_plano_como_texto(str(candidatos[0]), max_chars=MAX_CONTEXT_CHARS)
        logger.info(f"[Reasoner] Automatically detected blueprint: {candidatos[0].name}")

    return plano_texto


def _construir_prompt(perception: PerceptionOutput, contexto_rag: str, plano_texto: str) -> str:
    """Constructs the final prompt for the Reasoner Agent."""

    prompt = f"""FIELD TECHNICIAN DATA (from Perception Agent):
- Floor: {perception.piso or "Not detected"}
- Tower: {perception.torre or "Not detected"}
- Room/Unit: {perception.habitacion or "Not detected"}
- Identified Object: {perception.objeto_detectado or "Not detected"}
- Query Objective: {perception.objetivo_consulta}
- Discipline: {perception.disciplina or "Not determined"}
- Urgency Notes: {perception.notas_urgencia or "None"}

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
        "respuesta_candidata": (
            f"Based on the available blueprints for {perception.habitacion or 'the location'} "
            f"on floor {perception.piso or 'unspecified'}: "
            f"Found references related to '{perception.objetivo_consulta}'. "
            f"However, manual review is required to confirm exact technical specifications "
            f"(e.g., breaker numbers, wire gauges)."
        ),
        "sources": [],
        "circuito_trazado": None,
        "pasos_recomendados": ["Verify information on the physical panel", "Consult with a technical supervisor"],
        "materiales_mencionados": [],
        "normativa_citada": None,
        "confianza_inicial": 0.3,
        "advertencias_tecnicas": ["Response generated in fallback mode — requires manual verification"]
    }
