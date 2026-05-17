"""
FacilityMind — Agent 3: Validator
Verifies the candidate response against original blueprints using Gemini 2.5 Pro.
Detects hallucinations, evaluates safety risks, and makes final approval decisions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GeminiModels, Thresholds
from models.schemas import ReasonerOutput, ValidatorOutput, SourceCitation, SafetyAssessment
from ingestion.pdf_loader import cargar_plano_como_texto

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "validator.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

MODEL = GeminiModels.VALIDATOR


async def agente_validador(
    reasoner: ReasonerOutput,
    plano_completo_path: Optional[str] = None,
    contexto_rag: str = "",
    active_cache: Optional[str] = None
) -> ValidatorOutput:
    """
    Validator Agent: Verifies the candidate response for accuracy and safety.

    Args:
        reasoner: Output from the Reasoner Agent.
        plano_completo_path: Path to the original PDF for verification.
        contexto_rag: RAG context used by the reasoner.

    Returns:
        ValidatorOutput with the final approval or rejection decision.
    """
    logger.info(f"[Validator] Starting verification | initial_confidence={reasoner.confianza_inicial:.2f}")

    # --- Load full blueprint for verification ---
    plano_texto = ""
    if plano_completo_path and os.path.exists(plano_completo_path):
        plano_texto = cargar_plano_como_texto(plano_completo_path)
        logger.info(f"[Validator] Blueprint loaded for verification: {len(plano_texto)} characters")
    else:
        # Fallback to RAG context if full blueprint is unavailable
        plano_texto = contexto_rag if contexto_rag else "(Original blueprint text not available for verification)"
        logger.info("[Validator] Using RAG context as fallback for verification")

    # --- Serialize Reasoner Data ---
    sources_json = "[]"
    try:
        sources_list = [s.model_dump() for s in reasoner.sources]
        sources_json = json.dumps(sources_list, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[Validator] Error serializing citations: {e}")

    circuito_json = "null"
    if reasoner.circuito_trazado:
        try:
            circuito_json = json.dumps(reasoner.circuito_trazado.model_dump(), ensure_ascii=False)
        except Exception:
            pass

    # --- Build Prompt & Execute Gemini Pro ---
    
    # --- SAFETY CHECK: Auto-reject if no blueprint data available ---
    # If there's no cache AND no real blueprint text, the Validator cannot verify anything.
    # Calling Gemini would just rubber-stamp the Reasoner's (potentially hallucinated) answer.
    if not active_cache and plano_texto == "(Original blueprint text not available for verification)":
        logger.warning("[Validator] ABORT: No cache and no blueprint text. Auto-rejecting to prevent approval of unverified data.")
        return ValidatorOutput(
            aprobado=False,
            respuesta_final=reasoner.respuesta_candidata,
            confianza_final=0.0,
            sources_verificadas=[],
            advertencias=["No blueprint data available for verification — auto-rejected"],
            requiere_supervisor=True,
            motivo_rechazo="Cannot verify: no blueprint data loaded. Upload the PDF first.",
            alucinaciones_detectadas=["Unable to verify any claims — no source data"],
            safety=SafetyAssessment(
                nivel_riesgo="high",
                descripcion_riesgo="Response cannot be verified without blueprint access"
            )
        )
    
    prompt = f"""CANDIDATE RESPONSE TO VALIDATE:
{reasoner.respuesta_candidata}

SOURCES CITED BY REASONER:
{sources_json}

TRACED CIRCUIT PATH:
{circuito_json}

ORIGINAL BLUEPRINT TEXT:
{plano_texto[:150000] if not active_cache else "⚠️ IMPORTANT INSTRUCTION: The full, original blueprint document is already natively loaded into your memory via the Gemini Context Cache. You MUST use your cached memory to verify every citation, dimension, and note. Do not complain about missing text in this prompt."}

Generate your validation in the specified strict JSON format.
"""

    try:
        import asyncio
        from google import genai as genai_v2
        from google.genai import types
        
        v2_client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
        
        response = None
        if active_cache:
            # --- Native Long-Context via Gemini Cache ---
            logger.info(f"[Validator] Active Context Cache found ({active_cache}). Using native full-blueprint verification.")
            
            try:
                response = await asyncio.to_thread(
                    v2_client.models.generate_content,
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=active_cache,
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=ValidatorOutput
                    )
                )
            except Exception as e:
                logger.warning(f"[Validator] Cache request failed ({e}). Falling back to traditional verification.")
                active_cache = None
                
        if not active_cache:
            # --- Traditional Execution ---
            response = await asyncio.to_thread(
                v2_client.models.generate_content,
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ValidatorOutput,
                    system_instruction=SYSTEM_PROMPT
                )
            )
            
        raw = response.text.replace("```json", "").replace("```", "").strip()

        logger.info(f"[Validator] Raw response snippet: {raw[:400]}...")

        # Parse JSON output
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[Validator] Malformed JSON received, applying fallback logic")
            return _fallback_validation(reasoner)

        # Normalize and validate data structures
        if "sources_verificadas" in data and isinstance(data["sources_verificadas"], list):
            data["sources_verificadas"] = [SourceCitation(**s) for s in data["sources_verificadas"]]
        else:
            data["sources_verificadas"] = []

        if "safety" in data and isinstance(data["safety"], dict):
            data["safety"] = SafetyAssessment(**data["safety"])
        else:
            data["safety"] = SafetyAssessment()

        # Ensure required fields are populated
        data.setdefault("aprobado", False)
        data.setdefault("respuesta_final", reasoner.respuesta_candidata)
        data.setdefault("confianza_final", reasoner.confianza_inicial * 0.8)
        data.setdefault("advertencias", [])
        data.setdefault("requiere_supervisor", True)
        data.setdefault("motivo_rechazo", None)
        data.setdefault("alucinaciones_detectadas", [])

        output = ValidatorOutput(**data)

        # Apply additional threshold and safety rules
        output = _aplicar_thresholds(output, reasoner)

        logger.info(
            f"[Validator] Verification complete | approved={output.aprobado}, "
            f"final_conf={output.confianza_final:.2f}, supervisor={output.requiere_supervisor}, "
            f"hallucinations={len(output.alucinaciones_detectadas)}"
        )
        return output

    except Exception as e:
        logger.error(f"[Validator] Processing error: {e}")
        return _fallback_validation(reasoner, error=str(e))


def _aplicar_thresholds(output: ValidatorOutput, reasoner: ReasonerOutput) -> ValidatorOutput:
    """Applies additional threshold rules and safety logic to the validation decision."""

    # Rule: Reject automatically if hallucinations are detected
    if output.alucinaciones_detectadas:
        output.aprobado = False
        output.confianza_final = min(output.confianza_final, 0.4)
        output.requiere_supervisor = True
        if not output.motivo_rechazo:
            output.motivo_rechazo = f"Hallucinations detected: {len(output.alucinaciones_detectadas)}"

    # Rule: Critical risk always requires supervisor approval and rejection
    if output.safety and output.safety.nivel_riesgo == "critical":
        output.requiere_supervisor = True
        output.aprobado = False
        output.confianza_final = min(output.confianza_final, 0.3)
        if not output.motivo_rechazo:
            output.motivo_rechazo = "Critical safety risk detected"

    # Rule: Confidence Thresholds
    if output.confianza_final >= Thresholds.VALIDATOR_APPROVE:
        # High confidence: approve (unless high/critical risk is present)
        if not output.alucinaciones_detectadas and output.safety.nivel_riesgo not in ["high", "critical"]:
            output.aprobado = True
    elif output.confianza_final >= Thresholds.VALIDATOR_REJECT:
        # Medium confidence: approve but require supervisor review
        output.requiere_supervisor = True
    else:
        # Low confidence: reject
        output.aprobado = False
        output.requiere_supervisor = True
        if not output.motivo_rechazo:
            output.motivo_rechazo = f"Final confidence too low: {output.confianza_final:.2f}"

    # Rule: Safety detection for dangerous operations (e.g., bypassing breakers)
    texto_respuesta = (output.respuesta_final or "").lower()
    dangerous_keywords = ["bridge", "jumper", "bypass", "skip", "puentear", "puente"]
    if any(keyword in texto_respuesta for keyword in dangerous_keywords):
        output.safety.riesgo_electrico = True
        output.safety.nivel_riesgo = "critical"
        output.aprobado = False
        output.requiere_supervisor = True
        output.advertencias.append("DANGEROUS OPERATION DETECTED: Unauthorized electrical system modification requested")
        if not output.motivo_rechazo:
            output.motivo_rechazo = "Query involves dangerous and potentially illegal technical operations"

    return output


def _fallback_validation(reasoner: ReasonerOutput, error: str = "") -> ValidatorOutput:
    """Fallback mechanism for validation failures."""
    advertencias = ["Validation system error — manual technical review required"]
    if error:
        advertencias.append(f"Technical error: {error[:100]}")

    return ValidatorOutput(
        aprobado=False,
        respuesta_final=reasoner.respuesta_candidata,
        confianza_final=reasoner.confianza_inicial * 0.7,
        sources_verificadas=[],
        advertencias=advertencias,
        requiere_supervisor=True,
        motivo_rechazo="Automated validation relies on RAG context (full text not loaded)" if not error else f"Error: {error[:150]}",
        alucinaciones_detectadas=[],
        safety=SafetyAssessment(
            nivel_riesgo="medium",
            descripcion_riesgo="Unable to complete automated safety verification"
        )
    )
