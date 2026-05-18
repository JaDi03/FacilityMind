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
from ingestion.pdf_loader import load_blueprint_as_text

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
    logger.info(f"[Validator] Starting verification | initial_confidence={reasoner.initial_confidence:.2f}")

    # --- Load full blueprint for verification ---
    plano_texto = ""
    if plano_completo_path and os.path.exists(plano_completo_path):
        plano_texto = load_blueprint_as_text(plano_completo_path)
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
    if reasoner.traced_circuit:
        try:
            circuito_json = json.dumps(reasoner.traced_circuit.model_dump(), ensure_ascii=False)
        except Exception:
            pass

    # --- Build Prompt & Execute Gemini Pro ---
    
    # --- SAFETY CHECK: Auto-reject if no real blueprint data available ---
    # If there's no cache AND no RAG context with actual content, the Validator
    # cannot verify anything. Auto-reject regardless of what the Reasoner claimed.
    has_real_data = (
        active_cache or
        (contexto_rag and len(contexto_rag.strip()) > 50 and
         "no relevant document" not in contexto_rag.lower())
    )

    if not has_real_data:
        logger.warning("[Validator] ABORT: No verifiable data. Auto-rejecting.")
        return ValidatorOutput(
            approved=False,
            final_response=reasoner.candidate_response,
            final_confidence=0.0,
            verified_sources=[],
            warnings=["No blueprint data available for verification — auto-rejected"],
            requires_supervisor=True,
            rejection_reason="Cannot verify: no blueprint data loaded. Upload the PDF first.",
            detected_hallucinations=["Unable to verify any claims — no source data"],
            safety=SafetyAssessment(
                risk_level="high",
                risk_description="Response cannot be verified without blueprint access"
            )
        )
    
    prompt = f"""CANDIDATE RESPONSE TO VALIDATE:
{reasoner.candidate_response}

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
                
                # CRITICAL FIX: Reload the text for fallback if we cleared it!
                if plano_texto == "(Original blueprint text not available for verification)" or not plano_texto.strip():
                    if plano_completo_path and os.path.exists(plano_completo_path):
                        from ingestion.pdf_loader import load_blueprint_as_text
                        plano_texto = load_blueprint_as_text(plano_completo_path)
                    elif contexto_rag:
                        plano_texto = contexto_rag
                
        if not active_cache:
            # SAFETY CHECK FOR FALLBACK: If cache failed and we don't have text, do NOT call Gemini!
            if plano_texto == "(Original blueprint text not available for verification)" or not plano_texto.strip():
                logger.warning("[Validator] Cache failed and no fallback text available. Auto-rejecting.")
                return _fallback_validation(reasoner, error="Cache failed and no blueprint text available for manual verification.")
                
            # Rebuild prompt because the original one instructed Gemini to use the cache!
            fallback_prompt = f"""CANDIDATE RESPONSE TO VALIDATE:
{reasoner.candidate_response}

SOURCES CITED BY REASONER:
{sources_json}

TRACED CIRCUIT PATH:
{circuito_json}

ORIGINAL BLUEPRINT TEXT:
{plano_texto[:150000]}

Generate your validation in the specified strict JSON format.
"""
            # --- Traditional Execution ---
            response = await asyncio.to_thread(
                v2_client.models.generate_content,
                model=MODEL,
                contents=fallback_prompt,
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
        if "verified_sources" in data and isinstance(data["verified_sources"], list):
            data["verified_sources"] = [SourceCitation(**s) for s in data["verified_sources"]]
        else:
            data["verified_sources"] = []

        if "safety" in data and isinstance(data["safety"], dict):
            data["safety"] = SafetyAssessment(**data["safety"])
        else:
            data["safety"] = SafetyAssessment()

        # Ensure required fields are populated
        data.setdefault("approved", False)
        data.setdefault("final_response", reasoner.candidate_response)
        data.setdefault("final_confidence", reasoner.initial_confidence * 0.8)
        data.setdefault("warnings", [])
        data.setdefault("requires_supervisor", True)
        data.setdefault("rejection_reason", None)
        data.setdefault("detected_hallucinations", [])

        output = ValidatorOutput(**data)

        # Apply additional threshold and safety rules
        output = _aplicar_thresholds(output, reasoner)

        logger.info(
            f"[Validator] Verification complete | approved={output.approved}, "
            f"final_conf={output.final_confidence:.2f}, supervisor={output.requires_supervisor}, "
            f"hallucinations={len(output.detected_hallucinations)}"
        )
        return output

    except Exception as e:
        logger.error(f"[Validator] Processing error: {e}")
        return _fallback_validation(reasoner, error=str(e))


def _aplicar_thresholds(output: ValidatorOutput, reasoner: ReasonerOutput) -> ValidatorOutput:
    """Applies additional threshold rules and safety logic to the validation decision."""

    # Rule: Reject automatically if hallucinations are detected
    if output.detected_hallucinations:
        output.approved = False
        output.final_confidence = min(output.final_confidence, 0.4)
        output.requires_supervisor = True
        if not output.rejection_reason:
            output.rejection_reason = f"Hallucinations detected: {len(output.detected_hallucinations)}"

    # Rule: Critical risk always requires supervisor approval and rejection
    if output.safety and output.safety.risk_level == "critical":
        output.requires_supervisor = True
        output.approved = False
        output.final_confidence = min(output.final_confidence, 0.3)
        if not output.rejection_reason:
            output.rejection_reason = "Critical safety risk detected"

    # Rule: Confidence Thresholds
    if output.final_confidence >= Thresholds.VALIDATOR_APPROVE:
        # High confidence: approve (unless high/critical risk is present)
        if not output.detected_hallucinations and output.safety.risk_level not in ["high", "critical"]:
            output.approved = True
    elif output.final_confidence >= Thresholds.VALIDATOR_REJECT:
        # Medium confidence: approve but require supervisor review
        output.requires_supervisor = True
    else:
        # Low confidence: reject
        output.approved = False
        output.requires_supervisor = True
        if not output.rejection_reason:
            output.rejection_reason = f"Final confidence too low: {output.final_confidence:.2f}"

    # Rule: Safety detection for dangerous operations (e.g., bypassing breakers)
    texto_respuesta = (output.final_response or "").lower()
    dangerous_keywords = ["bridge", "jumper", "bypass", "skip", "puentear", "puente"]
    if any(keyword in texto_respuesta for keyword in dangerous_keywords):
        output.safety.electrical_risk = True
        output.safety.risk_level = "critical"
        output.approved = False
        output.requires_supervisor = True
        output.warnings.append("DANGEROUS OPERATION DETECTED: Unauthorized electrical system modification requested")
        if not output.rejection_reason:
            output.rejection_reason = "Query involves dangerous and potentially illegal technical operations"

    return output


def _fallback_validation(reasoner: ReasonerOutput, error: str = "") -> ValidatorOutput:
    """Fallback mechanism for validation failures."""
    warnings = ["Validation system error — manual technical review required"]
    if error:
        warnings.append(f"Technical error: {error[:100]}")

    return ValidatorOutput(
        approved=False,
        final_response=reasoner.candidate_response,
        final_confidence=reasoner.initial_confidence * 0.7,
        verified_sources=[],
        warnings=warnings,
        requires_supervisor=True,
        rejection_reason="Automated validation relies on RAG context (full text not loaded)" if not error else f"Error: {error[:150]}",
        detected_hallucinations=[],
        safety=SafetyAssessment(
            risk_level="medium",
            risk_description="Unable to complete automated safety verification"
        )
    )
