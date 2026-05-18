"""
FacilityMind - Security Layer (Lobster Trap + Spanish Pre-Filter)
Inspects every user query through a TWO-STAGE security system:
Stage 1: Python semantic filter (detects Spanish/English threats)
Stage 2: Lobster Trap DPI engine (existing Go binary)
"""

import subprocess
import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOBSTERTRAP_BIN = Path(__file__).parent.parent / "lobstertrap.exe"
POLICY_PATH = Path(__file__).parent.parent / "configs" / "facilitymind_policy.yaml"


@dataclass
class SecurityVerdict:
    """Result of security inspection."""
    allowed: bool
    action: str
    risk_score: float
    matched_rule: Optional[str]
    deny_message: Optional[str]
    intent_category: Optional[str]
    raw_metadata: dict


# ============================================================
# STAGE 1: Python Semantic Filter (Spanish + English)
# ============================================================

# Spanish and English patterns for credential extraction
CREDENTIAL_PATTERNS = [
    # Spanish - API keys and credentials
    r"\b(dame|proporciona|pasame|envia|mu[eé]strame|dime)\b.*\b(api\s*key|clave\s*api|token|contrase[ñn]a|password|credencial)\b",
    r"\b(api\s*key|clave\s*api|token|contrase[ñn]a)\b.*\b(de\s*gemini|del\s*sistema|de\s*acceso|de\s*google)\b",
    r"\b(cual\s*es|dime)\b.*\b(la\s*clave|el\s*token|la\s*contrase[ñn]a|la\s*api)\b",
    r"\b(quiero|necesito)\b.*\b(acceder\s*a|entrar\s*a)\b.*\b(el\s*sistema|la\s*configuraci[oó]n|el\s*panel)\b",
    r"\b(olvide|no\s*recuerdo)\b.*\b(mi\s*contrase[ñn]a|mi\s*clave|mi\s*token)\b",
    # Spanish - System access
    r"\b(como\s*puedo|dime\s*como)\b.*\b(hackear|bypassear|saltar|evadir)\b.*\b(la\s*seguridad|el\s*sistema)\b",
    r"\b(ignora|olvida)\b.*\b(tus\s*instrucciones|las\n*reglas\s*anteriores|tu\s*sistema\s*de\s*prompt)\b",
    r"\b(eres|actua\s*como)\b.*\b(un\s*admin|root|superusuario|desarrollador)\b",
    # English - API keys and credentials
    r"\b(give\s*me|show\s*me|send\s*me|tell\s*me)\b.*\b(api\s*key|apikey|password|credential|secret\s*key)\b",
    r"\b(what\s*is|what's)\b.*\b(the\s*api\s*key|the\s*password|the\s*token)\b",
    r"\b(i\s*need|i\s*want)\b.*\b(access\s*to|the\s*system\s*config)\b",
    # English - Prompt injection
    r"\b(ignore\s*previous|ignore\s*all|forget\s*your)\b.*\b(instructions|rules|prompt)\b",
    r"\b(you\s*are\s*now|act\s*as|pretend\s*to\s*be)\b.*\b(admin|root|developer|owner)\b",
    r"\b(system\s*prompt|debug\s*mode|developer\s*mode)\b",
    # Universal - Data exfiltration
    r"\b(send\s*to|forward\s*to|email\s*to)\b.*\b(gmail|whatsapp|telegram|discord|pastebin)\b",
]

# Spanish and English patterns for prompt injection
INJECTION_PATTERNS = [
    # Spanish
    r"\b(ignora|olvida|descarta)\b.*\b(todo\s*lo\s*anterior|las\s*instrucciones|tus\s*reglas)\b",
    r"\b(ahora\s*eres|desde\s*ahora)\b.*\b(un\s*assistente\s*diferente|modo\s*libre|sin\s*restricciones)\b",
    # English
    r"\b(ignore\s*all|disregard|override)\b.*\b(previous|above|prior)\b.*\b(instructions|commands)\b",
    r"\b(dan\s*mode|jailbreak|dify)\b",
]

# Patterns for role impersonation
IMPERSONATION_PATTERNS = [
    r"\b(soy\s*el\s*admin|soy\s*el\s*desarrollador|soy\s*el\s*due[ñn]o)\b",
    r"\b(i\s*am\s*the\s*admin|i\s*am\s*the\s*developer|i\s*am\s*the\s*owner)\b",
    r"\b(trabajo\s*para\s*veea|trabajo\s*en\s*facilitymind)\b",
]

# Compile all patterns
COMPILED_CREDENTIAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CREDENTIAL_PATTERNS]
COMPILED_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
COMPILED_IMPERSONATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in IMPERSONATION_PATTERNS]


def _spanish_pre_filter(prompt: str) -> Optional[SecurityVerdict]:
    """
    Stage 1: Python-based semantic filter for Spanish and English threats.
    Returns a DENY verdict if threat detected, None if clean.
    """
    prompt_lower = prompt.lower()

    # Check credential extraction
    for i, pattern in enumerate(COMPILED_CREDENTIAL_PATTERNS):
        if pattern.search(prompt):
            logger.warning(f"[Security] Stage 1: Credential extraction detected (pattern {i})")
            return SecurityVerdict(
                allowed=False,
                action="DENY",
                risk_score=0.95,
                matched_rule=f"spanish_pre_filter:credential_extraction:{i}",
                deny_message=(
                    "[SECURITY] Bloqueado: Intento de extraccion de credenciales "
                    "detectado. Este incidente ha sido registrado."
                ),
                intent_category="credential_extraction",
                raw_metadata={
                    "stage": "spanish_pre_filter",
                    "pattern_type": "credential_extraction",
                    "pattern_index": i,
                    "matched_text": pattern.search(prompt).group(0)
                }
            )

    # Check prompt injection
    for i, pattern in enumerate(COMPILED_INJECTION_PATTERNS):
        if pattern.search(prompt):
            logger.warning(f"[Security] Stage 1: Prompt injection detected (pattern {i})")
            return SecurityVerdict(
                allowed=False,
                action="DENY",
                risk_score=0.90,
                matched_rule=f"spanish_pre_filter:prompt_injection:{i}",
                deny_message=(
                    "[SECURITY] Bloqueado: Intento de inyeccion de prompt "
                    "detectado. Este incidente ha sido registrado."
                ),
                intent_category="prompt_injection",
                raw_metadata={
                    "stage": "spanish_pre_filter",
                    "pattern_type": "prompt_injection",
                    "pattern_index": i,
                    "matched_text": pattern.search(prompt).group(0)
                }
            )

    # Check role impersonation
    for i, pattern in enumerate(COMPILED_IMPERSONATION_PATTERNS):
        if pattern.search(prompt):
            logger.warning(f"[Security] Stage 1: Role impersonation detected (pattern {i})")
            return SecurityVerdict(
                allowed=False,
                action="DENY",
                risk_score=0.85,
                matched_rule=f"spanish_pre_filter:role_impersonation:{i}",
                deny_message=(
                    "[SECURITY] Bloqueado: Intento de suplantacion de rol "
                    "detectado. Este incidente ha sido registrado."
                ),
                intent_category="role_impersonation",
                raw_metadata={
                    "stage": "spanish_pre_filter",
                    "pattern_type": "role_impersonation",
                    "pattern_index": i,
                    "matched_text": pattern.search(prompt).group(0)
                }
            )

    # Stage 1 passed - no Spanish threats detected
    return None


# ============================================================
# STAGE 2: Lobster Trap DPI (existing Go binary)
# ============================================================

def _lobster_trap_inspect(prompt: str) -> SecurityVerdict:
    """Stage 2: Run Lobster Trap DPI. Returns verdict."""
    if not LOBSTERTRAP_BIN.exists():
        logger.warning("[Security] Lobster Trap binary not found. Allowing.")
        return SecurityVerdict(
            allowed=True, action="ALLOW", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={}
        )

    try:
        result = subprocess.run(
            [str(LOBSTERTRAP_BIN), "inspect", "--policy", str(POLICY_PATH),
             "--json", prompt],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace"
        )

        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()

        # Parse JSON output
        data = {}
        if output:
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                for line in output.split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue

        action = data.get("action", "ALLOW")
        allowed = action in ("ALLOW", "LOG")

        verdict = SecurityVerdict(
            allowed=allowed,
            action=action,
            risk_score=data.get("metadata", {}).get("risk_score", 0.0),
            matched_rule=data.get("matched_rule", None),
            deny_message=data.get("deny_message", None),
            intent_category=data.get("metadata", {}).get("intent_category", "general"),
            raw_metadata=data
        )

        level = logging.WARNING if not allowed else logging.INFO
        logger.log(level,
            f"[Security] Lobster Trap: {verdict.action} | "
            f"risk={verdict.risk_score:.2f} | rule={verdict.matched_rule}"
        )

        return verdict

    except subprocess.TimeoutExpired:
        logger.error("[Security] Lobster Trap timed out. Allowing.")
        return SecurityVerdict(
            allowed=True, action="ALLOW_TIMEOUT", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={"error": "timeout"}
        )
    except Exception as e:
        logger.error(f"[Security] Lobster Trap error: {e}. Allowing.")
        return SecurityVerdict(
            allowed=True, action="ALLOW_ERROR", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={"error": str(e)}
        )


# ============================================================
# PUBLIC API: Two-Stage Inspection
# ============================================================

def inspect_prompt(prompt: str) -> SecurityVerdict:
    """
    Two-stage security inspection:
    1. Spanish/English semantic pre-filter (Python regex)
    2. Lobster Trap DPI (Go binary)

    Stage 1 catches Spanish threats that Lobster Trap misses.
    Stage 2 catches threats that the regex filter might miss.
    """
    if not prompt or not prompt.strip():
        return SecurityVerdict(
            allowed=True, action="ALLOW_EMPTY", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={}
        )

    logger.info(f"[Security] Starting 2-stage inspection, prompt length: {len(prompt)}")

    # Stage 1: Spanish pre-filter
    stage1_result = _spanish_pre_filter(prompt)
    if stage1_result:
        logger.warning(f"[Security] Stage 1 DENIED: {stage1_result.intent_category}")
        return stage1_result

    logger.info("[Security] Stage 1 passed (Spanish pre-filter clean)")

    # Stage 2: Lobster Trap
    stage2_result = _lobster_trap_inspect(prompt)
    if not stage2_result.allowed:
        logger.warning(f"[Security] Stage 2 DENIED: {stage2_result.intent_category}")
    else:
        logger.info("[Security] Stage 2 passed (Lobster Trap clean)")

    return stage2_result
