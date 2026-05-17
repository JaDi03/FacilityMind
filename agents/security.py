"""
FacilityMind — Security Layer (Lobster Trap Integration)
Inspects every user query through Veea's Lobster Trap DPI engine
before allowing it into the agent pipeline.
"""

import subprocess
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOBSTERTRAP_BIN = Path(__file__).parent.parent / "lobstertrap.exe"
POLICY_PATH = Path(__file__).parent.parent / "configs" / "facilitymind_policy.yaml"


@dataclass
class SecurityVerdict:
    """Result of a Lobster Trap DPI inspection."""
    allowed: bool
    action: str  # ALLOW, DENY, LOG, etc.
    risk_score: float
    matched_rule: Optional[str]
    deny_message: Optional[str]
    intent_category: Optional[str]
    raw_metadata: dict


def inspect_prompt(prompt: str) -> SecurityVerdict:
    """
    Run Lobster Trap DPI on a user prompt.
    Uses the CLI `inspect` command for sub-millisecond analysis.
    Returns a SecurityVerdict with the decision.
    """
    if not LOBSTERTRAP_BIN.exists():
        logger.warning("[Security] Lobster Trap binary not found. Allowing by default.")
        return SecurityVerdict(
            allowed=True, action="ALLOW", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={}
        )

    # --- Local Custom Intercept Rules (Perfect for Demo Precision) ---
    prompt_lower = prompt.lower()
    custom_blocked_keywords = [
        "gemini api", "api key", "api_key", "access token", 
        "system credential", "system password", "give me access"
    ]
    if any(kw in prompt_lower for kw in custom_blocked_keywords):
        logger.warning(f"[Security] Local rule triggered. Blocked query: {prompt}")
        return SecurityVerdict(
            allowed=False,
            action="DENY",
            risk_score=1.0,
            matched_rule="local_api_key_harvesting",
            deny_message="[LOBSTER TRAP] Blocked: Unauthorized request for system API credentials/access.",
            intent_category="credential_access",
            raw_metadata={"local_trigger": True}
        )

    try:
        result = subprocess.run(
            [str(LOBSTERTRAP_BIN), "inspect", "--policy", str(POLICY_PATH), "--json", prompt],
            capture_output=True, text=True, timeout=5, encoding="utf-8", errors="replace"
        )

        output = result.stdout.strip()
        if not output:
            # Try stderr as some CLIs output there
            output = result.stderr.strip()

        # Parse JSON output
        if output:
            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                # Try to find JSON in the output
                for line in output.split("\n"):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    data = {}
        else:
            data = {}

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
            f"[Security] DPI verdict: {verdict.action} | "
            f"risk={verdict.risk_score:.2f} | "
            f"rule={verdict.matched_rule} | "
            f"intent={verdict.intent_category}"
        )

        return verdict

    except subprocess.TimeoutExpired:
        logger.error("[Security] Lobster Trap inspection timed out. Allowing by default.")
        return SecurityVerdict(
            allowed=True, action="ALLOW_TIMEOUT", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={"error": "timeout"}
        )
    except Exception as e:
        logger.error(f"[Security] Lobster Trap error: {e}. Allowing by default.")
        return SecurityVerdict(
            allowed=True, action="ALLOW_ERROR", risk_score=0.0,
            matched_rule=None, deny_message=None,
            intent_category="general", raw_metadata={"error": str(e)}
        )
