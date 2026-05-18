"""
FacilityMind — Agent 1: Perception
Analyzes audio + image + text from the field technician using Gemini 2.5 Flash.
Extracts structured data: location, identified object, and query objective.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GeminiModels
from models.schemas import PerceptionOutput

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load system prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "perception.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

# Model: Gemini 2.5 Flash — prioritized for low-latency multimodal analysis
MODEL = GeminiModels.PERCEPTION


async def agente_percepcion(
    audio_path: Optional[str] = None,
    photo_path: Optional[str] = None,
    text: str = ""
) -> PerceptionOutput:
    """
    Perception Agent: Processes multimodal inputs from the technician.

    Args:
        audio_path: Path to the audio file (ogg, mp3, wav).
        photo_path: Path to the image file (jpg, png).
        text: Additional text or manual transcription.

    Returns:
        PerceptionOutput containing structured query data.
    """
    logger.info(f"[Perception] Starting analysis | audio={audio_path is not None}, photo={photo_path is not None}, text={bool(text)}")

    model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)
    parts = []

    # --- Audio Input ---
    if audio_path and os.path.exists(audio_path):
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            # Detect MIME type
            mime = "audio/ogg"
            if audio_path.endswith(".mp3"):
                mime = "audio/mpeg"
            elif audio_path.endswith(".wav"):
                mime = "audio/wav"
            parts.append({"mime_type": mime, "data": audio_bytes})
            logger.info(f"[Perception] Audio loaded: {len(audio_bytes)} bytes ({mime})")
        except Exception as e:
            logger.error(f"[Perception] Error loading audio: {e}")

    # --- Image Input ---
    if photo_path and os.path.exists(photo_path):
        try:
            with open(photo_path, "rb") as f:
                img_bytes = f.read()
            mime = "image/jpeg"
            if photo_path.endswith(".png"):
                mime = "image/png"
            parts.append({"mime_type": mime, "data": img_bytes})
            logger.info(f"[Perception] Image loaded: {len(img_bytes)} bytes ({mime})")
        except Exception as e:
            logger.error(f"[Perception] Error loading image: {e}")

    # --- Query Text ---
    prompt_text = f"""
Technician Text: {text if text else "(no additional text provided)"}

Analyze all provided inputs and return ONLY the JSON object requested in the system prompt.
If audio is provided, use the transcription for data extraction.
If an image is provided, describe the visible object with technical precision.
"""
    parts.append(prompt_text)

    # --- Gemini Model Execution ---
    try:
        response = await model.generate_content_async(parts)
        raw = response.text

        # Clean potential markdown formatting
        raw = raw.replace("```json", "").replace("```", "").strip()

        logger.info(f"[Perception] Raw response snippet: {raw[:300]}...")

        # Parse JSON output
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[Perception] Malformed JSON, attempting fallback parsing: {e}")
            # Fallback: extract data using simple heuristic parsing
            data = _fallback_parsing(text, raw)

        # Ensure required fields are populated
        if "query_objective" not in data or not data["query_objective"]:
            data["query_objective"] = text if text else "Query objective not specified"
        if "perception_confidence" not in data:
            data["perception_confidence"] = 0.5

        output = PerceptionOutput(**data)
        logger.info(f"[Perception] Analysis complete | floor={output.floor}, room={output.room}, confidence={output.perception_confidence:.2f}")
        return output

    except Exception as e:
        logger.error(f"[Perception] Error during Gemini API call: {e}")
        # Minimum fallback to maintain pipeline integrity
        return PerceptionOutput(
            query_objective=text if text else "Processing error",
            perception_confidence=0.1,
            raw_transcription=text if text else None,
            urgency_notes=f"System error: {str(e)[:100]}"
        )


def _fallback_parsing(text: str, raw_response: str) -> dict:
    """
    Fallback mechanism when Gemini fails to return valid JSON.
    Attempts to extract data using regex patterns.
    """
    import re

    data = {
        "floor": None,
        "tower": None,
        "room": None,
        "detected_object": None,
        "query_objective": text,
        "discipline": None,
        "urgency_notes": None,
        "detected_language": "en",
        "perception_confidence": 0.3,
        "raw_transcription": text,
    }

    # Extract floor numbers
    piso_match = re.search(r'floor\s+(\d+|GF|B\d+|S\d+)', text, re.IGNORECASE)
    if piso_match:
        data["floor"] = piso_match.group(1).upper()

    # Extract tower identifiers
    torre_match = re.search(r'tower\s+(\d+|[A-Z])', text, re.IGNORECASE)
    if torre_match:
        data["tower"] = torre_match.group(1).upper()

    # Extract room/unit identifiers
    hab_match = re.search(r'(?:room|unit|local|office|chamber)\s+([\w-]+)', text, re.IGNORECASE)
    if hab_match:
        data["room"] = hab_match.group(1)

    # Discipline detection
    text_lower = text.lower()
    if any(w in text_lower for w in ["breaker", "outlet", "panel", "electrical", "cable", "power", "voltage", "amperage"]):
        data["discipline"] = "electrical"
    elif any(w in text_lower for w in ["piping", "drainage", "valve", "plumbing", "water", "leak", "cleanout"]):
        data["discipline"] = "plumbing"

    # Urgency detection
    if any(w in text_lower for w in ["urgent", "emergency", "danger", "sparks", "flooding", "burnt"]):
        data["urgency_notes"] = "Potential emergency situation detected"

    return data
