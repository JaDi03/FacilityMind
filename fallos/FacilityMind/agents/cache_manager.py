"""
FacilityMind — Gemini Context Cache Manager
Handles uploading blueprints directly to the Gemini File API and creating Context Caches.
This replaces traditional RAG with native Long-Context memory for maximum accuracy.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GeminiModels

logger = logging.getLogger(__name__)

# Initialize the v2 GenAI client
client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})

def cache_blueprint(pdf_path: str, display_name: str) -> Dict:
    """
    Uploads a PDF to Gemini and creates a Context Cache.
    """
    try:
        logger.info(f"[CacheManager] Uploading {display_name} to Gemini File API...")
        
        # 1. Upload File
        uploaded_file = client.files.upload(
            file=pdf_path, 
            config={'display_name': display_name}
        )
        
        logger.info(f"[CacheManager] File uploaded: {uploaded_file.name}. Waiting for processing...")
        
        # 2. Wait for processing (PDFs require brief processing on Google's side)
        file_info = client.files.get(name=uploaded_file.name)
        while file_info.state.name == "PROCESSING":
            time.sleep(2)
            file_info = client.files.get(name=uploaded_file.name)
            logger.info(f"[CacheManager] Processing state: {file_info.state.name}")
            
        if file_info.state.name == "FAILED":
            raise Exception("Google Gemini failed to process the PDF document.")

        # Paths to prompt files
        PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
        
        reasoner_prompt = ""
        reasoner_prompt_path = PROMPTS_DIR / "reasoner.txt"
        if reasoner_prompt_path.exists():
            reasoner_prompt = reasoner_prompt_path.read_text(encoding="utf-8")
            
        validator_prompt = ""
        validator_prompt_path = PROMPTS_DIR / "validator.txt"
        if validator_prompt_path.exists():
            validator_prompt = validator_prompt_path.read_text(encoding="utf-8")

        # 3. Create Reasoner Cache (with reasoner prompt baked in)
        logger.info(f"[CacheManager] Creating Reasoner Context Cache for {display_name}...")
        reasoner_cache = client.caches.create(
            model=GeminiModels.REASONER,
            config=types.CreateCachedContentConfig(
                contents=[uploaded_file],
                system_instruction=reasoner_prompt,
                ttl="3600s" # 1 hour
            )
        )
        logger.info(f"[CacheManager] Reasoner Cache created: {reasoner_cache.name}")

        # 4. Create Validator Cache (with validator prompt baked in)
        logger.info(f"[CacheManager] Creating Validator Context Cache for {display_name}...")
        validator_cache = client.caches.create(
            model=GeminiModels.VALIDATOR,
            config=types.CreateCachedContentConfig(
                contents=[uploaded_file],
                system_instruction=validator_prompt,
                ttl="3600s" # 1 hour
            )
        )
        logger.info(f"[CacheManager] Validator Cache created: {validator_cache.name}")

        return {
            "success": True,
            "file_name": uploaded_file.name,
            "cache_name_reasoner": reasoner_cache.name,
            "cache_name_validator": validator_cache.name
        }
        
    except Exception as e:
        logger.error(f"[CacheManager] Error caching blueprint: {e}")
        return {"success": False, "error": str(e)}

def delete_cache(cache_name: str):
    """Cleans up the cache when a blueprint is removed."""
    try:
        client.caches.delete(name=cache_name)
        logger.info(f"[CacheManager] Deleted cache: {cache_name}")
    except Exception as e:
        logger.warning(f"[CacheManager] Could not delete cache {cache_name}: {e}")
