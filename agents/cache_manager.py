"""
FacilityMind — Gemini Context Cache Manager
Handles uploading blueprints directly to the Gemini File API and creating Context Caches.
This replaces traditional RAG with native Long-Context memory for maximum accuracy.
"""

import os
import time
import logging
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

        # 3. Create Context Cache (Valid for 1 hour by default)
        logger.info(f"[CacheManager] Creating Context Cache for {display_name}...")
        
        system_instruction = (
            "You are FacilityMind's Technical Reasoner. You have been provided with a complete building blueprint. "
            "Analyze the entire document carefully. When answering user queries, you MUST cite the exact page number "
            "and quote the relevant text or diagram title from this document. Never invent circuits or pathways not found here."
        )
        
        cached_content = client.caches.create(
            model=GeminiModels.REASONER,
            config=types.CreateCachedContentConfig(
                contents=[uploaded_file],
                system_instruction=system_instruction,
                ttl="3600s" # 1 hour
            )
        )
        
        logger.info(f"[CacheManager] Success! Cache created: {cached_content.name}")
        
        return {
            "success": True,
            "file_name": uploaded_file.name,
            "cache_name": cached_content.name
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
