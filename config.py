"""
FacilityMind — Centralized Gemini Model Configuration
Provides centralized management for AI model selection and system thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── API Keys ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment variables. Example: export GEMINI_API_KEY='...'")

# ─── Model Selection ───
class GeminiModels:
    # Agent 1 — Perception
    PERCEPTION = "models/gemini-2.5-flash"
    # Agent 2 — Reasoner
    REASONER = "models/gemini-2.5-pro"
    # Agent 3 — Validator
    VALIDATOR = "models/gemini-2.5-pro"
    # Agent 4 — Visualizer
    VISUALIZER_IMAGE = "models/imagen-4.0-ultra-generate-001"
    # Embeddings
    EMBEDDING = "models/gemini-embedding-001"

# ─── Chroma DB Configuration ───
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/processed/chroma_db/blueprints")
COLLECTION_NAME = "blueprints_v1"

# ─── Paths ───
PDF_RAW_PATH = Path("./data/raw")
PDF_PROCESSED_PATH = Path("./data/processed")

# ─── Agent Thresholds ───
class Thresholds:
    # Perception
    PERCEPTION_MIN = 0.65
    # Reasoner
    REASONER_MIN = 0.70
    # Validator (The ones missing in the previous step)
    VALIDATOR_MIN = 0.85
    VALIDATOR_APPROVE = 0.80  # High confidence threshold
    VALIDATOR_REJECT = 0.50   # Low confidence threshold
    MAX_RETRIES = 2

# ─── System Flags & Limits ───
MAX_CONTEXT_CHARS = 1000000
VISUALIZATION_ENABLED = True

# ─── UI & API Settings ───
API_HOST = "0.0.0.0"
API_PORT = 8000
LOG_LEVEL = "INFO"
