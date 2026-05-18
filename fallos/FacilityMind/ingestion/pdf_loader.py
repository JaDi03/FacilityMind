"""
FacilityMind — PDF Blueprint Loader (Vision-Only)
Extracts text from construction blueprints using Gemini Vision OCR.
No PyMuPDF — sends PDF directly to Google Gemini for native processing.
"""

from pathlib import Path
from typing import List, Dict, Optional
import logging
import time
import re

logger = logging.getLogger(__name__)


def _count_pdf_pages(pdf_path: str) -> int:
    """Count pages in a PDF using pypdf (pure Python, no C dependencies)."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return len(reader.pages)


def infer_blueprint_type(blueprint_id: str) -> str:
    """
    Infers the blueprint type from the ID or filename.
    Conventions: A- = architectural, E- = electrical, P- = plumbing, S- = structural, M- = HVAC.
    """
    pid = blueprint_id.upper()
    if any(p in pid for p in ["E-", "ELEC", "POWER", "LIGTH", "LIGHTING"]):
        return "electrical"
    if any(p in pid for p in ["P-", "PLUM", "MEP-PD", "DRAIN", "SANITARY", "WATER"]):
        return "plumbing"
    if any(p in pid for p in ["A-", "ARCH", "FLOOR", "PLAN"]):
        return "architectural"
    if any(p in pid for p in ["S-", "STRUCT", "FOUND", "BEAM", "COLUMN"]):
        return "structural"
    if any(p in pid for p in ["M-", "MECH", "HVAC", "AC-", "DUCT"]):
        return "hvac"
    if any(p in pid for p in ["FP-", "FIRE", "SPRINK"]):
        return "fire_protection"
    return "general"


def infer_floor(blueprint_id: str, page_num: int, total_pages: int) -> Optional[str]:
    """
    Attempts to infer the floor covered by the blueprint from its ID.
    Example: 'E-14' → floor 14, 'P-B1' → basement 1, 'A-GF' → ground floor.
    """
    patterns = [
        r'[-_](\d+)$',      # E-14, P_03
        r'[-_](B\d+)$',     # E-B1, P-B2 (basements)
        r'[-_](GF)$',       # A-GF (ground floor)
        r'[-_](PB)$',       # A-PB (planta baja / ground floor)
        r'[-_](S\d+)$',     # E-S1 (service)
        r'[-_](R\d+)$',     # E-R1 (rooftop)
        r'\((\d+)\)',       # E(14), P(03)
    ]
    for pat in patterns:
        match = re.search(pat, blueprint_id.upper())
        if match:
            return match.group(1)
    return None


def infer_tower(blueprint_id: str) -> Optional[str]:
    """Infers building tower or wing. Example: E-14-T2 → Tower 2."""
    match = re.search(r'[-_]T(\d+|[A-Z])', blueprint_id.upper())
    if match:
        return match.group(1)
    return None


def extract_text_with_vision(pdf_path: str, page_num: int, uploaded_file=None) -> str:
    """
    Uses Gemini Flash Vision to extract structured text from a specific PDF page.
    Sends the PDF directly to Gemini — no PyMuPDF, no image rendering needed.

    Args:
        pdf_path: Path to the PDF file.
        page_num: 1-indexed page number to extract.
        uploaded_file: Optional pre-uploaded Gemini file reference (to avoid re-uploading).
    """
    from google import genai as genai_v2
    from google.genai import types
    from config import GeminiModels, GEMINI_API_KEY

    client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})

    try:
        # Upload PDF if not already uploaded
        if not uploaded_file:
            uploaded_file = client.files.upload(file=pdf_path, config={'display_name': f'ocr_page_{page_num}'})
            file_info = client.files.get(name=uploaded_file.name)
            while file_info.state.name == "PROCESSING":
                time.sleep(2)
                file_info = client.files.get(name=uploaded_file.name)
            if file_info.state.name == "FAILED":
                raise Exception("Google Gemini failed to process the PDF.")

        prompt = f"""You are analyzing page {page_num} of this construction blueprint PDF. Extract ALL text visible on PAGE {page_num} ONLY in a structured, readable format.

CRITICAL INSTRUCTIONS:
1. **Sheet Info**: Extract the sheet title and number (e.g., "E.3 - UNIT A ELECTRICAL PLAN")
2. **Room Labels**: List every room name visible (e.g., BEDROOM, LIVING ROOM, KITCHEN, BATHROOM, LAUNDRY)
3. **Electrical Annotations**: For EVERY electrical symbol (outlet, switch, light, etc.), extract:
   - The text label next to it (e.g., "A-9", "A-11", "C-1", "B-3")
   - WHICH ROOM it is physically located inside (e.g., "In BEDROOM: outlet labeled A-9")
4. **Panel Info**: Any panel labels, panel schedules, load schedules, breaker lists
5. **Dimensions**: ALL measurements and dimensions shown (e.g., 20'-0", 202'-0", 46'-0", etc.)
6. **Notes & Legends**: All written notes, specifications, legends, and general notes
7. **Equipment Labels**: HVAC units, plumbing fixtures, appliance labels (REF, DW, W/D, WH, etc.)
8. **Title Block**: Project name, drawing date, scale, architect info
9. **Area Data**: Any square footage data, room areas, building areas

FORMAT YOUR OUTPUT AS STRUCTURED TEXT with clear section headers.
Do NOT skip any annotation, label, or note — even small text matters for facility management.
For electrical plans, it is CRITICAL to associate each circuit label with the specific room it appears in."""

        response = client.models.generate_content(
            model=GeminiModels.VISION_OCR,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(temperature=0.1)
        )

        extracted = response.text if response.text else ""
        logger.info(f"  Page {page_num}: Vision OCR extracted {len(extracted)} chars")

        time.sleep(0.5)
        return extracted

    except Exception as e:
        logger.warning(f"  Page {page_num}: Vision OCR failed ({e})")
        return ""


def load_blueprint(
    pdf_path: str,
    blueprint_id: Optional[str] = None,
    building_id: str = "default"
) -> List[Dict]:
    """
    Loads a PDF blueprint and returns a list of page metadata entries.
    Text extraction is deferred to the background Vision OCR task.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not blueprint_id:
        blueprint_id = pdf_path.stem

    blueprint_type = infer_blueprint_type(blueprint_id)
    floor = infer_floor(blueprint_id, 0, 0)
    tower = infer_tower(blueprint_id)

    total_pages = _count_pdf_pages(str(pdf_path))
    documents = []

    for page_num in range(total_pages):
        page_floor = infer_floor(blueprint_id, page_num, total_pages) or floor

        doc_dict = {
            "text": "",
            "metadata": {
                "blueprint_id": str(blueprint_id),
                "blueprint_type": str(blueprint_type),
                "building_id": str(building_id),
                "floor": str(page_floor) if page_floor is not None else "",
                "tower": str(tower) if tower is not None else "",
                "page": int(page_num + 1),
                "total_pages": int(total_pages),
                "source": str(f"{blueprint_id}_p{page_num + 1}"),
                "file_name": str(pdf_path.name),
            },
            "id": f"{blueprint_id}_p{page_num + 1}"
        }
        documents.append(doc_dict)

    logger.info(f"  -> {len(documents)} page metadatas registered (Vision OCR pending)")
    return documents


def load_blueprint_as_text(pdf_path: str, max_chars: int = 500000) -> str:
    """Returns empty string — text extraction is handled entirely by Vision OCR."""
    return ""


def list_available_blueprints(directory: str = "./data/raw") -> List[Dict]:
    """
    Lists all available PDF blueprints in the specified directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    blueprints = []
    for pdf_file in sorted(dir_path.glob("*.pdf")):
        blueprint_id = pdf_file.stem
        try:
            total_pages = _count_pdf_pages(str(pdf_file))

            blueprints.append({
                "blueprint_id": blueprint_id,
                "blueprint_type": infer_blueprint_type(blueprint_id),
                "floor": infer_floor(blueprint_id, 0, 0),
                "tower": infer_tower(blueprint_id),
                "total_pages": total_pages,
                "file_path": str(pdf_file),
                "file_name": pdf_file.name,
            })
        except Exception as e:
            logger.error(f"Error reading {pdf_file}: {e}")

    return blueprints
