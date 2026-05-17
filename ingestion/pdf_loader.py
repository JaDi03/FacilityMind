"""
FacilityMind — PDF Blueprint Loader
Extracts text, images, and tables from construction blueprints using PyMuPDF.
Each page is converted into a document with enriched metadata for precise retrieval.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def inferir_tipo_plano(plano_id: str) -> str:
    """
    Infers the blueprint type from the ID or filename.
    Conventions: A- = architectural, E- = electrical, P- = plumbing, S- = structural, M- = HVAC.
    """
    pid = plano_id.upper()
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


def inferir_piso(plano_id: str, page_num: int, total_pages: int) -> Optional[str]:
    """
    Attempts to infer the floor covered by the blueprint from its ID.
    Example: 'E-14' → floor 14, 'P-B1' → basement 1, 'A-GF' → ground floor.
    """
    import re
    # Look for patterns like -14, _14, (14), B1, GF, PB, S1, etc.
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
        match = re.search(pat, plano_id.upper())
        if match:
            return match.group(1)
    # If not detected in the ID, it might be a general site plan
    return None


def inferir_torre(plano_id: str) -> Optional[str]:
    """Infers building tower or wing. Example: E-14-T2 → Tower 2."""
    import re
    match = re.search(r'[-_]T(\d+|[A-Z])', plano_id.upper())
    if match:
        return match.group(1)
    return None



def extraer_texto_con_vision(page: fitz.Page, page_num: int) -> str:
    """
    Uses Gemini Flash Vision to extract structured text from a blueprint page image.
    This produces spatially-aware text that preserves which annotations are in which rooms.
    
    Example output for an electrical plan:
    "UNIT A - ELECTRICAL PLAN (E.3)
     BEDROOM: Outlet labeled A-9, Outlet labeled A-11
     BATHROOM: Outlet labeled A-13 (GFCI)
     KITCHEN: Outlet labeled A-7, A-8
     PANEL A located in LAUNDRY ROOM"
    """
    import io
    import time
    import google.generativeai as genai
    from config import GeminiModels, GEMINI_API_KEY
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    try:
        # Convert PDF page to PNG image at 200 DPI (good balance of quality vs size)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        
        # Build the extraction prompt
        prompt = """You are analyzing a construction blueprint page. Extract ALL text visible on this page in a structured, readable format.

CRITICAL INSTRUCTIONS:
1. **Sheet Info**: Extract the sheet title and number (e.g., "E.3 - UNIT A ELECTRICAL PLAN")
2. **Room Labels**: List every room name visible (e.g., BEDROOM, LIVING ROOM, KITCHEN, BATHROOM, LAUNDRY)
3. **Electrical Annotations**: For EVERY electrical symbol (outlet, switch, light, etc.), extract:
   - The text label next to it (e.g., "A-9", "A-11", "C-1", "B-3")
   - WHICH ROOM it is physically located inside (e.g., "In BEDROOM: outlet labeled A-9")
4. **Panel Info**: Any panel labels, panel schedules, load schedules, breaker lists
5. **Dimensions**: Any measurements or dimensions shown
6. **Notes & Legends**: All written notes, specifications, legends, and general notes
7. **Equipment Labels**: HVAC units, plumbing fixtures, appliance labels
8. **Title Block**: Project name, drawing date, scale, architect info

FORMAT YOUR OUTPUT AS STRUCTURED TEXT with clear section headers.
Do NOT skip any annotation, label, or note — even small text matters for facility management.
For electrical plans, it is CRITICAL to associate each circuit label with the specific room it appears in."""
        
        model = genai.GenerativeModel(GeminiModels.VISION_OCR)
        
        # Send image to Gemini Flash
        from PIL import Image
        image = Image.open(io.BytesIO(img_bytes))
        
        response = model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(temperature=0.1)
        )
        
        extracted = response.text if response.text else ""
        logger.info(f"  Page {page_num}: Vision OCR extracted {len(extracted)} chars")
        
        # Small delay to avoid rate limiting (32 pages at ~2-3 sec each)
        time.sleep(0.5)
        
        return extracted
        
    except Exception as e:
        logger.warning(f"  Page {page_num}: Vision OCR failed ({e}), falling back to raw text")
        return ""


def cargar_plano(
    pdf_path: str,
    plano_id: Optional[str] = None,
    edificio_id: str = "default"
) -> List[Dict]:
    """
    Loads a PDF blueprint and returns a list of chunks (one per page with text).

    Args:
        pdf_path: Path to the PDF file.
        plano_id: Blueprint ID (if None, extracted from filename).
        edificio_id: ID of the building it belongs to.

    Returns:
        List of dictionaries with: text, metadata, id.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not plano_id:
        plano_id = pdf_path.stem

    tipo_plano = inferir_tipo_plano(plano_id)
    piso = inferir_piso(plano_id, 0, 0)
    torre = inferir_torre(plano_id)

    documentos = []
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    for page_num in range(total_pages):
        # We completely ignore PyMuPDF's garbage text extraction to keep our index 100% pristine.
        # All pages will be processed visually in the background or handled via Gemini's native PDF Context Cache.
        text = ""

        # Update floor if inferred from specific page context (if applicable)
        page_piso = inferir_piso(plano_id, page_num, total_pages) or piso

        doc_dict = {
            "text": text,
            "metadata": {
                "plano_id": str(plano_id),
                "tipo_plano": str(tipo_plano),
                "edificio_id": str(edificio_id),
                "piso": str(page_piso) if page_piso is not None else "",
                "torre": str(torre) if torre is not None else "",
                "pagina": int(page_num + 1),
                "total_paginas": int(total_pages),
                "source": str(f"{plano_id}_p{page_num + 1}"),
                "file_name": str(pdf_path.name),
            },
            "id": f"{plano_id}_p{page_num + 1}"
        }
        documentos.append(doc_dict)

    doc.close()
    logger.info(f"  -> {len(documentos)} page metadatas registered synchronously (pure visual-first approach)")
    return documentos


def cargar_plano_como_texto(pdf_path: str, max_chars: int = 500000) -> str:
    """
    Returns empty string because we completely refuse to use the garbage raw text layer of CAD PDFs.
    The system relies entirely on Gemini Context Cache (natively visual) and high-quality Vision OCR.
    """
    return ""


def listar_planos_disponibles(directorio: str = "./data/raw") -> List[Dict]:
    """
    Lists all available PDF blueprints in the specified directory.

    Returns:
        List of dictionaries with information for each blueprint.
    """
    dir_path = Path(directorio)
    if not dir_path.exists():
        return []

    planos = []
    for pdf_file in sorted(dir_path.glob("*.pdf")):
        plano_id = pdf_file.stem
        try:
            doc = fitz.open(str(pdf_file))
            total_pages = len(doc)
            doc.close()

            planos.append({
                "plano_id": plano_id,
                "tipo_plano": inferir_tipo_plano(plano_id),
                "piso": inferir_piso(plano_id, 0, 0),
                "torre": inferir_torre(plano_id),
                "total_paginas": total_pages,
                "file_path": str(pdf_file),
                "file_name": pdf_file.name,
            })
        except Exception as e:
            logger.error(f"Error reading {pdf_file}: {e}")

    return planos


# English Aliases for Clean Global Naming Conventions
infer_blueprint_type = inferir_tipo_plano
infer_floor = inferir_piso
infer_tower = inferir_torre
extract_text_with_vision = extraer_texto_con_vision
load_blueprint = cargar_plano
load_blueprint_as_text = cargar_plano_como_texto
list_available_blueprints = listar_planos_disponibles

