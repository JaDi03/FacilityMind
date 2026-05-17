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


def extraer_texto_pagina(page: fitz.Page) -> str:
    """Extracts text from a PDF page, attempting to preserve table formatting."""
    # Attempt simple text extraction first
    text = page.get_text("text")
    if not text.strip():
        # If no text is found, it might be a scanned image → Gemini-based OCR fallback in pipeline
        return ""
    return text


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

    logger.info(f"Loading blueprint {plano_id} ({tipo_plano}) - {total_pages} pages")

    for page_num in range(total_pages):
        page = doc[page_num]
        text = extraer_texto_pagina(page)

        if not text.strip():
            logger.warning(f"  Page {page_num + 1} has no selectable text (likely a scanned image)")
            continue

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
    logger.info(f"  -> {len(documentos)} pages with text extracted")
    return documentos


def cargar_plano_como_texto(pdf_path: str, max_chars: int = 500000) -> str:
    """
    Extracts ALL text from a PDF as a single continuous string.
    Used for passing the entire blueprint as long-context to Gemini.

    Args:
        pdf_path: Path to the PDF file.
        max_chars: Maximum characters allowed (truncates if longer).

    Returns:
        Full text content of the blueprint.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return ""

    doc = fitz.open(str(pdf_path))
    partes = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            partes.append(f"\n--- BLUEPRINT PAGE {i + 1} ---\n{text}")
    doc.close()

    full_text = "\n".join(partes)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n...[CONTENT TRUNCATED DUE TO CONTEXT LIMIT]"

    return full_text


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
