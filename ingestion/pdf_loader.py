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


def extract_text_with_vision(pdf_path: str, page_num: int = None, uploaded_file=None) -> str:
    """DEPRECATED: Use extract_all_pages_with_vision() for batch processing."""
    return ""


def extract_all_pages_with_vision(pdf_path: str, uploaded_file=None) -> dict:
    """
    Extracts structured text from all pages in parallel by splitting the PDF 
    and processing each page individually using Gemini 2.5 Flash.
    This guarantees 100% optical resolution and prevents output token truncation.
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pypdf import PdfReader, PdfWriter
    from google import genai as genai_v2
    from google.genai import types
    from config import GeminiModels, GEMINI_API_KEY
    from pathlib import Path

    client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    logger.info(f"[Vision OCR] Initializing high-fidelity parallel indexer for {total_pages} pages...")
    
    temp_dir = Path("scratch/temp_split")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    def process_single_page(page_idx):
        page_num = page_idx + 1
        temp_pdf_path = temp_dir / f"page_{page_num}.pdf"
        
        try:
            # 1. Isolate the single page
            writer = PdfWriter()
            writer.add_page(reader.pages[page_idx])
            with open(temp_pdf_path, "wb") as f:
                writer.write(f)
                
            # 2. Upload to Gemini
            up_file = client.files.upload(
                file=str(temp_pdf_path),
                config={'display_name': f'ocr_page_{page_num}'}
            )
            
            # Wait for processing
            file_info = client.files.get(name=up_file.name)
            while file_info.state.name == "PROCESSING":
                time.sleep(0.5)
                file_info = client.files.get(name=up_file.name)
                
            if file_info.state.name == "FAILED":
                raise Exception("Gemini processing failed")
                
            # 3. Meticulous extraction prompt
            prompt = f"""You are a master electrical and architectural indexing agent.
Perform a high-fidelity visual and textual audit of this isolated blueprint page (Page {page_num} of the PDF).

Extract all visible annotations with absolute technical precision:
1. Sheet Info: Sheet title, sheet number, building ID, and drawing scale.
2. Room Labels: Every room or unit name visible (e.g. UNIT D Bedroom, Kitchen, Living Room).
3. Electrical Circuits: Locate all outlets, equipment, and devices. Follow the dashed line conduit loops to find exactly which circuit (e.g. A-13, A-15, A-16, A-19, A-20, A-23) feeds each symbol.
4. Panel Board Info: Note any electrical panels (e.g. PANEL A) and schedule data.
5. Content Summary: Provide a highly detailed summary explaining which circuits physically power which appliances or areas.

Respond with the extracted text in a clean, highly structured format.
"""
            
            response = client.models.generate_content(
                model=GeminiModels.VISION_OCR,
                contents=[prompt, up_file],
                config=types.GenerateContentConfig(
                    temperature=0.1
                )
            )
            
            extracted_text = response.text.strip()
            
            # Clean up Gemini file
            try:
                client.files.delete(name=up_file.name)
            except Exception:
                pass
                
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
            logger.info(f"[Vision OCR] Successfully indexed Page {page_num}/{total_pages}")
            return page_num, extracted_text
            
        except Exception as e:
            logger.error(f"[Vision OCR] Error indexing Page {page_num}: {e}")
            if os.path.exists(temp_pdf_path):
                try:
                    os.remove(temp_pdf_path)
                except Exception:
                    pass
            return page_num, f"Error processing page: {e}"

    start_time = time.time()
    
    # Process up to 8 pages in parallel to stay within rate limits and optimize speed
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_page, idx) for idx in range(total_pages)]
        for future in futures:
            p_num, text = future.result()
            results[p_num] = text
            
    elapsed = time.time() - start_time
    logger.info(f"[Vision OCR] High-fidelity parallel indexing complete for {total_pages} pages in {elapsed:.1f}s")
    
    # Clean up temp dir
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass
        
    return results



def _fallback_page_parser(text: str) -> dict:
    """Fallback: parses plain text with PAGE markers into a page dict."""
    import re
    result = {}
    page_pattern = re.compile(r'(?:page|pagina|pagina)\s*(\d+)[:\s\n=\-]+', re.IGNORECASE)
    matches = list(page_pattern.finditer(text))

    if not matches:
        return {1: text}

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[page_num] = text[start:end].strip()

    return result


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
