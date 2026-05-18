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
        temp_img_path = temp_dir / f"page_{page_num}.png"
        
        try:
            # 1. Render the page to high-res PNG using PyMuPDF (fitz) at 4x zoom (288 DPI)
            import fitz
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_idx)
            zoom = 4.0  # 4x zoom provides extremely crisp text for blueprints (288 DPI)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(temp_img_path))
            doc.close()
                
            # 2. Upload to Gemini
            up_file = client.files.upload(
                file=str(temp_img_path),
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
            prompt = f"""You are a master electrical blueprint OCR agent analyzing Page {page_num}.

## YOUR TASK:
Read ALL text and annotations visible on this blueprint page. Report exactly what you see.

## RULES:
1. For circuit labels (like "A-9", "A-13", "A-15", "A-23"): Report the EXACT text you see written on the drawing. Read each character carefully.
2. For each circuit label you find, list which electrical symbols (outlets, switches, lights) are PHYSICALLY NEAR that label or connected to it by a dashed line on the drawing.
3. If you see outlet symbols that have NO circuit label near them and NO dashed line connecting to any label, report them as having "No circuit label visible".
4. DO NOT copy data from a panel schedule table and apply it to floor plan symbols. Panel schedules and floor plans are separate things.

## EXTRACT:
1. **Sheet Info**: Title, number, building ID, scale.
2. **Room Labels**: All room/unit names visible.
3. **Circuit Labels Found**: List every circuit label text visible on the floor plan drawing (e.g., "A-9", "A-13"). For each, describe what devices it appears to serve based on its position in the drawing.
4. **Panel Board Info**: Panel name, location, schedule data if visible.
5. **Summary**: Brief factual description of the page content.

Be thorough — report ALL labels you can read. Do not omit labels out of caution.
"""
            
            response = client.models.generate_content(
                model=GeminiModels.VISION_OCR,
                contents=[prompt, up_file],
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            
            extracted_text = response.text.strip()
            
            # Clean up Gemini file
            try:
                client.files.delete(name=up_file.name)
            except Exception:
                pass
                
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
            logger.info(f"[Vision OCR] Successfully indexed Page {page_num}/{total_pages}")
            return page_num, extracted_text
            
        except Exception as e:
            logger.error(f"[Vision OCR] Error indexing Page {page_num}: {e}")
            if os.path.exists(temp_img_path):
                try:
                    os.remove(temp_img_path)
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
    
    # Save the extracted texts to disk so they can be loaded by load_blueprint_as_text() (FIX 5)
    try:
        save_dir = Path("scratch/extracted_texts")
        save_dir.mkdir(parents=True, exist_ok=True)
        pdf_stem = Path(pdf_path).stem
        save_path = save_dir / f"{pdf_stem}.json"
        
        # Serialize with json
        import json
        save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[Vision OCR] Saved high-fidelity extracted texts to: {save_path}")
    except Exception as e:
        logger.warning(f"[Vision OCR] Could not persist extracted texts to disk: {e}")

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
    """Load extracted text from Vision OCR results stored on disk."""
    cache_path = Path("scratch/extracted_texts") / f"{Path(pdf_path).stem}.json"
    if cache_path.exists():
        import json
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            texts = [f"--- Page {p} ---\n{t}" for p, t in sorted(data.items(), key=lambda x: int(x[0]))]
            return "\n\n".join(texts)[:max_chars]
        except Exception as e:
            logger.warning(f"Error reading extracted text JSON: {e}")
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


def extract_metadata_from_text(text: str) -> dict:
    """
    Parses units, circuits, rooms, panels, and sheet_no from raw OCR text using regex.
    This provides super-fast metadata extraction without calling another LLM.
    """
    import re
    
    # Circuits: match A-13, C-14, P-5, etc.
    circuits = sorted(list(set(re.findall(r'\b[A-Z]-\d+\b', text) + re.findall(r'\bC-\d+\b', text))))
    
    # Units: match UNIT A, UNIT B, UNIT 102, etc.
    units = sorted(list(set(re.findall(r'\bUNIT\s+[A-G0-9]\b', text, re.IGNORECASE) + re.findall(r'\bUNIDAD\s+[A-G0-9]\b', text, re.IGNORECASE))))
    units = [u.upper() for u in units]
    
    # Panels: match PANEL A, PANEL B, TABLERO A, etc.
    panels = sorted(list(set(re.findall(r'\bPANEL\s+[A-Z0-9]\b', text, re.IGNORECASE) + re.findall(r'\bTABLERO\s+[A-Z0-9]\b', text, re.IGNORECASE))))
    panels = [p.upper() for p in panels]
    
    # Rooms: common room labels in Spanish/English
    room_keywords = r'\b(BEDROOM|KITCHEN|LIVING|DINING|LAUNDRY|BATHROOM|HALLWAY|RECAMARA|RECÁMARA|SALA|COMEDOR|COCINA|BAÑO|CUARTO|PASILLO|CLOSET)\b'
    rooms = sorted(list(set(re.findall(room_keywords, text, re.IGNORECASE))))
    rooms = [r.upper() for r in rooms]
    
    # Sheet number: e.g. Sheet E.14 or Sheet E-14
    sheet_match = re.search(r'\bSheet\s*([A-Z0-9.\-]+)\b', text, re.IGNORECASE)
    sheet_no = sheet_match.group(1) if sheet_match else ""
    
    return {
        "circuits": circuits,
        "units": units,
        "rooms": rooms,
        "panels": panels,
        "sheet_no": sheet_no
    }

