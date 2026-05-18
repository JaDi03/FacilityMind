"""
FacilityMind — Blueprint Metadata Re-indexer
Uses Gemini 2.5 Flash to extract a structured technical index for all pages
and updates Chroma DB with precise metadata (circuits, units, rooms, panels).
"""

import os
import sys
import json
import time
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai as genai_v2
from google.genai import types

# Setup path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CHROMA_PATH, COLLECTION_NAME, GEMINI_API_KEY, GeminiModels
from ingestion.vector_store import BlueprintVectorStore
from ingestion.pdf_loader import _count_pdf_pages, infer_blueprint_type, infer_floor, infer_tower

# Define Response Schema to force Gemini 100% syntactically perfect JSON
class PageIndexEntry(BaseModel):
    page_num: int = Field(..., description="The page number from the PDF (1-indexed)")
    sheet_no: str = Field(..., description="The sheet identifier (e.g. E.4, A103, A001) or empty string if not found")
    units: List[str] = Field(default_factory=list, description="List of residential/retail unit identifiers visible on this page (e.g. UNIT A, UNIT B)")
    circuits: List[str] = Field(default_factory=list, description="Exhaustive list of technical electrical circuits labeled on this page (e.g. A-9, A-10)")
    rooms: List[str] = Field(default_factory=list, description="List of rooms or space names on this page")
    panels: List[str] = Field(default_factory=list, description="List of electrical or service panel identifiers found on this page")
    content_summary: str = Field(..., description="Dense technical description summarizing the contents and key elements of this page")

class BlueprintIndexSchema(BaseModel):
    total_pages: int = Field(..., description="Total pages extracted")
    pages: List[PageIndexEntry] = Field(..., description="List of page index entries")

def run_structured_reindexing(pdf_path: str, blueprint_id: str):
    print(f"[START] Starting structured re-indexing for: {pdf_path}")
    total_pages = _count_pdf_pages(pdf_path)
    print(f"[INFO] Total pages in PDF: {total_pages}")

    client = genai_v2.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})
    
    # 1. Upload file for vision analysis
    print("[UPLOAD] Uploading blueprint to Gemini...")
    uploaded_file = client.files.upload(
        file=pdf_path,
        config={'display_name': f'reindex_vision_{blueprint_id}'}
    )
    
    file_info = client.files.get(name=uploaded_file.name)
    while file_info.state.name == "PROCESSING":
        print("[WAIT] Processing file on Google servers...")
        time.sleep(3)
        file_info = client.files.get(name=uploaded_file.name)
        
    if file_info.state.name == "FAILED":
        raise Exception("Failed to upload/process file on Gemini side.")
        
    print(f"[SUCCESS] Upload completed successfully: {uploaded_file.name}")

    # 2. Query Gemini Flash for Structured JSON
    prompt = """You are a highly precise construction blueprint indexer.
Analyze EVERY single page of this blueprint document.
For each page, perform an exhaustive visual scan of all layouts, electrical schedules, plumbing diagrams, and notes.

Extract:
1. "sheet_no": The sheet number/ID (e.g., "E.4", "A103", "A001") if printed on the page.
2. "units": A list of residential/retail unit identifiers visible on this page (e.g. ["UNIT A", "UNIT B", "RETAIL 1", "UNIT G"]).
3. "circuits": An exhaustive list of all electrical circuit/breaker numbers and technical codes labeled next to receptacles or inside panels/tables on this specific page (e.g., ["A-9", "A-10", "A-11", "A-12", "A-19", "A-20", "A-23", "A-17", "A-15", "A-16", "A-13"]). Look extremely closely for tiny, rotated, or vertical labels written adjacent to electrical symbols.
4. "rooms": A list of all rooms or spaces named on this page (e.g., ["BEDROOM", "KITCHEN", "LAUNDRY", "BATHROOM", "RESTROOM", "RETAIL AREA"]).
5. "panels": A list of electrical or service panel identifiers found on this page (e.g., ["PANEL A", "PANEL B", "REMOTE UNIT"]).
6. "content_summary": A brief, dense technical summary of this page (e.g., "Unit B Electrical & Lighting Floor Plan showing circuit connections to Panel A. Bedrooms are daisy-chained to circuit A-9. Kitchen outlets are on circuit A-10.").

Ensure no pages are left out or skipped. Ensure exact circuit and room matching.
"""

    print("[API_CALL] Querying Gemini 2.5 Flash for structured vision index...")
    start_time = time.time()
    
    response = client.models.generate_content(
        model=GeminiModels.VISION_OCR,  # Using Gemini Flash for Vision
        contents=[prompt, uploaded_file],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=BlueprintIndexSchema
        )
    )
    
    elapsed = time.time() - start_time
    print(f"[TIME] Vision analysis completed in {elapsed:.1f} seconds.")

    # 3. Save Structured Index to Disk
    raw_text = response.text.replace("```json", "").replace("```", "").strip()
    
    # Try parsing
    try:
        index_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        # Save raw to inspect
        with open("data/processed/raw_gemini_index_failed.txt", "w", encoding="utf-8") as f:
            f.write(raw_text)
        raise

    index_cache_path = Path("data/processed/blueprint_index.json")
    index_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_cache_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Structured blueprint index saved to: {index_cache_path}")

    # 4. Wipe and Re-index Chroma DB
    print("[CHROMA_WIPE] Wiping and re-initializing Chroma DB...")
    vector_store = BlueprintVectorStore()
    
    # Clear Chroma DB collection properly using IDs
    try:
        ids_to_delete = [f"{blueprint_id}_p{i}" for i in range(1, total_pages + 1)]
        vector_store.collection.delete(ids=ids_to_delete)
        print("[CHROMA_DELETE] Old blueprint chunks successfully purged from Chroma.")
    except Exception as e:
        print(f"[WARNING] Failed to delete old chunks: {e}")
        
    # Re-initialize collection
    vector_store = BlueprintVectorStore()

    chunks_to_add = []
    
    for page_entry in index_data.get("pages", []):
        page_num = page_entry.get("page_num", 0)
        sheet_no = page_entry.get("sheet_no", "")
        units = page_entry.get("units", [])
        circuits = page_entry.get("circuits", [])
        rooms = page_entry.get("rooms", [])
        panels = page_entry.get("panels", [])
        summary = page_entry.get("content_summary", "")

        # Infer basic properties
        page_floor = infer_floor(blueprint_id, page_num - 1, total_pages)
        page_type = infer_blueprint_type(sheet_no)
        tower = infer_tower(blueprint_id)

        # Standardize arrays to comma-separated strings for Chroma filtering compatibility
        metadata = {
            "blueprint_id": str(blueprint_id),
            "blueprint_type": str(page_type),
            "building_id": "default",
            "floor": str(page_floor) if page_floor is not None else "",
            "tower": str(tower) if tower is not None else "",
            "page": int(page_num),
            "total_pages": int(total_pages),
            "sheet_no": str(sheet_no),
            "source": f"{blueprint_id}_p{page_num}",
            "file_name": os.path.basename(pdf_path),
            # Precise structural tags for metadata filters
            "units": ",".join([u.upper() for u in units]),
            "circuits": ",".join([c.upper() for c in circuits]),
            "rooms": ",".join([r.upper() for r in rooms]),
            "panels": ",".join([p.upper() for p in panels]),
        }

        # Build highly structured searchable document text
        document_text = f"""[PDF Page {page_num} - Vision OCR Index]
Sheet: {sheet_no}
Floor: {metadata['floor']}
Units: {", ".join(units)}
Rooms: {", ".join(rooms)}
Panels: {", ".join(panels)}
Circuits Labeled: {", ".join(circuits)}

Summary:
{summary}
"""
        
        chunk = {
            "text": document_text,
            "metadata": metadata,
            "id": f"{blueprint_id}_p{page_num}"
        }
        chunks_to_add.append(chunk)

    print(f"[CHROMA_ADD] Adding {len(chunks_to_add)} enriched structured chunks to Chroma...")
    vector_store.add_chunks(chunks_to_add)
    print("[SUCCESS] Re-indexing complete! Every page is now a structured searchable entity.")

if __name__ == "__main__":
    pdf = r"c:\Users\USER\Downloads\Kimi_Agent_Gemini Hackathon\FacilityMind\data\raw\Mixed-Use.pdf"
    run_structured_reindexing(pdf, "Mixed-Use")
