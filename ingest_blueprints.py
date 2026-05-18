"""
FacilityMind — Blueprint Ingestion Script
Loads PDF blueprints into the system for vector database indexing.

Usage:
    # Ingest a single blueprint:
    python ingest_blueprints.py --blueprint data/raw/E-14.pdf --id E-14 --type electrical

    # Ingest all blueprints from a directory:
    python ingest_blueprints.py --dir data/raw/ --building torre_corp

    # List blueprints already loaded:
    python ingest_blueprints.py --list
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import GEMINI_API_KEY, CHROMA_PATH, COLLECTION_NAME
from ingestion.pdf_loader import load_blueprint, list_available_blueprints
from ingestion.chunker import create_intelligent_chunks
from ingestion.vector_store import BlueprintVectorStore


def ingest_file(file_path: str, blueprint_id: str = None, blueprint_type: str = None, building_id: str = "default"):
    """Ingests a single PDF file."""
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not configured. Create a .env file.")
        return False

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False

    print(f"📄 Ingesting: {file_path}")
    if blueprint_id:
        print(f"   ID: {blueprint_id}")
    if blueprint_type:
        print(f"   Type: {blueprint_type}")

    try:
        # 1. Load basic document layout
        documentos = load_blueprint(file_path, blueprint_id=blueprint_id, building_id=building_id)

        if not documentos:
            print("❌ Could not extract pages from PDF.")
            return False

        # Overwrite discipline type if specified
        if blueprint_type:
            for doc in documentos:
                doc["metadata"]["blueprint_type"] = blueprint_type

        print(f"   📑 {len(documentos)} pages registered. Running high-fidelity parallel Vision OCR...")

        # 2. Extract technical text page-by-page using parallel Vision OCR
        from ingestion.pdf_loader import extract_all_pages_with_vision, extract_metadata_from_text
        page_texts = extract_all_pages_with_vision(file_path)

        for doc in documentos:
            page_num = doc["metadata"]["page"]
            vision_text = page_texts.get(page_num, "")
            doc["text"] = f"[PDF Page {page_num} - Vision OCR]\n{vision_text}"
            
            # Enrich metadata with extracted structural tags (FIX 4)
            extracted = extract_metadata_from_text(vision_text)
            doc["metadata"]["circuits"] = ",".join(extracted["circuits"])
            doc["metadata"]["units"] = ",".join(extracted["units"])
            doc["metadata"]["rooms"] = ",".join(extracted["rooms"])
            doc["metadata"]["panels"] = ",".join(extracted["panels"])
            if extracted["sheet_no"]:
                doc["metadata"]["sheet_no"] = extracted["sheet_no"]

        print("   ✅ Vision OCR complete. Segmenting text...")

        # 3. Create intelligent chunks from populated text
        chunks = create_intelligent_chunks(documentos, incluir_tablas=True)
        print(f"   🧩 {len(chunks)} chunks generated (pages + tables)")

        # 4. Index in Chroma
        store = BlueprintVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
        num_agregados = store.add_chunks(chunks)

        print(f"   ✅ {num_agregados} chunks indexed in Chroma DB")
        print(f"   📦 Total chunks in DB: {store.count_chunks()}")

        return True

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


def ingest_directory(dir_path: str, building_id: str = "default"):
    """Ingests all PDFs from a directory."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"❌ Directory not found: {dir_path}")
        return

    pdfs = list(dir_path.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in: {dir_path}")
        return

    print(f"\n📁 Ingesting {len(pdfs)} blueprints from: {dir_path}")
    print("=" * 50)

    success_count = 0
    for pdf in sorted(pdfs):
        print()
        if ingest_file(str(pdf), building_id=building_id):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"✅ {success_count}/{len(pdfs)} blueprints ingested successfully")


def list_indexed():
    """Lists blueprints already indexed in Chroma DB."""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not configured")
        return

    store = BlueprintVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
    blueprints = store.list_blueprints()

    print(f"\n📦 Chroma DB: {CHROMA_PATH}")
    print(f"📊 Total indexed chunks: {store.count_chunks()}")
    print(f"📄 Loaded blueprints: {len(blueprints)}")

    if blueprints:
        print("\nList of blueprints:")
        for p in sorted(blueprints):
            print(f"  📄 {p}")


def main():
    parser = argparse.ArgumentParser(
        description="FacilityMind — PDF Blueprint Ingestion Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_blueprints.py --blueprint blueprints/E-14.pdf --id E-14 --type electrical
  python ingest_blueprints.py --dir blueprints/ --building corp_tower
  python ingest_blueprints.py --list
        """
    )

    parser.add_argument("--blueprint", help="Path to a blueprint PDF file")
    parser.add_argument("--id", help="Blueprint ID (e.g., E-14, P-01)")
    parser.add_argument("--type", help="Blueprint type: electrical, plumbing, architectural, structural, hvac, general")
    parser.add_argument("--dir", help="Directory containing PDF blueprints")
    parser.add_argument("--building", default="default", help="Building Identifier")
    parser.add_argument("--list", action="store_true", help="List blueprints already indexed")

    args = parser.parse_args()

    if args.list:
        list_indexed()
    elif args.blueprint:
        ingest_file(args.blueprint, args.id, args.type, args.building)
    elif args.dir:
        ingest_directory(args.dir, args.building)
    else:
        parser.print_help()
        print("\n⚠️ Use --blueprint, --dir, or --list to perform an action.")


if __name__ == "__main__":
    main()
