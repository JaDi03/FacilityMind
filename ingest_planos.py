"""
FacilityMind — Blueprint Ingestion Script
Loads PDF blueprints into the system for vector database indexing.

Usage:
    # Ingest a single blueprint:
    python ingest_planos.py --plano data/raw/E-14.pdf --id E-14 --tipo electrico

    # Ingest all blueprints from a directory:
    python ingest_planos.py --dir data/raw/ --edificio torre_corp

    # List blueprints already loaded:
    python ingest_planos.py --list
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import GEMINI_API_KEY, CHROMA_PATH, COLLECTION_NAME
from ingestion.pdf_loader import cargar_plano, listar_planos_disponibles
from ingestion.chunker import crear_chunks_inteligentes
from ingestion.vector_store import PlanoVectorStore


def ingestar_archivo(file_path: str, plano_id: str = None, tipo: str = None, edificio: str = "default"):
    """Ingests a single PDF file."""
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not configured. Create a .env file.")
        return False

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False

    print(f"📄 Ingesting: {file_path}")
    if plano_id:
        print(f"   ID: {plano_id}")
    if tipo:
        print(f"   Type: {tipo}")

    try:
        # 1. Load documents from PDF
        documentos = cargar_plano(file_path, plano_id=plano_id, edificio_id=edificio)

        if not documentos:
            print("❌ Could not extract text from PDF. Is it a scanned PDF without OCR?")
            return False

        # Overwrite discipline type if specified
        if tipo:
            for doc in documentos:
                doc["metadata"]["tipo_plano"] = tipo

        print(f"   📑 {len(documentos)} pages extracted")

        # 2. Create intelligent chunks
        chunks = crear_chunks_inteligentes(documentos, incluir_tablas=True)
        print(f"   🧩 {len(chunks)} chunks generated (pages + tables)")

        # 3. Index in Chroma
        store = PlanoVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
        num_agregados = store.add_chunks(chunks)

        print(f"   ✅ {num_agregados} chunks indexed in Chroma DB")
        print(f"   📦 Total chunks in DB: {store.contar_chunks()}")

        return True

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


def ingestar_directorio(dir_path: str, edificio: str = "default"):
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
        if ingestar_archivo(str(pdf), edificio=edificio):
            success_count += 1

    print(f"\n{'=' * 50}")
    print(f"✅ {success_count}/{len(pdfs)} blueprints ingested successfully")


def listar_indexados():
    """Lists blueprints already indexed in Chroma DB."""
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not configured")
        return

    store = PlanoVectorStore(persist_path=CHROMA_PATH, collection=COLLECTION_NAME)
    planos = store.listar_planos()

    print(f"\n📦 Chroma DB: {CHROMA_PATH}")
    print(f"📊 Total indexed chunks: {store.contar_chunks()}")
    print(f"📄 Loaded blueprints: {len(planos)}")

    if planos:
        print("\nList of blueprints:")
        for p in sorted(planos):
            print(f"  📄 {p}")


def main():
    parser = argparse.ArgumentParser(
        description="FacilityMind — PDF Blueprint Ingestion Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_planos.py --plano blueprints/E-14.pdf --id E-14 --tipo electrico
  python ingest_planos.py --dir blueprints/ --edificio corp_tower
  python ingest_planos.py --list
        """
    )

    parser.add_argument("--plano", help="Path to a blueprint PDF file")
    parser.add_argument("--id", help="Blueprint ID (e.g., E-14, P-01)")
    parser.add_argument("--tipo", help="Blueprint type: electrical, plumbing, architectural, structural, hvac, general")
    parser.add_argument("--dir", help="Directory containing PDF blueprints")
    parser.add_argument("--edificio", default="default", help="Building Identifier")
    parser.add_argument("--list", action="store_true", help="List blueprints already indexed")

    args = parser.parse_args()

    if args.list:
        listar_indexados()
    elif args.plano:
        ingestar_archivo(args.plano, args.id, args.tipo, args.edificio)
    elif args.dir:
        ingestar_directorio(args.dir, args.edificio)
    else:
        parser.print_help()
        print("\n⚠️ Use --plano, --dir, or --list to perform an action.")


if __name__ == "__main__":
    main()
