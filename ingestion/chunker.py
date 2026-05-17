"""
FacilityMind — Intelligent Chunker for Construction Blueprints
Divides blueprints into semantic chunks while maintaining table and section coherence.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def chunk_by_page(documents: List[Dict]) -> List[Dict]:
    """
    Simple strategy: each page is a chunk.
    Effective for blueprints where each page contains self-contained information.
    """
    return documents


def chunk_by_tables(text: str, base_metadata: Dict, blueprint_id: str, page: int) -> List[Dict]:
    """
    Detects tables in the text and extracts them as independent chunks.
    Technical tables (loads, specifications) are critical for accurate retrieval.
    """
    chunks = []

    lines = text.split('\n')
    in_table = False
    current_table = []
    table_num = 0

    for line in lines:
        columns = [c.strip() for c in re.split(r'\s{2,}', line) if c.strip()]

        if len(columns) >= 3:
            if not in_table:
                in_table = True
                table_num += 1
            current_table.append(line)
        else:
            if in_table and len(current_table) >= 3:
                table_text = '\n'.join(current_table)
                chunk_id = f"{blueprint_id}_p{page}_table{table_num}"
                chunks.append({
                    "text": f"[TECHNICAL TABLE]\n{table_text}",
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "table",
                        "table_num": table_num,
                        "source": chunk_id,
                    },
                    "id": chunk_id
                })
            in_table = False
            current_table = []

    if in_table and len(current_table) >= 3:
        table_text = '\n'.join(current_table)
        chunk_id = f"{blueprint_id}_p{page}_table{table_num}"
        chunks.append({
            "text": f"[TECHNICAL TABLE]\n{table_text}",
            "metadata": {
                **base_metadata,
                "chunk_type": "table",
                "table_num": table_num,
                "source": chunk_id,
            },
            "id": chunk_id
        })

    return chunks


def chunk_by_sections(text: str, base_metadata: Dict, blueprint_id: str, page: int) -> List[Dict]:
    """
    Detects blueprint sections (identified by uppercase titles, Roman numerals, etc.)
    and creates chunks per section.
    """
    chunks = []

    section_pattern = re.compile(
        r'^(?:SECTION|SECCION|SECCIÓN|DETAIL|DETALLE|SCHEDULE|NOTES|NOTAS|'
        r'LEGEND|LEYENDA|GENERAL NOTES|SCHEDULE OF|INDEX|INDICE)\s*[\dA-Z]*',
        re.IGNORECASE | re.MULTILINE
    )

    sections = list(section_pattern.finditer(text))

    if len(sections) < 2:
        return []

    for i, match in enumerate(sections):
        start = match.start()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) > 50:
            chunk_id = f"{blueprint_id}_p{page}_sec{i + 1}"
            chunks.append({
                "text": section_text,
                "metadata": {
                    **base_metadata,
                    "chunk_type": "section",
                    "section_title": match.group(0).strip(),
                    "source": chunk_id,
                },
                "id": chunk_id
            })

    return chunks


def create_intelligent_chunks(
    documents: List[Dict],
    include_tables: bool = True,
    include_sections: bool = False
) -> List[Dict]:
    """
    Creates intelligent chunks from page-level documents.
    """
    final_chunks = []

    for doc in documents:
        text = doc["text"]
        meta = doc["metadata"]
        blueprint_id = meta["blueprint_id"]
        page = meta["page"]

        final_chunks.append(doc)

        if include_tables:
            try:
                table_chunks = chunk_by_tables(text, meta, blueprint_id, page)
                final_chunks.extend(table_chunks)
            except Exception as e:
                logger.warning(f"Error extracting tables from {blueprint_id} p{page}: {e}")

        if include_sections:
            try:
                section_chunks = chunk_by_sections(text, meta, blueprint_id, page)
                final_chunks.extend(section_chunks)
            except Exception as e:
                logger.warning(f"Error extracting sections from {blueprint_id} p{page}: {e}")

    logger.info(f"Chunks created: {len(final_chunks)} (from {len(documents)} pages)")
    return final_chunks
