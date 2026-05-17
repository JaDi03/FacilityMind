"""
FacilityMind — Intelligent Chunker for Construction Blueprints
Divides blueprints into semantic chunks while maintaining table and section coherence.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def chunk_por_pagina(documentos: List[Dict]) -> List[Dict]:
    """
    Simple strategy: each page is a chunk.
    Effective for blueprints where each page contains self-contained information.
    """
    return documentos


def chunk_por_tablas(texto: str, metadata_base: Dict, plano_id: str, pagina: int) -> List[Dict]:
    """
    Detects tables in the text and extracts them as independent chunks.
    Technical tables (loads, specifications) are critical for accurate retrieval.
    """
    chunks = []

    # Pattern to detect tables: lines with multiple columns separated by spaces/tabs
    lineas = texto.split('\n')
    en_tabla = False
    tabla_actual = []
    tabla_num = 0

    for linea in lineas:
        # Heuristic: if the line has multiple segments separated by 2+ spaces
        columnas = [c.strip() for c in re.split(r'\s{2,}', linea) if c.strip()]

        if len(columnas) >= 3:  # Likely a table row
            if not en_tabla:
                en_tabla = True
                tabla_num += 1
            tabla_actual.append(linea)
        else:
            if en_tabla and len(tabla_actual) >= 3:  # Table with at least 3 rows
                tabla_texto = '\n'.join(tabla_actual)
                chunk_id = f"{plano_id}_p{pagina}_tabla{tabla_num}"
                chunks.append({
                    "text": f"[TECHNICAL TABLE]\n{tabla_texto}",
                    "metadata": {
                        **metadata_base,
                        "chunk_type": "tabla",
                        "tabla_num": tabla_num,
                        "source": chunk_id,
                    },
                    "id": chunk_id
                })
            en_tabla = False
            tabla_actual = []

    # Handle table at the end of text
    if en_tabla and len(tabla_actual) >= 3:
        tabla_texto = '\n'.join(tabla_actual)
        chunk_id = f"{plano_id}_p{pagina}_tabla{tabla_num}"
        chunks.append({
            "text": f"[TECHNICAL TABLE]\n{tabla_texto}",
            "metadata": {
                **metadata_base,
                "chunk_type": "tabla",
                "tabla_num": tabla_num,
                "source": chunk_id,
            },
            "id": chunk_id
        })

    return chunks


def chunk_por_secciones(texto: str, metadata_base: Dict, plano_id: str, pagina: int) -> List[Dict]:
    """
    Detects blueprint sections (identified by uppercase titles, Roman numerals, etc.)
    and creates chunks per section.
    """
    chunks = []

    # Section patterns: "SECTION 1", "DETAIL A", "SCHEDULE OF", etc.
    patron_seccion = re.compile(
        r'^(?:SECTION|SECCION|SECCIÓN|DETAIL|DETALLE|SCHEDULE|NOTES|NOTAS|'
        r'LEGEND|LEYENDA|GENERAL NOTES|SCHEDULE OF|INDEX|INDICE)\s*[\dA-Z]*',
        re.IGNORECASE | re.MULTILINE
    )

    secciones = list(patron_seccion.finditer(texto))

    if len(secciones) < 2:
        # No clear sections found → return empty list to fallback to full-page chunk
        return []

    for i, match in enumerate(secciones):
        inicio = match.start()
        fin = secciones[i + 1].start() if i + 1 < len(secciones) else len(texto)
        seccion_texto = texto[inicio:fin].strip()

        if len(seccion_texto) > 50:  # Ignore extremely short sections
            chunk_id = f"{plano_id}_p{pagina}_sec{i + 1}"
            chunks.append({
                "text": seccion_texto,
                "metadata": {
                    **metadata_base,
                    "chunk_type": "seccion",
                    "seccion_titulo": match.group(0).strip(),
                    "source": chunk_id,
                },
                "id": chunk_id
            })

    return chunks


def crear_chunks_inteligentes(
    documentos: List[Dict],
    incluir_tablas: bool = True,
    incluir_secciones: bool = False
) -> List[Dict]:
    """
    Creates intelligent chunks from page-level documents.

    Strategy:
    1. Each page serves as a base chunk (context).
    2. Detected tables are extracted as additional chunks (easier to retrieve).
    3. (Optional) Sections are extracted as additional chunks.

    Args:
        documentos: List of page-level documents (output from pdf_loader).
        incluir_tablas: Whether to extract tables as additional chunks.
        incluir_secciones: Whether to extract sections as additional chunks.

    Returns:
        Expanded list of chunks for indexing.
    """
    chunks_finales = []

    for doc in documentos:
        texto = doc["text"]
        meta = doc["metadata"]
        plano_id = meta["plano_id"]
        pagina = meta["pagina"]

        # Always include the full-page chunk for context
        chunks_finales.append(doc)

        # Extract tables as additional chunks
        if incluir_tablas:
            try:
                tabla_chunks = chunk_por_tablas(texto, meta, plano_id, pagina)
                chunks_finales.extend(tabla_chunks)
            except Exception as e:
                logger.warning(f"Error extracting tables from {plano_id} p{pagina}: {e}")

        # Extract sections as additional chunks
        if incluir_secciones:
            try:
                seccion_chunks = chunk_por_secciones(texto, meta, plano_id, pagina)
                chunks_finales.extend(seccion_chunks)
            except Exception as e:
                logger.warning(f"Error extracting sections from {plano_id} p{pagina}: {e}")

    logger.info(f"Chunks created: {len(chunks_finales)} (from {len(documentos)} pages)")
    return chunks_finales
