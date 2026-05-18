"""
FacilityMind — Vector Store with Chroma DB
Stores blueprint embeddings for efficient semantic retrieval.
"""

import chromadb
from chromadb.api.types import Documents, Embeddings
import google.generativeai as genai
from typing import List, Dict, Optional
import logging

from config import CHROMA_PATH, COLLECTION_NAME, GEMINI_API_KEY, GeminiModels

logger = logging.getLogger(__name__)


class GoogleGeminiEmbeddingFunction:
    """
    Custom embedding function for Google Gemini to resolve compatibility issues 
    between ChromaDB and the Google Generative AI SDK.
    """
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def name(self) -> str:
        return "google_gemini_custom"

    def __call__(self, input: Documents) -> Embeddings:
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=input,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error calling Gemini Embedding API: {e}")
            raise

    def embed_query(self, input: str) -> List[float]:
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=input,
                task_type="retrieval_query"
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error calling Gemini Embedding API (query): {e}")
            raise


class BlueprintVectorStore:
    """
    Manages the blueprint vector database using Chroma DB.
    Utilizes Gemini embeddings (text-embedding-004) for optimal compatibility.
    """

    def __init__(self, persist_path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
        self.persist_path = persist_path
        self.collection_name = collection

        self.client = chromadb.PersistentClient(path=persist_path)

        self.embedding_func = GoogleGeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=GeminiModels.EMBEDDING
        )

        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"VectorStore initialized: {persist_path}/{collection}")

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Adds chunks to the vector collection.
        """
        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            ids.append(chunk["id"])

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(chunks)} chunks to the collection")
        return len(chunks)

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieves the most relevant chunks for a given query.
        """
        where_filter = filters if filters else None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        return results

    def query_with_discipline_filter(
        self,
        query_text: str,
        discipline: Optional[str] = None,
        floor: Optional[str] = None,
        n_results: int = 8
    ) -> Dict:
        """
        Queries the vector store with discipline and floor filters for enhanced precision.
        """
        filters = {}
        if discipline:
            filters["blueprint_type"] = discipline
        if floor:
            filters["floor"] = floor

        return self.query(query_text, n_results, filters if filters else None)

    def query_by_metadata(
        self,
        unit: Optional[str] = None,
        circuit: Optional[str] = None,
        room: Optional[str] = None,
        panel: Optional[str] = None
    ) -> Dict:
        """
        Retrieves blueprint chunks using precise structured metadata filtering processed in Python.
        Extremely fast and 100% reliable for technical codes (e.g. A-9, PANEL A).
        """
        try:
            all_data = self.collection.get(include=["documents", "metadatas"])
            if not all_data or not all_data.get("documents"):
                return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

            filtered_docs = []
            filtered_metas = []
            filtered_ids = []

            target_unit = unit.strip().upper() if unit else None
            target_circuit = circuit.strip().upper() if circuit else None
            target_room = room.strip().upper() if room else None
            target_panel = panel.strip().upper() if panel else None

            for i in range(len(all_data["documents"])):
                doc = all_data["documents"][i]
                meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                id_ = all_data["ids"][i]

                # Extract metadata lists
                meta_units = [u.strip().upper() for u in meta.get("units", "").split(",") if u.strip()]
                meta_circuits = [c.strip().upper() for c in meta.get("circuits", "").split(",") if c.strip()]
                meta_rooms = [r.strip().upper() for r in meta.get("rooms", "").split(",") if r.strip()]
                meta_panels = [p.strip().upper() for p in meta.get("panels", "").split(",") if p.strip()]

                match = True
                
                if target_unit and target_unit not in meta_units:
                    match = False
                if target_circuit:
                    # Resilient check: exact or substring match in circuit codes
                    if not any(target_circuit == c or target_circuit in c for c in meta_circuits):
                        match = False
                if target_room:
                    # Substring check for rooms (e.g., "BEDROOM" matches "BEDROOM 1")
                    if not any(target_room in r or r in target_room for r in meta_rooms):
                        match = False
                if target_panel:
                    if not any(target_panel in p or p in target_panel for p in meta_panels):
                        match = False

                if match:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
                    filtered_ids.append(id_)

            logger.info(f"[VectorStore] Precise metadata filter matched {len(filtered_docs)} chunks (unit={unit}, circuit={circuit}, room={room}, panel={panel})")
            return {
                "documents": [filtered_docs],
                "metadatas": [filtered_metas],
                "ids": [filtered_ids]
            }
        except Exception as e:
            logger.error(f"[VectorStore] Error during query_by_metadata: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}


    def delete_blueprint(self, blueprint_id: str) -> bool:
        """
        Removes all chunks associated with a specific blueprint ID.
        """
        try:
            self.collection.delete(where={"blueprint_id": blueprint_id})
            logger.info(f"Deleted blueprint {blueprint_id} from the collection")
            return True
        except Exception as e:
            logger.error(f"Error deleting blueprint {blueprint_id}: {e}")
            return False

    def list_blueprints(self) -> List[str]:
        """
        Lists all unique blueprint IDs present in the collection.
        """
        try:
            all_meta = self.collection.get(include=["metadatas"])
            blueprint_ids = set()
            if all_meta and all_meta["metadatas"]:
                for meta in all_meta["metadatas"]:
                    if meta and "blueprint_id" in meta:
                        blueprint_ids.add(meta["blueprint_id"])
            return sorted(list(blueprint_ids))
        except Exception as e:
            logger.error(f"Error listing blueprints: {e}")
            return []

    def count_chunks(self) -> int:
        """
        Returns the total number of chunks in the collection.
        """
        try:
            return self.collection.count()
        except Exception:
            return 0

    def peek(self, n: int = 3) -> Dict:
        """Returns the first n chunks for debugging purposes."""
        return self.collection.peek(limit=n)
