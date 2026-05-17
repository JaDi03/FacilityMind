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
            # For a list of documents, it returns a list of embeddings.
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
        """Alias for embedding a single query string."""
        # For a single query, we use retrieval_query task type
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


class PlanoVectorStore:
    """
    Manages the blueprint vector database using Chroma DB.
    Utilizes Gemini embeddings (text-embedding-004) for optimal compatibility.
    """

    def __init__(self, persist_path: str = CHROMA_PATH, collection: str = COLLECTION_NAME):
        self.persist_path = persist_path
        self.collection_name = collection

        # Persistent Chroma client
        self.client = chromadb.PersistentClient(path=persist_path)

        # Gemini embedding function (Custom implementation to avoid ChromaDB bugs)
        self.embedding_func = GoogleGeminiEmbeddingFunction(
            api_key=GEMINI_API_KEY,
            model_name=GeminiModels.EMBEDDING
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic embeddings
        )

        logger.info(f"VectorStore initialized: {persist_path}/{collection}")

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Adds chunks to the vector collection.

        Args:
            chunks: List of dicts with keys: text, metadata, id.

        Returns:
            Number of chunks successfully added.
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

        # Chroma handles embeddings automatically via the specified embedding function
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

        Args:
            query_text: Query string.
            n_results: Number of results to return.
            filters: Optional metadata filters (e.g., {"tipo_plano": "electrical"}).

        Returns:
            Chroma retrieval results (documents, metadatas, distances, ids).
        """
        where_filter = filters if filters else None

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        return results

    def query_con_filtro_disciplina(
        self,
        query_text: str,
        disciplina: Optional[str] = None,
        piso: Optional[str] = None,
        n_results: int = 8
    ) -> Dict:
        """
        Queries the vector store with discipline and floor filters for enhanced precision.
        """
        filters = {}
        if disciplina:
            filters["tipo_plano"] = disciplina
        if piso:
            filters["piso"] = piso

        return self.query(query_text, n_results, filters if filters else None)

    def delete_plano(self, plano_id: str) -> bool:
        """
        Removes all chunks associated with a specific blueprint ID.
        """
        try:
            self.collection.delete(where={"plano_id": plano_id})
            logger.info(f"Deleted blueprint {plano_id} from the collection")
            return True
        except Exception as e:
            logger.error(f"Error deleting blueprint {plano_id}: {e}")
            return False

    def listar_planos(self) -> List[str]:
        """
        Lists all unique blueprint IDs present in the collection.
        """
        try:
            all_meta = self.collection.get(include=["metadatas"])
            plano_ids = set()
            if all_meta and all_meta["metadatas"]:
                for meta in all_meta["metadatas"]:
                    if meta and "plano_id" in meta:
                        plano_ids.add(meta["plano_id"])
            return sorted(list(plano_ids))
        except Exception as e:
            logger.error(f"Error listing blueprints: {e}")
            return []

    def contar_chunks(self) -> int:
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

    # English Aliases for Naming Consistency
    def count_chunks(self) -> int:
        return self.contar_chunks()

    def list_blueprints(self) -> List[str]:
        return self.listar_planos()

    def delete_blueprint(self, blueprint_id: str) -> bool:
        return self.delete_plano(blueprint_id)

    def query_with_discipline_filter(
        self,
        query_text: str,
        discipline: Optional[str] = None,
        floor: Optional[str] = None,
        n_results: int = 8
    ) -> Dict:
        return self.query_con_filtro_disciplina(query_text, discipline, floor, n_results)


# English Class Alias for Global Naming Conventions
BlueprintVectorStore = PlanoVectorStore

