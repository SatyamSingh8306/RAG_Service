import uuid
from typing import List, Dict, Any, Optional, Tuple
import logging

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
import os

# from backend.app.core.config import settings
from backend.app import EMBEDDING_MODEL_NAME,CHROMA_DB_PATH,CHROMA_COLLECTION_NAME

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreService:
    _instance = None
    _langchain_chroma_instance: Optional[Chroma] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        logger.info("Initializing VectorStoreService (Latest LangChain)...")

        try:
            logger.info(f"Loading HuggingFaceEmbeddings with model: {EMBEDDING_MODEL_NAME}")
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("HuggingFaceEmbeddings loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading HuggingFaceEmbeddings: {e}")
            self.embedding_function = None

        try:
            if self.embedding_function:
                logger.info(f"Initializing LangChain Chroma vector store at path: {CHROMA_DB_PATH}")
                logger.info(f"Using collection name: {CHROMA_COLLECTION_NAME}")
                self._langchain_chroma_instance = Chroma(
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                    persist_directory=CHROMA_DB_PATH
                )
                logger.info("LangChain Chroma vector store initialized successfully.")
            else:
                self._langchain_chroma_instance = None
                logger.warning("Embedding function not available, LangChain Chroma not initialized.")
        except Exception as e:
            logger.error(f"Error initializing LangChain Chroma vector store: {e}")
            self._langchain_chroma_instance = None

        # Initialize ChromaDB client with configuration from settings
        try:
            logger.info(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self._ensure_default_collection()
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            self.client = None

        self._initialized = True
        logger.info("VectorStoreService (Latest LangChain) initialization complete.")

    def _ensure_default_collection(self):
        """Ensure the default collection exists."""
        try:
            self.client.get_collection("default")
        except:
            self.client.create_collection("default")

    def create_collection(self, name: str) -> bool:
        """Create a new collection."""
        logger.info(f"Attempting to create collection: {name}")
        try:
            # Check if collection already exists
            try:
                logger.info(f"Checking if collection {name} already exists")
                self.client.get_collection(name)
                logger.warning(f"Collection {name} already exists")
                raise ValueError(f"Collection '{name}' already exists")
            except Exception as e:
                if "does not exists" in str(e).lower():
                    # This is what we want - collection doesn't exist, we can create it
                    logger.info(f"Collection {name} does not exist, proceeding with creation")
                else:
                    logger.error(f"Error checking collection existence: {e}")
                    raise e
            
            # Create the collection
            logger.info(f"Creating collection: {name}")
            self.client.create_collection(name)
            logger.info(f"Successfully created collection: {name}")
            return True
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Collection {name} already exists")
                raise ValueError(f"Collection '{name}' already exists")
            logger.error(f"Error creating collection {name}: {e}")
            raise e

    def list_collections(self) -> List[Tuple[str, int]]:
        """List all collections with their document counts."""
        collections = self.client.list_collections()
        return [(col.name, col.count()) for col in collections]

    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            return True
        except Exception as e:
            if "not found" in str(e):
                return False
            raise e

    def get_collection(self, name: str = "default"):
        """Get a collection by name."""
        return self.client.get_collection(name)

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], doc_ids: Optional[List[str]] = None, collection_name: Optional[str] = None) -> bool:
        """Add documents to the vector store."""
        if not chunks:
            print("No chunks provided to add.")
            return False
        if len(chunks) != len(metadatas):
            print("Error: The number of chunks and metadatas must be the same.")
            return False

        documents_to_add = []
        for i, chunk_text in enumerate(chunks):
            processed_metadata = {}
            for k, v in metadatas[i].items():
                if isinstance(v, (list, dict, tuple)):
                    processed_metadata[k] = str(v)
                elif v is None:
                    processed_metadata[k] = ""
                else:
                    processed_metadata[k] = v

            documents_to_add.append(
                LangchainDocument(page_content=chunk_text, metadata=processed_metadata)
            )

        if not doc_ids or len(doc_ids) != len(documents_to_add):
            doc_ids = [str(uuid.uuid4()) for _ in documents_to_add]

        try:
            # Use the specified collection or default to the configured one
            collection = collection_name or CHROMA_COLLECTION_NAME
            print(f"Adding {len(documents_to_add)} chunks to collection '{collection}'...")
            
            # Create a new Chroma instance for the specified collection
            chroma_instance = Chroma(
                collection_name=collection,
                embedding_function=self.embedding_function,
                persist_directory=CHROMA_DB_PATH
            )
            
            added_ids = chroma_instance.add_documents(
                documents=documents_to_add,
                ids=doc_ids
            )
            print(f"Successfully stored {len(added_ids)} chunks in collection '{collection}'.\n{'='*60}")
            return True
        except Exception as e:
            print(f"Error adding documents via LangChain Chroma: {e}")
            return False

    def query_documents_with_scores(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None) -> Optional[List[Tuple[LangchainDocument, float]]]:
        if not self._langchain_chroma_instance:
            print("Error: LangChain Chroma vector store not initialized.")
            return None

        try:
            # Use the specified collection name from the frontend
            if not collection_name:
                print("Error: No collection name provided")
                return None
                
            print(f"Querying LangChain Chroma collection '{collection_name}' with MMR (n_results={n_results}, filter={filter_dict})...")
            
            # Create a new Chroma instance for the specified collection
            chroma_instance = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=CHROMA_DB_PATH
            )
            
            # Debug: Check if collection exists and has documents
            collection_count = chroma_instance._collection.count()  # type: ignore
            print(f"Collection '{collection_name}' has {collection_count} documents")
            
            retriever = chroma_instance.as_retriever(
                search_type="mmr",
                search_kwargs={
                    'k': n_results,
                    'fetch_k': max(20, n_results * 3),
                    'lambda_mult': 0.7
                }
            )
            relevant_docs = retriever.get_relevant_documents(query_text, filter=filter_dict)
            print(f"MMR retrieved {len(relevant_docs)} docs. Reranking step would follow if implemented.")
            # If you have a reranker, you would pass relevant_docs to it here and get (doc, score) tuples.
            # For now, simulate scores as 0.0 for compatibility.
            mmr_results_with_simulated_scores = [(doc, 0.0) for doc in relevant_docs]
            return mmr_results_with_simulated_scores[:n_results]
        except Exception as e:
            print(f"Error querying LangChain Chroma with MMR: {e}")
            return None

    def get_collection_count(self) -> Optional[int]:
        if not self._langchain_chroma_instance:
            print("Error: LangChain Chroma vector store not initialized.")
            return None
        try:
            if hasattr(self._langchain_chroma_instance, '_collection') and self._langchain_chroma_instance._collection:
                return self._langchain_chroma_instance._collection.count()  # type: ignore
            print("Warning: Using less efficient method for collection count (getting all IDs).")
            chroma_collection = self._langchain_chroma_instance.get()
            return len(chroma_collection.get('ids', []))
        except Exception as e:
            print(f"Error getting LangChain Chroma collection count: {e}")
            return None

    def delete_documents(self, doc_ids: List[str]) -> bool:
        if not self._langchain_chroma_instance:
            print("Error: LangChain Chroma vector store not initialized.")
            return False
        if not doc_ids:
            print("No document IDs provided for deletion.")
            return False
        try:
            print(f"Attempting to delete {len(doc_ids)} documents from collection (IDs: {doc_ids})...")
            self._langchain_chroma_instance.delete(ids=doc_ids)
            print(f"Documents with IDs {doc_ids} delete call processed.")
            return True
        except Exception as e:
            print(f"Error or issue deleting documents from LangChain Chroma: {e}")
            return False

    def add_documents_to_collection(self, documents: List[Dict], collection_name: str = "default"):
        """Add documents to a collection."""
        collection = self.get_collection(collection_name)
        collection.add(
            documents=[doc["content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents],
            ids=[doc["id"] for doc in documents]
        )

    def query_collection(self, query_text: str, n_results: int = 5, collection_name: str = "default") -> List[Dict]:
        """Query documents from a collection."""
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return [
            {
                "id": id,
                "content": doc,
                "metadata": metadata,
                "distance": distance
            }
            for id, doc, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
