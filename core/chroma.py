import chromadb
from typing import List, Dict, Optional
from schemas import DocumentChunk
from logger import get_logger
from core.embeddings import GeminiEmbeddings
from config import settings

logger = get_logger("ChromaClient")

MAX_CHUNK_LENGTH = 30000

class ChromaClient:
    def __init__(self, collection_name: str = settings.CHROMA_COLLECTION):
        logger.info(f"Initializing ChromaClient with collection: {collection_name}")
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.embedding_fn = GeminiEmbeddings()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def split_large_chunks(self, chunks: List[DocumentChunk], max_len: int = MAX_CHUNK_LENGTH) -> List[DocumentChunk]:
        new_chunks = []
        for chunk in chunks:
            content = chunk.content
            if len(content) > max_len:
                parts = [content[i:i+max_len] for i in range(0, len(content), max_len)]
                for idx, part in enumerate(parts):
                    new_chunks.append(DocumentChunk(
                        content=part,
                        type="text",
                        page_number=chunk.page_number,
                        doc_id=chunk.doc_id
                    ))
            else:
                new_chunks.append(chunk)
        return new_chunks

    def contains_file_hash(self, file_hash: str) -> bool:
        logger.debug(f"Checking for existing file hash: {file_hash}")
        results = self.collection.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0

    def add_documents(self, chunks: List[DocumentChunk], document_id: str, file_hash: Optional[str] = None):
        logger.info(f"Adding {len(chunks)} chunks for document_id={document_id}")
        try:
            chunks = self.split_large_chunks(chunks)
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            documents = [c.content for c in chunks]
            embeddings = self.embedding_fn.embed_documents(documents)

            metadatas = []
            for chunk in chunks:
                metadata = {
                    "type": "text",
                    "page": chunk.page_number if chunk.page_number is not None else -1,
                    "doc_id": document_id,
                }
                if file_hash:
                    metadata["file_hash"] = file_hash
                metadatas.append(metadata)

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info("Chunks successfully added to ChromaDB.")

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}", exc_info=True)
            raise

    def retrieve_relevant_documents(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        logger.info("Running document-level semantic search")
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        doc_ids = []
        for meta in results["metadatas"][0]:
            doc_id = meta.get("doc_id")
            if doc_id and doc_id not in doc_ids:
                doc_ids.append(doc_id)
        logger.debug(f"Relevant documents found: {doc_ids}")
        return doc_ids

    def query(
    self,
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    document_id: Optional[str] = None,
    filter_types: Optional[List[str]] = None,
    n_results: int = 5,
) -> List[Dict]:
            
        filters = {}

        if document_id:
            filters["doc_id"] = document_id
        if filter_types:
            filters["type"] = {"$in": filter_types}

        query_args = {
            "n_results": n_results
        }

        # ONLY add "where" if filters is not empty
        if filters:
            query_args["where"] = filters

        if query_text:
            query_args["query_texts"] = [query_text]
        elif query_embedding:
            query_args["query_embeddings"] = [query_embedding]
        else:
            raise ValueError("Either query_text or query_embedding must be provided.")

        logger.info(f"ChromaDB query: filters={filters if filters else 'None'}, top_k={n_results}")
        
        results = self.collection.query(**query_args)

        return [
            {
                "id": id_,
                "document": doc,
                "metadata": meta,
                "score": 1 - dist
            }
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def delete_document(self, document_id: str) -> bool:
        logger.info(f"Deleting chunks for document_id={document_id}")
        try:
            result = self.collection.query(
                where={"doc_id": document_id},
                n_results=1000,
                include=["ids"]
            )
            ids = result.get("ids", [[]])[0]
            if not ids:
                logger.warning("No chunks found for deletion.")
                return False

            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks.")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document: {e}", exc_info=True)
            raise
