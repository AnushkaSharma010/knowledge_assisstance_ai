import chromadb
from typing import List, Dict, Optional
from schemas import DocumentChunk
from logger import get_logger
from core.embeddings import GeminiEmbeddings
from config import settings

logger = get_logger("ChromaClient")

MAX_CHUNK_LENGTH = 30000  # Gemini API max safe input length

class ChromaClient:
    def __init__(self, collection_name: str = settings.CHROMA_COLLECTION):
        logger.info(f"Initializing ChromaClient with collection: {collection_name}")
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.embedding_fn = GeminiEmbeddings() 
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_fn
        )
        logger.info("Chroma collection ready")

    def contains_file_hash(self, file_hash: str) -> bool:
        results = self.collection.get(where={"file_hash": file_hash}, limit=1)
        return len(results["ids"]) > 0

    def clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}
        for k, v in metadata.items():
            if v is None:
                cleaned[k] = "unknown"
            elif isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                cleaned[k] = v if v else ["unknown"]
            else:
                cleaned[k] = str(v)
        return cleaned

    def split_large_chunks(self, chunks: List[DocumentChunk], max_len: int = MAX_CHUNK_LENGTH) -> List[DocumentChunk]:
        new_chunks = []
        for chunk in chunks:
            content = chunk.content
            if len(content) > max_len:
                parts = [content[i:i+max_len] for i in range(0, len(content), max_len)]
                for idx, part in enumerate(parts):
                    part_id = f"{chunk.id}_part{idx}" if hasattr(chunk, 'id') and chunk.id else None
                    # Create new chunk with same metadata and updated content/id
                    new_chunk = DocumentChunk(
                        content=part,
                        metadata=chunk.metadata,
                        id=part_id,
                        type=chunk.type,
                        page_number=getattr(chunk, "page_number", -1),
                    )
                    new_chunks.append(new_chunk)
            else:
                new_chunks.append(chunk)
        return new_chunks

    def add_documents(
        self, 
        chunks: List[DocumentChunk], 
        document_id: str, 
        file_hash: Optional[str] = None
    ):
        logger.info(f"Adding {len(chunks)} document chunks for document_id: {document_id}")
        try:
            if not chunks:
                logger.warning("No chunks provided to add_documents")
                return

            # Split large text/table chunks first
            text_table_chunks = [c for c in chunks if c.type in ("text", "table")]
            text_table_chunks = self.split_large_chunks(text_table_chunks, MAX_CHUNK_LENGTH)

            # Separate image chunks (images not split)
            image_chunks = [c for c in chunks if c.type == "image"]

            # Add text/table chunks
            if text_table_chunks:
                ids = [f"{document_id}_txt_{i}" for i in range(len(text_table_chunks))]
                documents = [chunk.content for chunk in text_table_chunks]

                # Generate embeddings for text/table chunks
                embeddings = self.embedding_fn.embed_documents(documents)

                metadatas = []
                for chunk in text_table_chunks:
                    metadata = {
                        "type": chunk.type or "unknown",
                        "page": chunk.page_number if chunk.page_number is not None else -1,
                        "doc_id": document_id
                    }
                    if chunk.metadata:
                        if hasattr(chunk.metadata, "model_dump"):
                            raw_metadata = chunk.metadata.model_dump()
                        else:
                            raw_metadata = dict(chunk.metadata)
                        if file_hash:
                            raw_metadata["file_hash"] = file_hash
                        metadata.update(self.clean_metadata(raw_metadata))
                    else:
                        if file_hash:
                            metadata["file_hash"] = file_hash
                    metadatas.append(metadata)

                assert len(ids) == len(embeddings) == len(documents) == len(metadatas), \
                    f"Length mismatch in text/table chunks: ids={len(ids)}, embeddings={len(embeddings)}, documents={len(documents)}, metadatas={len(metadatas)}"

                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )

            # Add image chunks (no splitting, embeddings from uri/filename text)
            if image_chunks:
                ids = [f"{document_id}_img_{i}" for i in range(len(image_chunks))]

                uris = []
                image_texts_for_embedding = []
                metadatas = []

                for chunk in image_chunks:
                    metadata = {
                        "type": "image",
                        "page": chunk.page_number if chunk.page_number is not None else -1,
                        "doc_id": document_id
                    }
                    if chunk.metadata:
                        if hasattr(chunk.metadata, "model_dump"):
                            raw_metadata = chunk.metadata.model_dump()
                        else:
                            raw_metadata = dict(chunk.metadata)
                    else:
                        raw_metadata = {}

                    if file_hash:
                        raw_metadata["file_hash"] = file_hash

                    metadata.update(self.clean_metadata(raw_metadata))

                    uri = raw_metadata.get("uri") or raw_metadata.get("filename") or ""
                    uris.append(uri)
                    metadatas.append(metadata)

                    image_texts_for_embedding.append(uri or "image")

                embeddings = self.embedding_fn.embed_documents(image_texts_for_embedding)

                assert len(ids) == len(embeddings) == len(metadatas), \
                    f"Length mismatch in image chunks: ids={len(ids)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}"

                use_uris = all(uri and uri.strip() for uri in uris)

                if use_uris:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        uris=uris,
                        metadatas=metadatas
                    )
                else:
                    logger.warning("Some image URIs missing or empty; adding without URIs")
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas
                    )

            logger.info("Documents added successfully to ChromaDB")

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}", exc_info=True)
            raise

    def query(
    self,
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    document_id: Optional[str] = None,
    filter_types: Optional[List[str]] = None,
    n_results: int = 5,
) -> List[Dict]:
        logger.info(f"Querying ChromaDB for document_id: {document_id} | top_k={n_results}")
        try:
            filter_criteria = {}
            if document_id:
                filter_criteria["doc_id"] = document_id
            if filter_types:
                filter_criteria["type"] = {"$in": filter_types}

            if not query_text and not query_embedding:
                raise ValueError("Either query_text or query_embedding must be provided")

            query_args = {
                "where": filter_criteria,
                "n_results": n_results
            }
            if query_text:
                query_args["query_texts"] = [query_text]
            if query_embedding:
                query_args["query_embeddings"] = [query_embedding]

            results = self.collection.query(**query_args)

            logger.info(f"Query returned {len(results['ids'][0])} results")
            print("RESULT KEYS:", results.keys())
            print("DOCUMENTS:", results.get("documents"))
            print("IDS:", results.get("ids"))

            distances = (results.get("distances") or [[]])[0]
            ids = (results.get("ids") or [[]])[0]
            documents = (results.get("documents") or [[]])[0]
            metadatas = (results.get("metadatas") or [[]])[0]

            logger.info("Raw result scores before thresholding:")
            results_list = []

            for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
                # Use caption if document is None and caption is available
                if doc is None:
                    if meta and isinstance(meta, dict) and "caption" in meta:
                        doc = meta["caption"]
                    else:
                        logger.warning(f"Skipping result ID {id_} due to None document and no caption")
                        continue

                score = 1 - dist
                doc_preview = doc[:100] if doc else "[No document text]"
                logger.info(f"ID: {id_} | Score: {score:.4f} | Preview: {doc_preview}")

                results_list.append({
                    "id": id_,
                    "document": doc,
                    "metadata": meta,
                    "score": score
                })

            logger.info(f"{len(results_list)} results returned without threshold filtering")
            return results_list

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise


    def delete_document(self, document_id: str) -> bool:
        logger.info(f"Deleting document chunks for document_id: {document_id}")
        try:
            filter_criteria = {"doc_id": document_id}
            results = self.collection.query(
                where=filter_criteria,
                n_results=1000,
                include=["ids"]
            )
            chunk_ids = results.get("ids", [[]])[0]

            if not chunk_ids:
                logger.warning(f"No document chunks found for document_id: {document_id}")
                return False

            for chunk_id in chunk_ids:
                self.collection.delete(ids=[chunk_id])

            logger.info(f"Deleted {len(chunk_ids)} chunks for document_id: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document_id {document_id}: {str(e)}")
            raise
