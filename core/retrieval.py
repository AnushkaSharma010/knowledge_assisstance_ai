from typing import List, Optional, Dict, Tuple
import google.generativeai as genai
from config import settings
from schemas import DocumentChunk, QueryResponse
from core.chroma import ChromaClient
from core.embeddings import GeminiEmbeddings
from core.exceptions import RetrievalError, GenerationError
from logger import get_logger

logger = get_logger("MultimodalRetriever")


class MultimodalRetriever:
    def __init__(self, top_k: int = 5, top_docs: int = 3):
        logger.info("Initializing MultimodalRetriever")
        self.db = ChromaClient()
        self.embeddings = GeminiEmbeddings()
        self.generation_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.top_k = top_k
        self.top_docs = top_docs
        self.cache: Dict[Tuple[str, str], QueryResponse] = {}

    def _normalize_question(self, question: str) -> str:
        return question.strip().lower()

    async def correct_query(self, query: str) -> str:
        prompt = f"""
        You are an intelligent assistant that corrects typos and grammar in user questions.
        Return the corrected version of this question:

        Original: {query}
        """
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Query correction failed, using original query: {e}")
            return query

    def _format_results(self, results: List[Dict]) -> List[DocumentChunk]:
        valid_chunks = []
        for doc in results:
            content = doc.get("document")
            metadata = doc.get("metadata", {})
            if isinstance(content, str) and content.strip():
                valid_chunks.append(
                    DocumentChunk(
                        content=content,
                        type="text",
                        page_number=metadata.get("page", -1),
                        metadata=metadata
                    )
                )
            else:
                logger.warning(f"Skipping chunk with empty or invalid content: id={doc.get('id')}")
        return valid_chunks


    def get_top_documents(self, query_embedding: List[float]) -> List[str]:
        logger.info("Running document-level retrieval")
        results = self.db.query(
            query_embedding=query_embedding,
            n_results=self.top_docs
        )

        doc_scores: Dict[str, float] = {}
        for r in results:
            doc_id = r["metadata"].get("doc_id")
            score = r["score"]
            if doc_id:
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score

        top_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Top matching documents: {[doc[0] for doc in top_doc_ids]}")
        return [doc[0] for doc in top_doc_ids]

    def get_top_chunks_from_documents(self, query_embedding: List[float], doc_ids: List[str]) -> List[DocumentChunk]:
        logger.info(f"Retrieving top chunks from filtered documents: {doc_ids}")
        all_chunks = []
        for doc_id in doc_ids:
            results = self.db.query(
                query_embedding=query_embedding,
                document_id=doc_id,
                n_results=self.top_k
            )
            all_chunks.extend(results)

        sorted_chunks = sorted(all_chunks, key=lambda x: x["score"], reverse=True)
        top_chunks = self._format_results(sorted_chunks[:self.top_k])
        logger.info(f"Selected top {len(top_chunks)} chunks for generation")
        return top_chunks

    async def retrieve(self, query: str, document_ids: Optional[List[str]] = None) -> List[DocumentChunk]:
        logger.info(f"Retrieving relevant chunks for query: '{query}'")
        try:
            query_embedding = self.embeddings.embed_query(query)
            if document_ids:
                logger.info(f"Using user-specified document_ids: {document_ids}")
                doc_ids = document_ids
            else:
                doc_ids = self.get_top_documents(query_embedding)

            top_chunks = self.get_top_chunks_from_documents(query_embedding, doc_ids)
            return top_chunks

        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    async def generate_response(self, query: str, chunks: List[DocumentChunk]) -> QueryResponse:
        logger.info(f"Generating response from {len(chunks)} chunks")
        if not chunks:
            raise GenerationError("No relevant context found to answer the question.")

        try:
            context_str = "\n\n".join([
                f"[Page {chunk.page_number}]\n{chunk.content}" for chunk in chunks
            ])

            prompt = f"""
            You are a helpful assistant. Use the context below to answer the question.

            Question: {query}

            Context:
            {context_str}

            Instructions:
            - If any tables are mentioned, output them in markdown format.
            - Refer to any described images using [Image: description].
            - Use [Page X] to refer to pages if helpful.
            """

            response = self.generation_model.generate_content(prompt)

            sources = [
                {
                    "doc_id": chunk.metadata.get('doc_id', 'unknown'),
                    "page": chunk.page_number,
                    "type": chunk.type
                } for chunk in chunks
            ]

            return QueryResponse(
                answer=response.text.strip(),
                sources=sources,
                formatted_response=None
            )

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}", exc_info=True)
            raise GenerationError(f"Response generation failed: {str(e)}")

    async def end_to_end_query(self, query: str, document_ids: Optional[List[str]] = None) -> QueryResponse:
        logger.info("End-to-end query started")
        normalized_q = self._normalize_question(query)
        cache_key = (",".join(document_ids) if document_ids else "ALL_DOCS", normalized_q)

        if cache_key in self.cache:
            logger.info("Using cached response")
            return self.cache[cache_key]

        corrected_query = await self.correct_query(query)
        chunks = await self.retrieve(corrected_query, document_ids=document_ids)
        response = await self.generate_response(corrected_query, chunks)
        self.cache[cache_key] = response
        logger.info("End-to-end query completed")
        return response
