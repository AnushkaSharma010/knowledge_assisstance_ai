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
    def __init__(self, top_k: int = 5):
        logger.info("Initializing MultimodalRetriever")
        self.db = ChromaClient()
        self.embeddings = GeminiEmbeddings()
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        self.top_k = top_k
        self.cache: Dict[Tuple[str, str], QueryResponse] = {}  # (document_id, normalized_question) → response

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

    async def split_query_if_needed(self, query: str) -> List[str]:
        prompt = f"""
        The user may have asked a multi-part or long question. Split it into clear sub-questions, one per line. If it's a single simple question, return it as is.

        Question:
        {query}
        """
        try:
            response = self.generation_model.generate_content(prompt)
            sub_questions = [line.strip() for line in response.text.split("\n") if line.strip()]
            return sub_questions if sub_questions else [query]
        except Exception as e:
            logger.warning(f"Failed to split question: {e}. Proceeding with full query.")
            return [query]

    async def retrieve(
        self,
        query: str,
        document_id: str,
        filter_types: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        logger.info(f"Starting retrieval for query: '{query[:30]}...' and document_id: {document_id}")
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.db.query(
                query_text=query,
                query_embedding=query_embedding,
                document_id=document_id,
                filter_types=filter_types,
                n_results=self.top_k,
            )
            if not results:
                logger.warning("Chroma query returned no results.")
                return []

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    def _format_results(self, results: List[Dict]) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                content=item["document"],
                type=item["metadata"].get("type", "text"),
                page_number=item["metadata"].get("page", -1),
                metadata=item["metadata"]
            ) for item in results
        ]

    def _detect_special_content(self, text: str) -> Optional[Dict]:
        if "```markdown" in text:
            return {"type": "table", "content": text}
        elif "[Image:" in text:
            return {"type": "image", "content": text}
        return None

    def _filter_relevant_media(
        self,
        answer: str,
        media_chunks: List[DocumentChunk],
        top_k: int = 2
    ) -> List[DocumentChunk]:
        logger.info(f"[LLM Filtering] Filtering {len(media_chunks)} media chunks for relevance to the answer")

        prompt_template = """
            You are helping evaluate whether a media element (image or table) supports a given answer.
            Respond with a single word: "YES" if it is useful or supports the answer, or "NO" if it is unrelated.

            Answer:
            {answer}

            Media Content:
            {media_content}

            Does this media help explain or support the answer? YES or NO?
            """
        filtered = []
        for chunk in media_chunks:
            try:
                prompt = prompt_template.format(
                    answer=answer,
                    media_content=chunk.content[:1000]  # trim to avoid long payloads
                )
                response = self.generation_model.generate_content(prompt)
                result = response.text.strip().upper()

                logger.debug(f"[LLM Filtering] Media decision: {result} for chunk starting with {chunk.content[:30]}")
                if result.startswith("YES"):
                    filtered.append(chunk)
            except Exception as e:
                logger.warning(f"LLM filtering failed for media chunk: {e}")
                continue

        logger.info(f"[LLM Filtering] Retained {len(filtered)} media chunks after LLM filtering")
        return filtered[:top_k]

    async def generate_response(self, query: str, context_chunks: List[DocumentChunk]) -> QueryResponse:
        logger.info(f"Generating response for query: '{query[:30]}...' with {len(context_chunks)} context chunks")

        if not context_chunks:
            raise GenerationError("Your question doesn’t seem to be related to the uploaded document.")

        try:
            context_str = "\n\n".join([
                f"**{chunk.type.upper()}**:\n{chunk.content}"
                for chunk in context_chunks
            ])

            prompt = f"""
                You are a helpful assistant in a conversation with a user. Use the provided context to answer their question naturally and clearly.

                Speak in a friendly professional tone.

                Guidelines:
                - Mention sources using [Page X] if helpful.
                - If tables, use markdown formatting with ```markdown.
                - Refer to images as [Image: caption].
                - Avoid robotic repetition of the user's question.

                User’s Question: {query}

                Context:
                {context_str}
                """
            response = self.generation_model.generate_content(prompt)

            if not response.text or any(phrase in response.text.lower() for phrase in [
                "i don't know", "i'm not sure", "not enough information", "cannot determine"]):
                raise GenerationError("Sorry, no relevant answer could be found in the document.")

            sources = [
                {
                    "doc_id": chunk.metadata['doc_id'],
                    "page": chunk.page_number,
                    "type": chunk.type
                } for chunk in context_chunks
            ]

            return QueryResponse(
                answer=response.text.strip(),
                sources=sources,
                formatted_response=self._detect_special_content(response.text)
            )
        except Exception as e:
            raise GenerationError(f"Response generation failed: {str(e)}")

    async def end_to_end_query(self, query: str, document_id: str) -> QueryResponse:
        logger.info(f"Starting end-to-end query for document_id: {document_id}")

        normalized_q = self._normalize_question(query)
        cache_key = (document_id, normalized_q)

        if cache_key in self.cache:
            logger.info("Returning cached response.")
            return self.cache[cache_key]

        corrected_query = await self.correct_query(query)
        sub_questions = await self.split_query_if_needed(corrected_query)

        answers = []
        sources = []

        for sub_q in sub_questions:
            chunks = await self.retrieve(sub_q, document_id)
            try:
                sub_response = await self.generate_response(sub_q, chunks[:3])
                answers.append(sub_response.answer)
                sources.extend(sub_response.sources)
            except GenerationError as e:
                answers.append(str(e))

        final_answer = "\n\n".join(answers)

        final_response = QueryResponse(
            answer=final_answer,
            sources=sources,
            formatted_response=self._detect_special_content(final_answer)
        )

        self.cache[cache_key] = final_response
        logger.info("End-to-end query completed")
        return final_response
