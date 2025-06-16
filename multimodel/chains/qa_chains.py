from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from schemas import DocumentChunk, QueryResponse
from .formatters import MultimodalFormatter
from core.exceptions import GenerationError
from langchain.output_parsers import StrOutputParser
from logger import get_logger

logger = get_logger("MultimodalQAChain")

class MultimodalQAChain:
    """End-to-end QA chain using unified text chunks with Gemini + LangChain"""

    def __init__(self):
        logger.info("Initializing QA Chain with Gemini + LangChain")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )
        self.prompt = PromptTemplate.from_template(self._build_prompt_template())
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=StrOutputParser())
        logger.info("QA Chain ready.")

    def _build_prompt_template(self) -> str:
        """Constructs the QA prompt"""
        return """
        You are a helpful assistant. Use the following context to answer the user's question.

        Context:
        {context}

        Question: {question}

        Rules:
        - If tables are in the context, respond with a markdown table.
        - If images are described, refer to them using [Image: description].
        - Cite page numbers when available.
        - If answer is not found, say: "I'm sorry, I couldn't find that information in the document."
        """

    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context string from unified document chunks"""
        logger.debug("Formatting retrieved chunks into prompt context.")
        parts = []
        for chunk in chunks:
            page = chunk.page_number or "?"
            parts.append(f"[Page {page}] {chunk.content.strip()}")
        context_str = "\n\n---\n\n".join(parts)
        logger.info(f"Formatted context of length {len(context_str)} characters.")
        return context_str

    async def run(self, question: str, chunks: List[DocumentChunk]) -> QueryResponse:
        """Execute the QA chain with retrieved chunks"""
        logger.info(f"Running QA chain for question: '{question}' on {len(chunks)} chunks")
        if not chunks:
            logger.warning("No chunks available for QA. Returning empty answer.")
            raise GenerationError("No content found to generate an answer.")

        try:
            context_str = self._format_context(chunks[:3])
            result = await self.chain.arun({
                "question": question,
                "context": context_str
            })
            logger.info("QA chain output received successfully.")
            formatted = MultimodalFormatter.format_response(result)
            return formatted

        except Exception as e:
            logger.error(f"Failed to execute QA chain: {e}", exc_info=True)
            raise GenerationError("Could not generate answer from document.")
