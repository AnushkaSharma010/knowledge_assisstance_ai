from typing import List, Dict, Any
from langchain.chains import retrieval_qa
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from schemas import DocumentChunk, QueryResponse
from .formatters import MultimodalFormatter
from core.exceptions import GenerationError
import google.generativeai as genai
from langchain.output_parsers import StructuredOutputParser
from logger import get_logger

logger = get_logger("MultimodalQAChain")

class MultimodalQAChain:
    """End-to-end RAG pipeline with multimodal support"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            convert_system_message_to_human=True
        )
        self._setup_prompts()
        logger.info("MultimodalQAChain initialized with Gemini LLM and retriever.")
    
    def _setup_prompts(self):
        """Define prompt templates for different content types"""
        logger.debug("Setting up prompt templates.")
        self.base_prompt = PromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            
            Rules:
            1. For tables → respond with markdown
            2. For images → use [Image: description]
            3. Cite sources as [Page X]"""
        )
        
        self.table_prompt = PromptTemplate.from_template(
            """Analyze this table data and answer the question:
            {table_data}
            
            Question: {question}
            
            Required:
            - Respond in markdown table format
            - Highlight key insights"""
        )
        
        self.image_prompt = PromptTemplate.from_template(
            """Describe this image context to answer the question:
            Image Caption: {image_caption}
            
            Question: {question}
            
            Required:
            - Use [Image: description] format
            - Mention visual details"""
        )
    
    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        """Convert retrieved chunks to context string"""
        logger.debug(f"Formatting context from {len(chunks)} chunks.")
        context_parts = []
        for chunk in chunks:
            if chunk.type == "text":
                context_parts.append(
                    f"TEXT (Page {chunk.page_number}):\n{chunk.content}"
                )
            elif chunk.type == "table":
                context_parts.append(
                    f"TABLE (Page {chunk.page_number}):\n{chunk.content}"
                )
            elif chunk.type == "image":
                context_parts.append(
                    f"IMAGE (Page {chunk.page_number}):\n{chunk.content}"
                )
        formatted_context = "\n\n---\n\n".join(context_parts)
        logger.debug("Context formatting complete.")
        return formatted_context
    
    async def invoke_chain(self, question: str, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Execute the appropriate processing chain"""
        logger.info(f"Invoking LLM chain for question: {question}")
        context = self._format_context(chunks)
        
        try:
            # Check if we should use specialized prompts
            if any(chunk.type == "table" for chunk in chunks):
                logger.info("Detected table chunk(s); using table prompt.")
                chain = (
                    {"table_data": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.table_prompt
                    | self.llm
                    | StructuredOutputParser()
                )
                result = await chain.ainvoke({
                    "table_data": context,
                    "question": question
                })
                logger.debug("Table prompt chain invocation successful.")
                return result
                
            elif any(chunk.type == "image" for chunk in chunks):
                logger.info("Detected image chunk(s); using image prompt.")
                chain = (
                    {"image_caption": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.image_prompt
                    | self.llm
                    | StructuredOutputParser()
                )
                result = await chain.ainvoke({
                    "image_caption": context,
                    "question": question
                })
                logger.debug("Image prompt chain invocation successful.")
                return result
                
            else:  # Default text processing
                logger.info("No tables or images detected; using base prompt.")
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.base_prompt
                    | self.llm
                    | StructuredOutputParser()
                )
                result = await chain.ainvoke({
                    "context": context,
                    "question": question
                })
                logger.debug("Base prompt chain invocation successful.")
                return result
        
        except Exception as e:
            logger.error(f"Error invoking LLM chain: {e}", exc_info=True)
            raise
    
    async def run(self, question: str, document_id: str) -> QueryResponse:
        """Execute full RAG pipeline"""
        logger.info(f"Running full RAG pipeline for question: {question}, document_id: {document_id}")
        try:
            # Retrieve relevant chunks
            chunks = await self.retriever.aretrieve(
                question=question,
                document_id=document_id
            )
            logger.info(f"Retrieved {len(chunks)} chunks.")
            
            # Generate answer
            raw_output = await self.invoke_chain(question, chunks[:3])  # Use top 3 chunks
            logger.info("LLM chain executed successfully, formatting response.")
            
            # Format response
            formatted_response = MultimodalFormatter.format_response(raw_output)
            logger.info("Response formatted successfully.")
            return formatted_response
            
        except Exception as e:
            logger.error(f"QA chain failed: {e}", exc_info=True)
            raise GenerationError(f"QA chain failed: {str(e)}")
