import re
from typing import Dict, Any, Optional
from schemas import QueryResponse
from logger import get_logger

logger = get_logger("MultimodalFormatter")

class MultimodalFormatter:
    """Formats raw Gemini outputs into structured responses"""

    @staticmethod
    def extract_tables(text: str) -> Optional[Dict[str, Any]]:
        """Convert markdown tables to structured format"""
        logger.debug("Extracting tables from Gemini output.")
        table_pattern = r"```markdown\n(.*?)\n```"
        tables = re.findall(table_pattern, text, re.DOTALL)

        if not tables:
            logger.info("No tables found in Gemini output.")
            return None

        logger.info(f"Extracted {len(tables)} table(s) from output.")
        return {
            "type": "table",
            "content": tables[0].strip(),
            "format": "markdown"
        }

    @staticmethod
    def extract_images(text: str) -> Optional[Dict[str, Any]]:
        """Detect image references in text"""
        logger.debug("Extracting image references from Gemini output.")
        img_pattern = r'\[Image:\s?(.*?)\]'
        images = re.findall(img_pattern, text)

        if not images:
            logger.info("No image references found in Gemini output.")
            return None

        logger.info(f"Extracted {len(images)} image reference(s).")
        return {
            "type": "image",
            "references": images,
            "content": text
        }

    @staticmethod
    def format_response(gemini_output: str) -> QueryResponse:
        """Convert raw LLM output to structured QueryResponse"""
        logger.debug("Formatting Gemini output into structured QueryResponse.")

        table_data = MultimodalFormatter.extract_tables(gemini_output)
        if table_data:
            logger.info("Formatted response as table.")
            return QueryResponse(
                answer=gemini_output,
                formatted_response=table_data
            )

        image_data = MultimodalFormatter.extract_images(gemini_output)
        if image_data:
            logger.info("Formatted response as image references.")
            return QueryResponse(
                answer=gemini_output,
                formatted_response=image_data
            )

        logger.info("Formatted response as plain text.")
        return QueryResponse(
            answer=gemini_output,
            formatted_response={"type": "text", "content": gemini_output}
        )

    @staticmethod
    def format_error(error: Exception) -> Dict[str, Any]:
        """Standardize error responses"""
        logger.error(f"Error formatting Gemini output: {error}")
        return {
            "type": "error",
            "error": str(error),
            "content": "Unable to process request"
        }
