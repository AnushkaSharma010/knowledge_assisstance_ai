import google.generativeai as genai
from typing import Any, Dict, Optional, Union
from PIL import Image
import io
from config import settings
from schemas import ImageMetadata
from core.exceptions import DocumentProcessingError
from logger import get_logger

logger = get_logger("GeminiMultimodalProcessor")

class GeminiMultimodalProcessor:
    """
    Unified processor for handling both images and tables using Gemini 1.5 Flash.
    Features:
    - Image captioning with detailed metadata
    - Table analysis with semantic descriptions
    - Robust error handling
    - Configurable prompts
    """
    
    def __init__(self):
        logger.info("Configuring Gemini API and model")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self._configure_prompts()
        logger.info("GeminiMultimodalProcessor initialized")

    def _configure_prompts(self):
        """Centralized prompt templates for consistent outputs"""
        self.image_prompt = [
            "Analyze this image and generate a comprehensive caption including:",
            "1. Main subjects and their relationships",
            "2. Contextual elements and environment",
            "3. Any visible text or numerical data",
            "4. Overall purpose/meaning of the image",
            "{image}"
        ]
        
        self.table_prompt = """
        Analyze this table and generate a detailed description covering:
        1. Purpose and context of the table
        2. Column meanings and relationships
        3. Key trends/patterns in the data
        4. Notable outliers or important values

        Table Structure:
        {table_data}
        """
        logger.debug("Prompts configured")

    def process_image(self, image_bytes: bytes) -> ImageMetadata:
        """
        Process an image and generate comprehensive metadata.
        """
        try:
            logger.info("Processing image content")
            img = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"Image opened: format={img.format}, size={img.size}")

            response = self.model.generate_content(
                contents=self.image_prompt[:-1] + [img]
            )
            logger.info("Image caption generated")

            return ImageMetadata(
                width=img.width,
                height=img.height,
                format=img.format,
                caption=response.text
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise DocumentProcessingError(
                f"Image processing failed: {str(e)}"
            )

    def describe_table(self, table_data: Union[Dict[str, Any], str]) -> str:
        """
        Generate semantic description of tabular data.
        """
        try:
            logger.info("Describing table content")
            response = self.model.generate_content(
                self.table_prompt.format(table_data=table_data)
            )
            logger.info("Table description generated")
            return response.text
            
        except Exception as e:
            logger.error(f"Table analysis failed: {str(e)}")
            raise DocumentProcessingError(
                f"Table analysis failed: {str(e)}"
            )

    def process(self, content: bytes, content_type: str) -> Union[ImageMetadata, str]:
        """
        Unified processing interface for both images and tables.
        """
        logger.info(f"Processing content of type: {content_type}")
        if content_type == 'image':
            return self.process_image(content)
        elif content_type == 'table':
            return self.describe_table(content)
        else:
            logger.error(f"Unsupported content type: {content_type}")
            raise DocumentProcessingError(
                f"Unsupported content type: {content_type}"
            )
