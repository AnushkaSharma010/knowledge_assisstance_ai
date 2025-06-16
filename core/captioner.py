import google.generativeai as genai
from typing import Union, Dict, Any
from PIL import Image
import io
from config import settings
from core.exceptions import DocumentProcessingError
from logger import get_logger

logger = get_logger("GeminiMultimodalProcessor")


class GeminiMultimodalProcessor:
    """
    Processes images and tables using Gemini 1.5 Flash.
    Returns plain descriptive text for semantic embedding into chunks.
    """

    def __init__(self):
        logger.info("Initializing GeminiMultimodalProcessor...")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self._configure_prompts()
        logger.info("Gemini model initialized.")

    def _configure_prompts(self):
        """Prepare prompt templates for image and table processing"""
        self.image_prompt = [
            "Describe this image thoroughly so it can be understood in a text-only context.\n"
            "Mention: main objects, layout, purpose, any visible text or numbers.",
            "{image}"
        ]

        self.table_prompt = (
            "Analyze the following table and describe its structure, purpose, and key data trends:\n\n"
            "{table_data}"
        )
        logger.debug("Captioner prompts configured.")

    def process_image(self, image_bytes: bytes) -> str:
        """
        Generate a textual caption for an image.
        Returns plain text.
        """
        try:
            logger.info("Processing image to generate caption...")
            img = Image.open(io.BytesIO(image_bytes))
            logger.debug(f"Image loaded: format={img.format}, size={img.size}")

            response = self.model.generate_content(
                contents=self.image_prompt[:-1] + [img]
            )
            caption = response.text.strip()
            logger.info("Image captioning complete.")
            return caption

        except Exception as e:
            logger.error(f"Image captioning failed: {str(e)}", exc_info=True)
            raise DocumentProcessingError(f"Image captioning failed: {str(e)}")

    def process_table(self, table_data: Union[Dict[str, Any], str]) -> str:
        """
        Generate a textual description for a table.
        Returns plain text.
        """
        try:
            logger.info("Processing table to generate description...")
            if isinstance(table_data, dict):
                content = str(table_data)
            else:
                content = table_data

            prompt = self.table_prompt.format(table_data=content)
            response = self.model.generate_content(prompt)
            description = response.text.strip()
            logger.info("Table description complete.")
            return description

        except Exception as e:
            logger.error(f"Table description failed: {str(e)}", exc_info=True)
            raise DocumentProcessingError(f"Table description failed: {str(e)}")

    def process(self, content: bytes, content_type: str) -> str:
        """
        Unified processor for both image and table content.
        Returns caption/description in plain text.
        """
        logger.info(f"Running Gemini processor for type: {content_type}")
        if content_type == "image":
            return self.process_image(content)
        elif content_type == "table":
            return self.process_table(content)
        else:
            logger.error(f"Unsupported content type: {content_type}")
            raise DocumentProcessingError(f"Unsupported content type: {content_type}")

