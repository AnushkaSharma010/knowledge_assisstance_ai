from typing import List
import PyPDF2
import pdfplumber
import pandas as pd
from PIL import Image
import io
import pytesseract
import fitz  # PyMuPDF
from core.captioner import GeminiMultimodalProcessor
from schemas import DocumentChunk
from logger import get_logger

logger = get_logger("PDFProcessor")

class PDFProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.captioner = GeminiMultimodalProcessor()
        self.doc = fitz.open(self.file_path)

    def process(self) -> List[DocumentChunk]:
        logger.info(f"Processing PDF file: {self.file_path}")
        chunks = []

        with pdfplumber.open(self.file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                logger.debug(f"Processing page {page_num}")

                # Extract text
                text = page.extract_text() or ""
                logger.debug(f"[PDF] Text length on page {page_num}: {len(text)}")

                # Extract tables
                tables = page.extract_tables()
                logger.debug(f"[PDF] Page {page_num} has {len(tables)} tables.")
                table_str = ""
                for table in tables:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_str += f"\n\nTable:\n{df.to_markdown()}"
                    except Exception as e:
                        logger.warning(f"Failed to parse table on page {page_num}: {e}")

                # Extract images
                images = self.doc[i].get_images(full=True)
                logger.debug(f"[PDF] Page {page_num} has {len(images)} images.")
                image_str = ""
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = self.doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        caption_obj = self.captioner.process(image_bytes, 'image')
                        caption = getattr(caption_obj, 'caption', '')
                        image_str += f"\n\nImage: {caption}"
                    except Exception as e:
                        logger.warning(f"Failed to extract image on page {page_num}: {e}")

                combined = text.strip() + table_str + image_str
                if combined.strip():
                    chunks.append(DocumentChunk(
                        content=combined.strip(),
                        type="text",
                        page_number=page_num
                    ))
                    logger.info(f"[PDF] Created chunk for page {page_num}, length: {len(combined.strip())}")
                else:
                    logger.warning(f"[PDF] No content extracted from page {page_num}")

        logger.info(f"Extracted {len(chunks)} unified chunks from PDF")
        return chunks
