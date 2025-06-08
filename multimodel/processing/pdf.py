import PyPDF2
import pdfplumber
import pandas as pd
from PIL import Image
from typing import List, Dict, Union
import io
import pytesseract
import fitz  # PyMuPDF
from core.captioner import GeminiMultimodalProcessor
from schemas import DocumentChunk, TableMetadata, ImageMetadata
from logger import get_logger

logger = get_logger("PDFProcessor")

class PDFProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.captioner = GeminiMultimodalProcessor()
        self.content = {
            'text': '',
            'tables': [],
            'images': []
        }

    def extract_text(self) -> str:
        """Extract text from PDF using PyPDF2 and OCR fallback"""
        logger.debug("Starting text extraction from PDF.")
        text = ""

        try:
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        logger.debug(f"No text found on page {i+1} using PyPDF2.")
        except Exception as e:
            logger.error(f"Error reading PDF with PyPDF2: {e}", exc_info=True)

        # OCR fallback if no text found
        if not text.strip():
            logger.info("No extracted text found, falling back to OCR.")
            try:
                doc = fitz.open(self.file_path)
                for i, page in enumerate(doc):
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text + "\n"
                    logger.debug(f"OCR extracted text from page {i+1}, length: {len(ocr_text)}")
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}", exc_info=True)

        self.content['text'] = text.strip()
        logger.info(f"Extracted text length: {len(self.content['text'])} characters.")
        return self.content['text']

    def extract_tables(self) -> List[Dict[str, Union[pd.DataFrame, str]]]:
        """Extract tables from PDF with captions using pdfplumber"""
        logger.debug("Starting table extraction from PDF.")
        tables = []

        try:
            with pdfplumber.open(self.file_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    extracted_tables = page.extract_tables()
                    logger.debug(f"Found {len(extracted_tables)} tables on page {page_number}.")
                    for i, table in enumerate(extracted_tables):
                        try:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            caption = self.captioner.describe_table(df.to_markdown())
                            tables.append({
                                'table': df,
                                'caption': caption
                            })
                            logger.debug(f"Caption generated for table {i+1} on page {page_number}.")
                        except Exception as e:
                            logger.error(f"Table caption generation failed on page {page_number}, table {i+1}: {e}", exc_info=True)
                            tables.append({
                                'table': df,
                                'caption': ''
                            })
        except Exception as e:
            logger.error(f"Failed to extract tables from PDF: {e}", exc_info=True)

        self.content['tables'] = tables
        logger.info(f"Extracted {len(tables)} tables in total.")
        return tables

    def extract_images(self) -> List[Dict[str, Union[Image.Image, str, tuple, int]]]:
        """Extract images from PDF using PyMuPDF and generate captions"""
        logger.debug("Starting image extraction from PDF using PyMuPDF.")
        images = []

        try:
            doc = fitz.open(self.file_path)

            for page_number in range(len(doc)):
                page = doc[page_number]
                image_list = page.get_images(full=True)

                logger.debug(f"Page {page_number + 1} has {len(image_list)} images.")

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        # Caption generation using Gemini
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format="JPEG")
                        buffer.seek(0)
                        caption_obj = self.captioner.process(buffer.read(), 'image')
                        caption = getattr(caption_obj, 'caption', '')

                        images.append({
                            'image': img_pil,
                            'format': image_ext,
                            'size': img_pil.size,
                            'page': page_number + 1,
                            'caption': caption
                        })
                        logger.debug(f"Caption generated for image {img_index + 1} on page {page_number + 1}.")
                    except Exception as e:
                        logger.error(f"Image processing failed on page {page_number + 1}, image {img_index + 1}: {e}", exc_info=True)
                        images.append({
                            'image': None,
                            'format': '',
                            'size': (),
                            'page': page_number + 1,
                            'caption': ''
                        })
        except Exception as e:
            logger.error(f"Failed to extract images from PDF with PyMuPDF: {e}", exc_info=True)

        self.content['images'] = images
        logger.info(f"Extracted {len(images)} images in total.")
        return images

    def process(self) -> List[DocumentChunk]:
        logger.info(f"Starting full processing of PDF file: {self.file_path}")
        self.extract_text()
        self.extract_tables()
        self.extract_images()

        chunks = []

        # Add text chunk
        if self.content['text']:
            chunks.append(DocumentChunk(
                content=self.content['text'],
                type="text",
                page_number=None
            ))

        # Add table chunks
        for i, table in enumerate(self.content['tables']):
            if isinstance(table.get("table"), pd.DataFrame):
                markdown = table["table"].to_markdown()
            else:
                markdown = ""
            chunks.append(DocumentChunk(
                content=table.get("caption", ""),
                type="table",
                page_number=None,
                metadata=TableMetadata(table_markdown=markdown)
            ))

        # Add image chunks
        for img in self.content['images']:
            size = img.get("size", ())
            width, height = (size + (0, 0))[:2]
            if width == 0 or height == 0 or not img.get("caption"):
                continue 
            chunks.append(DocumentChunk(
                content=img.get("caption", ""),
                type="image",
                page_number=img.get("page"),
                metadata=ImageMetadata(
                    width=width,     # extract width from size tuple
                    height=height,    # extract height from size tuple
                    format=img.get("format"),
                    caption=img.get("caption", None)
                ),
                uri=img.get("filename")
            ))

        logger.info(f"Total document chunks created: {len(chunks)}")
        return chunks

