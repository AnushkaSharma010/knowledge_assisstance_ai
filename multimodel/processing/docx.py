import docx
from docx.document import Document as DocxDocument
from PIL import Image
import io
import pandas as pd
from typing import List, Dict, Union
import os
import zipfile
import xml.etree.ElementTree as ET
from core.captioner import GeminiMultimodalProcessor
from schemas import DocumentChunk, TableMetadata, ImageMetadata
from logger import get_logger

logger = get_logger("DocxProcessor")

class DocxProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.document = docx.Document(file_path)
        self.captioner = GeminiMultimodalProcessor()
        self.content = {
            'text': '',
            'tables': [],
            'images': []
        }

    def extract_text(self) -> str:
        """Extract all text content from the DOCX file"""
        logger.debug("Extracting text from DOCX.")
        full_text = []
        for para in self.document.paragraphs:
            full_text.append(para.text)

        for table in self.document.tables:
            for row in table.rows:
                for cell in row.cells:
                    full_text.append(cell.text)

        self.content['text'] = '\n'.join(full_text)
        logger.info(f"Extracted text length: {len(self.content['text'])} characters.")
        return self.content['text']

    def extract_tables(self) -> List[Dict[str, Union[str, pd.DataFrame]]]:
        """Extract tables from DOCX and return with captions"""
        logger.debug("Extracting tables from DOCX.")
        tables = []
        for idx, table in enumerate(self.document.tables):
            data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                data.append(row_data)

            df = pd.DataFrame(data[1:], columns=data[0]) if data else pd.DataFrame()

            try:
                caption = self.captioner.describe_table(df.to_dict())
                logger.debug(f"Generated caption for table {idx}: {caption}")
            except Exception as e:
                logger.error(f"Error generating table caption for table {idx}: {e}", exc_info=True)
                caption = ""

            tables.append({
                'data': df,
                'caption': caption
            })

        self.content['tables'] = tables
        logger.info(f"Extracted {len(tables)} tables.")
        return tables

    def extract_images(self) -> List[Dict[str, Union[str, Image.Image]]]:
        """Extract images from DOCX file with captions"""
        logger.debug("Extracting images from DOCX.")
        images = []

        try:
            with zipfile.ZipFile(self.file_path) as z:
                with z.open('word/document.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    ns = {
                        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
                    }

                    # Collect all embedded image rIds
                    blips = root.findall('.//a:blip', ns)
                    logger.debug(f"Found {len(blips)} image references in document.xml.")

                    # Read relationships to map rId to image paths
                    with z.open('word/_rels/document.xml.rels') as rels_file:
                        rels_tree = ET.parse(rels_file)
                        rels_root = rels_tree.getroot()
                        nsmap = {'pr': 'http://schemas.openxmlformats.org/package/2006/relationships'}

                        rels = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels_root.findall('pr:Relationship', nsmap)}

                    for i, elem in enumerate(blips):
                        r_id = elem.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if not r_id or r_id not in rels:
                            logger.warning(f"Image reference with rId={r_id} not found in relationships.")
                            continue
                        image_path = f'word/{rels[r_id]}'

                        try:
                            with z.open(image_path) as img_file:
                                img_data = img_file.read()
                                img = Image.open(io.BytesIO(img_data))

                                try:
                                    caption_obj = self.captioner.process(img_data, 'image')
                                    caption = getattr(caption_obj, 'caption', '')
                                    logger.debug(f"Generated caption for image {i}: {caption}")
                                except Exception as e:
                                    logger.error(f"Error generating image caption for image {i}: {e}", exc_info=True)
                                    caption = ""

                                images.append({
                                    'image': img,
                                    'format': img.format,
                                    'size': img.size,
                                    'filename': os.path.basename(image_path),
                                    'caption': caption
                                })
                        except Exception as e:
                            logger.error(f"Error processing image {image_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to extract images from DOCX: {e}", exc_info=True)

        self.content['images'] = images
        logger.info(f"Extracted {len(images)} images.")
        return images

    def process(self) -> List[DocumentChunk]:
        logger.info(f"Starting processing of DOCX file: {self.file_path}")
        self.extract_text()
        self.extract_tables()
        self.extract_images()

        chunks = []

        # Text chunk
        if self.content['text']:
            chunks.append(DocumentChunk(
                content=self.content['text'],
                type="text",
                page_number=None
            ))

        # Tables
        for i, table in enumerate(self.content['tables']):
            df = table.get("data")
            markdown = df.to_markdown() if isinstance(df, pd.DataFrame) else ""
            chunks.append(DocumentChunk(
                content=table.get("caption", ""),
                type="table",
                page_number=None,
                metadata=TableMetadata(table_markdown=markdown)
            ))

        # Images
        for img in self.content['images']:
            size = img.get("size", ())
            width, height = (size + (0, 0))[:2]
            chunks.append(DocumentChunk(
                content=img.get("caption", ""),
                type="image",
                page_number=None,
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

