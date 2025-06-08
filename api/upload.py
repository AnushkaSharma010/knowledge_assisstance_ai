import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from schemas import DocumentMetadata
from core.chroma import ChromaClient
from multimodel.processing import ProcessorFactory
from core.exceptions import DocumentProcessingError
import hashlib
from config import settings
from logger import get_logger
from core.file_utils import compute_file_hash

logger = get_logger("upload")

router = APIRouter()

async def validate_file(file: UploadFile):
    """Validate file size and extension"""
    content = await file.read()
    logger.debug(f"Validating file {file.filename} with size {len(content)} bytes")

    if len(content) > settings.MAX_FILE_SIZE:
        logger.warning(f"File too large: {file.filename} ({len(content)} bytes), max allowed is {settings.MAX_FILE_SIZE}")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE} bytes"
        )

    ext = os.path.splitext(file.filename)[1].lower()
    supported_extensions = [".pdf", ".docx"]

    if ext not in supported_extensions:
        logger.warning(f"Unsupported file extension: {ext} for file {file.filename}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension: {ext}"
        )

    await file.seek(0)  # Reset file pointer for next read
    return ext, content


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    document_id: Optional[str] = None
):
    """Handle document upload and processing"""
    logger.info(f"Upload request received: filename={file.filename}, document_id={document_id}")
    try:
        ext, file_bytes = await validate_file(file)
        file_type = ext.lstrip(".")
        file_hash = compute_file_hash(file_bytes=file_bytes)
        doc_id = document_id or file_hash[:16]
        logger.info(f"Using document ID: {doc_id}")

        chroma = ChromaClient()
        if chroma.contains_file_hash(file_hash):
            logger.warning(f"Duplicate upload detected for file: {file.filename}")
            raise HTTPException(
                status_code=409,
                detail="This document has already been uploaded."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        logger.info(f"Saved uploaded file to temp path: {tmp_file_path}")

        processor = ProcessorFactory.get_processor(tmp_file_path)
        chunks = processor.process()

        if not chunks:
            logger.warning(f"No valid chunks extracted from file: {file.filename}")
            raise HTTPException(
                status_code=422,
                detail="No valid content found in the uploaded document."
            )

        logger.info(f"Processed document into {len(chunks)} valid chunks")

        chroma.add_documents(chunks, doc_id, file_hash=file_hash)
        logger.info(f"Added document chunks to ChromaDB with document ID: {doc_id}")

        page_count = sum(1 for c in chunks if hasattr(c, 'page_number'))
        logger.info(f"Document page count: {page_count}")

        return {
            "document_id": doc_id,
            "file_hash": file_hash,
            "metadata": DocumentMetadata(
                source=file.filename,
                file_type=file_type,
                pages=page_count
            )
        }

    except DocumentProcessingError as e:
        logger.error(f"Document processing error for file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during upload of file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
