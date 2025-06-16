import os
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List
from schemas import DocumentMetadata
from core.chroma import ChromaClient
from multimodel.processing import ProcessorFactory
from core.exceptions import DocumentProcessingError
from core.file_utils import compute_file_hash
from config import settings
from logger import get_logger

logger = get_logger("upload")

router = APIRouter()

async def validate_file(file: UploadFile):
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(413, detail=f"File too large: {file.filename}")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.allowed_file_types:
        raise HTTPException(415, detail=f"Unsupported file extension: {ext}")

    await file.seek(0)
    return ext, content

@router.post("/upload")
async def upload_multiple_files(files: Optional[List[UploadFile]] = File(None)):
    if not files:
        raise HTTPException(400, detail="No files provided.")

    logger.info(f"Uploading {len(files)} document(s)...")
    chroma = ChromaClient()
    responses = []

    for file in files:
        try:
            ext, content = await validate_file(file)
            file_hash = compute_file_hash(content)
            document_id = file.filename.replace(" ", "_") + "_" + file_hash[:8]

            if chroma.contains_file_hash(file_hash):
                responses.append({
                    "document_id": document_id,
                    "status": "duplicate",
                    "message": f"{file.filename} already uploaded."
                })
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                file_path = tmp.name

            processor = ProcessorFactory.get_processor(file_path)
            chunks = processor.process()

            if not chunks:
                raise HTTPException(422, detail=f"No content extracted from {file.filename}")

            chroma.add_documents(chunks, document_id, file_hash=file_hash)
            logger.info(f"{file.filename} uploaded successfully → {len(chunks)} chunks.")

            responses.append({
                "document_id": document_id,
                "file_hash": file_hash,
                "status": "success",
                "metadata": DocumentMetadata(
                    source=file.filename,
                    file_type=ext.lstrip("."),
                    pages=len(chunks)
                ).model_dump()
            })

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            responses.append({
                "status": "error",
                "filename": file.filename,
                "message": str(e)
            })

    return {"results": responses}

