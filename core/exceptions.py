from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class APIError(BaseModel):
    detail: str
    code: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class DocumentProcessingError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )
        logger.error(f"Document Processing Error: {detail}")

class RetrievalError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail
        )
        logger.error(f"Retrieval Error: {detail}")

class GenerationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )
        logger.error(f"Generation Error: {detail}")

async def global_exception_handler(request, exc):
    """Centralized exception handler for FastAPI"""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIError(detail=exc.detail).model_dump()
        )
    
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIError(detail="Internal server error").model_dump()
    )