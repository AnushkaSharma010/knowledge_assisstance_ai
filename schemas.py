from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Union
from datetime import datetime

class DocumentMetadata(BaseModel):  
    source: str
    created_at: datetime = datetime.now()
    file_type: Literal["pdf", "docx"]
    pages: Optional[int] = None


class DocumentChunk(BaseModel):
    content: str                                     # Pure combined text: paragraph + table (as text) + image (as caption text)
    type: Literal["text"]                            # Everything is embedded as text now
    page_number: Optional[int] = None
    uri: Optional[str] = None                        # Keep only if you still want to use image references
    doc_id: Optional[str] = None  
    metadata: Optional[Dict[str, Union[str, int, float]]] = None                   # Link to document


class DocumentSummary(BaseModel):
    doc_id: str
    summary: str                                     # Document-level summarization (optional)
    embedding: Optional[List[float]] = None          # For Level-1 semantic search



class UploadRequest(BaseModel):
    document_id: Optional[str] = None
    content: bytes
    file_type: Literal["pdf", "docx"]

class UploadResponse(BaseModel):
    document_id: str
    file_hash: str
    metadata: DocumentMetadata


class QueryRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    question: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Union[str, int]]]        # [{doc_id: str, page: int, type: str}]
    formatted_response: Optional[Dict] = None
