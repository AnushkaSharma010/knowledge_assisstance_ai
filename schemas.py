from pydantic import BaseModel
from typing import Literal, Optional, List, Dict, Union
from datetime import datetime

class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    caption: Optional[str] = None

class TableMetadata(BaseModel):
    table_markdown: str


class DocumentMetadata(BaseModel):  
    source: str
    created_at: datetime = datetime.now()
    file_type: Literal["pdf", "docx"]
    pages: Optional[int] = None

class DocumentChunk(BaseModel):
    content: str             # Text, table description, or image caption
    type: Literal["text", "table", "image"]
    page_number: Optional[int] = None
    metadata: Optional[Union[Dict, ImageMetadata, TableMetadata]] = None
    uri: Optional[str] = None

class UploadRequest(BaseModel):
    document_id: Optional[str] = None  # Auto-generate if None
    content: bytes                     # Base64 encoded file
    file_type: Literal["pdf", "docx"]

class QueryRequest(BaseModel):
    document_id: Optional[str] = None
    question: Optional[str] = None
    include_images: Optional[bool] = True
    include_tables: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Union[str, int]]]  # {doc_id: str, page: int, type: str}
    formatted_response: Optional[Dict] = None  # For tables/images