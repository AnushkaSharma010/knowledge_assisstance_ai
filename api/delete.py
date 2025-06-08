from fastapi import APIRouter, HTTPException
from core.chroma import ChromaClient
from logger import get_logger

logger = get_logger("delete")
router = APIRouter()

@router.delete("/delete/{document_id}")
async def delete_document(document_id: str):
    chroma = ChromaClient()
    try:
        deleted = chroma.delete_document(document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Document with id '{document_id}' not found.")
        return {"message": f"Document with id '{document_id}' deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete document.")
