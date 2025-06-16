from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.chroma import ChromaClient
from logger import get_logger

logger = get_logger("delete")
router = APIRouter()

class DeleteRequest(BaseModel):
    document_id: str

@router.delete("/delete")
async def delete_document(request: DeleteRequest):
    """
    Delete all document chunks for a given document ID from ChromaDB.
    """
    logger.info(f"Received delete request for document_id: {request.document_id}")
    try:
        chroma = ChromaClient()
        deleted = chroma.delete_document(request.document_id)

        if not deleted:
            logger.warning(f"No document found with ID: {request.document_id}")
            raise HTTPException(status_code=404, detail=f"No document found with ID: {request.document_id}")

        logger.info(f"Successfully deleted all chunks for document ID: {request.document_id}")
        return {"message": f"Document '{request.document_id}' deleted successfully."}

    except Exception as e:
        logger.error(f"Failed to delete document ID {request.document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")
