from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse
from core.retrieval import MultimodalRetriever
from core.exceptions import RetrievalError, GenerationError
from config import settings
from typing import Optional
from logger import get_logger

logger = get_logger("query")

router = APIRouter()

@router.post("/query")
async def query_documents(
    request: QueryRequest,
    top_k: Optional[int] = 3,
) -> QueryResponse:
    """Handle multimodal document queries"""
   
    logger.info(f"Received query request: question='{request.question}', document_id='{request.document_id}', top_k={top_k}")
    
    if not request.question or not request.question.strip():
        logger.warning("Query failed: Empty or blank question received.")
        raise HTTPException(
            status_code=400,
            detail="Please ask a  question.!!"
        )
    
    if not request.document_id or not request.document_id.strip():
        logger.warning("Query failed: Missing or blank document ID.")
        raise HTTPException(
            status_code=400,
            detail="Document ID is required and cannot be blank."
        )

    try:
        retriever = MultimodalRetriever(
            top_k=top_k
        )
        # STEP 1: Correct the query using LLM
        corrected_query = await retriever.correct_query(request.question)
        if corrected_query != request.question:
            logger.info(f"Corrected query from '{request.question}' to '{corrected_query}'")

        
        response = await retriever.end_to_end_query(
            query= corrected_query,
            document_id=request.document_id
        )
        
        logger.info(f"Query successful, returning {len(response.answers) if hasattr(response, 'answers') else 'results'} results")
        return response
        
    except RetrievalError as e:
        logger.error(f"RetrievalError: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e))
    except GenerationError as e:
        logger.error(f"GenerationError: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}")
