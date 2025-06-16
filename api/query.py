from fastapi import APIRouter, HTTPException
from schemas import QueryRequest, QueryResponse
from core.retrieval import MultimodalRetriever
from core.exceptions import RetrievalError, GenerationError
from logger import get_logger
from typing import Optional

router = APIRouter()
logger = get_logger("query")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, top_k: Optional[int] = 5) -> QueryResponse:
    """
    Handles multimodal question answering from uploaded documents.
    Supports optional document_ids for focused querying, else runs full semantic match.
    """

    logger.info(f"Received query: '{request.question}' | document_ids={request.document_ids} | top_k={top_k}")

    if not request.question or not request.question.strip():
        logger.warning("Query failed: Question is blank.")
        raise HTTPException(status_code=400, detail="Please enter a non-empty question.")

    try:
        retriever = MultimodalRetriever(top_k=top_k)
        corrected_query = await retriever.correct_query(request.question)

        if corrected_query != request.question:
            logger.info(f"Query corrected: {request.question} → {corrected_query}")

        response = await retriever.end_to_end_query(
            query=corrected_query,
            document_ids=request.document_ids
        )

        logger.info("Query completed successfully.")
        return response

    except RetrievalError as re:
        logger.error(f"Retrieval failed: {re}")
        raise HTTPException(status_code=404, detail=str(re))

    except GenerationError as ge:
        logger.error(f"Answer generation failed: {ge}")
        raise HTTPException(status_code=500, detail=str(ge))

    except Exception as e:
        logger.exception("Unexpected error during query.")
        raise HTTPException(status_code=500, detail="Unexpected error during query.")
