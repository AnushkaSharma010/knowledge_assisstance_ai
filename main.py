import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from config import settings
from core import initialize_services
from core.exceptions import global_exception_handler
from api import upload, query, delete

# Setup logger
logger = logging.getLogger("multimodal_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Multimodal Document Assistant API...")
    try:
        initialize_services()
        logger.info("Services initialized successfully.")
        yield
    finally:
        logger.info("Cleaning up resources before shutdown.")


app = FastAPI(
    title="Multimodal Document Assistant API",
    description="API for processing documents with text, tables and images",
    version="1.0.0",
    docs_url=None,  # Disable Swagger UI
    redoc_url=None,  # Disable Redoc
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)

app.include_router(upload.router, prefix="/v1")
app.include_router(query.router, prefix="/v1")
app.include_router(delete.router, prefix="/v1")


@app.get("/health")
async def health():
    logger.info("Health check requested.")
    return {"status": "ready", "services": ["chroma", "gemini"]}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Running app on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "main:app",  # IMPORTANT: app as import string for reload support
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info"
    )
