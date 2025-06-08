import google.generativeai as genai
from typing import List
from config import settings
from logger import get_logger

logger = get_logger("GeminiEmbeddings")

MAX_CHUNK_LENGTH = 30000  
BATCH_SIZE = 5             

class GeminiEmbeddings:
    def __init__(self):
        logger.info("Configuring Gemini embeddings model")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = 'models/embedding-001'
        logger.info("Gemini embeddings model initialized")

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding total of {len(texts)} documents")
        embeddings = []
        batch = []

        def safe_embed(text_batch):
            batch_embeddings = []
            for text in text_batch:
                try:
                    emb = self.embed_query(text)
                    logger.info(f"Generated embedding dimension: {len(emb)}")
                    batch_embeddings.append(emb)
                except Exception as e:
                    logger.error(f"Failed to embed text: {text[:30]}... Error: {e}")
                    batch_embeddings.append([0.0] * 768)  # fallback embedding
            return batch_embeddings

        for text in texts:
            if len(text) > MAX_CHUNK_LENGTH:
                logger.warning(f"Splitting oversized text chunk of length {len(text)}")
                parts = [text[i:i+MAX_CHUNK_LENGTH] for i in range(0, len(text), MAX_CHUNK_LENGTH)]
                for part in parts:
                    batch.append(part)
                    if len(batch) == BATCH_SIZE:
                        embeddings.extend(safe_embed(batch))
                        batch = []
            else:
                batch.append(text)
                if len(batch) == BATCH_SIZE:
                    embeddings.extend(safe_embed(batch))
                    batch = []

        if batch:
            embeddings.extend(safe_embed(batch))

        logger.info("Completed embedding of all batches")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        logger.debug(f"Embedding single query text: {text[:30]}...")
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        embedding = result["embedding"]
        logger.debug("Embedding successful")
        return embedding
    
    def embed_for_media_relevance(self, answer: str, media_captions: List[str]) -> List[float]:
        logger.info("Embedding answer and media captions for media relevance filtering")
        all_texts = [answer] + media_captions
        embeddings = self.embed_documents(all_texts)
        return embeddings


    def name(self) -> str:
        return "gemini-embedding-001"
