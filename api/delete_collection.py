from chromadb import PersistentClient
from config import settings 

def delete_collection():
    client = PersistentClient(path=settings.CHROMA_PATH)
    collection_name = settings.CHROMA_COLLECTION
    try:
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Failed to delete collection '{collection_name}': {e}")

if __name__ == "__main__":
    delete_collection()

