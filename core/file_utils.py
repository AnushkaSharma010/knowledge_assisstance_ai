import hashlib

def compute_file_hash(file_bytes: bytes) -> str:
    """Generate SHA256 hash for file content"""
    return hashlib.sha256(file_bytes).hexdigest()
