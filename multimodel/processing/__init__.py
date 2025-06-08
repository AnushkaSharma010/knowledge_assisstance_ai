import magic
import os
from .docx import DocxProcessor
from .pdf import PDFProcessor

class ProcessorFactory:
    @staticmethod
    def get_processor(file_path: str):
        try:
            with open(file_path, 'rb') as f:
                initial_bytes = f.read(2048)
            file_type = magic.from_buffer(initial_bytes, mime=True)
        except Exception:
            # fallback to extension if magic fails
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".docx":
                return DocxProcessor(file_path)
            elif ext == ".pdf":
                return PDFProcessor(file_path)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        if file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocxProcessor(file_path)
        elif file_type == "application/pdf":
            return PDFProcessor(file_path)
        else:
            # fallback to extension if MIME not matched
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".docx":
                return DocxProcessor(file_path)
            elif ext == ".pdf":
                return PDFProcessor(file_path)
            else:
                raise ValueError(f"Unsupported file type or extension: {file_type}, {ext}")

    @staticmethod
    def process(file_path: str):
        processor = ProcessorFactory.get_processor(file_path)
        return processor.process()
