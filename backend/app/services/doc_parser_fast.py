import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Imports for semantic chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


# Local application imports
from backend.app.core.config import settings
from backend.app.services.vstore_svc import VectorStoreService

class DocParserFastService:
    """
    Service to parse documents (PDFs, images), extract text,
    perform STRUCTURE-AWARE semantic chunking, and add chunks to the vector store.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        self.ocr_processor = None  # Will be initialized on first use

        # Use Instructor embeddings model for more citation-aware and instruction-aligned chunks
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="hkunlp/instructor-xl",  # Free and powerful model from HuggingFace
                model_kwargs={'device': 'cpu'},
                encode_kwargs={"normalize_embeddings": True}
            )

            # NOTE: Using semantic chunking with INSTRUCTOR may require a custom splitter; fallback used here
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # smaller for citation-aware precision
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ". ", "? ", "! ", ",", " ", ""]
            )
        except Exception as e:
            print(f"Error initializing embedding model: {e}. Using fallback splitter.")
            self.embeddings = None
            self.semantic_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", ". ", "? ", "! ", ",", " ", ""]
            )

    def _is_heading(self, text: str) -> bool:
        """
        Detect if a line/paragraph is a heading based on heuristic:
        - Mostly uppercase or capitalized and short (< 8 words)
        """
        words = text.strip().split()
        if len(words) <= 8 and sum(1 for w in words if w.isupper()) >= len(words) * 0.6:
            return True
        return False

    def _extract_sections_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF and groups paragraphs into sections by detecting headings.
        Returns list of sections with 'title', 'page_number', and 'paragraphs'.
        """
        sections = []
        doc = fitz.open(file_path)
        section_counter = 0
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks", sort=True)
            current_section = {"title": f"page_{page_index+1}_untitled", "page_number": page_index+1, "paragraphs": []}
            for block in blocks:
                if block[6] != 0:
                    continue
                text = block[4].replace('\r', ' ').replace('\n', ' ').strip()
                if not text:
                    continue
                if self._is_heading(text):
                    # start new section
                    if current_section["paragraphs"]:
                        sections.append(current_section)
                    section_counter += 1
                    current_section = {"title": text, "page_number": page_index+1, "paragraphs": [], "section_index": section_counter}
                else:
                    current_section["paragraphs"].append(text)
            if current_section["paragraphs"]:
                # ensure section_index for untitled sections
                if "section_index" not in current_section:
                    section_counter += 1
                    current_section["section_index"] = section_counter
                sections.append(current_section)
        doc.close()
        return sections

    def _extract_text_from_image(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from an image using OCR.
        Treats the whole image as one section.
        """
        sections = []
        try:
            img = Image.open(file_path)
            text_from_ocr = pytesseract.image_to_string(img)
            
            if text_from_ocr.strip():
                # Split text into paragraphs
                paragraphs = [p.strip() for p in text_from_ocr.split('\n\n') if p.strip()]
                if not paragraphs:  # If no double newlines, try single newlines
                    paragraphs = [p.strip() for p in text_from_ocr.split('\n') if p.strip()]
                
                if paragraphs:
                    sections.append({
                        "title": "ocr_image_section",
                        "page_number": 1,
                        "paragraphs": paragraphs,
                        "section_index": 1
                    })
            print(f"Extracted text using OCR from image: {os.path.basename(file_path)}")
        except pytesseract.TesseractNotFoundError:
            print("TesseractNotFoundError: Tesseract is not installed or not in your PATH.")
            print("Please install Tesseract OCR and ensure it's accessible.")
        except Exception as e:
            print(f"Error extracting text from image {os.path.basename(file_path)} using OCR: {e}")
        return sections

    def _perform_semantic_chunking(self, text: str) -> List[str]:
        """
        Perform semantic chunking on text using embeddings, fallback if needed.
        Here we use RecursiveCharacterTextSplitter for INSTRUCTOR-compatible chunking.
        """
        try:
            return self.semantic_splitter.split_text(text)
        except Exception as e:
            print(f"Chunking failed: {e}")
            return [text]  # fallback to whole text if even fallback splitter fails

    def process_document(self, file_path: str, source_doc_id: str, collection_name: Optional[str] = None) -> bool:
        """Process a document using fast rule-based chunking."""
        try:
            # Extract text based on file type
            file_extension = Path(file_path).suffix.lower()
            if file_extension in ['.pdf']:
                sections = self._extract_sections_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                sections = self._extract_text_from_image(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return False

            if not sections:
                print(f"No text extracted from {file_path}")
                return False

            # Process the text
            all_chunks, all_metadatas, all_ids = [], [], []
            global_chunk_counter = 1
            for sec in sections:
                section_text = "\n\n".join(sec["paragraphs"])
                chunks = self._perform_semantic_chunking(section_text)
                for idx, chunk in enumerate(chunks):
                    sec_idx = sec.get("section_index", sec["page_number"])
                    chunk_id = f"{source_doc_id}_sec{sec_idx}_chunk{global_chunk_counter}"
                    metadata = {
                        "source_doc_id": source_doc_id,
                        "file_name": Path(file_path).name,
                        "section_title": sec["title"],
                        "page_number": sec["page_number"],
                        "section_index": sec_idx,
                        "chunk_index": global_chunk_counter,
                        "paragraph_number_in_page": idx + 1
                    }
                    all_chunks.append(chunk)
                    all_metadatas.append(metadata)
                    all_ids.append(chunk_id)
                    global_chunk_counter += 1

            # Add documents to vector store
            success = self.vector_store_service.add_documents(
                chunks=all_chunks,
                metadatas=all_metadatas,
                doc_ids=all_ids,
                collection_name=collection_name
            )

            return success
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return False

def process_document_background(
    file_path: str, 
    source_doc_id: str, 
    doc_parser_svc_instance: DocParserFastService,
    serial_no: Optional[int] = None, 
    total_count: Optional[int] = None,
    collection_name: Optional[str] = None
):
    parser_name = doc_parser_svc_instance.__class__.__name__
    progress_log = f"Document {serial_no}/{total_count}" if serial_no and total_count else "Single document"
    
    print(f"Background task started for: {Path(file_path).name}, Source ID: {source_doc_id}, Parser: {parser_name}, ({progress_log})")
    try:
        success = doc_parser_svc_instance.process_document(
            file_path=file_path, 
            source_doc_id=source_doc_id,
            collection_name=collection_name
        )
        if success:
            print(f"Background processing completed successfully for {source_doc_id} ({Path(file_path).name}) using {parser_name}")
        else:
            print(f"Background processing (using {parser_name}) had issues or no chunks generated for {source_doc_id} ({Path(file_path).name})")
    except Exception as e:
        print(f"Error during background document processing for {source_doc_id} ({Path(file_path).name}) (using {parser_name}): {e}")
