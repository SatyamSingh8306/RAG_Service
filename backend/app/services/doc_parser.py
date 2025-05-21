# backend/app/services/doc_parser.py

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import time # For potential retries or delays

import requests # For calling OpenRouter API

# Local application imports
from backend.app.core.config import settings, PROJECT_ROOT_DIR
from backend.app.services.vstore_svc import VectorStoreService # Assuming singleton instance

# Optional: If Tesseract is not in your PATH, you might need to specify its location
# Example for Windows, adjust if necessary:
# TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# if os.path.exists(TESSERACT_PATH):
#     pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
# else:
#     print(f"Warning: Tesseract OCR path not found at {TESSERACT_PATH}. Ensure Tesseract is in your system PATH.")


class DocParserService:
    """
    Service to parse documents (PDFs, images), extract text, 
    perform LLM-based semantic chunking, and add chunks to the vector store.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        print("DocParserService initialized (LLM-Only Chunking Mode).")

    def _extract_text_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of a PDF.
        Each item in the list represents a page, containing a list of paragraphs.
        """
        pages_content = []
        try:
            doc = fitz.open(file_path)
            print(f"PDF '{os.path.basename(file_path)}' has {doc.page_count} pages.")
            for page_num_human_readable in range(1, doc.page_count + 1):
                page = doc.load_page(page_num_human_readable - 1) # fitz uses 0-indexed pages
                blocks = page.get_text("blocks", sort=True) # Sort by y-coordinate
                
                page_paragraphs_data = []
                para_counter_on_page = 0
                for block in blocks:
                    if block[6] == 0: # Text block
                        paragraph_text = block[4].replace('\r', ' ').replace('\n', ' ').strip()
                        if paragraph_text:
                            para_counter_on_page += 1
                            page_paragraphs_data.append({
                                "paragraph_number_in_page": para_counter_on_page,
                                "text": paragraph_text
                            })
                
                if page_paragraphs_data:
                    pages_content.append({
                        "page_number": page_num_human_readable,
                        "paragraphs": page_paragraphs_data
                    })
            doc.close()
            print(f"Extracted {len(pages_content)} pages with paragraphs from PDF: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error extracting text from PDF {os.path.basename(file_path)}: {e}")
        return pages_content

    def _extract_text_from_image_ocr(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from an image using OCR.
        Treats the whole image as one page. Attempts to split by double newlines as paragraphs.
        """
        pages_content = []
        try:
            img = Image.open(file_path)
            text_from_ocr = pytesseract.image_to_string(img)
            
            if text_from_ocr.strip():
                raw_paragraphs = [p.strip() for p in text_from_ocr.split('\n\n') if p.strip()]
                if not raw_paragraphs: # If no double newlines, try single newlines
                    raw_paragraphs = [p.strip() for p in text_from_ocr.split('\n') if p.strip()]

                page_paragraphs_data = []
                for i, para_text in enumerate(raw_paragraphs):
                    page_paragraphs_data.append({
                        "paragraph_number_in_page": i + 1,
                        "text": para_text
                    })

                if page_paragraphs_data:
                    pages_content.append({
                        "page_number": 1, # OCR'd image is considered a single page
                        "paragraphs": page_paragraphs_data
                    })
            print(f"Extracted text using OCR from image: {os.path.basename(file_path)}")
        except pytesseract.TesseractNotFoundError:
            print("TesseractNotFoundError: Tesseract is not installed or not in your PATH.")
            print("Please install Tesseract OCR and ensure it's accessible.")
        except Exception as e:
            print(f"Error extracting text from image {os.path.basename(file_path)} using OCR: {e}")
        return pages_content

    def _chunk_paragraph_semantically_llm(self, paragraph_text: str, source_info: str) -> List[str]:
        """
        Chunks a single paragraph semantically using an LLM via OpenRouter.
        Returns a list of text chunks. If it fails for any reason, returns an empty list and logs a message.
        """
        if not settings.OPENROUTER_API_KEY:
            print(f"OpenRouter API key not configured. LLM semantic chunking skipped for: {source_info}")
            return []

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.PROJECT_NAME, 
            "X-Title": settings.PROJECT_NAME 
        }
        
        prompt = f"""You are an expert text processor. Your task is to break down the following text from '{source_info}' into semantically coherent and self-contained chunks. Each chunk should ideally be between 2 to 5 sentences long and focus on a distinct idea or topic within the paragraph. Ensure that the chunks are grammatically correct and retain the original meaning and important entities. Do not summarize or add any information not present in the original text. Output EACH chunk on a new line. Do NOT add any extra formatting, numbering, or commentary before or after the chunks. Just provide the raw text chunks, each on its own line. If the input text is too short or cannot be meaningfully chunked according to these rules, return the original text as a single chunk.

Text to process:
\"\"\"
{paragraph_text}
\"\"\"

Semantically coherent chunks (each on a new line):"""

        payload = {
            "model": settings.DEFAULT_LLM_MODEL, 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, 
            "max_tokens": 2000 
        }
        
        print(f"Attempting LLM semantic chunking for paragraph from {source_info} using {settings.DEFAULT_LLM_MODEL}...")
        try:
            response = requests.post(
                f"{settings.OPENROUTER_API_BASE}/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=90
            )
            response.raise_for_status() 
            
            response_json = response.json()
            if response_json.get("choices") and response_json["choices"][0].get("message"):
                content = response_json["choices"][0]["message"].get("content", "")
                llm_chunks = [chunk.strip() for chunk in content.split('\n') if chunk.strip()]
                
                if llm_chunks:
                    print(f"LLM generated {len(llm_chunks)} chunks for paragraph from {source_info}.")
                    return llm_chunks
                else:
                    print(f"LLM returned no usable chunks for {source_info}. Content: '{content}'. This paragraph will not be chunked.")
                    return [] 
            else:
                error_details = response_json.get("error", "Unknown error structure")
                print(f"LLM response malformed for {source_info}. Details: {error_details}. This paragraph will not be chunked.")
                return []
        except requests.exceptions.RequestException as e:
            print(f"API request failed for LLM semantic chunking ({source_info}): {e}. This paragraph will not be chunked.")
            return []
        except Exception as e:
            print(f"Unexpected error during LLM semantic chunking ({source_info}): {e}. This paragraph will not be chunked.")
            return []

    def process_document(self, file_path: str, source_doc_id: str) -> bool:
        """
        Processes a single document: extracts text, performs LLM semantic chunking, 
        and adds resulting chunks to the vector store.
        If LLM chunking is unavailable or fails for a paragraph, that paragraph yields no chunks.
        """
        print(f"\nProcessing document: {os.path.basename(file_path)} (Source ID: {source_doc_id}, Mode: LLM-Only Chunking)")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        extracted_pages_content: List[Dict[str, Any]] = []
        if file_extension == ".pdf":
            extracted_pages_content = self._extract_text_from_pdf(file_path)
        elif file_extension in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            extracted_pages_content = self._extract_text_from_image_ocr(file_path)
        else:
            print(f"Unsupported file type: {file_extension} for {file_path}")
            return False

        if not extracted_pages_content:
            print(f"No text could be extracted from {file_path}.")
            return False

        all_final_chunks_texts: List[str] = []
        all_final_chunks_metadatas: List[Dict[str, Any]] = []
        all_final_chunks_ids: List[str] = []
        
        total_paragraphs_processed = 0
        total_chunks_generated = 0

        for page_content in extracted_pages_content:
            page_num = page_content["page_number"]
            for paragraph_data in page_content.get("paragraphs", []):
                total_paragraphs_processed += 1
                para_num = paragraph_data["paragraph_number_in_page"]
                para_text = paragraph_data["text"]
                source_info_for_llm = f"Doc: {source_doc_id}, Page: {page_num}, Para: {para_num}"

                current_paragraph_chunks: List[str] = []

                # Always attempt LLM semantic chunking
                current_paragraph_chunks = self._chunk_paragraph_semantically_llm(para_text, source_info_for_llm)
                
                if not current_paragraph_chunks:
                     print(f"LLM semantic chunking yielded no chunks for {source_info_for_llm} (or API key was missing/call failed). This paragraph is skipped.")
                
                for chunk_seq, chunk_text in enumerate(current_paragraph_chunks):
                    total_chunks_generated +=1
                    # Create a unique ID for each chunk
                    chunk_id = f"{source_doc_id}_p{page_num}_pr{para_num}_c{chunk_seq+1}"
                    metadata = {
                        "source_doc_id": source_doc_id,
                        "file_name": os.path.basename(file_path),
                        "page_number": page_num,
                        "paragraph_number_in_page": para_num, # Paragraph from which this chunk originated
                        "chunk_sequence_in_paragraph": chunk_seq + 1, # Sequence of this chunk within the original paragraph
                        "original_paragraph_text_preview": para_text[:150] + "..." # For context during review
                    }
                    all_final_chunks_texts.append(chunk_text)
                    all_final_chunks_metadatas.append(metadata)
                    all_final_chunks_ids.append(chunk_id)
        
        print(f"\nFinished processing {os.path.basename(file_path)}.")
        print(f"Total paragraphs analyzed: {total_paragraphs_processed}")
        print(f"Total chunks generated via LLM: {total_chunks_generated}")

        if all_final_chunks_texts:
            print(f"Attempting to add {len(all_final_chunks_texts)} LLM-generated chunks to vector store...")
            success = self.vector_store_service.add_documents(
                chunks=all_final_chunks_texts,
                metadatas=all_final_chunks_metadatas,
                doc_ids=all_final_chunks_ids
            )
            if success:
                print(f"Successfully added LLM-generated chunks from {os.path.basename(file_path)} to vector store.")
                return True
            else:
                print(f"Failed to add LLM-generated chunks from {os.path.basename(file_path)} to vector store.")
                return False
        else:
            print(f"No LLM-generated chunks were produced from {os.path.basename(file_path)} to add to vector store.")
            # Consider this a successful processing run if no text was extractable or no chunks were made,
            # but nothing to add to DB. Or return False if chunks are mandatory.
            # For now, returning False if nothing to add.
            return False

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing DocParserService (LLM-Only Chunking) ---")

    print(f"OpenRouter API Key available for test: {'Yes' if settings.OPENROUTER_API_KEY else 'No'}")
    if not settings.OPENROUTER_API_KEY:
        print("WARNING: OpenRouter API Key is NOT configured. LLM chunking tests will be skipped or show no chunks.")
    print(f"Using LLM Model for test: {settings.DEFAULT_LLM_MODEL}")

    vstore_service_instance = VectorStoreService() 

    if not vstore_service_instance._langchain_chroma_instance:
        print("CRITICAL: VectorStoreService did not initialize correctly. Aborting test.")
    else:
        parser_service = DocParserService(vector_store_service=vstore_service_instance)

        test_data_dir = os.path.join(PROJECT_ROOT_DIR, "test_documents")
        os.makedirs(test_data_dir, exist_ok=True)

        dummy_pdf_path = os.path.join(test_data_dir, "dummy_llm_only.pdf")
        try:
            pdf_doc = fitz.open() 
            page = pdf_doc.new_page()
            page.insert_text((72, 72), "The first paragraph for LLM chunking. It talks about AI and its impact on modern software development. This field is rapidly evolving.")
            page.insert_text((72, 144), "Another distinct idea: renewable energy sources are crucial for a sustainable future. Solar and wind power are leading examples.")
            page = pdf_doc.new_page()
            page.insert_text((72, 72), "Page two has a very short paragraph. Just this one sentence.")
            pdf_doc.save(dummy_pdf_path)
            pdf_doc.close()
            print(f"Created dummy PDF for LLM-only test: {dummy_pdf_path}")
        except Exception as e:
            print(f"Could not create dummy PDF: {e}.")

        dummy_image_path = os.path.join(test_data_dir, "dummy_llm_only.png")
        try:
            img = Image.new('RGB', (700, 200), color = 'white')
            from PIL import ImageDraw # Import here to avoid error if Pillow not fully installed for main script
            d = ImageDraw.Draw(img)
            d.text((10,10), "OCR text for the LLM. This is the first sentence.\nThis is the second sentence of the first paragraph for OCR.\n\nA new paragraph begins here for OCR testing. It should be processed separately.", fill=(0,0,0))
            img.save(dummy_image_path)
            print(f"Created dummy PNG for LLM-only test: {dummy_image_path}")
        except Exception as e:
            print(f"Could not create dummy PNG: {e}.")

        # --- Test Case 1: Process PDF (Always uses LLM Chunking) ---
        print("\n--- Test Case 1: PDF Processing (LLM-Only Chunking) ---")
        if os.path.exists(dummy_pdf_path):
            parser_service.process_document(
                file_path=dummy_pdf_path,
                source_doc_id="llm_only_pdf_test"
            )
        else:
            print(f"Skipping Test Case 1: Dummy PDF not found at {dummy_pdf_path}")

        # --- Test Case 2: Process Image with OCR (Always uses LLM Chunking for OCR'd text) ---
        print("\n--- Test Case 2: Image with OCR (LLM-Only Chunking for OCR'd text) ---")
        if os.path.exists(dummy_image_path):
            parser_service.process_document(
                file_path=dummy_image_path,
                source_doc_id="llm_only_image_test"
            )
        else:
            print(f"Skipping Test Case 2: Dummy Image not found at {dummy_image_path}")
        
        print("\n--- DocParserService (LLM-Only Chunking) Test Complete ---")
 
