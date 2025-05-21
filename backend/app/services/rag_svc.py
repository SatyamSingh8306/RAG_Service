# backend/app/services/rag_svc.py

import requests
import json 
from typing import List, Dict, Any, Optional, Tuple
import time 

# Local application imports
from backend.app.core.config import settings 
from backend.app.services.vstore_svc import VectorStoreService
from langchain_core.documents import Document as LangchainDocument 

# For Cross-Encoder Reranking
from sentence_transformers import CrossEncoder

class RAGService:
    """
    Service to handle Retrieval Augmented Generation:
    - Retrieves relevant documents from the vector store (using MMR).
    - Reranks the retrieved documents using a Cross-Encoder.
    - Interacts with an LLM to generate answers and identify themes based on context.
    """
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        try:
            self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
            self.reranker = CrossEncoder(self.reranker_model_name, device='cpu') 
            print(f"RAGService initialized with Cross-Encoder: {self.reranker_model_name}.")
        except Exception as e:
            print(f"Error initializing CrossEncoder: {e}. Reranking will be skipped.")
            self.reranker = None


    def _call_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 3500) -> Optional[str]:
        if not settings.OPENROUTER_API_KEY:
            print("OpenRouter API key not configured. Cannot call LLM.")
            return None
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.PROJECT_NAME, "X-Title": settings.PROJECT_NAME
        }
        payload = {
            "model": settings.DEFAULT_LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens
        }
        print(f"\n--- Calling LLM ({settings.DEFAULT_LLM_MODEL}) ---")
        try:
            response = requests.post(
                f"{settings.OPENROUTER_API_BASE}/chat/completions",
                headers=headers, json=payload, timeout=180
            )
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("choices") and response_json["choices"][0].get("message"):
                content = response_json["choices"][0]["message"].get("content", "")
                print("LLM call successful.")
                return content.strip()
            else:
                print(f"LLM response malformed. Details: {response_json.get('error', 'Unknown')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"API request failed for LLM call: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during LLM call: {e}")
            return None

    def _rerank_documents(self, query: str, documents: List[LangchainDocument], top_n: int) -> List[Tuple[LangchainDocument, float]]:
        """Reranks documents using the CrossEncoder and returns top_n with scores."""
        if not self.reranker or not documents:
            print("Reranker not available or no documents to rerank. Returning original order (or empty).")
            return [(doc, 0.0) for doc in documents][:top_n]

        print(f"Reranking {len(documents)} documents with CrossEncoder...")
        sentence_pairs = [[query, doc.page_content] for doc in documents]
        
        try:
            scores = self.reranker.predict(sentence_pairs, show_progress_bar=False) # type: ignore
        except Exception as e:
            print(f"Error during reranker prediction: {e}. Returning original order.")
            return [(doc, 0.0) for doc in documents][:top_n]

        docs_with_reranker_scores = list(zip(documents, scores))
        docs_with_reranker_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Reranking complete. Top {top_n} selected.")
        return docs_with_reranker_scores[:top_n]


    def _format_context_for_prompt(self, 
                                   reranked_docs_with_scores: List[Tuple[LangchainDocument, float]],
                                   doc_to_ref_map: Dict[str, int]) -> str:
        """
        Formats reranked documents for the LLM prompt, using numerical references.
        doc_to_ref_map maps source_doc_id to its assigned reference number (e.g., 1, 2, 3).
        """
        context_str = ""
        if not reranked_docs_with_scores:
            return "No context snippets were available after reranking."
            
        for i, (doc, score) in enumerate(reranked_docs_with_scores):
            metadata = doc.metadata
            source_doc_id = metadata.get('source_doc_id', 'N/A')
            ref_num = doc_to_ref_map.get(source_doc_id, 0) 

            source_ref = (
                f"RefNum: [{ref_num}], "
                f"OrigSourceDocID: {source_doc_id}, "
                f"Paper: {metadata.get('file_name', 'N/A')}, "
                f"Page: {metadata.get('page_number', 'N/A')}, "
                f"Para: {metadata.get('paragraph_number_in_page', 'N/A')}, "
                f"ChunkInPara: {metadata.get('chunk_sequence_in_paragraph', 'N/A')}, "
                f"RerankScore: {score:.4f}" 
            )
            context_str += f"Context Snippet {i+1} ({source_ref}):\n"
            context_str += f"\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
        return context_str.strip()

    def get_answer_and_themes(self, query: str, collection_name: Optional[str] = None, n_final_docs_for_llm: int = 20, initial_mmr_k: int = 100, initial_mmr_fetch_k: int =200 ) -> Dict[str, Any]:
        """
        Main method: retrieves with MMR, reranks with CrossEncoder, then generates answer & themes.
        """
        print(f"\n--- RAGService: Processing query: '{query}' ---")
        print(f"Config: n_final_docs_for_llm={n_final_docs_for_llm}, initial_mmr_k={initial_mmr_k}, initial_mmr_fetch_k={initial_mmr_fetch_k}")

        # Prepare default empty response structure
        default_empty_response = {
            "answer": "Could not process the query.",
            "themes": [],
            "references": [],
            "retrieved_context_document_ids": [],
            "document_details": []
        }

        if not self.vector_store_service._langchain_chroma_instance:
            print("VectorStoreService not properly initialized.")
            default_empty_response["answer"] = "Error: Knowledge base connection unavailable."
            return default_empty_response
            
        mmr_retrieved_docs_tuples = self.vector_store_service.query_documents_with_scores(
            query_text=query,
            n_results=initial_mmr_k,
            collection_name=collection_name
        )

        if not mmr_retrieved_docs_tuples:
            print("MMR retrieval found no relevant documents.")
            default_empty_response["answer"] = "Could not find relevant information in the provided documents to answer your query."
            return default_empty_response

        mmr_retrieved_docs = [doc for doc, _ in mmr_retrieved_docs_tuples]
        reranked_docs_with_scores = self._rerank_documents(query, mmr_retrieved_docs, top_n=n_final_docs_for_llm)

        if not reranked_docs_with_scores:
            print("Reranking yielded no documents.")
            default_empty_response["answer"] = "Could not refine context after initial retrieval."
            return default_empty_response

        print(f"Top {len(reranked_docs_with_scores)} reranked document chunks for LLM context. Details:")
        # (Logging for reranked chunks as before) ...

        # Create a mapping of reference numbers to source document IDs
        doc_to_ref_map: Dict[str, int] = {}
        references_list_for_prompt: List[Dict[str, Any]] = []
        current_ref_number = 1
        final_context_doc_ids_for_tracking = []
        source_doc_id_map: Dict[int, str] = {}  # Map reference numbers to source_doc_ids
        
        for doc, _ in reranked_docs_with_scores:
            source_doc_id = doc.metadata.get('source_doc_id', 'N/A')
            final_context_doc_ids_for_tracking.append(source_doc_id)
            if source_doc_id not in doc_to_ref_map:
                doc_to_ref_map[source_doc_id] = current_ref_number
                source_doc_id_map[current_ref_number] = source_doc_id  # Store the mapping
                references_list_for_prompt.append({
                    "reference_number": current_ref_number,
                    "source_doc_id": source_doc_id,
                    "file_name": doc.metadata.get('file_name', 'N/A')
                })
                current_ref_number += 1

        # Extract document details from reranked documents
        document_details = []
        for doc, score in reranked_docs_with_scores:
            metadata = doc.metadata
            ref_num = doc_to_ref_map.get(metadata.get('source_doc_id', 'N/A'))
            if ref_num and ref_num in source_doc_id_map:
                document_details.append({
                    "source_doc_id": source_doc_id_map[ref_num],
                    "file_name": metadata.get('file_name', 'N/A'),
                    "extracted_answer": doc.page_content,
                    "page_number": metadata.get('page_number'),
                    "paragraph_number": metadata.get('paragraph_number_in_page')
                })

        formatted_context = self._format_context_for_prompt(reranked_docs_with_scores, doc_to_ref_map)
        
        # Updated prompt with more forceful instructions for themes and references
        prompt_template = f"""You are a highly proficient AI Research Assistant. Your expertise lies in analyzing and synthesizing information from academic research papers.
You will be provided with a user's query and a collection of context snippets. Each snippet is identified by a 'RefNum' (e.g., RefNum: [1]), its original SourceDocID, Paper name, Page, Paragraph, and RerankScore.
Your response must be strictly grounded in these provided snippets. DO NOT use any external knowledge.

Your primary objectives are:

1.  **Comprehensive Answer with Numerical Citations:**
    a.  Directly address the user's query using the most pertinent information from the context snippets.
    b.  When using information, you MUST cite it inline using the 'RefNum' and relevant page/paragraph details. Example: "Concept X is defined as... [1, Page: 3, Para: 2]." or "This is supported by multiple findings [2, Page: 5, Para: 1; 3, Page: 10, Para: 4]."
    c.  Synthesize information from multiple relevant snippets (and thus potentially multiple RefNums) to provide a comprehensive understanding.
    d.  If the context is insufficient, clearly state this limitation.

2.  **Cross-Document Theme Identification:**
    a.  Identify 1-3 overarching themes based on ALL provided context snippets relevant to the query.
    b.  For each theme, provide a concise summary.
    c.  You MUST list the 'RefNum's (e.g., [1], [3], [5]) of the snippets that contribute to each theme in the 'supporting_reference_numbers' field.

User Query: "{query}"

Provided Context Snippets:
{formatted_context}

IMPORTANT: Structure your entire response as a single JSON object with the exact keys: "answer", "identified_themes", "references".
The "references" list is MANDATORY.

-   "answer": (String) Your detailed, synthesized answer, with inline numerical citations (e.g., "[1, Page: 5, Para: 2]").
-   "identified_themes": (List of Objects) Each object MUST have "theme_summary" (String) AND "supporting_reference_numbers" (List of Integers, e.g., [1, 3]).
-   "references": (List of Objects) A bibliography mapping your numerical references to source details. For EVERY 'RefNum' [N] you assigned and used in your 'answer' or 'identified_themes', there MUST be a corresponding object in this list. Each object MUST have "reference_number" (Integer), "source_doc_id" (String), and "file_name" (String). Use the 'OrigSourceDocID' and 'Paper' name from the context snippets for this.

Begin JSON Response:


---
After completing the JSON response above, reflect as an expert AI research agent with internal knowledge. Using both the provided context and your own pretrained understanding, reason through the problem in a chain-of-thought manner and generate a second, separate response labeled as:

=== Synthesized Expert Answer (LLM + RAG) ===

This section should:
- Combine insights from the documents with your own LLM knowledge.
- Be analytical, nuanced, and expert-level.
- Avoid repeating the JSON response; instead, offer a refined and integrated expert answer.
- Also reference from the references in the JSON response, mentioning like the author or research paper name or using the document no. mentioned in the JSON response, to say it like as also said in that document. 
"""
        llm_response_str = self._call_llm(prompt_template)

        print(f"\n--- Raw LLM Response ---\n{llm_response_str}\n--- End Raw LLM Response ---")

        if not llm_response_str:
            print("LLM call failed or returned no response.")
            default_empty_response["answer"] = "There was an error processing your query with the language model."
            return default_empty_response

        try:
            if llm_response_str.strip().startswith("```json"):
                llm_response_str = llm_response_str.strip()[7:-3].strip()
            elif llm_response_str.strip().startswith("```"):
                 llm_response_str = llm_response_str.strip()[3:-3].strip()

            # Find the start and end of the JSON object
            json_start = llm_response_str.find('{')
            json_end = llm_response_str.rfind('}')

            synthesized_answer_marker = "=== Synthesized Expert Answer (LLM + RAG) ==="
            synthesized_answer_start = llm_response_str.find(synthesized_answer_marker)

            json_str = ""
            synthesized_expert_answer = ""

            if json_start != -1 and json_end != -1 and json_end > json_start:
                # Extract the JSON string
                json_str = llm_response_str[json_start : json_end + 1]
                # If the synthesized answer marker exists after the JSON, extract it
                if synthesized_answer_start != -1 and synthesized_answer_start > json_end:
                    synthesized_expert_answer = llm_response_str[synthesized_answer_start + len(synthesized_answer_marker):].strip()
                elif synthesized_answer_start != -1 and synthesized_answer_start < json_start: # Handle case where marker is before JSON
                     # If JSON is found, assume text after JSON is the synthesized answer if marker is before JSON
                     synthesized_expert_answer = llm_response_str[json_end + 1:].strip()
                else:
                     # If no marker found, or marker is within JSON, take everything after JSON as synthesized answer
                     synthesized_expert_answer = llm_response_str[json_end + 1:].strip()

            elif synthesized_answer_start != -1:
                 # If no JSON found but marker exists, take everything after marker
                 synthesized_expert_answer = llm_response_str[synthesized_answer_start + len(synthesized_answer_marker):].strip()

            else:
                print(f"Error: Could not find valid JSON object or synthesized answer marker in LLM response. Raw: {llm_response_str[:500]}...")
                default_empty_response["answer"] = f"LLM response did not contain a valid JSON object or synthesized answer. Snippet: {llm_response_str[:200]}..."
                return default_empty_response

            # Attempt to parse the extracted JSON string
            parsed_llm_response = json.loads(json_str) if json_str else {}

            final_answer = parsed_llm_response.get("answer", "LLM did not provide an answer in the expected format.")
            if not final_answer or not str(final_answer).strip():
                # Try to use synthesized_expert_answer if available
                if synthesized_expert_answer and synthesized_expert_answer.strip():
                    final_answer = synthesized_expert_answer.strip()
                else:
                    final_answer = "No answer was generated by the language model. Please try rephrasing your query or check back later."

            # Robust handling for identified_themes
            identified_themes_raw = parsed_llm_response.get("identified_themes", [])
            valid_themes = []
            if isinstance(identified_themes_raw, list):
                for theme_item in identified_themes_raw:
                    if isinstance(theme_item, dict) and \
                       "theme_summary" in theme_item and \
                       "supporting_reference_numbers" in theme_item and \
                       isinstance(theme_item["supporting_reference_numbers"], list):
                        valid_themes.append(theme_item)
                    else: 
                        print(f"Warning: Invalid theme item from LLM, attempting to fix or use default: {theme_item}")
                        # Provide a default structure if LLM messes up theme content but gets the key
                        valid_themes.append({
                            "theme_summary": theme_item.get("theme_summary", "Theme summary missing"),
                            "supporting_reference_numbers": theme_item.get("supporting_reference_numbers", []) if isinstance(theme_item.get("supporting_reference_numbers"), list) else []
                        })
            else:
                print(f"Warning: 'identified_themes' from LLM is not a list. Received: {identified_themes_raw}")
            identified_themes = valid_themes

            # Robust handling for references
            generated_references_raw = parsed_llm_response.get("references", [])
            valid_references = []
            if isinstance(generated_references_raw, list):
                for ref_item in generated_references_raw:
                    if isinstance(ref_item, dict) and \
                       "reference_number" in ref_item and \
                       "source_doc_id" in ref_item and \
                       "file_name" in ref_item:
                        valid_references.append(ref_item)
                    else: 
                        print(f"Warning: Invalid reference item from LLM: {ref_item}")
            else:
                print(f"Warning: 'references' from LLM is not a list. Received: {generated_references_raw}")
            generated_references = valid_references
            
            print("\n--- RAGService: Final Processed Output ---")
            print(f"Query: {query}")
            print(f"Synthesized Answer: {final_answer}")
            print("Identified Themes:")
            if identified_themes:
                for theme_obj in identified_themes:
                    print(f"  - Theme: {theme_obj['theme_summary']}")
                    print(f"    Supported by Refs: {theme_obj.get('supporting_reference_numbers', [])}") # Use .get for safety
            else:
                print("  No themes were identified or extracted in the expected format.")
            print(f"Generated References by LLM: {generated_references}")

            # --- New Step: Call LLM again as Research Assistant ---
            research_assistant_prompt = f"""You are now acting as a Research Assistant. Given the following user query and the previous LLM answer, synthesize a new expert answer that provides additional insights, clarifications, or a more comprehensive perspective.\n\nUser Query: {query}\n\nPrevious LLM Answer: {final_answer}\n\nYour task is to provide a synthesized expert answer that combines the information from both the user query and the previous answer."""
            synthesized_expert_answer = self._call_llm(research_assistant_prompt)

            # Add evidence snippets to themes
            for theme in identified_themes:
                theme['evidence_snippets'] = []
                for ref_num in theme.get('supporting_reference_numbers', []):
                    if ref_num in source_doc_id_map:
                        source_doc_id = source_doc_id_map[ref_num]
                        for doc, _ in reranked_docs_with_scores:
                            if doc.metadata.get('source_doc_id') == source_doc_id:
                                theme['evidence_snippets'].append({
                                    "text": doc.page_content,
                                    "page": doc.metadata.get('page_number'),
                                    "paragraph": doc.metadata.get('paragraph_number_in_page'),
                                    "source_doc_id": source_doc_id
                                })

            return {
                "answer": final_answer,
                "themes": identified_themes,
                "references": generated_references,
                "retrieved_context_document_ids": final_context_doc_ids_for_tracking,
                "synthesized_expert_answer": synthesized_expert_answer,
                "document_details": document_details
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON. Raw: {llm_response_str[:500]}... Error: {e}")
            default_empty_response["answer"] = f"LLM JSON format error. Snippet: {llm_response_str[:200]}..."
            return default_empty_response
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            # import traceback
            # traceback.print_exc()
            default_empty_response["answer"] = "Unexpected error processing LLM response."
            return default_empty_response

# --- Test Block ---
if __name__ == "__main__":
    # (Test block remains largely the same as in rag_svc_py_code_reranker_citation_fix)
    # Ensure it calls get_answer_and_themes and prints the full response including 'references'
    print("--- Testing RAGService (with Reranker and Numerical Citations - Robust Parsing) ---")
    vstore_service = VectorStoreService()
    if not vstore_service._langchain_chroma_instance:
        print("CRITICAL: VectorStoreService did not initialize. Aborting.")
        exit()
    
    test_chunks_for_rag = [
        "The novel attention mechanism, 'TransFusion', demonstrates a 15% improvement in NLP task benchmarks by integrating multi-modal inputs effectively.", 
        "Ethical AI frameworks must consider data privacy, algorithmic bias, and societal impact before large-scale deployment of autonomous systems.", 
        "Our proposed 'Contextual Embedding Alignment Protocol' (CEAP) significantly enhances cross-lingual information retrieval from diverse knowledge bases.", 
        "While TransFusion shows promise, its computational overhead for training remains a significant challenge for widespread adoption in resource-constrained environments.", 
        "Bias mitigation techniques in AI, such as adversarial debiasing and data augmentation, are critical for ensuring fairness (Johnson et al., 2023).", 
        "The CEAP method was validated on three distinct language pairs, showing consistent gains over existing SOTA models in zero-shot translation tasks.", 
        "Further research into optimized attention patterns, like those in TransFusion, is essential for next-generation language understanding." 
    ]
    test_metadatas_for_rag = [
        {"source_doc_id": "paper_A_2024", "file_name": "transfusion_nlp.pdf", "page_number": 5, "paragraph_number_in_page": 3, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_B_ethics", "file_name": "ethical_ai_frameworks.pdf", "page_number": 12, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_C_ceap", "file_name": "ceap_crosslingual.pdf", "page_number": 7, "paragraph_number_in_page": 4, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_A_2024", "file_name": "transfusion_nlp.pdf", "page_number": 8, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_B_ethics", "file_name": "ethical_ai_frameworks.pdf", "page_number": 15, "paragraph_number_in_page": 2, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_C_ceap", "file_name": "ceap_crosslingual.pdf", "page_number": 9, "paragraph_number_in_page": 2, "chunk_sequence_in_paragraph": 1},
        {"source_doc_id": "paper_D_attention", "file_name": "future_attention.pdf", "page_number": 2, "paragraph_number_in_page": 1, "chunk_sequence_in_paragraph": 1},
    ]
    test_ids_for_rag = [f"rerank_cite_test_v2_{i}" for i in range(len(test_chunks_for_rag))]

    print(f"Deleting existing test documents by IDs: {test_ids_for_rag}")
    vstore_service.delete_documents(doc_ids=test_ids_for_rag)
    time.sleep(0.5) 
    print("Adding test documents for RAG service...")
    vstore_service.add_documents(chunks=test_chunks_for_rag, metadatas=test_metadatas_for_rag, doc_ids=test_ids_for_rag)
    time.sleep(0.5)
    print(f"Document chunk count after adding test data: {vstore_service.get_collection_count()}")

    rag_service = RAGService(vector_store_service=vstore_service)

    print("\n--- RAG Test Query 1 (Reranker & Numerical Citations - Robust Parsing) ---")
    query1 = "What is TransFusion and its significance, including any limitations? Also discuss CEAP."
    if settings.OPENROUTER_API_KEY:
        response1 = rag_service.get_answer_and_themes(
            query1, 
            n_final_docs_for_llm=5, 
            initial_mmr_k=10 
        )
        if response1:
            print("\nFormatted Response for Query 1:")
            print(json.dumps(response1, indent=2))
    else:
        print(f"Skipping RAG Test Query 1: OpenRouter API Key not configured.")
    
    print("\n--- RAGService Test Complete ---")

