# frontend/ui.py

import streamlit as st
import requests
import os
from typing import List, Dict, Any, Optional
import time 

# --- Configuration ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
# The backend will now always use the fast parser.
# The `processing_mode` parameter will be removed from the API call.
DOCUMENTS_UPLOAD_URL = f"{BACKEND_URL}/api/v1/documents/upload-multiple" 
CHAT_QUERY_URL = f"{BACKEND_URL}/api/v1/chat/query"
COLLECTIONS_URL = f"{BACKEND_URL}/api/v1/collections"

# --- Helper Functions ---
def get_collections() -> List[Dict[str, Any]]:
    """Get list of collections from the API."""
    try:
        response = requests.get(COLLECTIONS_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        return []

def create_collection(name: str) -> bool:
    """Create a new collection."""
    try:
        print(f"Attempting to create collection with name: {name}")
        response = requests.post(
            COLLECTIONS_URL,
            json={"name": name},
            headers={"Content-Type": "application/json"}
        )
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Request error details: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Error response content: {e.response.text}")
        st.error(f"Error creating collection: {str(e)}")
        return False

def delete_collection(name: str) -> bool:
    """Delete a collection."""
    try:
        response = requests.delete(f"{COLLECTIONS_URL}/{name}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

def reset_chat_history():
    """Resets the chat history in the session state."""
    st.session_state.messages = []
    st.session_state.processed_doc_ids = set() 
    st.session_state.user_has_been_warned_about_processing = False
    if "processing_active_message_placeholder" in st.session_state and st.session_state.processing_active_message_placeholder is not None:
        st.session_state.processing_active_message_placeholder.empty()
        st.session_state.processing_active_message_placeholder = None

def query_chat(query: str, collection: Optional[str] = None) -> Dict[str, Any]:
    """Send a query to the chat API."""
    try:
        response = requests.post(
            CHAT_QUERY_URL,
            json={
                "query": query,
                "collection": collection,
                "n_results": 5
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        raise

def display_chat_message(role: str, content: Any):
    """Helper to display a chat message with a consistent avatar."""
    avatar_map = {"user": "👤", "assistant": "🤖"}
    avatar = avatar_map.get(role)
    with st.chat_message(role, avatar=avatar):
        if isinstance(content, str): 
            st.markdown(content)
        elif isinstance(content, dict): 
            # 1. Display the main answer
            st.markdown(f"**Answer:**\n{content.get('answer', 'No answer provided.')}")
            
            # 2. Display themes with evidence
            themes = content.get('themes', [])
            if themes:
                st.markdown("\n**Identified Themes:**")
                for i, theme_data in enumerate(themes):
                    theme_summary = theme_data.get('theme_summary', 'N/A')
                    supporting_refs = theme_data.get('supporting_reference_numbers', []) 
                    evidence_snippets = theme_data.get('evidence_snippets', [])
                    
                    # Display theme summary with references
                    if supporting_refs and all(isinstance(ref, int) for ref in supporting_refs):
                        refs_str = ", ".join([f"[{ref_num}]" for ref_num in supporting_refs])
                        st.markdown(f"  - **{i+1}. {theme_summary}** (Supported by Refs: {refs_str})")
                    else:
                        st.markdown(f"  - **{i+1}. {theme_summary}**")
                    
                    # Display evidence in collapsible section
                    if evidence_snippets:
                        with st.expander(f"Show Evidence for Theme {i+1}", expanded=False):
                            for evidence in evidence_snippets:
                                st.markdown(f"**Text:**\n_{evidence.get('text', 'No text available')}_")
                                st.markdown(f"**Location:** Doc ID: `{evidence.get('source_doc_id', 'N/A')}`, Page {evidence.get('page', 'N/A')}, Paragraph {evidence.get('paragraph', 'N/A')}")
                                st.markdown("---")
            else:
                st.markdown("\n*No specific themes were identified for this query.*")

            # 3. Display references
            references = content.get('references', [])
            if references:
                st.markdown("\n**References:**")
                sorted_references = sorted(references, key=lambda x: x.get('reference_number', 0))
                for ref_data in sorted_references:
                    ref_num = ref_data.get('reference_number', 'N/A')
                    file_name = ref_data.get('file_name', 'Unknown File')
                    source_id = ref_data.get('source_doc_id', 'Unknown Source ID')
                    st.markdown(f"  - **[{ref_num}]** {file_name} (Source ID: `{source_id}`)")

            # 4. Display LLM thought process
            synthesized_expert_answer = content.get('synthesized_expert_answer', None)
            if synthesized_expert_answer:
                with st.expander("Show LLM Thought Process", expanded=False):
                    st.markdown("### LLM Analysis and Synthesis")
                    st.markdown(synthesized_expert_answer)

            # 5. Display document details in a tabular format
            document_details = content.get('document_details', [])
            if document_details:
                with st.expander("Show Document Details", expanded=False):
                    st.markdown("### Document Details")
                    for doc in document_details:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            st.markdown(f"**Doc ID:**\n`{doc.get('source_doc_id', 'N/A')}`")
                        with col2:
                            st.markdown(f"**Content:**\n_{doc.get('extracted_answer', 'No content available')}_")
                        with col3:
                            st.markdown(f"**Citation:**\nPage: {doc.get('page_number', 'N/A')}\nPara: {doc.get('paragraph_number', 'N/A')}")
                        st.markdown("---")

            # Debug info
            retrieved_ids = content.get('retrieved_context_document_ids', [])
            if retrieved_ids:
                st.markdown(f"\n_(Debug: Context drawn from document IDs: {', '.join(retrieved_ids)})_")
        else: 
            st.markdown(str(content))

# --- Streamlit App ---
st.set_page_config(page_title="DocBot - Document Research & Theme ID", layout="wide")

st.title("📄 DocBot: Document Research & Theme Identifier")
st.caption("Upload your documents (PDFs, images). Documents will be processed using fast, rule-based chunking.")

# --- Initialize session state ---
default_session_state = {
    "messages": [],
    "processed_doc_ids": set(),
    "uploaded_file_details": [],
    "user_has_been_warned_about_processing": False,
    "processing_active_message_placeholder": None,
    "selected_collection": None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Sidebar for Collection and Document Management ---
with st.sidebar:
    st.header("📚 Collection & Document Management")
    
    # Collection Management Section
    st.subheader("Collection Management")
    
    # Get existing collections
    collections = get_collections()
    collection_names = [col["name"] for col in collections]
    
    # Create new collection
    with st.expander("Create New Collection", expanded=False):
        new_collection_name = st.text_input("Collection Name")
        if st.button("Create Collection", use_container_width=True):
            if new_collection_name:
                if create_collection(new_collection_name):
                    st.success(f"Collection '{new_collection_name}' created successfully!")
                    st.rerun()
            else:
                st.warning("Please enter a collection name")
    
    # Select collection
    if collection_names:
        st.session_state.selected_collection = st.selectbox(
            "Select Collection",
            options=collection_names,
            index=0
        )
        
        # Delete collection option
        with st.expander("Delete Collection", expanded=False):
            st.warning("⚠️ This action cannot be undone!")
            confirm_delete = st.checkbox("I confirm I want to delete this collection")
            if st.button("Delete Collection", disabled=not confirm_delete, use_container_width=True):
                if delete_collection(st.session_state.selected_collection):
                    st.success(f"Collection '{st.session_state.selected_collection}' deleted successfully!")
                    st.rerun()
    else:
        st.info("No collections available. Create a new collection to get started.")
    
    st.markdown("---")
    
    # Document Upload Section
    st.subheader("Document Upload")
    st.info("Using fast, rule-based document chunking.")

    if not st.session_state.selected_collection:
        st.warning("Please select or create a collection first")
    else:
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF, PNG, JPG, TIFF)",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"],
            accept_multiple_files=True,
            key="file_uploader" 
        )

    if uploaded_files:
        if st.button("Process Uploaded Documents", type="primary", use_container_width=True):
            files_to_send_for_api = []
            for uploaded_file_widget_instance in uploaded_files: 
                files_to_send_for_api.append(
                    ("files", (uploaded_file_widget_instance.name, uploaded_file_widget_instance.getvalue(), uploaded_file_widget_instance.type))
                )

            if files_to_send_for_api:
                if st.session_state.processing_active_message_placeholder is None:
                        st.session_state.processing_active_message_placeholder = st.empty()
                
                with st.session_state.processing_active_message_placeholder.container():
                    st.info(
                        (
                            "🚀 **Processing initiated using fast rule-based chunking!** "
                            "Your documents are being queued. This involves text extraction and OCR (for images). "
                            "Please monitor backend logs. You can query once processing seems complete."
                        )
                    )
                st.session_state.user_has_been_warned_about_processing = True

                try:
                    with st.spinner(f"Uploading & queueing {len(files_to_send_for_api)} document(s)..."):
                        response = requests.post(
                            DOCUMENTS_UPLOAD_URL, 
                            files=files_to_send_for_api,
                                params={"collection": st.session_state.selected_collection},
                                timeout=60
                        ) 
                        response.raise_for_status() 
                        
                        results = response.json()
                        current_uploads = []
                        successful_queues = 0
                        for res_item in results:
                            file_name = res_item.get("file_name", "Unknown file")
                            status = res_item.get("status", "failed")
                            doc_id = res_item.get("source_doc_id")
                            mode_used = res_item.get("processing_mode_used", "fast_rule_based") 
                            
                            if status == "queued_for_processing":
                                st.success(f"'{file_name}' (ID: {doc_id}) queued for {mode_used} processing.")
                                current_uploads.append({"name": file_name, "status": f"Queued ({mode_used})", "doc_id": doc_id})
                                successful_queues +=1
                            else:
                                st.error(f"Failed to queue '{file_name}': {res_item.get('message', 'Unknown error')}")
                                current_uploads.append({"name": file_name, "status": f"Failed: {res_item.get('message', '')[:50]}...", "doc_id": doc_id})
                        
                        existing_names = {f['name'] for f in st.session_state.uploaded_file_details}
                        for up_detail in current_uploads:
                            if up_detail['name'] not in existing_names:
                                st.session_state.uploaded_file_details.append(up_detail)
                                existing_names.add(up_detail['name'])
                            else: 
                                for i, existing_f in enumerate(st.session_state.uploaded_file_details):
                                    if existing_f['name'] == up_detail['name']:
                                        st.session_state.uploaded_file_details[i] = up_detail
                                        break
                        if successful_queues == 0 and files_to_send_for_api:
                             if st.session_state.processing_active_message_placeholder:
                                st.session_state.processing_active_message_placeholder.error("⚠️ No files were successfully queued.")
                except Exception as e:
                    st.error(f"Error during upload: {e}")
                    if st.session_state.processing_active_message_placeholder:
                        st.session_state.processing_active_message_placeholder.empty() 
            else:
                st.warning("No files selected to upload.")
    
    st.markdown("---")
    st.subheader("Uploaded Documents Log")
    if st.session_state.uploaded_file_details:
        for detail in st.session_state.uploaded_file_details:
            status_emoji = "⏳" if "Queued" in detail['status'] else "❌" 
            st.markdown(f"- {status_emoji} **{detail['name']}** (ID: `{detail.get('doc_id', 'N/A')}` | Status: {detail['status']})")
    else:
        st.info("No documents uploaded in this session yet.")
    
    if st.button("Clear Chat, Logs & Processing Message", use_container_width=True):
        reset_chat_history()
        st.session_state.uploaded_file_details = []
        st.session_state.processing_info_message = None 
        st.rerun()

# --- Main Chat Interface ---
st.header("💬 Chat with Your Documents")

if "processing_info_message" in st.session_state and st.session_state.processing_info_message:
    if st.session_state.processing_active_message_placeholder is None:
        st.session_state.processing_active_message_placeholder = st.empty()
    st.session_state.processing_active_message_placeholder.info(st.session_state.processing_info_message)

for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.processing_active_message_placeholder:
        st.session_state.processing_active_message_placeholder.empty()
        st.session_state.processing_active_message_placeholder = None 
        st.session_state.processing_info_message = None 
        st.session_state.user_has_been_warned_about_processing = False 

    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)

    with st.spinner("Thinking... (This may take a moment for complex queries)"):
        try:
            payload = {
                "query": prompt,
                "collection": st.session_state.selected_collection
            }
            response = requests.post(CHAT_QUERY_URL, json=payload, timeout=240) 
            response.raise_for_status()
            assistant_response_data = response.json()
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response_data})
            display_chat_message("assistant", assistant_response_data)

            if isinstance(assistant_response_data, dict):
                for doc_id in assistant_response_data.get('retrieved_context_document_ids', []):
                    st.session_state.processed_doc_ids.add(doc_id)
        except requests.exceptions.Timeout:
            st.error("The request to the backend timed out. The server might be busy or the query is too complex.")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I took too long to respond. Please try again or simplify your query."})
            display_chat_message("assistant", "Sorry, I took too long to respond. Please try again or simplify your query.")
        except requests.exceptions.RequestException as e: 
            st.error(f"Error communicating with the backend: {e}")
            error_message_ui = f"Sorry, I encountered an error trying to process your request. Please check the backend server. (Error: {str(e)[:100]})"
            st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
            display_chat_message("assistant", error_message_ui)
        except json.JSONDecodeError:
            st.error("Received an invalid response from the backend (not valid JSON). Please check server logs.")
            error_message_ui = "Sorry, the backend response was not in the expected format."
            st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
            display_chat_message("assistant", error_message_ui)
        except Exception as e: 
            st.error(f"An unexpected error occurred in the frontend: {e}")
            error_message_ui = f"An unexpected error occurred: {str(e)[:150]}..."
            st.session_state.messages.append({"role": "assistant", "content": error_message_ui})
            display_chat_message("assistant", error_message_ui)
