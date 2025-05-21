import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os
from datetime import datetime

# Constants
API_BASE_URL = "http://localhost:8000/api/v1"
COLLECTIONS_ENDPOINT = f"{API_BASE_URL}/collections"
DOCUMENTS_ENDPOINT = f"{API_BASE_URL}/documents"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

def get_collections() -> List[Dict[str, Any]]:
    """Get list of collections from the API."""
    try:
        response = requests.get(COLLECTIONS_ENDPOINT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        return []

def create_collection(name: str) -> bool:
    """Create a new collection."""
    try:
        response = requests.post(
            COLLECTIONS_ENDPOINT,
            json={"name": name}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return False

def delete_collection(name: str) -> bool:
    """Delete a collection."""
    try:
        response = requests.delete(f"{COLLECTIONS_ENDPOINT}/{name}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

def upload_document(file: Any, collection_name: str) -> bool:
    """Upload a document to the specified collection."""
    try:
        files = {"file": file}
        response = requests.post(
            f"{DOCUMENTS_ENDPOINT}/upload",
            files=files,
            params={"collection": collection_name}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error uploading document: {str(e)}")
        return False

def query_documents(query: str, collection_name: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Query documents from the specified collection."""
    try:
        response = requests.post(
            f"{DOCUMENTS_ENDPOINT}/query",
            json={
                "query": query,
                "collection": collection_name,
                "n_results": n_results
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return []

def chat_with_documents(query: str, collection_name: str) -> Dict[str, Any]:
    """Send a chat query to the specified collection."""
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={
                "query": query,
                "collection": collection_name
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error in chat: {str(e)}")
        return {"error": str(e)}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("Collection Management")
    
    # Get existing collections
    collections = get_collections()
    collection_names = [col["name"] for col in collections]
    
    # Collection selection
    selected_collection = st.selectbox(
        "Select Collection",
        options=collection_names,
        index=0 if collection_names else None
    )
    
    # Create new collection
    st.subheader("Create New Collection")
    new_collection_name = st.text_input("Collection Name")
    if st.button("Create Collection"):
        if new_collection_name:
            if create_collection(new_collection_name):
                st.success(f"Collection '{new_collection_name}' created successfully!")
                st.rerun()
        else:
            st.warning("Please enter a collection name")
    
    # Delete collection
    st.subheader("Delete Collection")
    if selected_collection:
        confirm_delete = st.checkbox("I confirm I want to delete this collection")
        if st.button("Delete Collection", disabled=not confirm_delete):
            if delete_collection(selected_collection):
                st.success(f"Collection '{selected_collection}' deleted successfully!")
                st.rerun()

# Main content
st.title("Document Research Assistant")

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
if uploaded_file and selected_collection:
    if st.button("Process Document"):
        if upload_document(uploaded_file, selected_collection):
            st.success("Document processed successfully!")

# Chat interface
st.subheader("Chat with Documents")
user_query = st.text_input("Ask a question about your documents")

if user_query and selected_collection:
    if st.button("Send"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Get response from API
        response = chat_with_documents(user_query, selected_collection)
        
        if "error" not in response:
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response["response"]})
        else:
            st.error(response["error"])

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Document search
st.subheader("Search Documents")
search_query = st.text_input("Search query")
n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

if search_query and selected_collection:
    if st.button("Search"):
        results = query_documents(search_query, selected_collection, n_results)
        
        if results:
            for i, result in enumerate(results, 1):
                with st.expander(f"Result {i}"):
                    st.write("Content:", result["content"])
                    st.write("Metadata:", result["metadata"])
                    st.write("Distance:", result["distance"])
        else:
            st.info("No results found") 