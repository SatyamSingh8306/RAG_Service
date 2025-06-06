---
title: Doc Theme Bot
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Doc Theme Bot

A document analysis and theme extraction bot built with Streamlit and LangChain.

## Features

- Document upload and processing
- Theme extraction and analysis
- Interactive UI for document exploration
- Powered by LangChain and Hugging Face models

## How to Use

1. Upload your document using the file uploader
2. Wait for the document to be processed
3. Explore the extracted themes and insights
4. Interact with the document through the chat interface

## Technical Details

- Built with Streamlit for the frontend
- Uses LangChain for document processing
- Implements Hugging Face models for analysis
- Docker-based deployment for reliability

🚀 DocBot: Document Research & Theme Identifier

A full-stack AI-powered system built for deep document research, intelligent theme identification, and citation-based answers.

This system enables users to upload large sets of documents, ask natural language questions, and get context-aware, theme-driven answers with fine-grained citations, powered by Retrieval-Augmented Generation (RAG).

🌟 Key Features

📂 Document & Collection Management

Upload and process multiple document types (PDFs, scanned images like PNG/JPG/TIFF)

Built-in OCR using Tesseract for scanned/image-based documents

Rule-based chunking and text preprocessing for high-fidelity semantic search

Named Collection System: Create, select, and delete custom collections to group documents by domain, topic, or project

Each collection is isolated within ChromaDB as its own semantic space

🤖 AI-Powered Research & Analysis

Retrieval-Augmented Generation (RAG) with LLMs via OpenRouter API

Natural language querying over 75+ documents

Theme Extraction: System identifies one or more coherent themes from document responses

Granular Citations: Includes doc ID, page number, and paragraph for every extracted fact

Cross-document synthesis: Aggregates info from all documents to create unified answers

LLM Thought Process Panel: Transparent reasoning section where the LLM elaborates on its understanding using both document content and its own pretrained knowledge

🔍 Advanced Retrieval Techniques

To ensure high-precision, context-aware retrieval, the system leverages two complementary techniques:

✅ Maximal Marginal Relevance (MMR)

Balances relevance and diversity in retrieval

Prevents redundant chunks from dominating

Ensures varied but contextually important information is surfaced

✅ Cross-Encoder Re-Ranking

Refines top-N retrieved chunks using a transformer-based cross-encoder (e.g., MiniLM, BERT)

Scores each query-passage pair deeply at token-level

Improves the semantic relevance of final context passed to the LLM

Result: A highly optimized context for LLMs that boosts factual accuracy, citation traceability, and response diversity.

💬 Frontend Interface (Streamlit)

Modern sidebar with collection creation, selection, and deletion

Drag-and-drop file upload with automatic parsing and storage

Real-time chat interface for user questions and interactive exploration

Collapsible UI panels for:

Themes

Citation breakdown

LLM analysis

Document detail mapping

Per-document result tables for traceability and comparison

📊 Presentation Format

Fully aligned with Wasserstoff's internship requirements:

Theme-wise grouped insights

Collapsible citation view for each theme

LLM Thought Process Section (custom bonus)

Tabular mapping of individual document responses

Visual mapping of reference document IDs supporting each insight

🔧 Tech Stack

Backend: FastAPI

Frontend: Streamlit

Vector Store: ChromaDB

OCR: Tesseract

LLM Access: OpenRouter API

Architecture: Modular, scalable, and cleanly separated by service (parsing, RAG, vector storage, API)

🌟 Outcome

This system goes beyond the core expectations of the internship task:

Adds a multi-collection manager

Includes fine-grained citation evidence

Offers explainability with LLM reasoning

Ready for real-world use cases in legal, research, compliance, and policy domains

🚀 Getting Started

Prerequisites

Python 3.8+

Tesseract OCR (for image processing)

OpenRouter API key (for LLM capabilities)

Installation

Clone the repository:

git clone https://github.com/yourusername/doc_theme_bot.git
cd doc_theme_bot

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

# Create a .env file in the project root
OPENROUTER_API_KEY=your_api_key_here

Running the Application

Start the backend server:

cd backend
uvicorn app.main:app --reload

In a new terminal, start the frontend:

cd frontend
streamlit run ui.py

Access the application at http://localhost:8501

💡 Usage

Create a Collection

Use the sidebar to create a new collection

Collections help organize related documents

Upload Documents

Select your collection

Upload PDF or image files

Monitor processing status in real-time

Query Documents

Type your question in the chat interface

View the AI-generated response

Explore identified themes and evidence

Check document details and citations

🔧 System Architecture

Backend (FastAPI)

Document processing service

RAG service for query handling

Vector store for document indexing

Collection management

Frontend (Streamlit)

User interface

Document upload handling

Chat interface

Results display

📅 API Endpoints

Document Management

POST /api/v1/documents/upload: Upload a single document

POST /api/v1/documents/upload-multiple: Upload multiple documents

GET /api/v1/documents/{doc_id}: Get document details

DELETE /api/v1/documents/{doc_id}: Delete a document

Chat and Query

POST /api/v1/chat/query: Query documents and get themes

Request body: { "query": "your question", "collection": "collection_name" }

Returns: Answer, themes, evidence, and document details

Collection Management

GET /api/v1/collections: List all collections

POST /api/v1/collections: Create a new collection

Request body: { "name": "collection_name" }

DELETE /api/v1/collections/{name}: Delete a collection

GET /api/v1/collections/{name}: Get collection details

GET /api/v1/collections/{name}/documents: List documents in a collection

System

GET /: Root endpoint - Welcome message

GET /api/v1/openapi.json: OpenAPI documentation

🔒 Security

API key management for LLM services

Secure document processing

Input validation and sanitization

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

