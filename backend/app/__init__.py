from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file at the project root
project_root = Path(__file__).resolve().parents[2].parents[1]
env_path = project_root / "backend" / ".env"
load_dotenv(dotenv_path=env_path)

# --- API Keys ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

# --- Model Names and URLs ---
GROQ_MODEL : str = os.environ.get("GROQ_MODEL")
OLLAMA_MODEL : str = os.environ.get("OLLAMA_MODEL")
OLLAMA_BASE_URL : str= os.environ.get("OLLAMA_BASE_URL")
DEFAULT_LLM_PROVIDER : str = os.environ.get("DEFAULT_LLM_PROVIDER", "openrouter")
OPENROUTER_API_BASE : str = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
DEFAULT_LLM_MODEL :str= os.environ.get("DEFAULT_LLM_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
EMBEDDING_MODEL_NAME : str = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Vector Store (ChromaDB) Configuration ---
PROJECT_ROOT_DIR = project_root
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", str(PROJECT_ROOT_DIR / "data" / "db"))
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "doc_embeddings_v1")

# --- Document Processing ---
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", str(PROJECT_ROOT_DIR / "data" / "uploads"))

# --- FastAPI Configuration ---
API_V1_STR = os.environ.get("API_V1_STR", "/api/v1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "Document Theme Bot")
