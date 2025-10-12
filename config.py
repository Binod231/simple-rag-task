import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
PINECONE_INDEX_NAME = "rag-project-index"

if not PINECONE_API_KEY or not PINECONE_API_ENV:
    raise ValueError("Pinecone API key and environment must be set in the .env file")

# --- Model Configuration ---
# Specifies the embedding model to use for vectorizing text.
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Specifies the local language model to use for generating answers.
# Make sure you have run 'ollama pull llama3.2' in your terminal.
LLM_MODEL = "llama3.2"