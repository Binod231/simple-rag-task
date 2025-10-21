import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Model Configuration ---
# Specifies the embedding model to use for vectorizing text.
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Specifies the local language model to use for generating answers.
# Make sure you have run 'ollama pull llama3' in your terminal.
LLM_MODEL = "llama3.2"