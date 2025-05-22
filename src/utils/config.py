import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMADB_PATH = os.getenv("CHROMADB_PATH", "data/vector_db")

# Validate critical variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env file")