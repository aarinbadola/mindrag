import os

from dotenv import load_dotenv

load_dotenv()

IS_HUGGINGFACE = os.path.exists("/data")

CHROMADB_PATH = "/data/chromadb" if IS_HUGGINGFACE else "./data/chromadb"
SQLITE_PATH = "/data/rag.db" if IS_HUGGINGFACE else "./data/rag.db"
DOCS_FOLDER = "./docs"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CONFIDENCE_THRESHOLD = 0.6
RERANKER_SCORE_THRESHOLD = 0.0
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
HISTORY_MESSAGES = 4
CLASSIFY_HISTORY = 1
SUMMARY_BATCH_SIZE = 20
MAX_DOCUMENTS = 10
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Document resolution for summarization intent — embedding cosine similarity
# against registered document titles.
DOC_RESOLUTION_LOW_FLOOR = 0.30       # below this top score, no document referenced
DOC_RESOLUTION_AUTO_MARGIN = 0.20     # gap between top two scores; below this is ambiguous
MAX_MULTI_DOCUMENTS = 3               # cap on documents auto-combined or shown in the popup
MULTI_DOCUMENT_KEYWORDS = ("and", "compare", "versus", "with", "both", "difference between")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL = "llama-3.1-8b-instant"


def get_retrieval_k(num_documents: int) -> int:
    return max(15, num_documents * 2)


def get_top_k(num_documents: int) -> int:
    if num_documents <= 3:
        return 3
    elif num_documents <= 6:
        return 5
    elif num_documents <= 10:
        return 7
    else:
        return 10
