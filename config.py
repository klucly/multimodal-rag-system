from dotenv import load_dotenv
import logging
import sys
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Chroma settings
CHROMA_PERSIST_DIRECTORY = "chroma_db"
DB_PATH = "database.sqlite"
CHROMA_COLLECTION_NAME = "the_batch_multimodal"

# Models
CLIP_MODEL_NAME = "ViT-L/14"
MODEL_NAME = "gemini-2.5-flash"

# Website
BASE_URL = "https://www.deeplearning.ai/the-batch/issue-{}/"
TITLE_URL = "https://www.deeplearning.ai/the-batch/tag/issue-{}/"
URL_INDEXES = range(331+1)

# Scraping & processing
TITLE_SELECTOR = r"h2.text-xl.lg\:text-2xl.font-semibold.tracking-tight.leading-tight.text-slate-800.font-primary.mb-2"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
REQUEST_DELAY = 5.0

# Text chunking
CHUNK_SIZE = 300    # Characters (approximate)

# Logger setup
logging_level = logging.WARNING
logger = logging.getLogger()
logger.setLevel(logging_level)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging_level)
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)

METADATA = dict[str, str | int]

RETRY_STATUSES = {408, 429}
SKIP_STATUSES = {400, 401, 403, 404, 405, 409, 410, 422, 500, 502, 504}

class RetrySignal(Exception): ...
class SkipSignal(Exception): ...
