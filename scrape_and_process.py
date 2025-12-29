"""
Local Multimodal preprocessing pipeline using Chroma
"""

import os
import re
import io
import time
import sqlite3
import hashlib
import trafilatura
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin
from dataclasses import dataclass



import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
import torch
from open_clip import create_model_and_transforms, get_tokenizer
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME,
    CLIP_MODEL_NAME, BASE_URL, RAW_HTML_DIR,
    REQUEST_DELAY, CHUNK_SIZE, DEFAULT_HEADERS, URL_INDEXES,
    TITLE_SELECTOR, TITLE_URL, DB_PATH, METADATA, logger,
    RETRY_STATUSES, SKIP_STATUSES, SkipSignal, RetrySignal
)


# Create directories
Path(CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Initialize SQLite for metadata
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE,
    title TEXT,
    date TEXT,
    clean_text TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    article_id TEXT,
    path TEXT,
    caption TEXT,
    FOREIGN KEY(article_id) REFERENCES articles(id)
)
""")
conn.commit()

# Initialize Chroma (persistent local DB)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIRECTORY,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(CLIP_MODEL_NAME, pretrained="laion2b_s32b_b82k")
tokenizer = get_tokenizer(CLIP_MODEL_NAME)
model.to(device)
model.eval()

url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
)

def is_valid_url(url: str) -> bool:
    """Return True if the url matches the given regex pattern."""
    return bool(url_pattern.fullmatch(url))

def get_url(index: int) -> str:
    url = BASE_URL.format(index)
    if not is_valid_url(url):
        logger.warning(f"Article url is formatted incorrectly: `{url}`")
    return url

def get_title_url(index: int) -> str:
    url = TITLE_URL.format(index)
    if not is_valid_url(url):
        logger.warning(f"Article title url is formatted incorrectly: `{url}`")
    return url

def fetch_title(index: int) -> str:
    title_url = get_title_url(index)
    title_response = requests.get(title_url)
    error_if_status(title_response.status_code)

    title_soup = BeautifulSoup(title_response.text, "html.parser")
    title = title_soup.select_one(TITLE_SELECTOR).text.strip()

    return title

def fetch_article(index: int) -> BeautifulSoup:
    url = get_url(index)
    response = requests.get(url)
    error_if_status(response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")
    
    return soup

def parse_date(soup: BeautifulSoup) -> str | None:
    raw_date = soup.find("div", class_="mt-1")
    if not raw_date: return

    date = raw_date.get_text(strip=True)
    if not validate_date(date): return

    return date

def validate_date(date_string: str) -> bool:
    try:
        # %b = Abbreviated Month (Jan, Feb, ...)
        # %d = Day (01, 31, ...)
        # %Y = 4-digit Year (2025)
        datetime.strptime(date_string, "%b %d, %Y")
        return True
    except ValueError:
        return False

def parse_text(soup: BeautifulSoup) -> str | None:
    text = trafilatura.extract(str(soup), include_images=False)
    return text

def break_into_chunks(text: str) -> list[str]:
    # Break for paragraphs first so that we dont break mid idea
    paragraphs = (p.strip() for p in text.split("\n") if p.strip())
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 > CHUNK_SIZE:
            if current:
                chunks.append(current)
            current = para
        else:
            current += ("\n" + para if current else para)
    if current:
        chunks.append(current)

    return chunks

def save_images_get_paths_captions(article_index: int,
                                   soup: BeautifulSoup) -> list[tuple[str, str]]:
    image_urls_captions = []
    img_tags = soup.find_all("img")
    for idx, img in enumerate(img_tags):
        img_url = img.get("src")
        if not is_valid_url(img_url):
            logger.info(f"Invalid image url: `{img_url}`")
            continue

        caption = img.get("alt", "") or ""
        img_id = hashlib.md5(img_url.encode()).hexdigest()
        cursor.execute("""
        INSERT OR REPLACE INTO images (id, article_id, path, caption)
        VALUES (?, ?, ?, ?)
        """, (img_id, article_index, img_url, caption))
        image_urls_captions.append((img_url, caption))

    conn.commit()
    return image_urls_captions

@dataclass
class ArticleInfo:
    article_id: int
    title: str
    date: str
    url: str
    text: str
    soup: BeautifulSoup

def add_to_chroma(text_chunks: list[str],
                  image_urls: list[str],
                  info: ArticleInfo) -> None:

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    base_metadata = {
        "type": None,
        "article_id": info.article_id,
        "title": info.title,
        "date": info.date,
        "url": info.url,
        "source": None
    }

    _add_text_to_chroma(
        text_chunks,
        info,
        base_metadata,
        documents,
        embeddings,
        metadatas,
        ids
    )
    _add_images_to_chroma(
        image_urls,
        info,
        base_metadata,
        documents,
        embeddings,
        metadatas,
        ids
    )
    if not ids: return
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def _add_text_to_chroma(chunks: list[str],
                        info: ArticleInfo,
                        base_metadata: METADATA,
                        documents: list[str],
                        embeddings: list[float],
                        metadatas: list[METADATA],
                        ids: list[int]) -> None:

    for i, chunk in enumerate(chunks):
        ids.append(f"text_{info.article_id}_{i}")
        documents.append(chunk)
        embeddings.append(get_text_embedding(chunk))
        metadata = base_metadata.copy()
        metadata["type"] = "text"
        metadata["source"] = "text"
        
        metadatas.append(metadata)

def fetch_image(url: str) -> bytes | None:
    try:
        response = requests.get(url)
    except Exception as e:
        logger.info(e)
        return

    if response.status_code != 200: return
    return response.content

def _add_images_to_chroma(image_ulrs_captions: list[tuple[str, str]],
                          info: ArticleInfo,
                          base_metadata: METADATA,
                          documents: list[str],
                          embeddings: list[float],
                          metadatas: list[METADATA],
                          ids: list[int]) -> None:
                        
    for i, (image_url, caption) in enumerate(image_ulrs_captions):
        image_data = fetch_image(image_url)
        if not image_data: continue

        ids.append(f"image_{info.article_id}_{i}")
        documents.append(caption or "Image from article")
        embeddings.append(get_image_embedding(image_data))

        metadata = base_metadata.copy()
        metadata["type"] = "image"
        metadata["source"] = "image"
        metadata["image_url"] = image_url
        metadata["caption"] = caption

        metadatas.append(metadata)

@torch.no_grad()
def get_text_embedding(text: str):
    tokens = tokenizer([text]).to(device)
    embedding = model.encode_text(tokens)
    return embedding.cpu().numpy()[0].tolist()

@torch.no_grad()
def get_image_embedding(img_bytes: bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    embedding = model.encode_image(image_input)
    return embedding.cpu().numpy()[0].tolist()

def error_if_status(status: int) -> None:
    if status // 100 == 2:
        return

    if status in RETRY_STATUSES:
        raise RetrySignal(status)

    if status in SKIP_STATUSES:
        raise SkipSignal(status)

    logger.warning(f"No status check has been found. Ignoring the following status: {status}")


def fetch_article_info(index) -> ArticleInfo:
    title = fetch_title(index)
    soup = fetch_article(index)

    date = parse_date(soup)
    text = parse_text(soup)
    if not date or not text:
        raise SkipSignal(200)

    return ArticleInfo(index, title, date, get_url(index), text, soup)

def unsafe_handle_article(index: int) -> bool:
    info = fetch_article_info(index)
    
    cursor.execute("""
    INSERT OR REPLACE INTO articles (id, url, title, date, clean_text)
    VALUES (?, ?, ?, ?, ?)
    """, (index, get_url(index), info.title, info.date, info.text))
    conn.commit()

    text_chunks = break_into_chunks(info.text)
    image_urls_captions = save_images_get_paths_captions(index, info.soup)
    add_to_chroma(text_chunks, image_urls_captions, info)
    return 0

def handle_article(index: int) -> bool:
    try:
        unsafe_handle_article(index)
        logger.info(f"Handled article {index}")
        return 1

    except SkipSignal as skip:
        status = skip.args[0]
        logger.error(f"Skipped article {index} with {status}")
        return 1

    except RetrySignal as retry:
        status = retry.args[0]
    
    logger.warning(f"Fetch attempt failed for {index} with {status}. Retrying...")
    return 0

def fetch_loop_article(index: int) -> None:
    stop = handle_article(index)
    if stop: return

    max_retries = 6
    timeout = 5
    time.sleep(timeout)

    for _ in range(max_retries):
        stop = handle_article(index)
        if stop: return
        
        logger.warning(f"Increasing fetch timeout to {timeout}s")
        timeout *= 2
        time.sleep(timeout)

def main():
    for i in tqdm(URL_INDEXES):
        fetch_loop_article(i)
        time.sleep(REQUEST_DELAY)


if __name__ == "__main__":
    main()
