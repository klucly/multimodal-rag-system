"""
Multimodal RAG Query Interface using the new Google GenAI SDK (google-genai).
"""

import sqlite3
import requests
from io import BytesIO
import os
import re
import textwrap
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings
import mimetypes
import hashlib
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CLIP_MODEL_NAME,
    GEMINI_API_KEY,
    DB_PATH,
    DEFAULT_HEADERS,
    MODEL_NAME
)
from open_clip import create_model_and_transforms, get_tokenizer
import torch
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

# (list of image urls, answer)
PARSED_RESPONSE = tuple[list[str], str]

SOURCE_TEXTS = list[dict[str, str]]
SOURCE_IMAGES = list[tuple[str, str]]
SOURCE_DATA = tuple[SOURCE_TEXTS, SOURCE_IMAGES]

# Create client (for Gemini Developer API)
client = genai.Client(api_key=GEMINI_API_KEY)

# Load Chroma (unchanged)
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)

# Load CLIP for query embedding (unchanged)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = create_model_and_transforms(CLIP_MODEL_NAME, pretrained="laion2b_s32b_b82k")
tokenizer = get_tokenizer(CLIP_MODEL_NAME)
clip_model.to(device)
clip_model.eval()


@torch.no_grad()
def embed_query(text: str):
    tokens = tokenizer([text]).to(device)
    embedding = clip_model.encode_text(tokens)
    return embedding.cpu().numpy()[0].tolist()

def search(query: str, article_count: int = 10) -> dict[str, ...]:
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=article_count,
        include=["documents", "metadatas", "distances"]
    )
    return results

def build_final_prompt(user_query: str, text_results: SOURCE_TEXTS) -> str:
    text_parts = [f"Source: {r['title']} ({r['date']})\nURL: {r['url']}\n{r['text']}\n"
                  for r in text_results]

    context_text = "\n\n".join(text_parts)

    prompt = f"""
### System Role
You are an expert AI assistant specialized in "The Batch" newsletter by DeepLearning.AI. Your task is to provide accurate, factual summaries based strictly on the provided context.

### Constraints
1. **Source Grounding**: Use ONLY the provided context. If the answer is not contained within the context, state: "I'm sorry, but the provided issues of The Batch do not contain information regarding this topic."
2. **No Hallucinations**: Do not use external knowledge or invent details.
3. **Images and Articles**: Extract image URLs and article links only if they are directly relevant to the specific answer. If no relevant images or article links exist in the context, leave the <[IMAGES]> and <[ARTICLES]> sections empty.

### Response Format
Your response must follow this exact structure:

<[IMAGES]>
https://example.com/image1.jpg
https://example.com/image2.png
<[IMAGES]>

<[ARTICLES]>
https://www.deeplearning.ai/the-batch/article-title-1
https://www.deeplearning.ai/the-batch/another-article-title
<[ARTICLES]>

<[RESPONSE]>
[Concise answer with inline citations, max 500 tokens]
---

### Context
{context_text}
---

### User Question
{user_query}
"""
    return prompt


@dataclass
class ParsedResponse:
    images: list[str]
    links: list[str]
    response: str

def extract_images_links_and_response(text: str) -> ParsedResponse:
    images_match = re.search(r'<\[IMAGES\]>\s*([\s\S]*?)\s*<\[IMAGES\]>', text)
    image_urls = []
    if images_match:
        content = images_match.group(1)
        image_urls = [
            line.strip() for line in content.splitlines()
            if line.strip().startswith(('http://', 'https://'))
        ]
    
    links_match = re.search(r'<\[ARTICLES\]>\s*([\s\S]*?)\s*<\[ARTICLES\]>', text)
    article_links = []
    if links_match:
        content = links_match.group(1)
        article_links = [
            line.strip() for line in content.splitlines()
            if line.strip().startswith(('http://', 'https://'))
        ]
    
    response_match = re.search(r'<\[RESPONSE\]>\s*([\s\S]*)', text)
    response_text = response_match.group(1).strip() if response_match else ""
    
    return ParsedResponse(
        images=image_urls,
        links=article_links,
        response=response_text
    )


def _parse_chunk_id(chunk_id: str) -> Tuple[int, int]:
    """Parse 'text_{article_id}_{chunk_index}' -> (article_id, chunk_index)"""
    parts = chunk_id.split('_')
    if len(parts) != 3 or parts[0] != 'text':
        raise ValueError(f"Invalid chunk ID format: {chunk_id}")
    return int(parts[1]), int(parts[2])

import hashlib
from typing import Dict, List, Set, Tuple


def find_data(query: str, article_count: int = 10) -> tuple[SOURCE_TEXTS, SOURCE_IMAGES]:
    print(f"\nSearching for: {query}")
    results = search(query, article_count=article_count * 5)  # Retrieve top 50 candidates

    # Step 1: Identify hit chunks and their article_ids + indices
    hit_records: List[Tuple[int, int, str, dict]] = []  # (article_id, chunk_idx, document, metadata)
    hit_article_ids: Set[int] = set()

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]  # ids are always returned
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]

        if meta["type"] != "text":
            continue

        try:
            article_id, chunk_idx = _parse_chunk_id(chunk_id)
        except ValueError:
            continue

        hit_article_ids.add(article_id)
        hit_records.append((article_id, chunk_idx, doc, meta))

    if not hit_article_ids:
        return [], []

    # Step 2: Fetch ALL text chunks for the relevant articles
    # Note: no "ids" in include — it's returned automatically
    full_results = collection.get(
        where={
            "$and": [
                {"type": {"$eq": "text"}},
                {"article_id": {"$in": list(hit_article_ids)}}
            ]
        },
        include=["documents", "metadatas"]  # Only these are needed
    )

    # Build article → list of (chunk_idx, doc, meta) using returned ids
    all_article_chunks: Dict[int, List[Tuple[int, str, dict]]] = {}
    for i, full_id in enumerate(full_results["ids"]):
        try:
            aid, cidx = _parse_chunk_id(full_id)
            doc = full_results["documents"][i]
            meta = full_results["metadatas"][i]
            if aid not in all_article_chunks:
                all_article_chunks[aid] = []
            all_article_chunks[aid].append((cidx, doc, meta))
        except:
            continue

    # Sort chunks by index within each article
    for aid in all_article_chunks:
        all_article_chunks[aid].sort(key=lambda x: x[0])

    # Step 3: Build expanded contexts
    expanded_contexts: List[dict] = []
    seen_signatures: Set[str] = set()
    WINDOW = 5

    hit_indices_per_article: Dict[int, Set[int]] = {}
    for aid, cidx, _, _ in hit_records:
        if aid not in hit_indices_per_article:
            hit_indices_per_article[aid] = set()
        hit_indices_per_article[aid].add(cidx)

    for article_id in hit_article_ids:
        if article_id not in all_article_chunks or article_id not in hit_indices_per_article:
            continue

        chunks = all_article_chunks[article_id]
        hit_idxs = hit_indices_per_article[article_id]

        if not hit_idxs:
            continue

        min_hit = min(hit_idxs)
        max_hit = max(hit_idxs)
        start_idx = max(0, min_hit - WINDOW)
        end_idx = min(len(chunks) - 1, max_hit + WINDOW)

        selected = chunks[start_idx : end_idx + 1]
        combined_text = "\n\n".join(doc for _, doc, _ in selected)

        # Deduplicate
        sig = hashlib.md5(combined_text.encode('utf-8', errors='ignore')).hexdigest()
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        # Metadata from first chunk in window
        _, _, sample_meta = selected[0]

        expanded_contexts.append({
            "title": sample_meta["title"],
            "date": sample_meta["date"],
            "url": sample_meta["url"],
            "text": combined_text
        })

        if len(expanded_contexts) >= article_count:
            break

    # Step 4: Fetch images
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    image_data = []
    if hit_article_ids:
        placeholders = ','.join(['?'] * len(hit_article_ids))
        cursor.execute(f"""
        SELECT path, caption FROM images WHERE article_id IN ({placeholders})
        """, tuple(hit_article_ids))
        image_data = cursor.fetchall()
    conn.close()

    print(f"Built {len(expanded_contexts)} expanded context passages (each up to {2*WINDOW + 1} chunks wide).")
    return expanded_contexts, image_data


def build_final_query(user_query: str, text_results: SOURCE_TEXTS, image_data: SOURCE_IMAGES) -> list[str | types.Part]:
    prompt_text = build_final_prompt(user_query, text_results)
    contents = [prompt_text]

    # Add images as inline bytes (up to reasonable limit)
    added = 0
    max_images = 15
    for img_url, caption in image_data[:20]:
        if added >= max_images:
            break
        try:
            response = requests.get(img_url, headers=DEFAULT_HEADERS, timeout=10)
            if response.status_code != 200:
                continue

            if caption:
                contents.append(f"Caption for the following media: `{caption}`. Url: `{img_url}`. media:")
            else:
                contents.append(f"No caption for the following media. Url: `{img_url}`. media:")

            mime_type, _ = mimetypes.guess_type(img_url)
            if mime_type not in {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}:
                continue

            contents.append(
                types.Part.from_bytes(
                    data=response.content,
                    mime_type=mime_type
                )
            )
            added += 1
        except Exception as e:
            print(f"Error fetching image {img_url}: {e}")

    print(f"Retrieved {len(text_results)} expanded context passages.")
    print(f"Attached {added} images from relevant articles.")

    return contents


def generate_response(query: list[str | types.Part]) -> str:
    return client.models.generate_content(
        model=MODEL_NAME,
        contents=query,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048
        )
    ).text


def rough_token_count(text: str) -> int:
    """Very rough estimate: Gemini counts ~4 chars per token on average"""
    return len(text) // 4

MAX_TOKENS_GEMINI_1_5_FLASH = 1_048_576   # context window
MAX_TOKENS_GEMINI_1_5_PRO   = 2_097_152
# Adjust based on your MODEL_NAME, or be conservative
SAFE_CONTEXT_LIMIT = 800_000  # leave headroom for images + response


def query_gemini(user_query: str, k: int = 10) -> tuple[ParsedResponse, SOURCE_DATA]:
    text_results, image_data = find_data(user_query, k)

    # Start with full data
    selected_text = text_results.copy()
    selected_images = image_data.copy()

    # Build prompt once to measure size
    prompt_text = build_final_prompt(user_query, selected_text)
    current_tokens = rough_token_count(prompt_text)

    # Estimate image cost: each image costs roughly 258 tokens + some per-tile cost
    # Conservative: ~1000 tokens per image to be safe
    image_token_cost = len(selected_images) * 1000
    total_estimated = current_tokens + image_token_cost

    print(f"Initial estimate: ~{current_tokens} text tokens + {len(selected_images)} images (~{total_estimated} total)")

    # Gracefully reduce if we're getting close to limits
    while total_estimated > SAFE_CONTEXT_LIMIT and (len(selected_text) > 1 or len(selected_images) > 0):
        if len(selected_text) > 2:  # Keep at least 1-2 articles
            # Remove the last (likely least relevant) article
            removed = selected_text.pop()
            removed_text = f"Source: {removed['title']}...\n{removed['text']}"
            current_tokens -= rough_token_count(removed_text)
        elif len(selected_images) > 0:
            selected_images.pop()  # drop least important images last
            image_token_cost -= 1000

        total_estimated = current_tokens + image_token_cost

    print(f"After pruning: {len(selected_text)} articles, {len(selected_images)} images (~{total_estimated} est. tokens)")

    # Now build the final query with reduced data
    final_query = build_final_query(user_query, selected_text, selected_images)

    # Try generating, with fallback on failure
    try:
        response = generate_response(final_query)
    except Exception as e:
        if "maximum context length" in str(e).lower() or "token" in str(e).lower():
            print("Gemini rejected due to size. Falling back to text-only with minimal context...")
            # Last resort: 1 article, no images
            minimal_text = selected_text[:1] if selected_text else []
            final_query = build_final_query(user_query, minimal_text, [])
            response = generate_response(final_query)
        else:
            raise  # re-raise if not a size issue

    parsed = extract_images_links_and_response(response)
    return parsed, (selected_text, selected_images)  # return actually used sources

def display_results(parsed_response: ParsedResponse) -> None:
    print("\n" + "="*80)
    print("GEMINI ANSWER:")
    print("="*80)
    print(textwrap.fill(parsed_response.response, width=90))
    print("\n" + "-"*80)
    print("SOURCES:")
    print("-"*80)
    for r in parsed_response.links:
        print(f"  {r}\n")

    if not parsed_response.images:
        return

    print("RELEVANT IMAGES:")
    for path in parsed_response.images:
        print(f"  Path: {path}")


def main():
    print("The Batch Multimodal RAG with Gemini (new GenAI SDK)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query:
            continue

        parsed_response, _ = query_gemini(query)
        display_results(parsed_response)


if __name__ == "__main__":
    main()