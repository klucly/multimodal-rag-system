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
from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    IMAGES_DIR,
    CLIP_MODEL_NAME,
    GEMINI_API_KEY,
    DB_PATH,
    DEFAULT_HEADERS,
    MODEL_NAME
)
from open_clip import create_model_and_transforms, get_tokenizer
import torch
import pickle
from dataclasses import dataclass

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
    # Extract content between <[IMAGES]> and <[IMAGES]> (or end if malformed)
    images_match = re.search(r'<\[IMAGES\]>\s*([\s\S]*?)\s*<\[IMAGES\]>', text)
    image_urls = []
    if images_match:
        content = images_match.group(1)
        image_urls = [
            line.strip() for line in content.splitlines()
            if line.strip().startswith(('http://', 'https://'))
        ]
    
    # Extract content between <[ARTICLES]> and <[ARTICLES]>
    links_match = re.search(r'<\[ARTICLES\]>\s*([\s\S]*?)\s*<\[ARTICLES\]>', text)
    article_links = []
    if links_match:
        content = links_match.group(1)
        article_links = [
            line.strip() for line in content.splitlines()
            if line.strip().startswith(('http://', 'https://'))
        ]
    
    # Extract everything after <[RESPONSE]> until the end
    response_match = re.search(r'<\[RESPONSE\]>\s*([\s\S]*)', text)
    response_text = response_match.group(1).strip() if response_match else ""
    
    return ParsedResponse(
        images=image_urls,
        links=article_links,
        response=response_text
    )
def find_data(query: str, article_count: int = 10) -> tuple[SOURCE_TEXTS, SOURCE_IMAGES]:
    print(f"\nSearching for: {query}")
    results = search(query, article_count=article_count)

    # Collect text results and unique article_ids
    text_results = []
    article_ids = set()

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        if meta["type"] == "text":
            text_results.append({
                "title": meta["title"],
                "date": meta["date"],
                "url": meta["url"],
                "text": doc
            })
            article_ids.add(meta["article_id"])

    # Fetch all image URLs from matching articles (from SQLite)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    image_data = []
    if article_ids:
        placeholders = ','.join(['?'] * len(article_ids))
        cursor.execute(f"""
        SELECT path, caption FROM images WHERE article_id IN ({placeholders})
        """, tuple(article_ids))
        image_data = cursor.fetchall()
    conn.close()

    return text_results, image_data

def build_final_query(user_query: str, text_results: SOURCE_TEXTS, image_data: SOURCE_IMAGES) -> list[str | types.Part]:
    prompt_text = build_final_prompt(user_query, text_results)
    contents = [prompt_text]

    # Add images as inline bytes (up to 10-15, Gemini handles well)
    for img_url, caption in image_data:
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
                contents.append(f"Media is in an unavailable format `{mime_type}`")
                continue

            contents.append(
                types.Part.from_bytes(
                    data=response.content,
                    mime_type=mime_type
                )
            )
        except Exception as e:
            print(f"Error fetching image {img_url}: {e}")

    print(f"Retrieved {len(text_results)} text chunks.")
    print(f"Attached {len(image_data)} images from those articles.")

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

def query_gemini(user_query: str, k: int = 10) -> tuple[ParsedResponse, SOURCE_DATA]:
    text, images = find_data(user_query, k)

    final_query = build_final_query(user_query, text, images)
    response = generate_response(final_query)

    return extract_images_links_and_response(response), (text, images)

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

        # try:
        parsed_response, _ = query_gemini(query)
        display_results(parsed_response)
        # except Exception as e:
        #     print(f"Error: {e}")

if __name__ == "__main__":
    main()