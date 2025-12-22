"""
Multimodal RAG Query Interface using the new Google GenAI SDK (google-genai).
"""

import sqlite3
import requests
from io import BytesIO
import os
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

    context_text = "\n\n".join(text_parts[:5])

    prompt = f"""
### System Role
You are an expert AI assistant specialized in "The Batch" newsletter by DeepLearning.AI. Your task is to provide accurate, factual summaries based strictly on the provided context.

### Constraints
1. **Source Grounding**: Use ONLY the provided context. If the answer is not contained within the context, state: "I'm sorry, but the provided issues of The Batch do not contain information regarding this topic."
2. **No Hallucinations**: Do not use external knowledge or invent details.
3. **Citations**: You MUST cite the article title and publication date for every claim made. Format as: (Title, Date).
4. **Images**: Extract image URLs only if they are directly relevant to the specific answer. If no relevant images exist in the context, leave the <[IMAGES]> section empty.

### Response Format
Your response must follow this exact structure:

<[IMAGES]>
https://www.youtube.com/watch?v=KsZ6tROaVOQ
https://www.youtube.com/watch?v=-s7TCuCpB5c

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

def extract_images_and_response(text: str) -> PARSED_RESPONSE:
    images_pattern = re.compile(r'<\[IMAGES\]>\s*(.*?)\s*<\[RESPONSE\]>', re.DOTALL)
    images_match = images_pattern.search(text)
    
    image_urls = []
    if images_match:
        images_block = images_match.group(1)
        image_urls = [
            line.strip() for line in images_block.splitlines()
            if line.strip().startswith(('http://', 'https://'))
        ]
    
    response_pattern = re.compile(r'<\[RESPONSE\]>\s*(.*)', re.DOTALL)
    response_match = response_pattern.search(text)
    
    response_text = response_match.group(1).strip() if response_match else ""
    
    return image_urls, response_text


def build_search_gemini_query(user_query: str) -> list[str]:
    return [f"""You are an expert in crafting optimized search queries for retrieving relevant content from AI-focused newsletters, specifically "The Batch" by DeepLearning.AI. Your goal is to generate a concise, effective search query string based on the user's prompt. This query will be used for semantic or keyword-based retrieval from a vector database or search index containing issues of The Batch.

### Guidelines for Query Optimization:
- **Relevance**: Extract key topics, entities (e.g., companies, technologies, people), dates, events, or concepts from the user prompt. Focus on AI/ML themes like models, breakthroughs, ethics, applications, or industry news.
- **Conciseness**: Keep the query under 100 words. Use natural language phrasing that captures the essence for semantic search, or keyword combinations for exact matches (e.g., "GPT-4 advancements ethics 2023").
- **Expansion**: Include synonyms, related terms, or abbreviations to broaden recall without diluting precision (e.g., for "large language models", include "LLMs" or "generative AI").
- **Structure**: If the prompt involves time, add date ranges (e.g., "since 2022"). If comparative or specific, incorporate operators like AND/OR if the search backend supports them.
- **No Assumptions**: Stick to the user prompt—do not add unrelated details or external knowledge.
- **Output Format**: Respond only with the generated search query string, nothing else.

User Prompt: {user_query}"""]

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
    for img_url, caption in image_data[:15]:
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

def query_gemini(user_query: str, k: int = 10) -> tuple[PARSED_RESPONSE, SOURCE_DATA]:
    # gemini_search_query = build_search_gemini_query(user_query)
    # db_search_query = generate_response(gemini_search_query)
    # with open("search_query.txt", "w") as file:
    #     file.write(db_search_query)
    with open("search_query.txt", "r") as file:
        db_search_query = file.read()

    text, images = find_data(db_search_query, k)

    final_query = build_final_query(user_query, text, images)
    response = generate_response(final_query)

    with open("response.txt", "w") as file:
        file.write(response)

    return extract_images_and_response(response), (text_results[:5], image_data[:5])

def display_results(answer: str, text_results, image_infos):
    print("\n" + "="*80)
    print("GEMINI ANSWER:")
    print("="*80)
    print(textwrap.fill(answer, width=90))
    print("\n" + "-"*80)
    print("SOURCES:")
    print("-"*80)
    for r in text_results:
        print(f"• {r['title']} ({r['date']})")
        print(f"  {r['url']}\n")

    if not image_infos:
        return

    print("RELEVANT IMAGES:")
    for path, info in image_infos:
        print(f"• Image: {os.path.basename(path)}")
        print(f"  Caption: {info}")
        print(f"  Path: {path}")
        # if info.get("caption"):
        #     print(f"  Caption: {info['caption']}")
        # print(f"  From: {info['title']} ({info['date']})\n")

def main():
    print("The Batch Multimodal RAG with Gemini (new GenAI SDK)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Your question: ").strip()
        # query = "What updates in graphics related topics are there?"
        # query = "Give me an image with an example of Style Upgrade"
        if query.lower() in ["exit", "quit", "q"]:
            break
        if not query:
            continue

        try:
            answer, texts, images_meta = query_gemini(query)
            display_results(answer, texts, images_meta)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()