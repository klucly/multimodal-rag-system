# Multimodal RAG System for The Batch Newsletter

This repository contains a complete Multimodal Retrieval-Augmented Generation (RAG) system for querying AI/ML news from "The Batch" newsletter by DeepLearning.AI. It supports text and image retrieval, grounded responses via Google Gemini, and an interactive Streamlit UI.

## Features

- **Data Ingestion**: Scrapes articles, extracts text/images, embeds with CLIP, stores in ChromaDB + SQLite.
- **Querying**: Semantic search for text/images; generates responses with citations and visuals.
- **Evaluation**: Automated framework for retrieval/generation metrics.
- **UI**: Streamlit app for user-friendly interaction.
- **Maintenance**: Re-chunking tool for database optimization.

## Prerequisites

- Python 3.13.11
- NVIDIA GPU (recommended for CLIP embeddings; fallback to CPU)
- Google Gemini API key (free tier available at [Google AI Studio](https://aistudio.google.com/))

## Setup Instructions

1. **Clone the Repository**:

   ```
   git clone https://github.com/klucly/multimodal-rag-system.git
   cd multimodal-rag-system
   ```
1. **Install Dependencies**:

   Use the provided `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```

   Note: If using GPU, ensure CUDA is installed (e.g., via `pip install torch --index-url https://download.pytorch.org/whl/cu121`).
1. **Environment Configuration**:

   - Create a `.env` file in the root directory:

     ```
     GEMINI_API_KEY=your_google_gemini_api_key_here
     ```
   - The system uses this for Gemini API access.
1. **Data Directories**:

   - ChromaDB persists in `chroma_db`; SQLite in `database.sqlite`.

## Running the System

### 1. Data Ingestion (Scraping and Processing)

Populate the database:

```
python scrape_and_process.py
```

- This fetches ~200 articles (configurable in `config.py` via `URL_INDEXES`).
- Runs with retries; may take time due to delays (respect site policies).

### 2. Querying via CLI

Test queries interactively:

```
python query.py
```

- Enter questions; type 'exit' to quit.
- Outputs answers, sources, and images.

### 3. Web Interface

Launch the Streamlit UI:

```
streamlit run interface.py
```

- Access at http://localhost:8501.
- Enter queries, adjust settings, view history.

### 4. Evaluation

Run automated tests:

```
python evaluation.py --auto --queries 50
```

- Options: `--auto` for DB-generated queries, `--sample N` to limit, `--load file.json` for pre-saved queries.
- Outputs metrics to console and `evaluation_results.json`.

### 5. Database Maintenance (Re-chunking)

Adjust chunk sizes (e.g., to 100 chars):

```
python rechunker.py
```

- Prompts for confirmation; creates backups automatically.

## Project Structure

- `config.py`: Configurations (URLs, models, paths).
- `scrape_and_process.py`: Data ingestion pipeline.
- `query.py`: Core querying logic (retrieval + generation).
- `interface.py`: Streamlit UI.
- `evaluation.py`: Evaluation framework.
- `rechunker.py`: Database re-chunking tool.
- `requirements.txt`: Dependencies.
- `.env`: API keys (not committed).

## Troubleshooting

- **API Key Issues**: Ensure `.env` is loaded; check Gemini quota.
- **Embedding Speed**: Use GPU for faster CLIP processing.
- **Scraping Errors**: Adjust `REQUEST_DELAY` in `config.py` if rate-limited.
- **ChromaDB Issues**: Delete `chroma_db` and re-ingest if corrupted.