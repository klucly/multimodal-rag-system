"""
Safe Database Re-chunking Tool
Modifies chunk sizes in existing Chroma database without re-scraping
"""

import sqlite3
import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from open_clip import create_model_and_transforms, get_tokenizer

from config import (
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
    CLIP_MODEL_NAME,
    DB_PATH,
    CHUNK_SIZE
)


class DatabaseReChunker:
    """Safely re-chunk and re-embed text in existing database"""
    
    def __init__(self, new_chunk_size: int, backup: bool = True):
        self.new_chunk_size = new_chunk_size
        self.backup = backup
        
        # Backup paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_db_path = f"{DB_PATH}.backup_{timestamp}"
        self.backup_chroma_path = f"{CHROMA_PERSIST_DIRECTORY}_backup_{timestamp}"
        
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = create_model_and_transforms(
            CLIP_MODEL_NAME, 
            pretrained="laion2b_s32b_b82k"
        )
        self.tokenizer = get_tokenizer(CLIP_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Using device: {self.device}")
        print(f"New chunk size: {self.new_chunk_size}")
    
    def create_backups(self) -> None:
        """Create backups of database and Chroma collection"""
        print("\n" + "="*80)
        print("CREATING BACKUPS")
        print("="*80)
        
        # Backup SQLite database
        print(f"Backing up SQLite database to: {self.backup_db_path}")
        shutil.copy2(DB_PATH, self.backup_db_path)
        
        # Backup Chroma directory
        print(f"Backing up Chroma collection to: {self.backup_chroma_path}")
        shutil.copytree(CHROMA_PERSIST_DIRECTORY, self.backup_chroma_path)
        
        print("✓ Backups created successfully")
    
    def verify_backups(self) -> bool:
        """Verify that backups were created correctly"""
        print("\nVerifying backups...")
        
        db_exists = Path(self.backup_db_path).exists()
        chroma_exists = Path(self.backup_chroma_path).exists()
        
        if db_exists and chroma_exists:
            print("✓ Backups verified")
            return True
        else:
            print("✗ Backup verification failed!")
            return False
    
    def get_all_articles(self) -> List[Dict]:
        """Retrieve all articles from SQLite database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, url, title, date, clean_text 
            FROM articles 
            ORDER BY id
        """)
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row[0],
                'url': row[1],
                'title': row[2],
                'date': row[3],
                'text': row[4]
            })
        
        conn.close()
        return articles
    
    def break_into_chunks(self, text: str) -> List[str]:
        """Break text into chunks with new chunk size"""
        paragraphs = (p.strip() for p in text.split("\n") if p.strip())
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) + 1 > self.new_chunk_size:
                if current:
                    chunks.append(current)
                current = para
            else:
                current += ("\n" + para if current else para)
        
        if current:
            chunks.append(current)
        
        return chunks
    
    @torch.no_grad()
    def get_text_embedding(self, text: str) -> List[float]:
        """Generate CLIP embedding for text"""
        tokens = self.tokenizer([text]).to(self.device)
        embedding = self.model.encode_text(tokens)
        return embedding.cpu().numpy()[0].tolist()
    
    def get_existing_images(self, article_id: int) -> List[Dict]:
        """Get all image embeddings for an article from Chroma"""
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        
        # Query all items for this article using $and operator
        results = collection.get(
            where={
                "$and": [
                    {"article_id": {"$eq": article_id}},
                    {"type": {"$eq": "image"}}
                ]
            },
            include=["embeddings", "metadatas", "documents"]
        )
        
        images = []
        if results['ids']:
            for i in range(len(results['ids'])):
                images.append({
                    'id': results['ids'][i],
                    'embedding': results['embeddings'][i],
                    'metadata': results['metadatas'][i],
                    'document': results['documents'][i]
                })
        
        return images
    
    def delete_article_from_chroma(self, article_id: int) -> None:
        """Delete all entries for an article from Chroma"""
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        
        # Get all IDs for this article using proper operator syntax
        results = collection.get(
            where={"article_id": {"$eq": article_id}},
            include=[]
        )
        
        if results['ids']:
            collection.delete(ids=results['ids'])
    
    def add_to_chroma(self, 
                      text_chunks: List[str],
                      images: List[Dict],
                      article: Dict) -> None:
        """Add re-chunked text and preserved images to Chroma"""
        
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        # Ensure article_id is string for consistency
        article_id_str = int(article['id'])
        
        base_metadata = {
            "type": None,
            "article_id": article_id_str,
            "title": article['title'],
            "date": article['date'],
            "url": article['url'],
            "source": None
        }
        
        # Add text chunks with new embeddings
        for i, chunk in enumerate(text_chunks):
            ids.append(f"text_{article_id_str}_{i}")
            documents.append(chunk)
            embeddings.append(self.get_text_embedding(chunk))
            
            metadata = base_metadata.copy()
            metadata["type"] = "text"
            metadata["source"] = "text"
            metadatas.append(metadata)
        
        # Add preserved image embeddings
        for i, img in enumerate(images):
            ids.append(f"image_{article_id_str}_{i}")
            documents.append(img['document'])
            embeddings.append(img['embedding'])
            
            # Use original metadata but ensure article_id is string
            metadata = img['metadata'].copy()
            metadata["article_id"] = article_id_str
            metadatas.append(metadata)
        
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
    
    def process_article(self, article: Dict) -> Tuple[int, int]:
        """Process a single article: re-chunk text, preserve images"""
        
        # Convert article ID to string for consistency with Chroma metadata
        article_id_str = int(article['id'])
        
        # Get new text chunks
        new_chunks = self.break_into_chunks(article['text'])
        
        # Get existing image embeddings (to preserve)
        existing_images = self.get_existing_images(article_id_str)
        
        # Delete old entries
        self.delete_article_from_chroma(article_id_str)
        
        # Add new entries (with original article dict)
        self.add_to_chroma(new_chunks, existing_images, article)
        
        return len(new_chunks), len(existing_images)
    
    def rechunk_database(self) -> None:
        """Main re-chunking process"""
        
        print("\n" + "="*80)
        print(f"STARTING RE-CHUNKING PROCESS")
        print(f"Old chunk size: {CHUNK_SIZE}")
        print(f"New chunk size: {self.new_chunk_size}")
        print("="*80)
        
        # Create and verify backups
        if self.backup:
            self.create_backups()
            if not self.verify_backups():
                print("ERROR: Backup verification failed. Aborting.")
                return
        else:
            print("\n⚠️  WARNING: Running without backups!")
            response = input("Are you sure you want to continue? (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return
        
        # Get all articles
        print("\nLoading articles from database...")
        articles = self.get_all_articles()
        print(f"Found {len(articles)} articles to process")
        
        # Process statistics
        total_text_chunks = 0
        total_images = 0
        errors = []
        
        # Process each article
        print("\nProcessing articles...")
        for article in tqdm(articles, desc="Re-chunking"):
            try:
                num_chunks, num_images = self.process_article(article)
                total_text_chunks += num_chunks
                total_images += num_images
            except Exception as e:
                error_msg = f"Error processing article {article['id']}: {e}"
                errors.append(error_msg)
                print(f"\n✗ {error_msg}")
        
        # Print summary
        self.print_summary(len(articles), total_text_chunks, total_images, errors)
    
    def print_summary(self, 
                      num_articles: int,
                      total_chunks: int,
                      total_images: int,
                      errors: List[str]) -> None:
        """Print processing summary"""
        
        print("\n" + "="*80)
        print("RE-CHUNKING SUMMARY")
        print("="*80)
        print(f"Articles processed:     {num_articles}")
        print(f"New text chunks:        {total_chunks}")
        print(f"Avg chunks per article: {total_chunks/num_articles:.1f}")
        print(f"Images preserved:       {total_images}")
        print(f"Errors:                 {len(errors)}")
        
        if errors:
            print("\n⚠️  Errors encountered:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        if self.backup:
            print("\n📁 Backups saved to:")
            print(f"  - {self.backup_db_path}")
            print(f"  - {self.backup_chroma_path}")
        
        print("\n✓ Re-chunking complete!")
    
    def verify_rechunking(self) -> None:
        """Verify that re-chunking was successful"""
        print("\n" + "="*80)
        print("VERIFICATION")
        print("="*80)
        
        # Connect to Chroma
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        
        # Get counts
        all_items = collection.get(include=["metadatas"])
        text_items = [m for m in all_items['metadatas'] if m['type'] == 'text']
        image_items = [m for m in all_items['metadatas'] if m['type'] == 'image']
        
        print(f"Total items in Chroma:  {len(all_items['ids'])}")
        print(f"Text chunks:            {len(text_items)}")
        print(f"Image embeddings:       {len(image_items)}")
        
        # Check chunk sizes
        print("\nSampling chunk sizes...")
        sample_size = min(100, len(text_items))
        
        # Get sample of text documents
        text_ids = [all_items['ids'][i] for i, m in enumerate(all_items['metadatas']) 
                   if m['type'] == 'text'][:sample_size]
        
        sample_results = collection.get(
            ids=text_ids,
            include=["documents"]
        )
        
        chunk_sizes = [len(doc) for doc in sample_results['documents']]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        
        print(f"Average chunk size:     {avg_size:.0f} chars")
        print(f"Min chunk size:         {min_size} chars")
        print(f"Max chunk size:         {max_size} chars")
        print(f"Target chunk size:      {self.new_chunk_size} chars")
        
        if max_size <= self.new_chunk_size * 1.1:  # Allow 10% tolerance
            print("\n✓ Chunk sizes within expected range")
        else:
            print(f"\n⚠️  Warning: Some chunks exceed target size")
        
        print("\n" + "="*80)
    
    def rollback(self) -> None:
        """Restore from backup if something went wrong"""
        print("\n" + "="*80)
        print("ROLLING BACK TO BACKUP")
        print("="*80)
        
        if not Path(self.backup_db_path).exists():
            print("✗ No backup found!")
            return
        
        # Restore SQLite
        print(f"Restoring SQLite from {self.backup_db_path}...")
        shutil.copy2(self.backup_db_path, DB_PATH)
        
        # Restore Chroma
        print(f"Restoring Chroma from {self.backup_chroma_path}...")
        if Path(CHROMA_PERSIST_DIRECTORY).exists():
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        shutil.copytree(self.backup_chroma_path, CHROMA_PERSIST_DIRECTORY)
        
        print("✓ Rollback complete")


def main():
    """Main execution"""
    print("="*80)
    print("DATABASE RE-CHUNKING TOOL")
    print("="*80)
    
    # Configuration
    print("\nCurrent configuration:")
    print(f"  Database: {DB_PATH}")
    print(f"  Chroma directory: {CHROMA_PERSIST_DIRECTORY}")
    print(f"  Current chunk size: {CHUNK_SIZE}")
    
    # Get new chunk size
    print("\nEnter new chunk size (or press Enter for default 1000):")
    user_input = input("> ").strip()
    
    if user_input:
        try:
            new_chunk_size = int(user_input)
        except ValueError:
            print("Invalid input. Using default: 1000")
            new_chunk_size = 1000
    else:
        new_chunk_size = 1000
    
    # Confirm
    print(f"\n⚠️  This will:")
    print(f"  1. Create backups of your database and Chroma collection")
    print(f"  2. Re-chunk all text to {new_chunk_size} characters")
    print(f"  3. Re-generate embeddings for text (images preserved)")
    print(f"\nThis process may take several minutes depending on database size.")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    # Initialize re-chunker
    rechunker = DatabaseReChunker(new_chunk_size=new_chunk_size, backup=True)
    
    try:
        # Run re-chunking
        rechunker.rechunk_database()
        
        # Verify results
        rechunker.verify_rechunking()
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print("Your database has been successfully re-chunked.")
        print("Backups have been preserved in case you need to rollback.")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("\nWould you like to rollback to backup? (yes/no)")
        response = input("> ")
        if response.lower() == 'yes':
            rechunker.rollback()
        else:
            print("Backup preserved at:")
            print(f"  - {rechunker.backup_db_path}")
            print(f"  - {rechunker.backup_chroma_path}")


if __name__ == "__main__":
    main()