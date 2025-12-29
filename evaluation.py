"""
Comprehensive Evaluation Framework for Multimodal RAG System
Evaluates retrieval quality, answer quality, and system performance
Supports both manual and automatic query generation
"""

import json
import time
import sqlite3
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict
import re

# Import from your query.py
from query import query_gemini, search, find_data, ParsedResponse, SOURCE_TEXTS, SOURCE_IMAGES
from config import DB_PATH  # Make sure this exists and points to your SQLite DB


@dataclass
class EvaluationQuery:
    """Test query with ground truth"""
    query: str
    expected_topics: List[str]  # Topics that should be covered
    expected_urls: List[str]  # Expected source URLs (partial matches ok)
    relevance_threshold: float = 0.7  # Minimum relevance score
    query_type: str = "general"  # Optional metadata: factual, multimodal, etc.


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float  # Mean Reciprocal Rank
    avg_similarity_score: float
    retrieval_time: float


@dataclass
class GenerationMetrics:
    """Metrics for answer generation quality"""
    answer_length: int
    includes_expected_topics: Dict[str, bool]
    source_grounding_score: float
    hallucination_indicators: List[str]
    images_retrieved: int
    links_retrieved: int
    generation_time: float


class AutoQueryGenerator:
    """Automatically generates diverse test queries from database content"""
    
    def __init__(self, db_path: str, num_queries: int = 100):
        self.db_path = db_path
        self.num_queries = num_queries
        self.articles = []
        self.topics = set()
        self.entities = set()
        
    def load_articles(self) -> None:
        """Load articles from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT article_id, title, date, url, summary 
                FROM articles 
                ORDER BY date DESC
                LIMIT 500
            """)
            
            rows = cursor.fetchall()
            for row in rows:
                article = {
                    'id': row[0],
                    'title': row[1] or "",
                    'date': row[2],
                    'url': row[3] or "",
                    'summary': row[4] or ""
                }
                self.articles.append(article)
            
            conn.close()
            print(f"Loaded {len(self.articles)} articles from database")
        except Exception as e:
            print(f"Warning: Could not load articles from DB: {e}")
            print("Falling back to empty article list")
            self.articles = []
    
    def extract_topics_and_entities(self) -> None:
        """Extract key topics and entities from articles"""
        
        # Predefined AI/ML vocabulary
        ai_topics = {
            'transformer', 'neural network', 'deep learning', 'machine learning',
            'LLM', 'large language model', 'GPT', 'Claude', 'Gemini', 'LLaMA',
            'computer vision', 'NLP', 'diffusion', 'GAN', 'reinforcement learning',
            'fine-tuning', 'prompt engineering', 'multimodal', 'RAG'
        }
        
        entities = {
            'OpenAI', 'Google', 'DeepMind', 'Anthropic', 'Meta', 'Microsoft',
            'NVIDIA', 'Tesla', 'xAI', 'Andrew Ng', 'Yann LeCun', 'Geoffrey Hinton'
        }
        
        self.topics.update(ai_topics)
        self.entities.update(entities)
        
        # Extract from article titles and summaries
        for article in self.articles:
            text = f"{article['title']} {article.get('summary', '')}".lower()
            
            # Extract potential entities (capitalized phrases)
            caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', article['title'])
            self.entities.update(caps[:5])
            
            # Extract common technical terms
            words = re.findall(r'\b[a-z]{5,}\b', text)
            self.topics.update([w for w in words if w not in {'https', 'com', 'www'}][:15])

    def generate_article_specific_queries(self, n: int) -> List[EvaluationQuery]:
        """Generate queries directly based on real article titles"""
        queries = []
        random.shuffle(self.articles)
        
        for article in self.articles[:n]:
            title = article['title']
            if not title:
                continue
                
            # Create variations
            templates = [
                f"Tell me about {title}",
                f"What is discussed in the article titled '{title}'?",
                f"Summarize {title}",
                f"Explain the key points from {title}"
            ]
            
            query = random.choice(templates)
            main_topic = title.split(':')[0].split('|')[0].strip()
            
            queries.append(EvaluationQuery(
                query=query,
                expected_topics=[main_topic.lower(), title.lower()],
                expected_urls=[article['url']] if article['url'] else [],
                query_type="article-specific"
            ))
        
        return queries
    
    def generate_general_queries(self, n: int) -> List[EvaluationQuery]:
        """Generate broader AI topic queries"""
        templates = [
            "What are recent advances in {topic}?",
            "How does {topic} work?",
            "What are the applications of {topic}?",
            "Explain {topic}",
            "Recent developments in {topic}",
            "Show examples of {topic}"
        ]
        
        topics = list(self.topics)[:50]
        entities = list(self.entities)[:30]
        candidates = topics + entities
        
        queries = []
        for _ in range(n):
            item = random.choice(candidates)
            template = random.choice(templates)
            query = template.format(topic=item)
            
            qtype = "multimodal" if "show" in query.lower() or "example" in query.lower() else "factual"
            
            queries.append(EvaluationQuery(
                query=query,
                expected_topics=[item.lower()],
                expected_urls=[],
                query_type=qtype
            ))
        
        return queries
    
    def generate_queries(self) -> List[EvaluationQuery]:
        """Generate full diverse test suite"""
        print(f"Generating {self.num_queries} automatic test queries...")
        
        self.load_articles()
        self.extract_topics_and_entities()
        
        queries = []
        
        # Prioritize article-specific queries for strong ground truth
        article_queries = self.generate_article_specific_queries(
            min(40, len(self.articles), self.num_queries // 2)
        )
        queries.extend(article_queries)
        
        # Fill remaining with general queries
        remaining = self.num_queries - len(queries)
        if remaining > 0:
            queries.extend(self.generate_general_queries(remaining))
        
        # Shuffle and trim
        random.shuffle(queries)
        queries = queries[:self.num_queries]
        
        print(f"Generated {len(queries)} queries")
        print(f"  - Article-specific: {len(article_queries)}")
        print(f"  - General: {len(queries) - len(article_queries)}")
        
        return queries
    
    def save_queries(self, queries: List[EvaluationQuery], filename: str = "auto_test_queries.json"):
        """Save generated queries"""
        with open(filename, 'w') as f:
            json.dump([asdict(q) for q in queries], f, indent=2)
        print(f"Saved queries to {filename}")
    
    def load_queries(self, filename: str = "auto_test_queries.json") -> List[EvaluationQuery]:
        """Load previously saved queries"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return [EvaluationQuery(**q) for q in data]


class RAGEvaluator:
    """Comprehensive RAG system evaluator"""
    
    def __init__(self, test_queries: List[EvaluationQuery]):
        self.test_queries = test_queries
        
    def evaluate_retrieval(self, query: str, expected_urls: List[str], k: int = 10) -> RetrievalMetrics:
        start_time = time.time()
        text_results, _ = find_data(query, article_count=k)
        retrieval_time = time.time() - start_time
        
        retrieved_urls = [r['url'] for r in text_results if 'url' in r]
        
        precision_at_k = {}
        recall_at_k = {}
        
        for k_val in [1, 3, 5, 10]:
            top_k = retrieved_urls[:k_val]
            relevant = sum(
                1 for url in top_k
                if any(exp in url for exp in expected_urls if exp)
            )
            precision_at_k[k_val] = relevant / k_val if k_val > 0 else 0
            recall_at_k[k_val] = relevant / len(expected_urls) if expected_urls else 1.0
        
        # MRR
        mrr = 0.0
        for i, url in enumerate(retrieved_urls, 1):
            if any(exp in url for exp in expected_urls if exp):
                mrr = 1.0 / i
                break
        
        # Similarity
        search_results = search(query, article_count=k)
        distances = search_results.get("distances", [[]])[0]
        avg_similarity = 1.0 - np.mean(distances) if distances else 0.0
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            avg_similarity_score=float(avg_similarity),
            retrieval_time=retrieval_time
        )
    
    def evaluate_generation(self, parsed_response: ParsedResponse, expected_topics: List[str], source_texts: SOURCE_TEXTS) -> GenerationMetrics:
        answer = parsed_response.response or ""
        
        topics_covered = {
            topic: bool(re.search(re.escape(topic), answer, re.IGNORECASE))
            for topic in expected_topics
        }
        
        # Grounding score
        source_terms = set()
        for text in source_texts:
            words = re.findall(r'\b\w{4,}\b', text['text'].lower())
            source_terms.update(words[:50])
        
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        grounding_score = len(answer_words & source_terms) / len(answer_words) if answer_words else 0.0
        
        # Hallucinations
        hallucination_phrases = [
            "based on my knowledge", "as far as i know", "i believe", "typically",
            "generally", "i don't have", "not contained", "my training"
        ]
        hallucination_indicators = [p for p in hallucination_phrases if p in answer.lower()]
        
        return GenerationMetrics(
            answer_length=len(answer.split()),
            includes_expected_topics=topics_covered,
            source_grounding_score=grounding_score,
            hallucination_indicators=hallucination_indicators,
            images_retrieved=len(parsed_response.images or []),
            links_retrieved=len(parsed_response.links or []),
            generation_time=0.0
        )
    
    def run_evaluation(self, sample_size: int = None) -> Dict[str, Any]:
        queries = self.test_queries
        if sample_size:
            queries = random.sample(queries, min(sample_size, len(queries)))
        
        print("=" * 80)
        print(f"STARTING EVALUATION ON {len(queries)} QUERIES")
        print("=" * 80)
        
        retrieval_metrics = []
        generation_metrics = []
        errors = 0
        total_time = 0.0
        
        for i, q in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] {q.query_type.upper()}: {q.query[:80]}...")
            
            try:
                start = time.time()
                
                ret_metrics = self.evaluate_retrieval(q.query, q.expected_urls, k=10)
                
                gen_start = time.time()
                parsed_response, (source_texts, _) = query_gemini(q.query, k=10)
                gen_time = time.time() - gen_start
                
                gen_metrics = self.evaluate_generation(parsed_response, q.expected_topics, source_texts)
                gen_metrics.generation_time = gen_time
                
                total_time += time.time() - start
                
                retrieval_metrics.append(ret_metrics)
                generation_metrics.append(gen_metrics)
                
                print(f"  Retrieval: Sim={ret_metrics.avg_similarity_score:.3f} | P@5={ret_metrics.precision_at_k.get(5,0):.2f}")
                print(f"  Generation: Topics={sum(gen_metrics.includes_expected_topics.values())}/{len(q.expected_topics)} | "
                      f"Images={gen_metrics.images_retrieved} | Grounding={gen_metrics.source_grounding_score:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
        
        return self._aggregate_results(retrieval_metrics, generation_metrics, errors, total_time, len(queries))
    
    def _aggregate_results(self, ret: List[RetrievalMetrics], gen: List[GenerationMetrics], 
                           errors: int, total_time: float, total_queries: int) -> Dict[str, Any]:
        if not ret:
            return {"error": "No successful evaluations"}
        
        topic_rates = [sum(g.includes_expected_topics.values()) / len(g.includes_expected_topics) for g in gen if g.includes_expected_topics]
        
        return {
            "retrieval_metrics": {
                "precision_at_1": np.mean([r.precision_at_k.get(1, 0) for r in ret]),
                "precision_at_5": np.mean([r.precision_at_k.get(5, 0) for r in ret]),
                "precision_at_10": np.mean([r.precision_at_k.get(10, 0) for r in ret]),
                "recall_at_5": np.mean([r.recall_at_k.get(5, 0) for r in ret]),
                "mrr": np.mean([r.mrr for r in ret]),
                "avg_similarity": np.mean([r.avg_similarity_score for r in ret]),
                "avg_time": np.mean([r.retrieval_time for r in ret])
            },
            "generation_metrics": {
                "avg_answer_length": np.mean([g.answer_length for g in gen]),
                "topic_coverage": np.mean(topic_rates),
                "avg_grounding_score": np.mean([g.source_grounding_score for g in gen]),
                "hallucination_rate": sum(bool(g.hallucination_indicators) for g in gen) / len(gen),
                "avg_images": np.mean([g.images_retrieved for g in gen]),
                "avg_links": np.mean([g.links_retrieved for g in gen]),
                "avg_time": np.mean([g.generation_time for g in gen])
            },
            "system_metrics": {
                "total_queries": total_queries,
                "successful": len(ret),
                "failed": errors,
                "success_rate": len(ret) / total_queries,
                "total_time": total_time,
                "avg_time_per_query": total_time / total_queries
            }
        }
    
    def print_report(self, results: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("MULTIMODAL RAG EVALUATION REPORT")
        print("=" * 80)
        
        s = results["system_metrics"]
        print(f"\nSYSTEM SUMMARY")
        print(f"Queries: {s['total_queries']} | Success: {s['success_rate']:.1%} | "
              f"Time: {s['total_time']:.1f}s ({s['avg_time_per_query']:.2f}s/query)")
        
        r = results["retrieval_metrics"]
        print(f"\nRETRIEVAL")
        print(f"P@1: {r['precision_at_1']:.3f} | P@5: {r['precision_at_5']:.3f} | "
              f"MRR: {r['mrr']:.3f} | Sim: {r['avg_similarity']:.3f}")
        
        g = results["generation_metrics"]
        print(f"\nGENERATION")
        print(f"Topic Coverage: {g['topic_coverage']:.1%} | Grounding: {g['avg_grounding_score']:.1%}")
        print(f"Images/query: {g['avg_images']:.1f} | Hallucination: {g['hallucination_rate']:.1%}")
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)


# ==================== MAIN ====================

def create_sample_test_suite() -> List[EvaluationQuery]:
    """Fallback manual test suite"""
    return [
        EvaluationQuery(
            query="What are the latest developments in multimodal AI?",
            expected_topics=["multimodal", "vision", "language", "image"],
            expected_urls=["the-batch"],
            query_type="factual"
        ),
        EvaluationQuery(
            query="Show me examples of AI-generated images",
            expected_topics=["generated", "image", "diffusion"],
            expected_urls=[],
            query_type="multimodal"
        ),
        EvaluationQuery(
            query="How is RAG used to improve LLM accuracy?",
            expected_topics=["RAG", "retrieval", "grounding"],
            expected_urls=["the-batch"],
            query_type="factual"
        ),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG Evaluation System")
    parser.add_argument("--auto", action="store_true", help="Generate queries automatically from database")
    parser.add_argument("--queries", type=int, default=50, help="Number of auto-generated queries")
    parser.add_argument("--sample", type=int, default=None, help="Evaluate only N random queries")
    parser.add_argument("--load", type=str, default=None, help="Load queries from JSON file")
    args = parser.parse_args()
    
    if args.load:
        print(f"Loading queries from {args.load}...")
        generator = AutoQueryGenerator(DB_PATH)
        test_queries = generator.load_queries(args.load)
    elif args.auto:
        print("Generating automatic test suite...")
        generator = AutoQueryGenerator(DB_PATH, num_queries=args.queries)
        test_queries = generator.generate_queries()
        generator.save_queries(test_queries)
    else:
        print("Using manual sample test suite")
        test_queries = create_sample_test_suite()
    
    evaluator = RAGEvaluator(test_queries)
    results = evaluator.run_evaluation(sample_size=args.sample)
    evaluator.print_report(results)
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")