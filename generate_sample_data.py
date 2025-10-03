#!/usr/bin/env python3
"""
Sample Data Generator for Drift Detection Testing

This script generates synthetic data to test the drift monitoring pipeline:
1. Documents with embeddings for knowledge base
2. User interaction logs with various behaviors
3. Query embeddings that simulate drift scenarios

Usage:
    python generate_sample_data.py --scenario baseline
    python generate_sample_data.py --scenario drift
    python generate_sample_data.py --scenario mixed
"""

import os
import sys
import argparse
import random
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load environment
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Sample topics and their characteristic embeddings
TOPICS = {
    "music": {
        "keywords": ["song", "album", "artist", "genre", "melody", "lyrics", "concert", "band"],
        "base_embedding": None  # Will be generated
    },
    "programming": {
        "keywords": ["code", "python", "javascript", "algorithm", "function", "variable", "debug", "api"],
        "base_embedding": None
    },
    "cooking": {
        "keywords": ["recipe", "ingredient", "cook", "bake", "flavor", "kitchen", "meal", "dish"],
        "base_embedding": None
    },
    "sports": {
        "keywords": ["game", "team", "player", "score", "match", "championship", "athlete", "training"],
        "base_embedding": None
    },
    "travel": {
        "keywords": ["destination", "flight", "hotel", "vacation", "explore", "culture", "tourist", "adventure"],
        "base_embedding": None
    }
}

class SampleDataGenerator:
    """Generate synthetic data for drift detection testing"""
    
    def __init__(self):
        self.conn = None
        self.setup_database()
        self.initialize_topic_embeddings()
    
    def setup_database(self):
        """Setup database connection"""
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL environment variable is required")
        
        self.conn = psycopg2.connect(DATABASE_URL)
        register_vector(self.conn)
        print("Connected to database")
    
    def initialize_topic_embeddings(self):
        """Generate base embeddings for each topic"""
        print("Initializing topic embeddings...")
        
        for topic_name, topic_data in TOPICS.items():
            # Generate a random base embedding for this topic
            # In real life, these would be actual text embeddings
            base_embedding = np.random.normal(0, 1, EMBEDDING_DIM).astype(np.float32)
            
            # Normalize to unit vector
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            topic_data["base_embedding"] = base_embedding
            print(f"  - {topic_name}: embedding shape {base_embedding.shape}")
    
    def generate_embedding_for_topic(self, topic: str, noise_level: float = 0.1) -> np.ndarray:
        """Generate an embedding similar to the topic's base embedding"""
        base = TOPICS[topic]["base_embedding"]
        
        # Add noise
        noise = np.random.normal(0, noise_level, EMBEDDING_DIM).astype(np.float32)
        embedding = base + noise
        
        # Normalize
        return embedding / np.linalg.norm(embedding)
    
    def generate_query_text(self, topic: str) -> str:
        """Generate a realistic query text for a topic"""
        keywords = TOPICS[topic]["keywords"]
        
        templates = [
            "How do I {action} {keyword1} and {keyword2}?",
            "What is the best way to {action} {keyword1}?",
            "Can you explain {keyword1} and {keyword2}?",
            "I need help with {keyword1} {keyword2}",
            "Tell me about {keyword1}",
        ]
        
        actions = ["learn", "understand", "use", "create", "find", "choose"]
        
        template = random.choice(templates)
        return template.format(
            action=random.choice(actions),
            keyword1=random.choice(keywords),
            keyword2=random.choice(keywords)
        )
    
    def generate_response_text(self, topic: str, refusal: bool = False, toxic: bool = False) -> str:
        """Generate a model response"""
        if refusal:
            refusal_responses = [
                "I'm sorry, I cannot help with that request.",
                "I don't think I can assist with that.",
                "That's not something I can help you with.",
                "I'm unable to provide information on that topic."
            ]
            return random.choice(refusal_responses)
        
        if toxic:
            # In a real scenario, you wouldn't want to generate actually toxic content
            # This is just a marker for testing
            return f"[FLAGGED_TOXIC] Here's some information about {topic}..."
        
        # Normal helpful response
        keywords = TOPICS[topic]["keywords"]
        responses = [
            f"Here's what I can tell you about {random.choice(keywords)}...",
            f"Great question! {random.choice(keywords).title()} is...",
            f"I'd be happy to help with {random.choice(keywords)}.",
            f"Let me explain {random.choice(keywords)} for you."
        ]
        return random.choice(responses)
    
    def insert_documents(self, count: int = 50):
        """Insert sample documents"""
        print(f"Inserting {count} sample documents...")
        
        with self.conn.cursor() as cur:
            for i in range(count):
                topic = random.choice(list(TOPICS.keys()))
                keywords = TOPICS[topic]["keywords"]
                
                title = f"Guide to {random.choice(keywords).title()}"
                content = f"This is a comprehensive guide about {topic} covering {', '.join(keywords[:3])}."
                source = f"doc_{i:03d}_{topic}.txt"
                
                embedding = self.generate_embedding_for_topic(topic, noise_level=0.05)
                
                cur.execute("""
                    INSERT INTO documents (source, title, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    source, title, content, embedding.tolist(),
                    json.dumps({"topic": topic, "generated": True})
                ))
                
                # Also log the document embedding
                cur.execute("""
                    INSERT INTO embeddings_log (type, embedding, metadata)
                    VALUES ('doc', %s, %s)
                """, (
                    embedding.tolist(),
                    json.dumps({"topic": topic, "document_id": i})
                ))
        
        self.conn.commit()
        print(f"✅ Inserted {count} documents")
    
    def insert_baseline_interactions(self, count: int = 200, days_ago: int = 35):
        """Insert baseline interactions (for comparison)"""
        print(f"Inserting {count} baseline interactions from {days_ago} days ago...")
        
        start_time = datetime.utcnow() - timedelta(days=days_ago)
        end_time = start_time + timedelta(days=7)  # 7-day window
        
        with self.conn.cursor() as cur:
            for i in range(count):
                # Baseline: mostly music and programming topics
                topic = random.choices(
                    ["music", "programming", "cooking"],
                    weights=[50, 40, 10]
                )[0]
                
                # Generate interaction
                timestamp = start_time + timedelta(
                    seconds=random.randint(0, int((end_time - start_time).total_seconds()))
                )
                
                query = self.generate_query_text(topic)
                
                # Low refusal/toxicity rates in baseline
                refusal_flag = random.random() < 0.02  # 2%
                toxicity_flag = random.random() < 0.01  # 1%
                
                response = self.generate_response_text(topic, refusal_flag, toxicity_flag)
                
                # Good baseline accuracy
                feedback_score = np.random.beta(8, 2) if random.random() < 0.3 else None  # 80% good scores
                
                cur.execute("""
                    INSERT INTO interaction_log 
                    (timestamp, user_query, model_response, refusal_flag, toxicity_flag, 
                     user_feedback_score, topic, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp, query, response, refusal_flag, toxicity_flag,
                    feedback_score, topic, json.dumps({"scenario": "baseline"})
                ))
                
                # Log query embedding
                query_embedding = self.generate_embedding_for_topic(topic, noise_level=0.1)
                cur.execute("""
                    INSERT INTO embeddings_log (timestamp, type, embedding, metadata)
                    VALUES (%s, 'query', %s, %s)
                """, (
                    timestamp, query_embedding.tolist(),
                    json.dumps({"topic": topic, "scenario": "baseline"})
                ))
        
        self.conn.commit()
        print(f"✅ Inserted {count} baseline interactions")
    
    def insert_recent_interactions(self, scenario: str, count: int = 150):
        """Insert recent interactions based on scenario"""
        print(f"Inserting {count} recent interactions for scenario: {scenario}")
        
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        with self.conn.cursor() as cur:
            for i in range(count):
                timestamp = start_time + timedelta(
                    seconds=random.randint(0, int((end_time - start_time).total_seconds()))
                )
                
                if scenario == "baseline":
                    # Same distribution as baseline
                    topic = random.choices(
                        ["music", "programming", "cooking"],
                        weights=[50, 40, 10]
                    )[0]
                    refusal_rate = 0.02
                    toxicity_rate = 0.01
                    accuracy_beta = (8, 2)  # Good scores
                    
                elif scenario == "drift":
                    # New topics appearing (sports, travel)
                    topic = random.choices(
                        ["music", "programming", "cooking", "sports", "travel"],
                        weights=[20, 20, 10, 25, 25]
                    )[0]
                    refusal_rate = 0.08  # Higher refusal rate
                    toxicity_rate = 0.03  # Higher toxicity
                    accuracy_beta = (4, 6)  # Lower scores
                    
                elif scenario == "mixed":
                    # Gradual shift
                    topic = random.choices(
                        ["music", "programming", "cooking", "sports", "travel"],
                        weights=[35, 30, 10, 15, 10]
                    )[0]
                    refusal_rate = 0.05
                    toxicity_rate = 0.015
                    accuracy_beta = (6, 4)  # Moderate scores
                
                query = self.generate_query_text(topic)
                
                refusal_flag = random.random() < refusal_rate
                toxicity_flag = random.random() < toxicity_rate
                
                response = self.generate_response_text(topic, refusal_flag, toxicity_flag)
                
                feedback_score = np.random.beta(*accuracy_beta) if random.random() < 0.3 else None
                
                cur.execute("""
                    INSERT INTO interaction_log 
                    (timestamp, user_query, model_response, refusal_flag, toxicity_flag, 
                     user_feedback_score, topic, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    timestamp, query, response, refusal_flag, toxicity_flag,
                    feedback_score, topic, json.dumps({"scenario": scenario})
                ))
                
                # Log query embedding with appropriate noise for drift
                if scenario == "drift":
                    # More noise for drift scenario
                    noise_level = 0.2
                else:
                    noise_level = 0.1
                    
                query_embedding = self.generate_embedding_for_topic(topic, noise_level=noise_level)
                cur.execute("""
                    INSERT INTO embeddings_log (timestamp, type, embedding, metadata)
                    VALUES (%s, 'query', %s, %s)
                """, (
                    timestamp, query_embedding.tolist(),
                    json.dumps({"topic": topic, "scenario": scenario})
                ))
        
        self.conn.commit()
        print(f"✅ Inserted {count} recent interactions for {scenario}")
    
    def show_statistics(self):
        """Show data statistics"""
        print("\n📊 Data Statistics:")
        
        with self.conn.cursor() as cur:
            # Document count
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            print(f"Documents: {doc_count}")
            
            # Interaction count by time period
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN timestamp >= NOW() - INTERVAL '7 days' THEN 'Recent'
                        WHEN timestamp >= NOW() - INTERVAL '37 days' AND timestamp < NOW() - INTERVAL '30 days' THEN 'Baseline'
                        ELSE 'Other'
                    END as period,
                    COUNT(*) as count,
                    AVG(CASE WHEN refusal_flag THEN 1.0 ELSE 0.0 END) as refusal_rate,
                    AVG(CASE WHEN toxicity_flag THEN 1.0 ELSE 0.0 END) as toxicity_rate,
                    AVG(user_feedback_score) as avg_feedback
                FROM interaction_log 
                GROUP BY 1
                ORDER BY 1
            """)
            
            print("\nInteraction Summary:")
            for row in cur.fetchall():
                period, count, refusal_rate, toxicity_rate, avg_feedback = row
                print(f"  {period}: {count} interactions, "
                      f"refusal_rate={refusal_rate:.3f}, "
                      f"toxicity_rate={toxicity_rate:.3f}, "
                      f"avg_feedback={avg_feedback:.3f}")
            
            # Topic distribution
            cur.execute("""
                SELECT topic, COUNT(*) as count
                FROM interaction_log 
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY topic
                ORDER BY count DESC
            """)
            
            print("\nRecent Topic Distribution:")
            for topic, count in cur.fetchall():
                print(f"  {topic}: {count}")
    
    def cleanup_existing_data(self):
        """Clean up existing generated data"""
        print("🧹 Cleaning up existing generated data...")
        
        with self.conn.cursor() as cur:
            # Clean up generated documents
            cur.execute("DELETE FROM documents WHERE metadata->>'generated' = 'true'")
            deleted_docs = cur.rowcount
            
            # Clean up generated interactions
            cur.execute("DELETE FROM interaction_log WHERE metadata->>'scenario' IS NOT NULL")
            deleted_interactions = cur.rowcount
            
            # Clean up generated embeddings
            cur.execute("DELETE FROM embeddings_log WHERE metadata->>'scenario' IS NOT NULL")
            deleted_embeddings = cur.rowcount
        
        self.conn.commit()
        print(f"✅ Cleaned up {deleted_docs} docs, {deleted_interactions} interactions, {deleted_embeddings} embeddings")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description="Generate sample data for drift detection")
    parser.add_argument("--scenario", choices=["baseline", "drift", "mixed"], default="baseline",
                       help="Scenario to simulate")
    parser.add_argument("--documents", type=int, default=50,
                       help="Number of documents to create")
    parser.add_argument("--interactions", type=int, default=200,
                       help="Number of interactions to create")
    parser.add_argument("--clean", action="store_true",
                       help="Clean up existing generated data first")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics, don't generate data")
    
    args = parser.parse_args()
    
    generator = SampleDataGenerator()
    
    try:
        if args.stats_only:
            generator.show_statistics()
            return
        
        if args.clean:
            generator.cleanup_existing_data()
        
        # Always generate documents and baseline data (for comparison)
        if not args.clean:  # Skip if we're just cleaning
            print(f"\n🎯 Generating sample data for scenario: {args.scenario}")
            
            # Generate documents
            generator.insert_documents(args.documents)
            
            # Generate baseline interactions (30+ days ago)
            generator.insert_baseline_interactions(args.interactions, days_ago=35)
            
            # Generate recent interactions based on scenario
            generator.insert_recent_interactions(args.scenario, args.interactions)
            
            print(f"\n✅ Sample data generation completed!")
            generator.show_statistics()
            
            print(f"\n🔍 Next steps:")
            print(f"1. Set your DATABASE_URL environment variable")
            print(f"2. Run: python drift_monitor.py")
            print(f"3. Check metrics at http://localhost:8000/metrics")
            print(f"4. View Grafana dashboard at http://localhost:3000")
            print(f"\nTo simulate different scenarios:")
            print(f"  python generate_sample_data.py --scenario drift --clean")
            print(f"  python generate_sample_data.py --scenario mixed --clean")
    
    finally:
        generator.close()

if __name__ == "__main__":
    main()