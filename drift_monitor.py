#!/usr/bin/env python3
"""
Drift-Aware Retraining Pipeline - Main Monitor Service

This service continuously monitors for:
1. Embedding drift (query/document embedding distribution changes)
2. Behavior drift (refusal rates, toxicity rates)
3. Accuracy drops (user feedback scores)

When drift is detected, it triggers appropriate actions:
- Re-indexing documents for embedding drift
- Fine-tuning/retraining for behavior/accuracy drift

Exposes Prometheus metrics on /metrics endpoint.
"""

import os
import sys
import time
import logging
import json
from typing import Tuple, Optional, List, Dict, Any
from datetime import datetime, timedelta

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from prometheus_client import start_http_server, Gauge, Counter, Info
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("drift_monitor")

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "900"))  # 15 min
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))

# Drift thresholds (configurable via environment)
EMBEDDING_COSINE_THRESHOLD = float(os.getenv("EMBEDDING_COSINE_THRESHOLD", "0.1"))
REFUSAL_RATE_THRESHOLD = float(os.getenv("REFUSAL_RATE_THRESHOLD", "0.05"))  # 5%
TOXICITY_RATE_THRESHOLD = float(os.getenv("TOXICITY_RATE_THRESHOLD", "0.02"))  # 2%
ACCURACY_DROP_POINTS = float(os.getenv("ACCURACY_DROP_POINTS", "0.05"))  # 5 points

# Prometheus metrics
embedding_drift_score = Gauge("embedding_drift_score", "Cosine distance between recent and baseline query centroids")
embedding_variance_score = Gauge("embedding_variance_score", "Variance change in embedding distribution")
refusal_rate_gauge = Gauge("refusal_rate", "Recent refusal rate (rolling 7 days)")
toxicity_rate_gauge = Gauge("toxicity_rate", "Recent toxicity rate (rolling 7 days)")
accuracy_gauge = Gauge("recent_accuracy", "Recent moving average accuracy")
baseline_accuracy_gauge = Gauge("baseline_accuracy", "Baseline accuracy for comparison")

# Event counters
retrain_events_total = Counter("retrain_events_total", "Number of retraining events triggered", ["reason"])
reindex_events_total = Counter("reindex_events_total", "Number of document reindex events triggered")
drift_detection_runs_total = Counter("drift_detection_runs_total", "Total drift detection runs")
drift_detection_errors_total = Counter("drift_detection_errors_total", "Drift detection errors")

# System info
system_info = Info("drift_monitor_info", "Information about the drift monitoring system")


class DriftMonitor:
    """Main drift monitoring service"""
    
    def __init__(self):
        self.validate_config()
        system_info.info({
            "version": "1.0.0",
            "embedding_dim": str(EMBEDDING_DIM),
            "check_interval": str(CHECK_INTERVAL_SECONDS),
            "cosine_threshold": str(EMBEDDING_COSINE_THRESHOLD),
            "refusal_threshold": str(REFUSAL_RATE_THRESHOLD),
            "toxicity_threshold": str(TOXICITY_RATE_THRESHOLD),
            "accuracy_drop_threshold": str(ACCURACY_DROP_POINTS)
        })
    
    def validate_config(self):
        """Validate required configuration"""
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL environment variable is required")
        log.info(f"Configuration loaded - Check interval: {CHECK_INTERVAL_SECONDS}s")
    
    def db_connect(self) -> psycopg2.extensions.connection:
        """Create database connection with pgvector support"""
        try:
            conn = psycopg2.connect(DATABASE_URL)
            register_vector(conn)
            return conn
        except Exception as e:
            log.error(f"Database connection failed: {e}")
            raise
    
    def fetch_embeddings(self, conn, embedding_type: str, start: datetime, end: datetime) -> np.ndarray:
        """
        Fetch embeddings of specified type within time range
        
        Args:
            conn: Database connection
            embedding_type: 'query' or 'doc'
            start: Start timestamp
            end: End timestamp
            
        Returns:
            numpy array of embeddings (N x embedding_dim)
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT embedding
                FROM embeddings_log
                WHERE type = %s AND timestamp >= %s AND timestamp < %s
                ORDER BY timestamp DESC
                """,
                (embedding_type, start, end),
            )
            rows = cur.fetchall()
        
        if not rows:
            log.warning(f"No {embedding_type} embeddings found between {start} and {end}")
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        
        # Convert pgvector to numpy arrays
        embeddings = []
        for (embedding_vec,) in rows:
            if embedding_vec is not None:
                embeddings.append(np.array(embedding_vec, dtype=np.float32))
        
        if not embeddings:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            
        return np.array(embeddings)
    
    def safe_centroid(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Compute centroid safely, handling empty arrays"""
        if X.size == 0:
            return None
        return X.mean(axis=0)
    
    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two vectors"""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        similarity = float(np.dot(a, b) / denom)
        return 1.0 - similarity
    
    def compute_variance_change(self, recent: np.ndarray, baseline: np.ndarray) -> float:
        """Compute relative change in embedding variance"""
        if recent.size == 0 or baseline.size == 0:
            return 0.0
        
        recent_var = np.mean(np.var(recent, axis=0))
        baseline_var = np.mean(np.var(baseline, axis=0))
        
        if baseline_var == 0:
            return 0.0
        
        return abs(recent_var - baseline_var) / baseline_var
    
    def detect_embedding_drift(self, conn) -> Tuple[float, float, bool]:
        """
        Detect embedding drift using centroid shift and variance change
        
        Returns:
            (drift_score, variance_change, is_drift_detected)
        """
        now = datetime.utcnow()
        recent_start = now - timedelta(days=7)  # Last 7 days
        baseline_end = now - timedelta(days=30)  # 30 days ago
        baseline_start = baseline_end - timedelta(days=7)  # 30-37 days ago
        
        # Fetch query embeddings for drift analysis
        recent_embeddings = self.fetch_embeddings(conn, "query", recent_start, now)
        baseline_embeddings = self.fetch_embeddings(conn, "query", baseline_start, baseline_end)
        
        log.info(f"Embedding drift analysis: {len(recent_embeddings)} recent, {len(baseline_embeddings)} baseline")
        
        # Compute centroids
        recent_centroid = self.safe_centroid(recent_embeddings)
        baseline_centroid = self.safe_centroid(baseline_embeddings)
        
        if recent_centroid is None or baseline_centroid is None:
            log.warning("Cannot compute embedding drift - insufficient data")
            return 0.0, 0.0, False
        
        # Compute drift metrics
        drift_score = self.cosine_distance(recent_centroid, baseline_centroid)
        variance_change = self.compute_variance_change(recent_embeddings, baseline_embeddings)
        
        # Detect drift
        drift_detected = drift_score > EMBEDDING_COSINE_THRESHOLD
        
        log.info(f"Embedding drift - Score: {drift_score:.4f}, Variance change: {variance_change:.4f}, Drift: {drift_detected}")
        
        return drift_score, variance_change, drift_detected
    
    def detect_behavior_drift(self, conn) -> Tuple[float, float, bool]:
        """
        Detect behavior drift using refusal and toxicity rates
        
        Returns:
            (refusal_rate, toxicity_rate, is_drift_detected)
        """
        now = datetime.utcnow()
        start = now - timedelta(days=7)  # Last 7 days
        
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN refusal_flag THEN 1 ELSE 0 END) as refusals,
                    SUM(CASE WHEN toxicity_flag THEN 1 ELSE 0 END) as toxicity
                FROM interaction_log
                WHERE timestamp >= %s AND timestamp < %s
                """,
                (start, now),
            )
            row = cur.fetchone()
        
        total, refusals, toxicity = row if row else (0, 0, 0)
        
        if total == 0:
            log.warning("No interactions found for behavior drift analysis")
            return 0.0, 0.0, False
        
        refusal_rate = refusals / total
        toxicity_rate = toxicity / total
        
        # Detect drift
        drift_detected = (refusal_rate > REFUSAL_RATE_THRESHOLD) or (toxicity_rate > TOXICITY_RATE_THRESHOLD)
        
        log.info(f"Behavior drift - Refusal rate: {refusal_rate:.4f}, Toxicity rate: {toxicity_rate:.4f}, Drift: {drift_detected}")
        
        return refusal_rate, toxicity_rate, drift_detected
    
    def detect_accuracy_drop(self, conn) -> Tuple[float, float, bool]:
        """
        Detect accuracy drops using user feedback scores
        
        Returns:
            (recent_accuracy, baseline_accuracy, is_drop_detected)
        """
        now = datetime.utcnow()
        recent_start = now - timedelta(days=7)
        baseline_end = now - timedelta(days=30)
        baseline_start = baseline_end - timedelta(days=7)
        
        def get_average_score(start_time: datetime, end_time: datetime) -> Optional[float]:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT AVG(user_feedback_score), COUNT(*)
                    FROM interaction_log
                    WHERE user_feedback_score IS NOT NULL 
                    AND timestamp >= %s AND timestamp < %s
                    """,
                    (start_time, end_time),
                )
                row = cur.fetchone()
                avg_score, count = row if row else (None, 0)
                log.debug(f"Score query {start_time} to {end_time}: avg={avg_score}, count={count}")
                return float(avg_score) if avg_score is not None else None
        
        recent_accuracy = get_average_score(recent_start, now)
        baseline_accuracy = get_average_score(baseline_start, baseline_end)
        
        if recent_accuracy is None or baseline_accuracy is None:
            log.warning("Cannot compute accuracy drop - insufficient feedback data")
            return 0.0, 0.0, False
        
        # Compute absolute drop
        accuracy_drop = baseline_accuracy - recent_accuracy
        drop_detected = accuracy_drop >= ACCURACY_DROP_POINTS
        
        log.info(f"Accuracy analysis - Recent: {recent_accuracy:.4f}, Baseline: {baseline_accuracy:.4f}, Drop: {accuracy_drop:.4f}, Drop detected: {drop_detected}")
        
        return recent_accuracy, baseline_accuracy, drop_detected
    
    def log_drift_event(self, conn, kind: str, details: Dict[str, Any]):
        """Log drift detection event to database"""
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO drift_events (kind, details) VALUES (%s, %s)",
                    (kind, json.dumps(details)),
                )
            conn.commit()
            log.info(f"Logged drift event: {kind}")
        except Exception as e:
            log.error(f"Failed to log drift event: {e}")
    
    def reindex_documents(self, conn):
        """
        Trigger document re-indexing
        
        This is a placeholder - implement your actual document ingestion pipeline:
        1. Fetch new/updated documents from source
        2. Generate embeddings
        3. Update documents and embeddings_log tables
        4. Refresh any in-memory indexes
        """
        log.info("🔄 Triggering document re-indexing...")
        
        # TODO: Implement actual reindexing logic
        # Example steps:
        # 1. Query external data sources for new content
        # 2. Generate embeddings for new documents
        # 3. Update database tables
        # 4. Refresh vector search indexes
        
        reindex_events_total.inc()
        log.info("✅ Document re-indexing triggered successfully")
    
    def fine_tune_model(self, conn, reason: str, details: Dict[str, Any]):
        """
        Trigger model fine-tuning or adaptation
        
        This is a placeholder - implement your actual retraining pipeline:
        1. Prepare training data from recent interactions
        2. Launch fine-tuning job (OpenAI API, Hugging Face, etc.)
        3. Wait for completion and validate
        4. Deploy new model version
        """
        log.warning(f"🔧 Triggering model retraining due to: {reason}")
        
        # TODO: Implement actual fine-tuning logic
        # Example steps:
        # 1. Extract recent poor-performing interactions
        # 2. Create training dataset
        # 3. Launch fine-tuning (e.g., OpenAI API)
        # 4. Validate new model performance
        # 5. Deploy if improved
        
        retrain_events_total.labels(reason=reason).inc()
        
        # Log model version for tracking
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_versions (model_name, source, notes)
                    VALUES (%s, %s, %s)
                    """,
                    (f"retrain_{int(time.time())}", reason, json.dumps(details))
                )
            conn.commit()
        except Exception as e:
            log.error(f"Failed to log model version: {e}")
        
        log.info("✅ Model retraining triggered successfully")
    
    def run_drift_detection(self):
        """Run one complete drift detection cycle"""
        drift_detection_runs_total.inc()
        
        try:
            with self.db_connect() as conn:
                log.info("🔍 Running drift detection cycle...")
                
                # 1. Embedding drift detection
                emb_score, emb_variance, emb_drift = self.detect_embedding_drift(conn)
                embedding_drift_score.set(emb_score)
                embedding_variance_score.set(emb_variance)
                
                # 2. Behavior drift detection
                refusal_rate, toxicity_rate, behavior_drift = self.detect_behavior_drift(conn)
                refusal_rate_gauge.set(refusal_rate)
                toxicity_rate_gauge.set(toxicity_rate)
                
                # 3. Accuracy drop detection
                recent_acc, baseline_acc, accuracy_drop = self.detect_accuracy_drop(conn)
                accuracy_gauge.set(recent_acc)
                baseline_accuracy_gauge.set(baseline_acc)
                
                # 4. Trigger appropriate actions
                actions_taken = []
                
                if emb_drift:
                    log.warning(f"🚨 Embedding drift detected! Score: {emb_score:.4f} > {EMBEDDING_COSINE_THRESHOLD}")
                    self.reindex_documents(conn)
                    self.log_drift_event(conn, "embedding_drift", {
                        "drift_score": emb_score,
                        "variance_change": emb_variance,
                        "threshold": EMBEDDING_COSINE_THRESHOLD
                    })
                    actions_taken.append("reindex")
                
                # Combine behavior and accuracy drift into one retraining action
                retrain_reasons = []
                retrain_details = {}
                
                if behavior_drift:
                    if refusal_rate > REFUSAL_RATE_THRESHOLD:
                        retrain_reasons.append("high_refusal_rate")
                        retrain_details["refusal_rate"] = refusal_rate
                        log.warning(f"🚨 High refusal rate detected! {refusal_rate:.4f} > {REFUSAL_RATE_THRESHOLD}")
                    
                    if toxicity_rate > TOXICITY_RATE_THRESHOLD:
                        retrain_reasons.append("high_toxicity_rate")
                        retrain_details["toxicity_rate"] = toxicity_rate
                        log.warning(f"🚨 High toxicity rate detected! {toxicity_rate:.4f} > {TOXICITY_RATE_THRESHOLD}")
                
                if accuracy_drop:
                    retrain_reasons.append("accuracy_drop")
                    retrain_details.update({
                        "recent_accuracy": recent_acc,
                        "baseline_accuracy": baseline_acc,
                        "drop_amount": baseline_acc - recent_acc
                    })
                    log.warning(f"🚨 Accuracy drop detected! {baseline_acc:.4f} → {recent_acc:.4f}")
                
                if retrain_reasons:
                    reason = " & ".join(retrain_reasons)
                    self.fine_tune_model(conn, reason, retrain_details)
                    self.log_drift_event(conn, "model_retrain", {
                        "reasons": retrain_reasons,
                        "details": retrain_details
                    })
                    actions_taken.append("retrain")
                
                if not actions_taken:
                    log.info("✅ No drift detected - all metrics within thresholds")
                else:
                    log.info(f"🎯 Actions taken: {', '.join(actions_taken)}")
                
        except Exception as e:
            drift_detection_errors_total.inc()
            log.exception(f"❌ Drift detection cycle failed: {e}")
            raise
    
    def run_monitoring_loop(self):
        """Main monitoring loop"""
        log.info(f"🚀 Starting drift monitoring service on port {PROMETHEUS_PORT}")
        log.info(f"📊 Metrics available at http://localhost:{PROMETHEUS_PORT}/metrics")
        log.info(f"⏰ Check interval: {CHECK_INTERVAL_SECONDS} seconds")
        
        # Start Prometheus metrics server
        start_http_server(PROMETHEUS_PORT)
        
        while True:
            try:
                self.run_drift_detection()
                log.info(f"💤 Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
                time.sleep(CHECK_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                log.info("🛑 Received interrupt signal - shutting down gracefully")
                break
            except Exception as e:
                log.exception(f"❌ Monitoring loop error: {e}")
                log.info(f"🔄 Retrying in {CHECK_INTERVAL_SECONDS} seconds...")
                time.sleep(CHECK_INTERVAL_SECONDS)


def main():
    """Main entry point"""
    log.info("🎯 Drift-Aware Retraining Pipeline Starting...")
    
    try:
        monitor = DriftMonitor()
        monitor.run_monitoring_loop()
    except Exception as e:
        log.exception(f"❌ Service startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()