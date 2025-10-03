-- Drift-Aware Retraining Pipeline Database Schema
-- Run this in your Supabase SQL editor or via psql

-- Enable pgvector extension for vector operations
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table for storing knowledge base
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL, -- URL, file path, or source identifier
    title TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL, -- OpenAI text-embedding-ada-002 dimension
    metadata JSONB DEFAULT '{}', -- Additional document metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on embedding for fast similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create embeddings log for tracking embedding distributions over time
CREATE TABLE IF NOT EXISTS embeddings_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    type TEXT NOT NULL CHECK (type IN ('query', 'doc')), -- Type of embedding
    ref_id BIGINT, -- Optional reference to documents.id or interaction_log.id
    embedding VECTOR(1536) NOT NULL,
    metadata JSONB DEFAULT '{}' -- Additional context (topic, user_id, etc.)
);

-- Create index on embeddings_log for time-based queries
CREATE INDEX IF NOT EXISTS embeddings_log_timestamp_type_idx 
ON embeddings_log (timestamp DESC, type);

-- Create interaction log for storing user interactions and behavior flags
CREATE TABLE IF NOT EXISTS interaction_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_query TEXT,
    model_response TEXT,
    model_name TEXT DEFAULT 'gpt-3.5-turbo', -- Track which model version was used
    
    -- Behavior flags
    refusal_flag BOOLEAN DEFAULT FALSE, -- Did the model refuse to answer?
    toxicity_flag BOOLEAN DEFAULT FALSE, -- Was the output flagged as toxic?
    
    -- Performance metrics
    user_feedback_score NUMERIC(3,2), -- User rating 0.00-1.00
    response_time_ms INTEGER, -- Response time in milliseconds
    token_count INTEGER, -- Number of tokens used
    
    -- Additional context
    topic TEXT, -- Detected or assigned topic
    user_id TEXT, -- Optional user identifier
    session_id TEXT, -- Optional session identifier
    metadata JSONB DEFAULT '{}' -- Additional interaction context
);

-- Create indexes for behavior analysis queries
CREATE INDEX IF NOT EXISTS interaction_log_timestamp_idx 
ON interaction_log (timestamp DESC);

CREATE INDEX IF NOT EXISTS interaction_log_flags_idx 
ON interaction_log (timestamp, refusal_flag, toxicity_flag);

CREATE INDEX IF NOT EXISTS interaction_log_feedback_idx 
ON interaction_log (timestamp, user_feedback_score) 
WHERE user_feedback_score IS NOT NULL;

-- Create drift events table for logging when drift is detected
CREATE TABLE IF NOT EXISTS drift_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kind TEXT NOT NULL, -- 'embedding_drift', 'behavior_drift', 'accuracy_drop', 'model_retrain'
    severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high')),
    details JSONB NOT NULL DEFAULT '{}', -- Drift metrics, thresholds, etc.
    action_taken TEXT, -- What action was triggered
    resolved_at TIMESTAMPTZ -- When the issue was resolved
);

-- Create index for querying recent drift events
CREATE INDEX IF NOT EXISTS drift_events_timestamp_kind_idx 
ON drift_events (timestamp DESC, kind);

-- Create model versions table for tracking model deployments
CREATE TABLE IF NOT EXISTS model_versions (
    id BIGSERIAL PRIMARY KEY,
    deployed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name TEXT NOT NULL, -- e.g., 'gpt-3.5-turbo-ft-123', 'claude-v1.3'
    version TEXT, -- Semantic version or identifier
    source TEXT, -- 'fine-tune', 'adapter', 'prompt-change', 'base-model'
    base_model TEXT, -- Original model this was derived from
    training_data_count INTEGER, -- Number of training examples used
    performance_metrics JSONB DEFAULT '{}', -- Evaluation scores
    notes TEXT, -- Human-readable description
    active BOOLEAN DEFAULT TRUE -- Is this the currently deployed model?
);

-- Create index for tracking active model
CREATE INDEX IF NOT EXISTS model_versions_active_idx 
ON model_versions (deployed_at DESC, active);

-- Create a view for easy drift monitoring queries
CREATE OR REPLACE VIEW recent_drift_metrics AS
SELECT 
    -- Recent time windows
    NOW() - INTERVAL '7 days' as week_ago,
    NOW() - INTERVAL '30 days' as month_ago,
    
    -- Embedding stats
    (SELECT COUNT(*) FROM embeddings_log 
     WHERE type = 'query' AND timestamp >= NOW() - INTERVAL '7 days') as recent_query_count,
    
    -- Behavior stats  
    (SELECT COUNT(*) FROM interaction_log 
     WHERE timestamp >= NOW() - INTERVAL '7 days') as recent_interaction_count,
     
    (SELECT AVG(CASE WHEN refusal_flag THEN 1.0 ELSE 0.0 END) 
     FROM interaction_log 
     WHERE timestamp >= NOW() - INTERVAL '7 days') as recent_refusal_rate,
     
    (SELECT AVG(CASE WHEN toxicity_flag THEN 1.0 ELSE 0.0 END) 
     FROM interaction_log 
     WHERE timestamp >= NOW() - INTERVAL '7 days') as recent_toxicity_rate,
     
    -- Accuracy stats
    (SELECT AVG(user_feedback_score) 
     FROM interaction_log 
     WHERE user_feedback_score IS NOT NULL 
     AND timestamp >= NOW() - INTERVAL '7 days') as recent_accuracy,
     
    (SELECT AVG(user_feedback_score) 
     FROM interaction_log 
     WHERE user_feedback_score IS NOT NULL 
     AND timestamp BETWEEN NOW() - INTERVAL '37 days' AND NOW() - INTERVAL '30 days') as baseline_accuracy;

-- Create a function to clean up old data (optional)
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up old embeddings_log entries (keep recent ones for drift detection)
    DELETE FROM embeddings_log 
    WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Note: Don't auto-delete interaction_log as it contains valuable historical data
    -- Clean manually or archive to separate table if needed
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create some example queries for drift analysis

-- Query 1: Get recent vs baseline embedding centroids (for manual drift calculation)
/*
-- Recent query embeddings centroid
SELECT AVG(embedding) as recent_centroid
FROM embeddings_log 
WHERE type = 'query' 
AND timestamp >= NOW() - INTERVAL '7 days';

-- Baseline query embeddings centroid  
SELECT AVG(embedding) as baseline_centroid
FROM embeddings_log 
WHERE type = 'query' 
AND timestamp BETWEEN NOW() - INTERVAL '37 days' AND NOW() - INTERVAL '30 days';
*/

-- Query 2: Behavior drift analysis
/*
SELECT 
    DATE_TRUNC('day', timestamp) as day,
    COUNT(*) as total_interactions,
    AVG(CASE WHEN refusal_flag THEN 1.0 ELSE 0.0 END) as refusal_rate,
    AVG(CASE WHEN toxicity_flag THEN 1.0 ELSE 0.0 END) as toxicity_rate,
    AVG(user_feedback_score) as avg_feedback
FROM interaction_log 
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY day;
*/

-- Query 3: Recent drift events
/*
SELECT 
    timestamp,
    kind,
    severity,
    details->>'drift_score' as drift_score,
    details->>'threshold' as threshold,
    action_taken
FROM drift_events 
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC;
*/

-- Insert some metadata for tracking schema version
INSERT INTO drift_events (kind, details, action_taken) 
VALUES ('schema_setup', '{"version": "1.0.0", "created_at": "' || NOW() || '"}', 'database_initialized')
ON CONFLICT DO NOTHING;

-- Success message
-- Note: This won't show in Supabase SQL editor, but would show in psql
SELECT 'Database schema setup completed successfully! 🎉' as status;