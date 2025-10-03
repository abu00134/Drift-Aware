# Drift-Aware 🎯

A comprehensive drift monitoring system for machine learning models that detects embedding drift, behavior drift, and accuracy drops in real-time.

## 🚀 Quick Deploy

### Deploy to Railway (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https%3A%2F%2Fgithub.com%2Fabu00134%2FDrift-Aware)

**Live Demo**: https://drift-aware-production.up.railway.app/metrics

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/abu00134/Drift-Aware)

### Deploy to Heroku

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/abu00134/Drift-Aware)

## ✨ Features

- 🎯 **Embedding Drift Detection**: Monitors query/document embedding distribution changes
- 🛡️ **Behavior Drift Detection**: Tracks refusal rates and toxicity rates  
- 📈 **Accuracy Drop Detection**: Monitors user feedback scores
- 📊 **Prometheus Metrics**: Exposes metrics for monitoring and alerting
- 🔄 **Auto-Retraining**: Triggers model retraining when drift is detected
- 🗄️ **PostgreSQL + pgvector**: Stores embeddings and interaction logs
- 🐳 **Docker Support**: Easy containerized deployment
- 📈 **Grafana Dashboards**: Pre-configured monitoring dashboards

## 📚 Quick Start

### 1. One-Click Deploy
Click the Railway deploy button above. It will automatically:
- Deploy PostgreSQL with pgvector extension
- Deploy the drift monitoring application  
- Set up environment variables
- Provide you with a live URL

### 2. Configure Environment Variables

After deployment, set these environment variables in your Railway dashboard:

```env
DATABASE_URL=postgresql://user:pass@host:port/db  # Auto-generated
OPENAI_API_KEY=your_openai_api_key_here
CHECK_INTERVAL_SECONDS=900
EMBEDDING_DIM=1536
EMBEDDING_COSINE_THRESHOLD=0.1
REFUSAL_RATE_THRESHOLD=0.05
TOXICITY_RATE_THRESHOLD=0.02
ACCURACY_DROP_POINTS=0.05
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
```

### 3. Initialize Database

Run the SQL setup script in your Railway PostgreSQL console:

```sql
-- Execute the contents of setup_database.sql
-- This creates all necessary tables and indexes
```

### 4. Start Monitoring

Your drift monitor will be available at:
- **Metrics Endpoint**: `https://your-app.railway.app/metrics`
- **Health Check**: Same endpoint

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │  Drift Monitor  │    │   Prometheus    │
│   (pgvector)    │◄───│   Service       │───►│   Metrics       │
│                 │    │                 │    │                 │
│ • Documents     │    │ • Embedding     │    │ • Grafana       │
│ • Interactions  │    │   Analysis      │    │ • Alerting      │
│ • Embeddings    │    │ • Auto-Retrain  │    │ • Dashboards    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Monitoring

The system exposes Prometheus metrics at `/metrics`:

- `embedding_drift_score` - Cosine distance between recent vs baseline embeddings
- `refusal_rate` - Recent refusal rate (rolling 7 days)
- `toxicity_rate` - Recent toxicity rate
- `recent_accuracy` - User feedback scores
- `retrain_events_total` - Counter of retraining events
- `reindex_events_total` - Counter of reindex events

## 🔧 Integration

### Log Interactions

```python
# In your application, log each interaction:
cur.execute("""
    INSERT INTO interaction_log 
    (user_query, model_response, refusal_flag, toxicity_flag, user_feedback_score)
    VALUES (%s, %s, %s, %s, %s)
""", (query, response, is_refusal, is_toxic, user_score))
```

### Log Embeddings

```python
# Log query embeddings for drift analysis:
query_embedding = get_embedding(user_query)
cur.execute("""
    INSERT INTO embeddings_log (type, embedding)
    VALUES ('query', %s)
""", (query_embedding.tolist(),))
```

## 🧪 Testing

Generate sample data to test the system:

```bash
# Generate baseline data
python generate_sample_data.py --scenario baseline

# Generate drift scenario  
python generate_sample_data.py --scenario drift --clean
```

## 🔧 Local Development

```bash
# Clone and setup
git clone https://github.com/abu00134/Drift-Aware.git
cd Drift-Aware
cp .env.example .env

# Install dependencies
pip install -r requirements.txt

# Run with Docker Compose (includes Prometheus & Grafana)
docker-compose up -d

# Or run locally
python drift_monitor.py
```

Access points:
- Drift Monitor: http://localhost:8000/metrics
- Prometheus: http://localhost:9090  
- Grafana: http://localhost:3000 (admin/admin)

## 🚨 Drift Detection

The system monitors three types of drift:

1. **Embedding Drift**: Compares recent query embeddings to baseline distributions using cosine distance
2. **Behavior Drift**: Tracks refusal and toxicity rates in model responses
3. **Accuracy Drift**: Monitors user feedback scores over time

When drift is detected:
- **Re-indexing** triggered for embedding drift
- **Model retraining** triggered for behavior/accuracy drift
- **Prometheus metrics** updated
- **Events logged** to database

## 🛠️ Configuration

Adjust drift thresholds via environment variables:

```bash
# Stricter monitoring
EMBEDDING_COSINE_THRESHOLD=0.05
REFUSAL_RATE_THRESHOLD=0.03
ACCURACY_DROP_POINTS=0.03

# More relaxed  
EMBEDDING_COSINE_THRESHOLD=0.15
REFUSAL_RATE_THRESHOLD=0.08
ACCURACY_DROP_POINTS=0.08
```

## 🔍 API Endpoints

- `GET /metrics` - Prometheus metrics endpoint
- Health check available at the same endpoint

## 🤝 Contributing

Feel free to open issues and pull requests!

## 📄 License

MIT License - see LICENSE file for details.

---

**Need help?** Check the `deploy.md` file for detailed deployment instructions or open an issue.