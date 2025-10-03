# Drift-Aware Retraining Pipeline 🎯

A comprehensive system for monitoring AI model drift and automatically triggering retraining when performance degrades or user behavior patterns change.

## 🎉 What You Have

This system monitors three types of drift:

1. **Embedding Drift** - Changes in query/document embedding distributions
2. **Behavior Drift** - Increased refusal rates or toxicity in model outputs  
3. **Accuracy Drift** - Declining user satisfaction or performance metrics

When drift is detected, it automatically triggers:
- **Document re-indexing** for embedding drift
- **Model fine-tuning/retraining** for behavior/accuracy drift

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supabase      │    │  Drift Monitor  │    │  Prometheus     │
│   (pgvector)    │◄───│   Service       │───►│   + Grafana     │
│                 │    │                 │    │                 │
│ • Documents     │    │ • Embedding     │    │ • Metrics       │
│ • Interactions  │    │   Analysis      │    │ • Dashboards    │
│ • Embeddings    │    │ • Trigger Logic │    │ • Alerting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
C:\Users\rohim\
├── docker-compose.yml          # Prometheus + Grafana setup
├── prometheus.yml              # Prometheus configuration  
├── .env.example               # Environment variables template
├── requirements.txt           # Python dependencies
├── setup_database.sql         # Database schema setup
├── drift_monitor.py          # Main monitoring service
├── generate_sample_data.py   # Test data generation
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Setup Environment

Create a virtual environment and install dependencies:

```powershell
# Create virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the environment template and fill in your values:

```powershell
Copy-Item .env.example .env
# Edit .env with your database credentials
```

Required variables:
```bash
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME
# Optional:
OPENAI_API_KEY=your_openai_key_here
```

### 3. Setup Database Schema

Run the SQL setup script in your Supabase SQL editor:

```sql
-- Copy and paste the contents of setup_database.sql
-- This creates all necessary tables and indexes
```

Or via psql:
```powershell
psql $env:DATABASE_URL -f setup_database.sql
```

### 4. Start Monitoring Infrastructure

Launch Prometheus and Grafana:

```powershell
docker compose up -d
```

Access points:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 5. Generate Sample Data (Optional)

Create test data to see the system in action:

```powershell
# Generate baseline data
python generate_sample_data.py --scenario baseline

# Generate drift scenario
python generate_sample_data.py --scenario drift --clean
```

### 6. Start the Drift Monitor

Run the main monitoring service:

```powershell
python drift_monitor.py
```

The service will:
- ✅ Start metrics server on http://localhost:8000/metrics
- 🔍 Run drift detection every 15 minutes (configurable)
- 📊 Update Prometheus metrics continuously

## 📊 Monitoring & Dashboards

### Grafana Dashboard Setup

1. Open Grafana at http://localhost:3000 (admin/admin)
2. Add Prometheus data source: `http://prometheus:9090`
3. Create panels for key metrics:

#### Essential Metrics to Track:
- `embedding_drift_score` - Cosine distance between recent vs baseline embeddings
- `refusal_rate` - Recent refusal rate (rolling 7 days)
- `toxicity_rate` - Recent toxicity rate  
- `recent_accuracy` - User feedback scores
- `retrain_events_total` - Counter of retraining events
- `reindex_events_total` - Counter of reindex events

#### Sample Dashboard Panels:

**Embedding Drift Over Time:**
```promql
embedding_drift_score
```

**Behavior Metrics:**
```promql
rate(refusal_rate[1h])
rate(toxicity_rate[1h])
```

**Accuracy Comparison:**
```promql
recent_accuracy
baseline_accuracy
```

**Action Events:**
```promql
increase(retrain_events_total[24h])
increase(reindex_events_total[24h])
```

### Alerting Rules

Set up alerts in Grafana for:
- Embedding drift score > 0.1
- Refusal rate > 5%
- Toxicity rate > 2%
- Accuracy drop > 5 points

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | - | PostgreSQL connection string (required) |
| `CHECK_INTERVAL_SECONDS` | 900 | How often to run drift checks (15 min) |
| `EMBEDDING_DIM` | 1536 | Embedding dimension (OpenAI ada-002) |
| `EMBEDDING_COSINE_THRESHOLD` | 0.1 | Embedding drift threshold |
| `REFUSAL_RATE_THRESHOLD` | 0.05 | Max acceptable refusal rate (5%) |
| `TOXICITY_RATE_THRESHOLD` | 0.02 | Max acceptable toxicity rate (2%) |
| `ACCURACY_DROP_POINTS` | 0.05 | Max acceptable accuracy drop |
| `PROMETHEUS_PORT` | 8000 | Metrics server port |

### Drift Detection Thresholds

Adjust thresholds based on your use case:

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

## 🔧 Integration with Your App

### Data Logging Requirements

Your main application needs to log data to the database tables:

#### 1. Log User Interactions
```python
# After each model interaction:
cur.execute("""
    INSERT INTO interaction_log 
    (user_query, model_response, refusal_flag, toxicity_flag, user_feedback_score)
    VALUES (%s, %s, %s, %s, %s)
""", (query, response, is_refusal, is_toxic, user_score))
```

#### 2. Log Query Embeddings
```python
# Log query embeddings for drift analysis:
query_embedding = get_embedding(user_query)  # Your embedding function
cur.execute("""
    INSERT INTO embeddings_log (type, embedding)
    VALUES ('query', %s)
""", (query_embedding.tolist(),))
```

#### 3. Log Document Embeddings
```python
# When indexing documents:
doc_embedding = get_embedding(document_text)
cur.execute("""
    INSERT INTO documents (source, content, embedding)
    VALUES (%s, %s, %s)
""", (source, content, doc_embedding.tolist()))

# Also log to embeddings_log:
cur.execute("""
    INSERT INTO embeddings_log (type, embedding)
    VALUES ('doc', %s)
""", (doc_embedding.tolist(),))
```

### Refusal Detection

Detect refusals automatically:
```python
refusal_patterns = [
    "I cannot", "I'm sorry, I can't", "I don't think I can",
    "I'm unable to", "I cannot assist", "I'm not able to"
]

def is_refusal(response: str) -> bool:
    return any(pattern.lower() in response.lower() for pattern in refusal_patterns)
```

### Toxicity Detection

Using OpenAI Moderation API:
```python
import openai

def is_toxic(text: str) -> bool:
    response = openai.moderations.create(input=text)
    return response.results[0].flagged
```

## 🎯 Testing Drift Scenarios

### Scenario 1: Baseline (No Drift)
```powershell
python generate_sample_data.py --scenario baseline
python drift_monitor.py
# Should show: "No drift detected - all metrics within thresholds"
```

### Scenario 2: Embedding Drift
```powershell
python generate_sample_data.py --scenario drift --clean
# Wait for next drift check cycle
# Should trigger: "Embedding drift detected! Triggering re-index"
```

### Scenario 3: Behavior/Accuracy Drift
The drift scenario also includes higher refusal rates and lower accuracy scores, which should trigger model retraining.

### View Results
Monitor the results in:
- **Logs**: Real-time drift detection output
- **Metrics**: http://localhost:8000/metrics
- **Grafana**: http://localhost:3000 dashboards
- **Database**: Query `drift_events` table for history

## 🛠️ Customization & Extensions

### 1. Advanced Drift Detection

Replace simple cosine distance with more sophisticated methods:

```python
# Add to drift_monitor.py
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

def detect_advanced_embedding_drift(self, recent_embeddings, baseline_embeddings):
    """Use EvidentlyAI for advanced drift detection"""
    recent_df = pd.DataFrame(recent_embeddings)
    baseline_df = pd.DataFrame(baseline_embeddings)
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline_df, current_data=recent_df)
    
    # Extract drift score from report
    return report.as_dict()
```

### 2. Custom Action Triggers

Implement your specific retraining pipeline:

```python
def fine_tune_model(self, conn, reason: str, details: Dict[str, Any]):
    """Custom implementation for your model retraining"""
    
    if "high_refusal_rate" in reason:
        # Update system prompts or safety filters
        self.update_system_prompt()
    
    elif "accuracy_drop" in reason:
        # Extract recent interactions for fine-tuning
        training_data = self.prepare_training_data(conn)
        
        # Launch fine-tuning job
        fine_tune_job = self.launch_openai_fine_tune(training_data)
        
        # Deploy when ready
        self.deploy_model_when_ready(fine_tune_job)
```

### 3. Cost Tracking

Add cost monitoring:

```python
# Add to Prometheus metrics:
cost_gauge = Gauge("total_cost_usd", "Total API cost in USD")

def track_api_cost(self, tokens_used: int, model: str):
    """Track API costs"""
    cost_per_token = {
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-4": 0.03 / 1000
    }
    
    cost = tokens_used * cost_per_token.get(model, 0)
    cost_gauge.inc(cost)
```

## 🔍 Troubleshooting

### Common Issues

1. **"DATABASE_URL is not set"**
   - Check your `.env` file
   - Ensure environment variables are loaded: `load_dotenv()`

2. **"pgvector extension not found"**
   - Run `CREATE EXTENSION IF NOT EXISTS vector;` in Supabase
   - Ensure pgvector is enabled in your database

3. **"No data for drift analysis"**
   - Generate sample data: `python generate_sample_data.py`
   - Ensure your app is logging interactions correctly

4. **Prometheus can't scrape metrics**
   - Check if drift_monitor.py is running
   - Verify port 8000 is accessible
   - Check Windows firewall settings

5. **Grafana shows no data**
   - Verify Prometheus is scraping successfully (check targets page)
   - Ensure correct data source URL: `http://prometheus:9090`

### Debug Mode

Run with debug logging:
```powershell
$env:LOG_LEVEL = "DEBUG"
python drift_monitor.py
```

## 📈 Performance Considerations

- **Database Indexing**: The setup script includes optimized indexes for time-based queries
- **Data Retention**: Use the `cleanup_old_data()` function to manage database size
- **Monitoring Frequency**: Adjust `CHECK_INTERVAL_SECONDS` based on your volume
- **Embedding Storage**: Consider dimensionality reduction for very high-dimensional embeddings

## 🚀 Production Deployment

### Option 1: Cloud VM (Recommended)

```powershell
# Deploy to AWS EC2, DigitalOcean, etc.
# 1. Copy files to server
# 2. Install Docker and Python
# 3. Run: docker compose up -d
# 4. Run: python drift_monitor.py (as a service)
```

### Option 2: Container Deployment

Create a Dockerfile:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "drift_monitor.py"]
```

### Option 3: Serverless (Partial)

Use scheduled functions for periodic checks:
- **AWS Lambda** + **CloudWatch Events**
- **Google Cloud Functions** + **Cloud Scheduler**  
- **Supabase Edge Functions** + **Cron**

## 🎉 Success! 

Your drift-aware retraining pipeline is now ready. The system will:

✅ Continuously monitor your AI system for drift  
✅ Automatically trigger retraining when needed  
✅ Provide rich metrics and dashboards  
✅ Log all events for debugging and analysis  
✅ Scale with your application needs  

**Next Steps:**
1. Integrate with your main application for data logging
2. Customize action triggers for your specific use case  
3. Set up alerting for critical drift events
4. Deploy to production environment

## 📚 Additional Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/) - Advanced drift detection
- [pgvector Documentation](https://github.com/pgvector/pgvector) - Vector database operations
- [Prometheus Monitoring](https://prometheus.io/docs/) - Metrics and alerting
- [Grafana Dashboards](https://grafana.com/docs/) - Visualization and alerting

---

**Questions or issues?** Check the troubleshooting section or review the logs for detailed error information.# Drift-Aware
