# Drift-Aware Deployment Guide

## Quick Deploy to Railway

### Prerequisites
1. GitHub account (already done ✅)
2. Railway account: https://railway.app

### Step 1: Deploy Database
1. Go to https://railway.app/new
2. Click "Deploy PostgreSQL"
3. Wait for deployment to complete
4. Click on the PostgreSQL service
5. Go to "Variables" tab
6. Copy the `DATABASE_URL` value

### Step 2: Deploy Application
1. Click "New Project" in Railway
2. Select "Deploy from GitHub repo"
3. Connect your GitHub account and select `abu00134/Drift-Aware`
4. Railway will automatically detect the Dockerfile and deploy

### Step 3: Configure Environment Variables
In your Railway app service, go to Variables and add:

```
DATABASE_URL=<paste the PostgreSQL DATABASE_URL from step 1>
CHECK_INTERVAL_SECONDS=900
EMBEDDING_DIM=1536
EMBEDDING_COSINE_THRESHOLD=0.1
REFUSAL_RATE_THRESHOLD=0.05
TOXICITY_RATE_THRESHOLD=0.02
ACCURACY_DROP_POINTS=0.05
PROMETHEUS_PORT=8000
LOG_LEVEL=INFO
```

### Step 4: Enable pgvector Extension
1. In Railway, go to your PostgreSQL service
2. Click on "Data" tab
3. Run this SQL command:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 5: Initialize Database Schema
Run the setup_database.sql script in your PostgreSQL instance.

### Your URLs will be:
- **Drift Monitor API**: `https://your-app.railway.app/metrics`
- **Database**: Automatically connected via DATABASE_URL

## Alternative: Render.com Deployment

If you prefer Render:
1. Connect your GitHub repo to Render
2. Choose "Web Service"
3. Use these settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python drift_monitor.py`
   - Add the same environment variables as above

## Local Development with Docker

```bash
# Clone your repo
git clone https://github.com/abu00134/Drift-Aware.git
cd Drift-Aware

# Create .env file (copy from .env.example)
cp .env.example .env

# Edit .env with your database credentials

# Run with Docker Compose (includes Prometheus & Grafana)
docker-compose up -d

# Or run just the app
docker build -t drift-aware .
docker run -p 8000:8000 --env-file .env drift-aware
```

Access points:
- Drift Monitor: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)