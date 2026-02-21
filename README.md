# 🎭 Sentiment Analysis MLOps Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-blue)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-model%20registry-red)](https://mlflow.org/)
[![AWS](https://img.shields.io/badge/AWS-deployment-orange)](https://aws.amazon.com/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-blue)](https://github.com/features/actions)

A **production-ready MLOps pipeline** for sentiment classification of YouTube comments with **automated CI/CD**, **DVC pipelines**, **MLflow tracking**, and **AWS cloud deployment**. Features multi-model training (SVM, DistilBERT), automated evaluation, model registry, and containerized serving with comprehensive testing and deployment automation.

## 🎯 Key Features

- 🤖 **Multi-Model Training**: SVM (baseline), SVM+SMOTE (balanced), DistilBERT (transformer)
- 🔄 **DVC Pipeline**: Reproducible ML workflows with data versioning and S3 storage
- 📊 **MLflow + DagsHub**: Experiment tracking, model registry, and artifact versioning
- 🚀 **GitHub Actions CI/CD**: Automated testing, building, and deployment pipeline
- ☁️ **AWS Architecture**: Multi-EC2 deployment with RDS PostgreSQL and load balancing
- 🐳 **Docker Compose**: Containerized backend (Flask) and frontend (Nginx) services
- ✅ **Comprehensive Testing**: Unit tests, smoke tests, and deployment validation
- 📈 **Monitoring**: Dashboard with model metrics, prediction history, and analytics

---

## 🏗️ Complete Architecture

![MLOps Architecture](pic/diagram.png)

---

## 🛠️ Tech Stack

### 🤖 ML & Data Science
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Text vectorization |
| **Classical ML** | scikit-learn | SVM models |
| **Class Balancing** | imbalanced-learn (SMOTE) | Handle imbalanced data |
| **Transformers** | Hugging Face Transformers | DistilBERT fine-tuning |
| **Deep Learning** | PyTorch | Backend for transformers |

### 📊 MLOps & Tracking
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Experiment Tracking** | MLflow + DagsHub | Log metrics, params, artifacts |
| **Model Registry** | MLflow Model Registry | Version and stage models |
| **Data Versioning** | DVC + S3 | Track datasets and pipelines |
| **Pipeline Orchestration** | DVC (dvc.yaml) | Define reproducible workflows |

### 🌐 Serving & API
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Flask 3.0+ | REST API backend |
| **WSGI Server** | Gunicorn | Production-grade server |
| **Reverse Proxy** | Nginx | Frontend & load balancing |
| **Database** | PostgreSQL (AWS RDS) | Prediction logging & analytics |
| **Containerization** | Docker + Docker Compose | Service orchestration |

### ☁️ Cloud & Infrastructure
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Cloud Platform** | AWS | Infrastructure hosting |
| **Compute** | EC2 (t2.small/medium) | Frontend & backend servers |
| **Database** | RDS PostgreSQL 15 | Managed database service |
| **Container Registry** | Docker Hub | Image storage & distribution |
| **Networking** | VPC, Security Groups | Network isolation & security |

### 🔄 CI/CD & DevOps
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Code Quality** | flake8, black, isort | Linting & formatting |
| **Testing** | pytest + pytest-cov | Unit & integration tests |
| **Secrets Management** | GitHub Secrets | Secure credential storage |

### 📦 Key Dependencies
```bash
# ML & Data
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0

# MLOps
mlflow>=2.5.0
dvc>=3.0.0
dvc[s3]

# API & Serving
flask>=3.0.0
gunicorn>=21.0.0
psycopg2-binary>=2.9.0

# DevOps
python-dotenv>=1.0.0
pyyaml>=6.0
boto3>=1.28.0
```

---

## 📁 Project Structure

```
sentiment_mlops/
│
├── .github/workflows/              # GitHub Actions CI/CD
│   ├── deploy.yml                 # Main deployment pipeline (10 stages)
│   └── deploy-prod.yml            # Production-only workflow
│
├── data/                          # Data directory (DVC tracked)
│   ├── raw/                       # Original dataset
│   │   ├── YoutubeCommentsDataSet.csv
│   │   └── YoutubeCommentsDataSet.csv.dvc  # DVC pointer
│   └── processed/                 # Preprocessed data
│       ├── train_vectors.npy, train_labels.npy
│       ├── test_vectors.npy, test_labels.npy
│       ├── metadata.json
│       └── smote/                 # SMOTE-balanced training data
│
├── src/                           # Training & preprocessing scripts
│   ├── pretrain.py               # Data preprocessing + optional SMOTE
│   ├── train_svm_noSMOTE.py      # SVM baseline training
│   ├── train_svm.py              # SVM + SMOTE training
│   ├── train_bert.py             # DistilBERT fine-tuning
│   ├── evaluate.py               # Model comparison & MLflow registration
│   └── predict.py                # Batch prediction utility
│
├── aws_serving/                   # AWS deployment files
│   ├── app/                       # Backend application
│   │   ├── app.py                # Flask API with /predict, /health, /reload
│   │   ├── dashboard.py          # Dashboard routes & analytics
│   │   ├── database.py           # PostgreSQL RDS integration
│   │   ├── pull_model.py         # Download model from MLflow
│   │   ├── Dockerfile            # Backend container
│   │   ├── requirements.txt      # Python dependencies
│   │   └── artifacts/            # Model artifacts (from MLflow)
│   │       └── best_model/       # Downloaded production model
│   ├── frontend/                  # Frontend Nginx proxy
│   │   ├── index.html            # Landing page
│   │   ├── nginx.conf            # Reverse proxy config
│   │   └── Dockerfile            # Frontend container
│   ├── scripts/                   # Deployment utilities
│   │   ├── init_rds.py           # Initialize RDS database
│   │   ├── inspect_run.py        # Debug MLflow runs
│   │   └── pull_model.py         # Download from registry
│   ├── docker-compose.yml         # Local development setup
│   ├── docker-compose.backend.yml # Backend-only deployment
│   ├── docker-compose.frontend.yml # Frontend-only deployment
│   └── AWS_DEPLOYMENT_GUIDE.md    # Step-by-step AWS setup
│
├── serving/                       # Legacy FastAPI server (optional)
│   ├── app.py                    # FastAPI application
│   ├── model_loader.py           # Load models from S3/MLflow/disk
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Logging utilities
│   ├── requirements.txt          # Serving dependencies
│   └── Dockerfile                # Container for inference
│
├── models/                        # Trained model artifacts (local)
│   ├── svm_sentiment.joblib      # SVM+SMOTE model
│   ├── svm_scaler.joblib         # Feature scaler
│   ├── svm_baseline_no_smote.joblib
│   ├── distilbert_sentiment/     # DistilBERT checkpoint
│   ├── svm_baseline_info.json    # Metrics for baseline
│   ├── svm_model_info.json       # Metrics for SVM+SMOTE
│   ├── distilbert_info.json      # Metrics for DistilBERT
│   └── model_comparison.csv      # Evaluation results
│
├── mlartifacts/                   # MLflow local storage
│   └── 1/                         # Experiment ID
│       └── <run_id>/              # Individual run artifacts
│
├── notebooks/                     # Jupyter notebooks (exploratory)
│   ├── preprocess.ipynb
│   ├── train_svm_baseline.ipynb
│   ├── train_svm.ipynb
│   └── train_distilbert.ipynb
│
├── tests/                         # Unit & integration tests
│   ├── conftest.py               # pytest fixtures
│   ├── test_app.py               # FastAPI endpoint tests
│   ├── test_config.py            # Configuration tests
│   └── test_model_loader.py      # Model loading tests
│
├── scripts/                       # Utility scripts
│   ├── pull_best_model.py        # Download from MLflow registry
│   ├── promote_model.py          # Upload model to S3
│   ├── deploy_hf_space.sh        # Hugging Face Space deployment
│   ├── deploy.sh                 # General deployment script
│   ├── setup_ec2.sh              # EC2 initial setup
│   └── smoke_test.sh             # Manual smoke testing
│
├── dvc.yaml                       # DVC pipeline definition (6 stages)
├── params.yaml                    # Pipeline parameters (single source of truth)
├── .dvc/config                    # DVC remote configuration (S3)
├── pyproject.toml                # Project metadata & tool config
├── requirements.txt              # Python dependencies
├── CI_CD_SETUP.md                # GitHub Actions setup guide
├── DEPLOYMENT_ARCHITECTURE.md     # AWS architecture options
├── GITHUB_SECRETS_SETUP.md        # Required secrets documentation
└── README.md                      # This file
```
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Git** & **DVC**
- **Docker** & **Docker Compose**
- **AWS Account** (for deployment)
- **DagsHub Account** (for MLflow tracking)

---

### 1️⃣ Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/sairam030/sentiment_mlops.git
cd sentiment_mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2️⃣ Configure Credentials

Create a `.env` file in the project root:

```bash
# ─────────────────────────────────────────────────
# MLflow Tracking (DagsHub)
# ─────────────────────────────────────────────────
export MLFLOW_TRACKING_USERNAME="your_dagshub_username"
export MLFLOW_TRACKING_PASSWORD="your_dagshub_token"
export MLFLOW_TRACKING_URI="https://dagshub.com/your_username/mlops_sentimentAnalysis.mlflow"

# ─────────────────────────────────────────────────
# AWS Credentials (for DVC & S3 model storage)
# ─────────────────────────────────────────────────
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"

# ─────────────────────────────────────────────────
# Database (AWS RDS PostgreSQL)
# ─────────────────────────────────────────────────
export DATABASE_URL="postgresql://user:password@sentiment-db.xxxxx.us-east-1.rds.amazonaws.com:5432/sentiment_db"
export DATABASE_SSL="require"

# ─────────────────────────────────────────────────
# Model Serving Configuration
# ─────────────────────────────────────────────────
export MODEL_STAGE="production"  # or "staging"
export S3_MODEL_BUCKET="sentiment-mlops-models"
```

Load environment variables:
```bash
source .env
```

---

### 3️⃣ Setup DVC & Pull Data

```bash
# Initialize DVC (if not already done)
dvc remote add -d myremote s3://your-bucket-name/dvc-storage
dvc remote modify myremote region us-east-1

# Pull data from remote storage
dvc pull

# Verify data
ls -lh data/raw/
```

---

### 4️⃣ Run the DVC Pipeline

**Run the entire pipeline (all 6 stages):**
```bash
dvc repro
```

This will execute:
1. `preprocess` - Generate embeddings from raw data
2. `train_svm_baseline` - Train SVM without SMOTE
3. `preprocess_smote` - Apply SMOTE balancing
4. `train_svm_smote` - Train SVM on balanced data
5. `train_bert` - Fine-tune DistilBERT
6. `evaluate` - Compare models & register best to MLflow

**Or run individual stages:**
```bash
# Preprocess data
dvc repro preprocess

# Train specific model
dvc repro train_svm_baseline
dvc repro train_svm_smote
dvc repro train_bert

# Evaluate all models
dvc repro evaluate
```

**View pipeline DAG:**
```bash
dvc dag
```

---

### 5️⃣ Check MLflow Experiments

```bash
# View in DagsHub (recommended)
open https://dagshub.com/sairam030/mlops_sentimentAnalysis

# Or run local MLflow UI
mlflow ui --port 5000
open http://localhost:5000
```

**Registered Models:** Navigate to "Models" tab to see:
- `sentiment-best-model`
- Model versions with aliases: `staging`, `production`
- Metrics: accuracy, F1, precision, recall

---

### 6️⃣ Test Locally with Docker

**Build and run backend:**
```bash
cd aws_serving/app

# Pull model from MLflow first
python pull_model.py --alias production --output artifacts/

# Build Docker image
docker build -t sentiment-backend:local .

# Run container
docker run -p 5000:5000 \
  -e DATABASE_URL="${DATABASE_URL}" \
  -e DATABASE_SSL=require \
  sentiment-backend:local
```

**Test the API:**
```bash
# Health check
curl http://localhost:5000/health

# Prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'

# Dashboard
open http://localhost:5000/dashboard
```

---

### 7️⃣ Deploy to AWS (CI/CD)

**Setup GitHub Secrets** (see [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md)):
```
DAGSHUB_USERNAME, DAGSHUB_TOKEN
DOCKERHUB_TOKEN
TEST_FRONTEND_HOST, TEST_FRONTEND_USER, TEST_FRONTEND_SSH_KEY
TEST_BACKEND_HOST, TEST_BACKEND_USER
RDS_DATABASE_URL
PROD_FRONTEND_HOST, PROD_BACKEND_HOST, ...
```

**Trigger deployment:**
```bash
git add .
git commit -m "Deploy to AWS"
git push origin main
```

GitHub Actions will automatically:
1. Pull model from MLflow
2. Build Docker images
3. Run linting & tests
4. Push to Docker Hub
5. Deploy to TEST environment
6. Run smoke tests
7. Wait for manual approval
8. Deploy to PRODUCTION

---

## 📊 DVC Pipeline Details

The pipeline is defined in [dvc.yaml](dvc.yaml) with 6 stages orchestrated by DVC:

### Stage 1: `preprocess`
**Script:** [src/pretrain.py](src/pretrain.py)  
**Purpose:** Generate sentence embeddings and split data

```yaml
deps:
  - src/pretrain.py
  - data/raw/YoutubeCommentsDataSet.csv
params:
  - preprocess.embedding_model: all-MiniLM-L6-v2
  - preprocess.test_size: 0.2
  - preprocess.random_state: 42
outs:
  - data/processed/{train,test}_{vectors,labels}.npy
  - data/processed/metadata.json
```

**What it does:**
- Loads raw YouTube comments
- Generates 384-dim embeddings using Sentence-BERT
- Splits into 80% train, 20% test
- Saves as NumPy arrays

---

### Stage 2a: `train_svm_baseline`
**Script:** [src/train_svm_noSMOTE.py](src/train_svm_noSMOTE.py)  
**Purpose:** Baseline SVM on imbalanced data

```yaml
deps:
  - data/processed/train_vectors.npy
  - data/processed/train_labels.npy
params:
  - train.svm.kernel: rbf
  - train.svm.C: 1.0
  - train.svm.gamma: scale
outs:
  - models/svm_baseline_no_smote.joblib
  - models/svm_baseline_scaler.joblib
metrics:
  - models/svm_baseline_info.json
```

**What it does:**
- Trains SVM with RBF kernel
- No class balancing (baseline)
- Logs metrics to MLflow

---

### Stage 2b: `preprocess_smote`
**Script:** [src/pretrain.py](src/pretrain.py) `--smote`  
**Purpose:** Apply SMOTE to balance classes

```yaml
deps:
  - src/pretrain.py
params:
  - train.smote.random_state: 42
outs:
  - data/processed/smote/train_vectors.npy
  - data/processed/smote/train_labels.npy
```

**What it does:**
- Applies SMOTE oversampling to minority classes
- Balances training data distribution

---

### Stage 2c: `train_svm_smote`
**Script:** [src/train_svm.py](src/train_svm.py)  
**Purpose:** SVM on balanced data

```yaml
deps:
  - data/processed/smote/train_vectors.npy
  - data/processed/smote/train_labels.npy
outs:
  - models/svm_sentiment.joblib
  - models/svm_scaler.joblib
metrics:
  - models/svm_model_info.json
```

**What it does:**
- Trains SVM on SMOTE-balanced data
- Better performance on minority classes

---

### Stage 2d: `train_bert`
**Script:** [src/train_bert.py](src/train_bert.py)  
**Purpose:** Fine-tune DistilBERT

```yaml
deps:
  - src/train_bert.py
  - data/raw/YoutubeCommentsDataSet.csv
params:
  - train.bert.base_model: distilbert-base-uncased
  - train.bert.epochs: 3
  - train.bert.learning_rate: 2e-5
  - train.bert.batch_size: 16
  - train.bert.max_length: 128
outs:
  - models/distilbert_sentiment/
metrics:
  - models/distilbert_info.json
```

**What it does:**
- Fine-tunes DistilBERT transformer
- 3 epochs, batch size 16
- Saves in HuggingFace format

---

### Stage 3: `evaluate`
**Script:** [src/evaluate.py](src/evaluate.py)  
**Purpose:** Compare all models and register best

```yaml
deps:
  - models/svm_baseline_info.json
  - models/svm_model_info.json
  - models/distilbert_info.json
params:
  - mlflow.experiment_name: sentiment-classification
  - mlflow.registered_model_name: sentiment-best-model
outs:
  - models/model_comparison.csv
```

**What it does:**
- Loads all 3 trained models
- Evaluates on test set
- Compares: accuracy, F1, precision, recall
- Selects best model
- **Registers to MLflow Model Registry**
- Creates comparison CSV

---

## 📈 MLflow Integration

All training runs are tracked in MLflow with the following logged:

**Metrics Logged:**
- Accuracy, Precision, Recall, F1 (macro)
- Confusion matrix
- Training time
- Model size

**Parameters Logged:**
- SVM: kernel, C, gamma
- SMOTE: random_state
- DistilBERT: epochs, learning_rate, batch_size, max_length

**Artifacts Logged:**
- Model files (.joblib or HuggingFace format)
- Metrics JSON files
- Confusion matrix plots

**Model Registry:**
- Best model registered as `sentiment-best-model`
- Aliases: `staging`, `production`
- Versioned for rollback capability

**Access MLflow:**
```bash
# DagsHub (remote)
https://dagshub.com/sairam030/mlops_sentimentAnalysis

# Local UI
mlflow ui --port 5000
```

---

## 🔄 GitHub Actions CI/CD Pipeline

The complete CI/CD pipeline is defined in [.github/workflows/deploy.yml](.github/workflows/deploy.yml)

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  GITHUB ACTIONS CI/CD - 10 STAGE PIPELINE                        │
└──────────────────────────────────────────────────────────────────┘

Trigger: git push to main

  [1] 🎯 Pull Best Model from MLflow
      └─ Download production model from DagsHub
      └─ Copy to aws_serving/app/artifacts/
      └─ Upload as GitHub Actions artifact
      
  [2] 🏗️ Build Docker Images
      └─ Backend: Flask + Gunicorn + ML models
      └─ Frontend: Nginx reverse proxy
      └─ Tag as :test
      
  [3] 🔍 Lint & Code Quality
      └─ flake8 (PEP8 compliance)
      └─ black (code formatting)
      └─ isort (import sorting)
      
  [4] 🧪 Unit Tests
      └─ pytest with PostgreSQL service
      └─ Coverage report generation
      └─ Test: endpoints, model loading, database
      
  [5] 📤 Push to Docker Hub (:test)
      └─ Push sairam030/sentiment-backend:test
      └─ Push sairam030/sentiment-frontend:test
      
  [6] 🚀 Deploy to Test Environment
      └─ SSH to Frontend EC2 (18.61.98.223)
      └─ Jump to Backend EC2 (private subnet)
      └─ docker-compose up -d
      └─ Inject DATABASE_URL secret
      
  [7] 💨 Smoke Tests
      └─ Health check: GET /api/health
      └─ Prediction test: POST /api/predict
      └─ Dashboard access: GET /dashboard
      
  [8] ⏸️ Manual Approval
      └─ GitHub Environments protection rule
      └─ Human approval required for production
      
  [9] 🏷️ Tag Production Images
      └─ Retag :test → :prod
      └─ Push sairam030/sentiment-backend:prod
      └─ Push sairam030/sentiment-frontend:prod
      
 [10] 🌐 Deploy to Production
      └─ SSH to Production EC2s
      └─ Pull :prod images
      └─ Zero-downtime deployment
      └─ Validate health checks
```

### Required GitHub Secrets

See [GITHUB_SECRETS_SETUP.md](GITHUB_SECRETS_SETUP.md) for complete setup guide.

**MLflow / DagsHub:**
- `DAGSHUB_USERNAME` - Your DagsHub username
- `DAGSHUB_TOKEN` - DagsHub personal access token

**Docker Hub:**
- `DOCKERHUB_TOKEN` - Docker Hub access token

**Test Environment:**
- `TEST_FRONTEND_HOST` - Public IP of test frontend EC2
- `TEST_FRONTEND_USER` - SSH username (ubuntu)
- `TEST_FRONTEND_SSH_KEY` - Private SSH key content
- `TEST_BACKEND_HOST` - Private IP of test backend EC2
- `TEST_BACKEND_USER` - SSH username

**Production Environment:**
- `PROD_FRONTEND_HOST` - Public IP of prod frontend EC2
- `PROD_FRONTEND_USER` - SSH username
- `PROD_FRONTEND_SSH_KEY` - Private SSH key
- `PROD_BACKEND_HOST` - Private IP of prod backend EC2
- `PROD_BACKEND_USER` - SSH username

**Database:**
- `RDS_DATABASE_URL` - PostgreSQL connection string

---

## 🌐 API Endpoints

### `GET /`
Landing page with API information

**Response:**
```json
{
  "message": "Sentiment Analysis API",
  "endpoints": ["/predict", "/health", "/reload", "/dashboard"]
}
```

---

### `GET /health`
Health check and model status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "DistilBERT",
  "model_info": {
    "accuracy": 0.8956,
    "f1_macro": 0.8723,
    "version": "v2"
  },
  "database_connected": true
}
```

---

### `POST /predict`
Predict sentiment for text input

**Request:**
```json
{
  "text": "This movie was absolutely amazing! I loved every minute of it."
}
```

**Response:**
```json
{
  "text": "This movie was absolutely amazing! I loved every minute of it.",
  "prediction": "POSITIVE",
  "confidence": 0.9876,
  "all_scores": {
    "POSITIVE": 0.9876,
    "NEUTRAL": 0.0098,
    "NEGATIVE": 0.0026
  },
  "model_type": "DistilBERT",
  "latency_ms": 45.23
}
```

---

### `POST /reload`
Hot-reload model without restarting server

**Request:**
```json
{
  "alias": "production"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model reloaded successfully",
  "model_type": "DistilBERT",
  "version": "v3"
}
```

---

### `GET /dashboard`
Interactive dashboard with:
- Model comparison metrics
- Prediction history (last 100 predictions)
- Sentiment distribution chart
- System health indicators

**Access:** `http://18.61.98.223/dashboard`

---

## 🧪 Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=serving --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Test Coverage

**Unit Tests:**
- ✅ API endpoint testing ([tests/test_app.py](tests/test_app.py))
- ✅ Configuration validation ([tests/test_config.py](tests/test_config.py))
- ✅ Model loading from disk/S3/MLflow ([tests/test_model_loader.py](tests/test_model_loader.py))

**Integration Tests:**
- ✅ PostgreSQL database connection
- ✅ MLflow artifact download
- ✅ End-to-end prediction flow

**Smoke Tests (CI/CD):**
- ✅ Health endpoint validation
- ✅ Sample prediction verification
- ✅ Dashboard accessibility

---

## 🚢 Deployment Options

### Option 1: AWS Multi-EC2 (Current)

**Architecture:** Separate EC2s for frontend and backend per environment

**Advantages:**
- ✅ Independent scaling
- ✅ Better isolation
- ✅ Production-grade setup

**Setup Guide:** See [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md)

---

### Option 2: Docker Compose (Local/Single Server)

```bash
cd aws_serving

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services Started:**
- Backend: http://localhost:5000
- Frontend: http://localhost:80
- Dashboard: http://localhost:80/dashboard

---

### Option 3: Hugging Face Spaces

```bash
# Deploy to HF Spaces
bash scripts/deploy_hf_space.sh

# Or manually
cd hf_space
huggingface-cli login
huggingface-cli repo create sentiment-analysis --type space --space_sdk docker
git push hf main
```

---

## 📝 Configuration

All pipeline parameters are centralized in [params.yaml](params.yaml):

```yaml
# ─────────────────────────────────────────────────
# Paths Configuration
# ─────────────────────────────────────────────────
paths:
  raw_data: data/raw/YoutubeCommentsDataSet.csv
  processed_dir: data/processed/
  models_dir: models/

# ─────────────────────────────────────────────────
# Preprocessing Configuration
# ─────────────────────────────────────────────────
preprocess:
  embedding_model: all-MiniLM-L6-v2      # Sentence-BERT model
  test_size: 0.2                         # 80/20 train/test split
  random_state: 42                       # Reproducibility

# ─────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────
train:
  # SVM Hyperparameters
  svm:
    kernel: rbf                          # RBF kernel
    C: 1.0                               # Regularization
    gamma: scale                         # Kernel coefficient
    random_state: 42
  
  # SMOTE Configuration
  smote:
    random_state: 42                     # For reproducibility
  
  # DistilBERT Configuration
  bert:
    base_model: distilbert-base-uncased  # HuggingFace model
    epochs: 3                            # Training epochs
    learning_rate: 2e-5                  # AdamW learning rate
    batch_size: 16                       # Batch size
    max_length: 128                      # Max token length
    weight_decay: 0.01                   # L2 regularization
    warmup_ratio: 0.1                    # LR warmup

# ─────────────────────────────────────────────────
# MLflow Configuration
# ─────────────────────────────────────────────────
mlflow:
  experiment_name: sentiment-classification
  tracking_uri: https://dagshub.com/sairam030/mlops_sentimentAnalysis.mlflow
  registered_model_name: sentiment-best-model
```

**To modify parameters:**
1. Edit `params.yaml`
2. Run `dvc repro` to re-execute affected stages
3. DVC automatically tracks parameter changes

---

## 🔧 Advanced Usage

### Train with GPU

```bash
# Check GPU availability
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run DistilBERT training
dvc repro train_bert
```

---

### Skip Certain Stages

```bash
# Train only SVM, skip BERT
dvc repro --single-item train_svm_smote

# Run until specific stage
dvc repro train_svm_baseline
```

---

### View Metrics

```bash
# Show all metrics
dvc metrics show

# Compare metrics across branches
dvc metrics diff main experiment-branch
```

---

### Push/Pull DVC Data

```bash
# Push data to remote (S3)
dvc push

# Pull data from remote
dvc pull

# Check remote status
dvc status --remote
```

---

### Debug Pipeline

```bash
# Dry run (show what will execute)
dvc repro --dry

# Force re-run specific stage
dvc repro -f train_bert

# Show pipeline dependency graph
dvc dag --md
```

---

### Manual Model Promotion

```bash
# Download from MLflow
python scripts/pull_best_model.py --alias production --output artifacts/

# Promote to S3 (optional)
python scripts/promote_model.py --latest --alias production
```

---

### Database Management

```bash
# Initialize RDS database
cd aws_serving/scripts
python init_rds.py

# Inspect predictions
python -c "from database import get_all_predictions; print(get_all_predictions(limit=10))"
```

---

## 📚 Dataset Information

**Source:** YouTube Comments Dataset  
**Size:** ~15,000 comments  
**Labels:** 
- 🟢 **Positive** (1)
- 🟡 **Neutral** (0)  
- 🔴 **Negative** (-1)

**Location:** `data/raw/YoutubeCommentsDataSet.csv`

**Sample Data:**
```csv
text,label
"This video is amazing!",1
"Not sure what to think about this",0
"Terrible content, waste of time",-1
```

**Class Distribution:**
- Imbalanced dataset (use SMOTE to balance)
- Majority class: Positive
- Minority class: Negative

---

## 🔒 Security Best Practices

**Secrets Management:**
- ✅ Never commit `.env` files
- ✅ Use GitHub Secrets for CI/CD
- ✅ Rotate credentials regularly
- ✅ Use IAM roles for AWS access

**Network Security:**
- ✅ Backend in private subnet
- ✅ RDS in private subnet
- ✅ Security groups restrict access
- ✅ SSL/TLS for database connections

**Docker Security:**
- ✅ Use non-root users in containers
- ✅ Scan images for vulnerabilities
- ✅ Keep base images updated
- ✅ Minimize attack surface

---

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes and test**
   ```bash
   pytest tests/ -v
   flake8 src/ serving/
   black src/ serving/ tests/
   ```
4. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

**Code Quality Checks:**
- Linting: `flake8`
- Formatting: `black`
- Import sorting: `isort`
- Type checking: `mypy` (optional)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DagsHub** for MLflow hosting and version control
- **Hugging Face** for transformer models and datasets
- **Sentence-Transformers** for embedding generation
- **AWS** for cloud infrastructure
- **GitHub Actions** for CI/CD automation

---

## 👤 Author

**Sairam**
- GitHub: [@sairam030](https://github.com/sairam030)
- DagsHub: [@sairam030](https://dagshub.com/sairam030)
- Project: [sentiment_mlops](https://github.com/sairam030/sentiment_mlops)

---

## 🔗 Related Documentation

- [CI/CD Setup Guide](CI_CD_SETUP.md) - Complete GitHub Actions setup
- [Deployment Architecture](DEPLOYMENT_ARCHITECTURE.md) - AWS infrastructure options
- [GitHub Secrets Setup](GITHUB_SECRETS_SETUP.md) - Required secrets configuration
- [AWS Deployment Guide](aws_serving/AWS_DEPLOYMENT_GUIDE.md) - Step-by-step AWS deployment

---

## 📞 Support

For issues, questions, or contributions:
- 🐛 [Open an issue](https://github.com/sairam030/sentiment_mlops/issues)
- 💬 [Start a discussion](https://github.com/sairam030/sentiment_mlops/discussions)
- 📧 Contact via GitHub profile

---

## 🎯 Project Status

- ✅ DVC Pipeline: **Production Ready**
- ✅ MLflow Tracking: **Active**
- ✅ CI/CD Pipeline: **Operational**
- ✅ AWS Deployment: **Running**
- ✅ Test Coverage: **>80%**
- 🚧 Monitoring Dashboard: **In Progress**

---

**Last Updated:** February 2026  
**Version:** 2.0.0  
**License:** MIT



