# 🎭 Sentiment Analysis MLOps Pipeline

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)
[![DVC](https://img.shields.io/badge/DVC-data%20versioning-blue)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-model%20registry-red)](https://mlflow.org/)

A production-ready MLOps pipeline for sentiment classification of YouTube comments using multiple deep learning and traditional ML approaches, with automated training, evaluation, model registry, and FastAPI serving.

**Features:**
- 🚀 Three model architectures: SVM (baseline), SVM+SMOTE (class-balanced), DistilBERT (transformer-based)
- 📊 Automated DVC pipeline with reproducible data versioning
- 🔄 MLflow experiment tracking and model registry integration
- 🌐 FastAPI inference server with hot-reload capability
- 🐳 Docker containerization for both training and serving
- 📈 Model comparison and automatic best-model promotion
- ✅ Unit tests and integration tests included

---

## 📐 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SENTIMENT ANALYSIS PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────┘

INPUT DATA
    │
    └─→ data/raw/YoutubeCommentsDataSet.csv
            │
            ├─ Preprocessing (pretrain.py)
            │   │ • Sentence-BERT embedding (all-MiniLM-L6-v2)
            │   │ • Train/test split (80/20)
            │   └─→ data/processed/{train,test}_{vectors,labels}.npy
            │
            └─→ DUAL PROCESSING PATHS
                │
                ├─→ PATH A: SVM MODELS
                │   │
                │   ├─ SVM Baseline (train_svm_noSMOTE.py)
                │   │   └─→ models/svm_baseline_{no_smote.joblib, scaler.joblib}
                │   │       └─→ metrics: models/svm_baseline_info.json
                │   │
                │   └─ SMOTE Balancing (pretrain.py --smote)
                │       │
                │       └─ SVM+SMOTE (train_svm.py)
                │           └─→ models/svm_{sentiment.joblib, scaler.joblib}
                │               └─→ metrics: models/svm_model_info.json
                │
                └─→ PATH B: TRANSFORMER MODEL
                    │
                    └─ DistilBERT Fine-tuning (train_bert.py)
                        • Base: distilbert-base-uncased
                        • 3 epochs, batch_size=16, max_length=128
                        └─→ models/distilbert_sentiment/ (HuggingFace format)
                            └─→ metrics: models/distilbert_info.json

EVALUATION & DEPLOYMENT
    │
    ├─→ evaluate.py
    │   • Compare all 3 models on test set
    │   • Select best model (accuracy/F1)
    │   • Register to MLflow Model Registry
    │   └─→ models/model_comparison.csv
    │
    ├─→ promote_model.py
    │   • Upload best model to S3
    │   • Tag as production/staging
    │
    └─→ FastAPI Inference Server (serving/app.py)
        • Auto-load best model on startup
        • POST /predict — inference endpoint
        • GET /health — model status
        • POST /reload — hot-reload without restart
        └─→ Docker container

EXPERIMENT TRACKING
    │
    └─→ MLflow + DagsHub
        • Log all hyperparameters
        • Track metrics (accuracy, F1, precision, recall)
        • Version all artifacts
        • Register production-ready models

LABELS: [Positive, Neutral, Negative]
```

---

## 🛠️ Tech Stack

### Core ML/Data Science
| Component | Technology | Version |
|-----------|-----------|---------|
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Latest |
| **Classical ML** | scikit-learn | Latest |
| **Class Balancing** | imbalanced-learn (SMOTE) | Latest |
| **Transformers** | Hugging Face Transformers | Latest |
| **Transformer Base** | DistilBERT | distilbert-base-uncased |

### ML Workflow & Registry
| Component | Technology | Version |
|-----------|-----------|---------|
| **Experiment Tracking** | MLflow | Latest |
| **Model Registry** | MLflow Registry (DagsHub) | - |
| **Data Versioning** | DVC (Data Version Control) | Latest |
| **Pipeline Orchestration** | DVC Stages (dvc.yaml) | - |

### Serving & API
| Component | Technology | Version |
|-----------|-----------|---------|
| **Web Framework** | FastAPI | 0.100+ |
| **ASGI Server** | Uvicorn | Latest |
| **Containerization** | Docker | - |

### DevOps & Cloud
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Storage** | AWS S3 | Model versioning & deployment |
| **Code Quality** | Ruff (linter & formatter) | Python linting |
| **Testing** | pytest | Unit & integration tests |
| **Python Version** | 3.12+ | Latest stable |

### Package Dependencies
```
pandas, numpy, matplotlib, seaborn          # Data handling
scikit-learn, imbalanced-learn              # ML models
sentence-transformers                       # Embeddings
transformers, accelerate, datasets, huggingface_hub  # Transformers
pyyaml                                      # Config
mlflow                                      # Experiment tracking
python-dotenv                               # Environment variables
dvc, dvc[s3]                                # Data versioning & S3 integration
```

---

## 📁 Project Structure

```
sentiment_mlops/
│
├── data/                          # Data directory (DVC tracked)
│   ├── raw/                       # Original dataset
│   │   └── YoutubeCommentsDataSet.csv
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
│   ├── evaluate.py               # Model comparison & evaluation
│   └── predict.py                # Batch prediction utility
│
├── serving/                       # FastAPI inference server
│   ├── app.py                    # Main FastAPI application
│   ├── model_loader.py           # Load models (S3/MLflow/disk)
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Logging utilities
│   ├── requirements.txt          # Serving dependencies
│   └── Dockerfile                # Container for inference
│
├── models/                        # Trained model artifacts
│   ├── svm_sentiment.joblib      # SVM+SMOTE model
│   ├── svm_scaler.joblib         # Feature scaler
│   ├── distilbert_sentiment/     # DistilBERT checkpoint
│   └── model_comparison.csv      # Evaluation results
│
├── mlartifacts/                   # MLflow artifacts & models
│
├── hf_space/                      # Hugging Face Space deployment
│   ├── Dockerfile
│   └── requirements.txt
│
├── notebooks/                     # Jupyter notebooks
│   ├── preprocess.ipynb
│   ├── train_svm.ipynb
│   └── train_distilbert.ipynb
│
├── tests/                         # Unit & integration tests
│   ├── conftest.py
│   ├── test_app.py               # FastAPI endpoint tests
│   ├── test_config.py            # Configuration tests
│   └── test_model_loader.py      # Model loading tests
│
├── scripts/                       # Utility scripts
│   ├── promote_model.py           # Upload model to S3
│   ├── deploy.sh                 # Deploy inference server
│   └── setup_ec2.sh              # EC2 setup
│
├── dvc.yaml                       # DVC pipeline definition
├── params.yaml                    # Pipeline parameters (single source of truth)
├── pyproject.toml                # Project metadata & tool configuration
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **pip** or **conda**
- **Git**
- **DVC** (for data versioning)
- **AWS S3 credentials** (for model promotion — optional)
- **DagsHub credentials** (for MLflow tracking)

### 1️⃣ Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-user/sentiment_mlops.git
cd sentiment_mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install pytest pytest-cov
```

### 2️⃣ Configure Credentials

Create a `.env` file in the project root:

```bash
# MLflow (DagsHub)
export MLFLOW_TRACKING_USERNAME="your_dagshub_username"
export MLFLOW_TRACKING_PASSWORD="your_dagshub_token"
export MLFLOW_TRACKING_URI="https://dagshub.com/your_username/mlops_sentimentAnalysis.mlflow"

# AWS S3 (for model promotion)
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"

# Serving
export MODEL_STAGE="staging"  # or "production"
export S3_MODEL_BUCKET="sentiment-mlops-models"
```

Load environment variables:
```bash
source .env
```

### 3️⃣ Run the DVC Pipeline

Pull data (if DVC remote is configured):
```bash
dvc pull
```

Run the entire pipeline:
```bash
dvc repro
```

Or run individual stages:
```bash
# Preprocess data
dvc repro preprocess

# Train SVM baseline
dvc repro train_svm_baseline

# Train SVM with SMOTE
dvc repro preprocess_smote
dvc repro train_svm_smote

# Fine-tune DistilBERT
dvc repro train_bert

# Evaluate all models
dvc repro evaluate
```

### 4️⃣ Start FastAPI Server

**Local Development:**
```bash
cd serving
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`
Interactive API docs: `http://localhost:8000/docs`

**Docker:**
```bash
# Build image
docker build -f serving/Dockerfile -t sentiment-api:latest .

# Run container
docker run -p 8000:8000 \
  -e MODEL_STAGE=staging \
  -e MLFLOW_TRACKING_USERNAME="your_username" \
  -e MLFLOW_TRACKING_PASSWORD="your_token" \
  -e MLFLOW_TRACKING_URI="https://dagshub.com/..." \
  sentiment-api:latest
```

### 5️⃣ Make Predictions

**Via cURL:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["This video is amazing!", "Terrible content", "It is okay"]
  }'
```

**Via Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "texts": ["Great movie!", "I did not like it"]
    }
)

print(response.json())
```

**Response Format:**
```json
{
  "predictions": [
    {
      "text": "This video is amazing!",
      "prediction": "POSITIVE",
      "confidence": 0.9876,
      "all_scores": {
        "POSITIVE": 0.9876,
        "NEUTRAL": 0.0098,
        "NEGATIVE": 0.0026
      }
    }
  ],
  "model_stage": "staging",
  "latency_ms": 45.23
}
```

---

## 📊 Pipeline Details

### Stage 1: Data Preprocessing (`pretrain.py`)
- Loads raw YouTube comments dataset
- Generates sentence embeddings using Sentence-Transformers (all-MiniLM-L6-v2)
- Splits into 80/20 train/test sets
- Outputs: train/test vectors (`.npy`) and labels
- Optional: Apply SMOTE for class balancing (imbalanced-learn)

**Parameters:**
```yaml
preprocess:
  embedding_model: all-MiniLM-L6-v2
  test_size: 0.2
  random_state: 42
```

### Stage 2a: SVM Baseline (`train_svm_noSMOTE.py`)
- Trains SVM on imbalanced data (no class balancing)
- Kernel: RBF
- Logs metrics: accuracy, precision, recall, F1, AUC
- Baseline for comparison

### Stage 2b: SVM with SMOTE (`train_svm.py`)
- Applies SMOTE to training data for class balancing
- Trains SVM on balanced dataset
- Better performance on minority classes
- Logs all metrics to MLflow

### Stage 2c: DistilBERT Fine-tuning (`train_bert.py`)
- Fine-tunes DistilBERT (distilbert-base-uncased)
- Batch size: 16, Max length: 128
- Learning rate: 2e-5, Epochs: 3
- Warmup ratio: 10%, Weight decay: 0.01
- Saved in HuggingFace format

### Stage 3: Evaluation & Model Selection (`evaluate.py`)
- Loads all three trained models
- Evaluates on test set
- Compares metrics (accuracy, F1 macro, precision, recall)
- Selects best model
- Registers to MLflow Model Registry
- Outputs: `model_comparison.csv`

---

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=serving --cov-report=html
```

Tests included:
- ✅ FastAPI endpoint tests (`test_app.py`)
- ✅ Configuration validation (`test_config.py`)
- ✅ Model loading from disk/S3/MLflow (`test_model_loader.py`)

---

## 📈 MLflow Experiment Tracking

View experiments and models:
```bash
# Start MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

Or use DagsHub directly:
```
https://dagshub.com/sairam030/mlops_sentimentAnalysis
```

**Tracked Metrics:**
- Accuracy, Precision, Recall, F1 (macro)
- Confusion matrix
- Training time
- Model size
- Hyperparameters

---

## 🔄 Model Promotion to S3

Promote the best model to S3 for production deployment:

```bash
python scripts/promote_model.py \
  --model-stage staging \
  --s3-bucket sentiment-mlops-models \
  --s3-prefix models
```

This script:
1. Loads best model from MLflow Registry
2. Uploads to S3 with versioning
3. Tags as `staging` or `production`
4. Updates model metadata

---

## 🌐 API Endpoints

### `GET /`
Interactive landing page with API documentation

### `GET /health`
Health check and model status
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_stage": "staging",
  "model_version": "v1",
  "model_source": "s3",
  "model_info": {
    "model_type": "DistilBERT",
    "accuracy": 0.89,
    "f1_macro": 0.87
  }
}
```

### `POST /predict`
Predict sentiment for one or more texts
- **Input:** List of strings (max 100 per request)
- **Output:** Predictions with confidence scores

### `POST /reload`
Hot-reload model without restarting the server

### `GET /metrics`
Request metrics (Prometheus-compatible format)

---

## 📝 Configuration

Edit `params.yaml` for pipeline parameters:

```yaml
# Preprocessing
preprocess:
  embedding_model: all-MiniLM-L6-v2
  test_size: 0.2
  random_state: 42

# SVM hyperparameters
train:
  svm:
    kernel: rbf
    C: 1.0
    gamma: scale
  
  # SMOTE for class balancing
  smote:
    random_state: 42
  
  # DistilBERT fine-tuning
  bert:
    base_model: distilbert-base-uncased
    epochs: 3
    learning_rate: 2e-5
    batch_size: 16
    max_length: 128
    weight_decay: 0.01
    warmup_ratio: 0.1

# MLflow configuration
mlflow:
  experiment_name: sentiment-classification
  tracking_uri: https://dagshub.com/sairam030/mlops_sentimentAnalysis.mlflow
  registered_model_name: sentiment-best-model
```

---

## 🔧 Advanced Usage

### Train with GPU
Export before running:
```bash
export CUDA_VISIBLE_DEVICES=0
dvc repro train_bert
```

### Skip Certain Stages
```bash
# Train only SVM, skip BERT
dvc repro --single-item train_svm_smote
```

### View Pipeline DAG
```bash
dvc dag
```

### Clear Cache
```bash
dvc cache remove --keep-remote
```

---

## 🚢 Deployment

### Docker Compose (Local)
```bash
docker-compose up -d
```

### AWS EC2
```bash
bash scripts/setup_ec2.sh
```

### Hugging Face Spaces
```bash
bash scripts/deploy_hf_space.sh
```

---

## 📚 Dataset

**Source:** YouTube Comments Dataset
- **Size:** ~15K comments
- **Labels:** Positive, Neutral, Negative
- **Location:** `data/raw/YoutubeCommentsDataSet.csv`

---

## 🤝 Contributing

1. Create a feature branch
2. Make changes
3. Run tests: `pytest tests/`
4. Format code: `ruff format src/ serving/ tests/`
5. Lint: `ruff check src/ serving/ tests/`
6. Submit PR

---

## 📄 License

This project is licensed under the MIT License — see LICENSE file for details.

---

## 👤 Author

**Sairam**
- GitHub: [@sairam030](https://github.com/sairam030)
- DagsHub: [@sairam030](https://dagshub.com/sairam030)



and i want to make a firefox and chrome extension for this project.. so that when that extention is added and then when the user is on a youtube video page, the extension can analyze the comments and show the sentiment distribution in a small popup or sidebar. The extension will use the FastAPI backend to get predictions for the comments.