---
title: Sentiment Analysis API
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
tags:
  - sentiment-analysis
  - distilbert
  - fastapi
  - mlops
short_description: Sentiment prediction for YouTube comments
---

# 🎭 Sentiment Analysis API

A production-ready FastAPI inference server for YouTube comment sentiment analysis.

## Model

**DistilBERT** fine-tuned on YouTube comments — 3-class classification:
- 🟢 **Positive**
- 🟡 **Neutral**
- 🔴 **Negative**

| Metric | Score |
|--------|-------|
| Accuracy | 85.5% |
| F1 Macro | 80.9% |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check + model info |
| `/predict` | POST | Predict sentiment |
| `/metrics` | GET | Request metrics |

## Usage

```bash
curl -X POST https://sairam030-sentiment-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This video is amazing!", "Terrible content"]}'
```

## Built With

- [DistilBERT](https://huggingface.co/distilbert-base-uncased) — Base model
- [FastAPI](https://fastapi.tiangolo.com/) — Inference server
- [MLflow](https://mlflow.org/) — Experiment tracking
- [DVC](https://dvc.org/) — Pipeline management
