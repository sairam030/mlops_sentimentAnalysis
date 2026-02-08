"""
app.py — FastAPI inference server for Sentiment Analysis.
==========================================================
Endpoints:
    GET  /health       — Health check + model info
    POST /predict      — Predict sentiment for one or more texts
    POST /reload       — Hot-reload model without restarting the container
    GET  /metrics      — Basic request metrics (Prometheus-compatible)
"""

import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from serving.config import settings
from serving.logger import get_logger
from serving.model_loader import model_loader

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Request/Response schemas
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=[["This video is amazing!", "Terrible content", "It's okay I guess"]],
        description="List of texts to classify (max 100 per request)",
    )


class PredictionItem(BaseModel):
    text: str
    prediction: str
    confidence: float
    all_scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model_stage: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_stage: str
    model_version: str | None
    model_source: str | None
    model_info: dict | None
    timestamp: str


# ─────────────────────────────────────────────
# Metrics tracking (simple in-memory)
# ─────────────────────────────────────────────
class Metrics:
    def __init__(self):
        self.total_requests: int = 0
        self.total_predictions: int = 0
        self.total_errors: int = 0
        self.total_latency_ms: float = 0.0
        self.started_at: str = datetime.now(UTC).isoformat()

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_requests, 1)


metrics = Metrics()


# ─────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info(f"Starting up — MODEL_STAGE={settings.model_stage}")
    try:
        model_loader.load()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        logger.warning("Server starting without a loaded model. Use POST /reload to retry.")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive/neutral/negative) for YouTube comments",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """Interactive landing page for the Sentiment Analysis API."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🎭 Sentiment Analysis API</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem 1rem}
h1{font-size:2rem;margin-bottom:.3rem;background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:2rem;font-size:.95rem}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:1.5rem;width:100%;max-width:640px;margin-bottom:1.5rem}
textarea{width:100%;min-height:100px;background:#12122a;border:1px solid #333;border-radius:8px;color:#e0e0e0;padding:.75rem;font-size:.95rem;resize:vertical;font-family:inherit}
textarea:focus{outline:none;border-color:#6366f1}
button{background:linear-gradient(135deg,#6366f1,#a855f7);color:#fff;border:none;border-radius:8px;padding:.65rem 1.5rem;font-size:1rem;cursor:pointer;margin-top:.75rem;transition:opacity .2s}
button:hover{opacity:.88}
button:disabled{opacity:.5;cursor:not-allowed}
.results{margin-top:1rem}
.result-item{background:#12122a;border-radius:8px;padding:1rem;margin-bottom:.75rem;border-left:4px solid #444}
.result-item.positive{border-left-color:#22c55e}
.result-item.negative{border-left-color:#ef4444}
.result-item.neutral{border-left-color:#eab308}
.label{font-weight:700;font-size:1.05rem;text-transform:uppercase;letter-spacing:.5px}
.positive .label{color:#22c55e}
.negative .label{color:#ef4444}
.neutral .label{color:#eab308}
.text-preview{color:#aaa;font-size:.85rem;margin-top:.3rem;font-style:italic}
.confidence{color:#bbb;font-size:.85rem;margin-top:.25rem}
.bar-bg{background:#222;border-radius:4px;height:6px;margin-top:.35rem;overflow:hidden}
.bar{height:100%;border-radius:4px;transition:width .5s}
.latency{color:#666;font-size:.8rem;text-align:right;margin-top:.5rem}
.info{display:flex;gap:1.5rem;flex-wrap:wrap;justify-content:center;margin-bottom:1.5rem;font-size:.85rem;color:#888}
.info a{color:#6366f1;text-decoration:none}
.info a:hover{text-decoration:underline}
.error{color:#ef4444;margin-top:.75rem}
.hint{color:#666;font-size:.8rem;margin-top:.4rem}
</style>
</head>
<body>
<h1>🎭 Sentiment Analysis API</h1>
<p class="subtitle">DistilBERT fine-tuned on YouTube comments &mdash; positive · neutral · negative</p>
<div class="info">
  <span>📖 <a href="/docs">API Docs</a></span>
  <span>❤️ <a href="/health">Health Check</a></span>
  <span>📊 <a href="/metrics">Metrics</a></span>
</div>
<div class="card">
  <textarea id="input" placeholder="Type or paste text here...&#10;&#10;Tip: put each comment on a new line to analyze multiple at once.">This video is amazing! Best tutorial ever.&#10;Terrible quality, total waste of time.&#10;It was okay, nothing special.</textarea>
  <button id="btn" onclick="analyze()">Analyze Sentiment</button>
  <p class="hint">POST /predict — send {"texts": ["..."]} for programmatic access</p>
  <div id="results" class="results"></div>
</div>
<script>
async function analyze(){
  const btn=document.getElementById('btn'),box=document.getElementById('results'),
        lines=document.getElementById('input').value.split('\\n').map(s=>s.trim()).filter(Boolean);
  if(!lines.length){box.innerHTML='<p class="error">Enter at least one line of text.</p>';return}
  btn.disabled=true;btn.textContent='Analyzing...';box.innerHTML='';
  try{
    const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texts:lines})});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||r.statusText)}
    const d=await r.json();
    box.innerHTML=d.predictions.map(p=>{
      const c=(p.confidence*100).toFixed(1),cls=p.prediction.toLowerCase(),
            colors={positive:'#22c55e',negative:'#ef4444',neutral:'#eab308'};
      return `<div class="result-item ${cls}">
        <span class="label">${p.prediction} &nbsp;${cls==='positive'?'😊':cls==='negative'?'😠':'😐'}</span>
        <span class="confidence">${c}% confidence</span>
        <div class="bar-bg"><div class="bar" style="width:${c}%;background:${colors[cls]||'#888'}"></div></div>
        <p class="text-preview">"${p.text}"</p>
      </div>`}).join('')+`<p class="latency">⚡ ${d.latency_ms.toFixed(0)}ms</p>`;
  }catch(e){box.innerHTML=`<p class="error">Error: ${e.message}</p>`}
  btn.disabled=false;btn.textContent='Analyze Sentiment';
}
document.getElementById('input').addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')analyze()});
</script>
</body>
</html>"""


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns model status and metadata."""
    status_info = model_loader.status()
    return HealthResponse(
        status="healthy" if status_info["loaded"] else "degraded",
        model_loaded=status_info["loaded"],
        model_stage=status_info["model_stage"],
        model_version=status_info["model_version"],
        model_source=status_info["source"],
        model_info=status_info["model_info"],
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for one or more texts."""
    if not model_loader.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Send POST /reload to load the model.",
        )

    metrics.total_requests += 1
    start = time.time()

    try:
        predictions = model_loader.predictor.predict(request.texts)
        latency_ms = (time.time() - start) * 1000

        metrics.total_predictions += len(predictions)
        metrics.total_latency_ms += latency_ms

        logger.info(
            f"Predicted {len(predictions)} texts in {latency_ms:.1f}ms "
            f"(avg {latency_ms / len(predictions):.1f}ms/text)"
        )

        return PredictResponse(
            predictions=[PredictionItem(**p) for p in predictions],
            model_stage=settings.model_stage,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        metrics.total_errors += 1
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") from e


@app.post("/reload")
async def reload_model():
    """Hot-reload the model without restarting the container."""
    logger.info(f"Reloading model for stage='{settings.model_stage}'...")
    try:
        model_loader.load()
        status_info = model_loader.status()
        logger.info(f"Model reloaded: {status_info}")
        return {
            "status": "reloaded",
            "model_info": status_info,
        }
    except Exception as e:
        logger.error(f"Reload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}") from e


@app.get("/metrics")
async def get_metrics():
    """Basic request metrics."""
    return {
        "total_requests": metrics.total_requests,
        "total_predictions": metrics.total_predictions,
        "total_errors": metrics.total_errors,
        "avg_latency_ms": round(metrics.avg_latency_ms, 2),
        "total_latency_ms": round(metrics.total_latency_ms, 2),
        "started_at": metrics.started_at,
        "model_status": model_loader.status(),
    }


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serving.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
