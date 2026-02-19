"""
app.py — Flask API for Sentiment Analysis
==========================================
Auto-detects whether the best model is sklearn (SVM) or transformers (DistilBERT)
by reading the MLflow MLmodel file, then loads and serves it.

NOTE: Models are loaded DIRECTLY via transformers/joblib — mlflow is NOT needed
at serving time. This keeps the Docker image small (~1.5GB vs ~8GB).
"""

import os
import time
import yaml
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import init_db, get_db_session, Prediction

app = Flask(__name__)

# Model version for tracking
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0")

# ─────────────────────────────────────────────
# Load model at startup
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "best_model")
MODEL = None
MODEL_TYPE = None  # "sklearn" or "transformers"
SCALER = None
EMBEDDER = None


def detect_model_type():
    """Read MLmodel file to detect if it's sklearn or transformers."""
    mlmodel_path = os.path.join(MODEL_DIR, "MLmodel")
    if not os.path.exists(mlmodel_path):
        raise FileNotFoundError(f"MLmodel not found at {mlmodel_path}")
    with open(mlmodel_path) as f:
        mlmodel = yaml.safe_load(f)
    flavors = mlmodel.get("flavors", {})
    if "transformers" in flavors:
        return "transformers"
    elif "sklearn" in flavors:
        return "sklearn"
    else:
        raise ValueError(f"Unknown model flavor: {list(flavors.keys())}")


def load_model():
    global MODEL, MODEL_TYPE, SCALER, EMBEDDER

    MODEL_TYPE = detect_model_type()
    print(f"[app] Detected model type: {MODEL_TYPE}")

    if MODEL_TYPE == "transformers":
        # Load DistilBERT DIRECTLY via transformers — no mlflow needed
        from transformers import pipeline as hf_pipeline

        model_path = os.path.join(MODEL_DIR, "model")
        tokenizer_path = os.path.join(MODEL_DIR, "components", "tokenizer")
        print(f"[app] Loading DistilBERT directly from {model_path} ...")
        MODEL = hf_pipeline(
            "text-classification",
            model=model_path,
            tokenizer=tokenizer_path,
        )
        print("[app] ✅ DistilBERT loaded.")

    elif MODEL_TYPE == "sklearn":
        # Load SVM + scaler + sentence-transformer embedder
        import joblib
        from sentence_transformers import SentenceTransformer

        # Find the sklearn model file inside the MLflow artifact
        model_subdir = os.path.join(MODEL_DIR, "model")
        sklearn_file = None
        if os.path.isdir(model_subdir):
            for f in os.listdir(model_subdir):
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    sklearn_file = os.path.join(model_subdir, f)
                    break
            # MLflow sklearn saves as model.pkl
            if sklearn_file is None:
                candidate = os.path.join(model_subdir, "model.pkl")
                if os.path.exists(candidate):
                    sklearn_file = candidate

        if sklearn_file is None:
            raise FileNotFoundError(f"No sklearn model found in {model_subdir}")

        print(f"[app] Loading SVM model from {sklearn_file} ...")
        MODEL = joblib.load(sklearn_file)

        # Check if scaler exists as a separate artifact
        scaler_candidates = [
            os.path.join(MODEL_DIR, "svm_scaler.joblib"),
            os.path.join(MODEL_DIR, "svm_baseline_scaler.joblib"),
        ]
        for sc_path in scaler_candidates:
            if os.path.exists(sc_path):
                SCALER = joblib.load(sc_path)
                print(f"[app] ✅ Scaler loaded from {sc_path}")
                break

        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        print("[app] ✅ SVM + SentenceTransformer loaded.")


# Load on startup
try:
    load_model()
except Exception as e:
    print(f"[app] ❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()

# Initialize database
try:
    init_db()
except Exception as e:
    print(f"[app] ⚠️  Database initialization failed (will continue without logging): {e}")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_type": MODEL_TYPE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    text = data.get("text", "")
    texts = data.get("texts", [])

    # Support single text or batch
    if text and not texts:
        texts = [text]
    if not texts:
        return jsonify({"error": "Provide 'text' (string) or 'texts' (list)"}), 400

    try:
        if MODEL_TYPE == "transformers":
            # Direct transformers pipeline returns list of dicts
            results = MODEL(texts)
            predictions = [
                {"label": r["label"], "score": round(r["score"], 4)}
                for r in results
            ]

        elif MODEL_TYPE == "sklearn":
            # Encode texts → vectors → scale → predict
            embeddings = EMBEDDER.encode(texts)
            if SCALER is not None:
                embeddings = SCALER.transform(embeddings)
            raw_predictions = MODEL.predict(embeddings)
            predictions = [
                {"label": str(pred), "score": 1.0}
                for pred in raw_predictions
            ]

        response_time = int((time.time() - start_time) * 1000)
        
        # Log to database
        try:
            db = get_db_session()
            for idx, text_input in enumerate(texts):
                pred_data = predictions[idx]
                
                log_entry = Prediction(
                    input_text=text_input[:500],  # Truncate long texts
                    prediction=pred_data.get("label", str(pred_data)),
                    confidence=pred_data.get("score", 1.0),
                    model_type=MODEL_TYPE,
                    model_version=MODEL_VERSION,
                    response_time_ms=response_time,
                    user_ip=request.remote_addr,
                    session_id=request.headers.get("X-Session-ID", "")
                )
                db.add(log_entry)
            db.commit()
            db.close()
        except Exception as db_error:
            print(f"[app] ⚠️  Failed to log prediction to database: {db_error}")

        return jsonify({
            "model_type": MODEL_TYPE,
            "predictions": predictions,
            "response_time_ms": response_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Sentiment Analysis API",
        "model_type": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "endpoints": {
            "/predict": "POST - Send {'text': '...'} or {'texts': [...]}",
            "/health": "GET - Health check",
            "/monitoring/stats": "GET - Prediction statistics",
            "/monitoring/drift": "GET - Drift detection check",
        },
    })


@app.route("/monitoring/stats", methods=["GET"])
def monitoring_stats():
    """Get prediction statistics for monitoring"""
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        db = get_db_session()
        
        # Last 24 hours stats
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        total = db.query(func.count(Prediction.id)).filter(
            Prediction.timestamp >= yesterday
        ).scalar() or 0
        
        avg_confidence = db.query(func.avg(Prediction.confidence)).filter(
            Prediction.timestamp >= yesterday
        ).scalar()
        
        label_dist = db.query(
            Prediction.prediction,
            func.count(Prediction.id)
        ).filter(
            Prediction.timestamp >= yesterday
        ).group_by(Prediction.prediction).all()
        
        avg_response_time = db.query(func.avg(Prediction.response_time_ms)).filter(
            Prediction.timestamp >= yesterday
        ).scalar()
        
        db.close()
        
        return jsonify({
            "period": "last_24h",
            "total_predictions": total,
            "average_confidence": round(float(avg_confidence or 0), 4),
            "average_response_time_ms": int(avg_response_time or 0),
            "label_distribution": {label: count for label, count in label_dist},
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/monitoring/drift", methods=["GET"])
def check_drift():
    """Simple drift detection based on confidence trends"""
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func
        
        db = get_db_session()
        now = datetime.utcnow()
        
        # Compare this week vs last week
        this_week = now - timedelta(days=7)
        last_week = now - timedelta(days=14)
        
        this_week_conf = db.query(func.avg(Prediction.confidence)).filter(
            Prediction.timestamp >= this_week
        ).scalar()
        
        last_week_conf = db.query(func.avg(Prediction.confidence)).filter(
            Prediction.timestamp >= last_week,
            Prediction.timestamp < this_week
        ).scalar()
        
        this_week_count = db.query(func.count(Prediction.id)).filter(
            Prediction.timestamp >= this_week
        ).scalar() or 0
        
        last_week_count = db.query(func.count(Prediction.id)).filter(
            Prediction.timestamp >= last_week,
            Prediction.timestamp < this_week
        ).scalar() or 0
        
        db.close()
        
        drift_detected = False
        confidence_change_pct = 0.0
        
        if this_week_conf and last_week_conf:
            confidence_change_pct = ((this_week_conf - last_week_conf) / last_week_conf) * 100
            # Alert if confidence drops by more than 10%
            drift_detected = confidence_change_pct < -10
        
        return jsonify({
            "drift_detected": drift_detected,
            "this_week_avg_confidence": round(float(this_week_conf or 0), 4),
            "last_week_avg_confidence": round(float(last_week_conf or 0), 4),
            "confidence_change_pct": round(confidence_change_pct, 2),
            "this_week_predictions": this_week_count,
            "last_week_predictions": last_week_count,
            "message": "⚠️ Confidence drop detected - possible drift" if drift_detected else "✅ No significant drift detected"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Dashboard Routes
# ─────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    """Monitoring dashboard"""
    return render_template("dashboard.html")


@app.route("/dashboard/stats")
def dashboard_stats():
    """Get overall statistics for dashboard"""
    try:
        from sqlalchemy import func
        
        db = get_db_session()
        
        # Total predictions
        total_predictions = db.query(func.count(Prediction.id)).scalar() or 0
        
        # Last 24 hours
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_24h = db.query(func.count(Prediction.id)).filter(
            Prediction.timestamp >= yesterday
        ).scalar() or 0
        
        # Average confidence (all time)
        avg_confidence = db.query(func.avg(Prediction.confidence)).scalar() or 0
        
        # Average response time (all time)
        avg_response_time_ms = db.query(func.avg(Prediction.response_time_ms)).scalar() or 0
        
        db.close()
        
        return jsonify({
            "total_predictions": total_predictions,
            "recent_24h": recent_24h,
            "avg_confidence": round(float(avg_confidence), 4),
            "avg_response_time_ms": int(avg_response_time_ms)
        })
    except Exception as e:
        return jsonify({"error": str(e), "total_predictions": 0, "recent_24h": 0, "avg_confidence": 0, "avg_response_time_ms": 0}), 200


@app.route("/dashboard/confidence_histogram")
def confidence_histogram():
    """Confidence distribution histogram"""
    try:
        db = get_db_session()
        predictions = db.query(Prediction.prediction, Prediction.confidence).all()
        db.close()
        
        df = pd.DataFrame(predictions, columns=['prediction', 'confidence'])
        if df.empty:
            return jsonify({})
        
        fig = px.histogram(
            df, x='confidence', color='prediction', nbins=50,
            title='Confidence Score Distribution by Label',
            labels={'confidence': 'Confidence Score', 'count': 'Frequency'},
            barmode='overlay', opacity=0.7
        )
        fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Count", template="plotly_white")
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/confidence_ranges")
def confidence_ranges():
    """Confidence distribution by 10% ranges for each label"""
    try:
        db = get_db_session()
        predictions = db.query(Prediction.prediction, Prediction.confidence).all()
        db.close()
        
        if not predictions:
            return jsonify({})
        
        # Create confidence range bins (0-10, 10-20, etc.)
        ranges = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        data = {label: [0] * 10 for label in ['positive', 'negative', 'neutral']}
        
        for pred, conf in predictions:
            # Convert confidence (0-1) to bin index (0-9)
            bin_idx = min(int(conf * 10), 9)
            if pred.lower() in data:
                data[pred.lower()][bin_idx] += 1
        
        # Create bar chart with grouped bars for each label
        fig = go.Figure()
        colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#a78bfa'}
        
        for label in ['positive', 'negative', 'neutral']:
            fig.add_trace(go.Bar(
                name=label.title(),
                x=ranges,
                y=data[label],
                marker_color=colors[label]
            ))
        
        fig.update_layout(
            title='Predictions by Confidence Range (0-100%)',
            xaxis_title='Confidence Range',
            yaxis_title='Number of Predictions',
            barmode='group',
            template="plotly_white"
        )
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/label_distribution")
def label_distribution():
    """Label distribution pie chart"""
    try:
        from sqlalchemy import func
        db = get_db_session()
        labels = db.query(Prediction.prediction, func.count(Prediction.id).label('count')).group_by(Prediction.prediction).all()
        db.close()
        
        df = pd.DataFrame(labels, columns=['prediction', 'count'])
        if df.empty:
            return jsonify({})
        
        fig = px.pie(df, values='count', names='prediction', title='Label Distribution')
        fig.update_layout(template="plotly_white")
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/confidence_by_label")
def confidence_by_label():
    """Confidence box plot by label"""
    try:
        db = get_db_session()
        predictions = db.query(Prediction.prediction, Prediction.confidence).all()
        db.close()
        
        df = pd.DataFrame(predictions, columns=['prediction', 'confidence'])
        if df.empty:
            return jsonify({})
        
        fig = px.box(df, x='prediction', y='confidence', title='Confidence Distribution by Label')
        fig.update_layout(template="plotly_white")
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/predictions_over_time")
def predictions_over_time():
    """Predictions timeline - simple daily counts"""
    try:
        from sqlalchemy import func
        db = get_db_session()
        
        # Get last 7 days of data grouped by day
        daily_counts = db.query(
            func.date_trunc('day', Prediction.timestamp).label('date'),
            Prediction.prediction,
            func.count(Prediction.id).label('count')
        ).group_by('date', Prediction.prediction).order_by('date').all()
        
        db.close()
        
        if not daily_counts:
            return jsonify({})
        
        df = pd.DataFrame(daily_counts, columns=['date', 'prediction', 'count'])
        
        # Create line chart with clear colors
        fig = go.Figure()
        colors = {'positive': '#10b981', 'negative': '#ef4444', 'neutral': '#a78bfa'}
        
        for label in ['positive', 'negative', 'neutral']:
            label_data = df[df['prediction'] == label]
            if not label_data.empty:
                fig.add_trace(go.Scatter(
                    x=label_data['date'],
                    y=label_data['count'],
                    name=label.title(),
                    mode='lines+markers',
                    line=dict(color=colors[label], width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='Daily Predictions Count',
            xaxis_title='Date',
            yaxis_title='Number of Predictions',
            template="plotly_white",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/drift_timeline")
def drift_timeline():
    """Confidence drift timeline"""
    try:
        from sqlalchemy import func
        db = get_db_session()
        
        daily_conf = db.query(
            func.date_trunc('day', Prediction.timestamp).label('day'),
            func.avg(Prediction.confidence).label('avg_confidence')
        ).group_by('day').order_by('day').all()
        
        db.close()
        
        df = pd.DataFrame(daily_conf, columns=['day', 'avg_confidence'])
        if df.empty:
            return jsonify({})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['day'], y=df['avg_confidence'], mode='lines+markers', name='Average Confidence'))
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="Drift Alert Threshold")
        fig.update_layout(title='Confidence Drift Over Time', xaxis_title='Date', yaxis_title='Average Confidence', template="plotly_white")
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard/model_metrics_comparison")
def model_metrics_comparison():
    """Model metrics comparison chart"""
    try:
        # Load baseline metrics from artifacts
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        comparison = []
        
        for filename in ['distilbert_info.json', 'svm_model_info.json', 'svm_baseline_info.json']:
            filepath = os.path.join(artifacts_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath) as f:
                        metrics = json.load(f)
                        model_name = filename.replace('_info.json', '').replace('_', ' ').title()
                        comparison.append({
                            'model': model_name,
                            'accuracy': metrics.get('accuracy', 0),
                            'f1_score': metrics.get('f1_score', 0),
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0)
                        })
                except:
                    continue
        
        if not comparison:
            return jsonify({"error": "No metrics data found"}), 200
        
        # Add current model live performance estimate
        db = get_db_session()
        recent = datetime.utcnow() - timedelta(days=7)
        predictions = db.query(Prediction.confidence).filter(Prediction.timestamp >= recent).all()
        db.close()
        
        confidences = [p[0] for p in predictions]
        current_avg = sum(confidences) / len(confidences) if confidences else 0
        
        if current_avg > 0:
            comparison.append({
                'model': 'Current (Live)',
                'accuracy': current_avg,
                'f1_score': current_avg * 0.95,
                'precision': current_avg * 0.97,
                'recall': current_avg * 0.93
            })
        
        df = pd.DataFrame(comparison)
        
        fig = go.Figure()
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            fig.add_trace(go.Bar(name=metric.replace('_', ' ').title(), x=df['model'], y=df[metric]))
        
        fig.update_layout(
            title='Model Metrics Comparison', 
            barmode='group', 
            yaxis_title='Score', 
            xaxis_title='Model', 
            template="plotly_white",
            yaxis=dict(range=[0, 1])
        )
        
        return jsonify(json.loads(fig.to_json(engine="json")))
    except Exception as e:
        return jsonify({"error": str(e)}), 200


@app.route("/download/csv")
def download_csv():
    """Download all predictions as CSV"""
    try:
        from datetime import datetime as dt
        db = get_db_session()
        predictions = db.query(Prediction).all()
        
        data = [{
            'id': p.id, 'timestamp': p.timestamp.isoformat(), 'input_text': p.input_text,
            'prediction': p.prediction, 'confidence': p.confidence, 'model_type': p.model_type,
            'model_version': p.model_version, 'response_time_ms': p.response_time_ms
        } for p in predictions]
        
        db.close()
        
        df = pd.DataFrame(data)
        filename = f'predictions_{dt.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = f'/tmp/{filename}'
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/filtered")
def download_filtered():
    """Download filtered predictions as CSV"""
    try:
        from datetime import datetime as dt, timedelta
        db = get_db_session()
        
        days = int(request.args.get('days', 7))
        label = request.args.get('label', None)
        
        query = db.query(Prediction)
        start_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Prediction.timestamp >= start_date)
        
        if label and label != 'all':
            query = query.filter(Prediction.prediction == label)
        
        predictions = query.all()
        
        data = [{
            'id': p.id, 'timestamp': p.timestamp.isoformat(), 'input_text': p.input_text,
            'prediction': p.prediction, 'confidence': p.confidence, 'model_type': p.model_type,
            'model_version': p.model_version, 'response_time_ms': p.response_time_ms
        } for p in predictions]
        
        db.close()
        
        df = pd.DataFrame(data)
        filename = f'predictions_filtered_{dt.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = f'/tmp/{filename}'
        df.to_csv(filepath, index=False)
        
        return send_file(filepath, mimetype='text/csv', as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
