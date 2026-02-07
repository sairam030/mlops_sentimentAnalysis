"""
config.py — Environment-based configuration for the serving app.
=================================================================
All settings are read from environment variables.
Defaults are provided for local development.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Model source ───────────────────────────────────────────
    model_stage: str = field(
        default_factory=lambda: os.getenv("MODEL_STAGE", "staging")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "sentiment-best-model")
    )

    # ── Local dev mode (skip S3/MLflow, load from disk) ──────
    local_model_path: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_PATH", "")
    )

    # ── S3 model cache ─────────────────────────────────────────
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("S3_MODEL_BUCKET", "sentiment-mlops-models")
    )
    s3_prefix: str = field(
        default_factory=lambda: os.getenv("S3_MODEL_PREFIX", "models")
    )
    local_model_dir: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_DIR", "/tmp/sentiment_model")
    )

    # ── MLflow (DagsHub) ───────────────────────────────────────
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            "https://dagshub.com/sairam030/mlops_sentimentAnalysis.mlflow",
        )
    )
    mlflow_tracking_username: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_USERNAME", "")
    )
    mlflow_tracking_password: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    )

    # ── Server ─────────────────────────────────────────────────
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def s3_model_path(self) -> str:
        """s3://bucket/prefix/staging/  or  s3://bucket/prefix/production/"""
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/{self.model_stage}/"

    def validate(self):
        """Raise if critical settings are missing."""
        errors = []
        if not self.mlflow_tracking_username:
            errors.append("MLFLOW_TRACKING_USERNAME is not set")
        if not self.mlflow_tracking_password:
            errors.append("MLFLOW_TRACKING_PASSWORD is not set")
        if errors:
            raise OSError(
                "Missing required environment variables:\n  - " + "\n  - ".join(errors)
            )


settings = Settings()
