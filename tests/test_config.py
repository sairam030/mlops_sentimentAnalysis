"""
test_config.py — Test serving configuration.
"""



def test_settings_defaults(monkeypatch):
    """Settings should pick up defaults when env vars are unset."""
    monkeypatch.delenv("MODEL_STAGE", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_PATH", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    # Re-import to pick up new env
    from serving.config import Settings

    s = Settings()
    assert s.model_stage == "staging"
    assert s.local_model_path == ""
    assert s.port == 8000


def test_settings_from_env(monkeypatch):
    """Settings should read from environment variables."""
    monkeypatch.setenv("MODEL_STAGE", "production")
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/my/model")
    monkeypatch.setenv("PORT", "7860")

    from serving.config import Settings

    s = Settings()
    assert s.model_stage == "production"
    assert s.local_model_path == "/my/model"
    assert s.port == 7860


def test_s3_model_path():
    """s3_model_path property should compose correctly."""
    from serving.config import Settings

    s = Settings()
    expected = f"s3://{s.s3_bucket}/{s.s3_prefix}/{s.model_stage}/"
    assert s.s3_model_path == expected


def test_validate_raises_without_credentials(monkeypatch):
    """validate() should raise when MLflow credentials are missing."""
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "")

    from serving.config import Settings

    import pytest as _pt

    s = Settings()
    with _pt.raises(EnvironmentError, match="Missing required"):
        s.validate()


def test_validate_passes_with_credentials(monkeypatch):
    """validate() should pass when credentials are set."""
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "pass")

    from serving.config import Settings

    s = Settings()
    s.validate()  # should not raise
