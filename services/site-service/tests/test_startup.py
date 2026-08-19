import pytest
from fastapi.testclient import TestClient
from shared import config

from site_service.main import app


def test_startup_fails_fast_when_object_storage_is_unconfigured(monkeypatch):
    """Otherwise the first upload request dies deep inside boto3 with
    "Invalid endpoint:" and no hint that the environment was never loaded."""
    monkeypatch.setattr(config, "S3_ENDPOINT_URL", "")

    with pytest.raises(RuntimeError) as exc:
        with TestClient(app):
            pass

    assert "S3_ENDPOINT_URL" in str(exc.value)
