import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """TestClient без будь-яких моків — застосунок як є."""
    return TestClient(app)
