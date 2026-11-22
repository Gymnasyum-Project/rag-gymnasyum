# Tes ringan — TIDAK memuat model (tidak butuh GPU).
# Jalankan: pytest -v
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["service"] == "rag-gymnasyum"


def test_generate_summary_rejects_empty_features():
    r = client.post("/generate-summary", json={"customer_id": "CUST00053", "features": {}})
    assert r.status_code == 422


def test_generate_summary_rejects_missing_customer_id():
    r = client.post("/generate-summary", json={"features": {"attendance_frequency_7d": 1}})
    assert r.status_code == 422
