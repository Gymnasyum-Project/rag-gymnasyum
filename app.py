# -*- coding: utf-8 -*-
"""
RAG service — Gymnasyum. HANYA bagian RAG.

Jalankan:
    pip install -r requirements.txt
    cp .env.example .env        # isi RAG_API_KEY kalau mau pakai kunci
    uvicorn app:app --host 0.0.0.0 --port 8001

Endpoint:
    GET  /health            -> {"status": "ok", ...} (tidak memuat model)
    POST /generate-summary  -> dipakai backend Express (llm.client.js)
    POST /followup          -> alias, dipakai rag.client.js (RAG_URL/followup)

Contoh request POST /generate-summary (dari backend):
    {
      "customer_id": "CUST00053",
      "customer_name": "Rina Susanti",          // opsional
      "label": "At Risk",                        // label hasil /predict
      "predicted_label": "At Risk",              // alias, boleh salah satu
      "features": {
        "attendance_frequency_7d": 1,
        "missed_session_count_7d": 1,
        "reschedule_frequency_7d": 1,
        "remaining_package_sessions": 10,
        "days_until_package_expiry": 7,
        "days_since_last_visit": 4,
        "reminder_count_7d": 1,
        "reminder_response_count_7d": 0,
        "package_total_sessions": 24,           // opsional, kalau ada
        "package_duration_days": 90,            // opsional
        "days_since_package_purchase": 83,      // opsional
        "total_sessions_attended": 14,          // opsional
        "has_ever_visited": 1                   // opsional
      },
      "question": "Buatkan draft follow-up."    // opsional
    }

Respons (200):
    {
      "customer_id": "CUST00053",
      "label": "At Risk",
      "summary": "A. Customer Information\n...",
      "ringkasan": "A. Customer Information\n...",   // alias, sama isinya
      "whatsapp_draft": "A. Customer Information\n...", // alias, sama isinya
      "conditions": ["paket_hampir_berakhir", "reminder_tanpa_respons"],
      "notes": [],
      "sources": [{"source_pdf": "03_...", "section_title": "..."}],
      "elapsed_seconds": 48.1
    }

Catatan:
- "summary" di sini adalah teks lengkap format A-F dari LLM. Backend yang
  memotong bagian D (draf WhatsApp) kalau perlu menampilkannya terpisah.
  Layanan ini sengaja TIDAK mencoba memotong draf dengan regex rapuh.
- Auth: kalau env RAG_API_KEY diisi, setiap request harus membawa header
  "x-api-key: <kunci yang sama>". Kalau env kosong, tidak ada auth
  (hanya untuk tes lokal — jangan begitu di server publik).
- Model dimuat malas (lazy): /health tidak memuat model, request pertama
  yang butuh LLM akan memuat embedding + Chroma + Qwen dan bisa makan
  waktu menit pertama. Di GPU T4, 1 jawaban ~48 detik (percobaan asli).
"""

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import rag_service

app = FastAPI(title="Gymnasyum RAG Service", version="1.0.0")

API_KEY = os.environ.get("RAG_API_KEY", "").strip()


class GenerateSummaryRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    customer_name: str | None = None
    label: str | None = None
    predicted_label: str | None = None
    segmen: str | None = None
    features: dict = Field(default_factory=dict)
    question: str | None = None


def _check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="x-api-key tidak cocok")


def _label_of(req: GenerateSummaryRequest):
    return req.label or req.predicted_label or req.segmen


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-gymnasyum", **rag_service.model_status()}


@app.get("/api/health")
def health_alias():
    return health()


@app.post("/generate-summary")
def generate_summary(req: GenerateSummaryRequest, x_api_key: str | None = Header(default=None)):
    _check_api_key(x_api_key)
    if not req.features:
        raise HTTPException(status_code=422, detail="features wajib diisi (dict, boleh 8 fitur prediksi)")
    try:
        result = rag_service.ask_for_customer(
            customer_id=req.customer_id,
            label=_label_of(req),
            features=req.features,
            question=req.question,
            customer_name=req.customer_name,
        )
    except Exception as e:  # noqa: BLE001 - sampaikan error apa adanya, jangan ditelan
        raise HTTPException(status_code=500, detail=f"RAG gagal: {e}") from e
    result["ringkasan"] = result["summary"]
    result["whatsapp_draft"] = result["summary"]
    return result


@app.post("/followup")
def followup(req: GenerateSummaryRequest, x_api_key: str | None = Header(default=None)):
    """Alias untuk rag.client.js yang memanggil RAG_URL/followup."""
    return generate_summary(req, x_api_key=x_api_key)


@app.post("/api/generate-summary")
def generate_summary_alias(req: GenerateSummaryRequest, x_api_key: str | None = Header(default=None)):
    return generate_summary(req, x_api_key=x_api_key)
