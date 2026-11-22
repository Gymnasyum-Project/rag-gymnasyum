# RAG Service — Gymnasyum (HANYA RAG)

Layanan RAG untuk ringkasan AI + draf follow-up WhatsApp. Diadaptasi dari
`percobaanrag1.py` (PercobaanRAG1). **Zip ini hanya berisi RAG** — tidak ada
kode prediksi (`.joblib`), tidak ada backend Express, tidak ada frontend.
File lain (backend, frontend) diubah terpisah kalau memang butuh.

## Isi zip

```
rag-service/
├── app.py               # FastAPI: /health, POST /generate-summary, POST /followup
├── rag_service.py       # Inti RAG: chunk PDF, embedding, ChromaDB, analyze, LLM
├── requirements.txt
├── .env.example         # salin ke .env
├── pdf/                 # 5 PDF knowledge base (KB-01 s.d. KB-05, 19 halaman)
├── tests/test_api.py    # tes ringan, tanpa GPU
└── README.md            # file ini
```

## Cara jalan (lokal / server GPU)

```bash
pip install -r requirements.txt
cp .env.example .env      # isi RAG_API_KEY kalau mau pakai kunci
uvicorn app:app --host 0.0.0.0 --port 8001
```

Cek: `curl http://localhost:8001/health` → `{"status": "ok", ...}`.
Request pertama yang butuh LLM akan memuat model (menit pertama) — itu normal.

Tes ringan (tanpa GPU, tanpa model):

```bash
pip install pytest httpx
pytest -v
```

## Cara menyambung dari backend Express

Backend sudah punya jalurnya — tinggal arahkan ke layanan ini:

1. Di backend, isi `.env`:
   ```
   RAG_URL=http://<host-layanan-ini>:8001
   RAG_API_KEY=<isi yang sama dengan .env layanan ini>
   ```
   Kalau layanan ini jalan di server yang sama dengan backend, host-nya
   `http://localhost:8001`.
2. Di `aiSummary.service.js`, ganti pemanggilan dari `llm.client.js`
   (`LLM_SERVICE_URL/generate-summary`) ke `rag.client.js`
   (`RAG_URL/followup`). `rag.client.js` sudah ada di backend — yang
   diganti hanya `require`-nya dan isi env di atas. Frontend tidak diubah:
   frontend tetap `POST /ai/summary/:id` ke backend, seperti sekarang.
3. Backend HARUS mengirim data customer asli di setiap request
   (lihat contoh di `app.py`). Layanan ini tidak punya database customer
   dan tidak membaca database backend.

Contoh request:

```bash
curl -X POST http://localhost:8001/generate-summary \
  -H "Content-Type: application/json" \
  -H "x-api-key: <kunci>" \
  -d '{"customer_id": "CUST00053", "label": "At Risk",
       "features": {"attendance_frequency_7d": 1, "missed_session_count_7d": 1,
       "reschedule_frequency_7d": 1, "remaining_package_sessions": 10,
       "days_until_package_expiry": 7, "days_since_last_visit": 4,
       "reminder_count_7d": 1, "reminder_response_count_7d": 0}}'
```

Respons berisi `summary` (teks lengkap format A–F), `conditions`,
`sources` (PDF + bagian), dan `elapsed_seconds`. Field `ringkasan` dan
`whatsapp_draft` adalah alias dari `summary` yang sama — backend yang
memotong bagian D kalau perlu menampilkan draf saja.

## Yang berubah dari percobaan asli

- **Data dummy dihapus.** Percobaan asli punya 6 baris customer contoh
  (C-0001 s.d. C-0006) di dalam file. Di sini tidak ada — data customer
  datang dari request backend.
- **Bukan lagi script Colab.** Path `/content/...`, `!pip install`, dan
  `print` percobaan dihapus. PDF dibaca dari folder `pdf/` di samping
  `app.py`, ChromaDB disimpan di `chroma_db/`.
- **Model dimuat malas (lazy).** `/health` tidak memuat model, supaya
  health check dan tes validasi tetap jalan tanpa GPU.
- **Dua endpoint.** `POST /generate-summary` (untuk `llm.client.js`) dan
  `POST /followup` (untuk `rag.client.js`, `RAG_URL/followup`).
- Logika `analyze()`, `CFG`, `SYSTEM_PROMPT`, dan format A–F **tidak diubah**
  dari percobaan asli.

## Batasan — baca sebelum presentasi / deploy

- 5 PDF di `pdf/` tertulis **"REKOMENDASI AWAL — bukan kebijakan resmi
  Gymnasium, perlu divalidasi"**. Angka batas waktu/frekuensi di dalamnya
  adalah contoh awal. Begitu juga `CFG` di `rag_service.py`.
- 1 jawaban di percobaan asli butuh **48,1 detik** (GPU T4, Qwen 3B 4-bit,
  3,2 GB VRAM). Backend lama timeout 15 detik — pastikan timeout backend
  untuk `/generate-summary` cukup panjang (backend yang sekarang 180 detik),
  dan frontend menampilkan loading, jangan kirim ulang berkali-kali.
- Hanya 1 dari 8 pemanggilan `ask()` di percobaan asli yang outputnya
  tersimpan. Tes ujung-ke-ujung dengan data asli dari backend belum pernah
  dilakukan — lakukan itu sebelum dipakai presentasi.
- Error 422 yang muncul di frontend waktu "Generate Ulang" kemarin
  (Detail Customer Rina Susanti) kemungkinan karena payload backend tidak
  cocok dengan skema layanan yang dituju. Setelah layanan ini jalan, tes
  Generate Ulang lagi dan cek log backend kalau masih 422.
