# -*- coding: utf-8 -*-
"""
RAG service core — Gymnasyum (diadaptasi dari percobaanrag1.py / PercobaanRAG1).
Menggunakan Groq API untuk generasi LLM (cepat, ringan, tanpa GPU).
"""

import glob
import math
import os
import re
import threading
import time

# ---------------------------------------------------------------------------
# Konfigurasi (bisa diubah lewat .env, lihat .env.example)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.environ.get("RAG_PDF_DIR", os.path.join(BASE_DIR, "pdf"))
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", os.path.join(BASE_DIR, "chroma_db"))
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")
LLM_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Ambang kondisi — CONTOH, perlu divalidasi Gymnasium (dari percobaan asli)
CFG = dict(
    lama_tidak_hadir_hari=14,
    sesi_terlewat_min=2,
    reschedule_min=2,
    sesi_hampir_habis_max=2,
    paket_hampir_berakhir_hari=7,
    pemakaian_lambat_selisih=30,
    toleransi_hari=2,
    toleransi_sesi=1,
)

_lock = threading.Lock()
_embedder = None
_groq_client = None
_chroma_col = None
_chunks_loaded = False


def _load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if device == "cuda":
            _embedder.half()
        _embedder.max_seq_length = 512
    return _embedder


def _embed(texts):
    emb = _load_embedder()
    return emb.encode(
        texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


HEADING = re.compile(r"^\d{1,2}(\.\d{1,2})?\.?\s+[A-Z]")


def _load_pdf_lines(path):
    import fitz

    lines = []
    for page in fitz.open(path):
        for ln in page.get_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("KB-0") or re.fullmatch(r"Halaman \d+", ln):
                continue
            lines.append(ln)
    return lines


def _chunk_pdf(path, max_chars=1200):
    fname = os.path.basename(path)
    doc_title = os.path.splitext(fname)[0].replace("_", " ")
    sections, cur_title, cur = [], "Pendahuluan", []
    for ln in _load_pdf_lines(path):
        if HEADING.match(ln):
            sections.append((cur_title, cur))
            cur_title, cur = ln, []
        else:
            cur.append(ln)
    sections.append((cur_title, cur))

    chunks = []
    for title, body in sections:
        if title == "Pendahuluan" or not body:
            continue
        parts, buf = [], ""
        for ln in body:
            if len(buf) + len(ln) > max_chars and buf:
                parts.append(buf)
                buf = ""
            buf += ln + "\n"
        if buf:
            parts.append(buf)
        for p in parts:
            chunks.append(
                {
                    "id": f"{fname}-{len(chunks)}",
                    "text": f"{doc_title} | {title}\n{p}".strip(),
                    "meta": {"source_pdf": fname, "section_title": title},
                }
            )
    return chunks


def _load_chroma():
    """Bangun / buka koleksi ChromaDB. Dipanggil sekali, di bawah lock."""
    global _chroma_col, _chunks_loaded
    if _chroma_col is not None:
        return _chroma_col
    import chromadb

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("gym_kb")
    except Exception:
        pass
    col = client.create_collection("gym_kb", metadata={"hnsw:space": "cosine"})

    all_chunks = []
    for path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        all_chunks += _chunk_pdf(path)
    if not all_chunks:
        raise RuntimeError(f"Tidak ada chunk dari PDF di {PDF_DIR}")
    col.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        embeddings=_embed([c["text"] for c in all_chunks]),
        metadatas=[c["meta"] for c in all_chunks],
    )
    _chroma_col = col
    _chunks_loaded = True
    return col


def retrieve(query, k=3):
    col = _load_chroma()
    res = col.query(query_embeddings=_embed([query]), n_results=k)
    return [
        {
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "dist": res["distances"][0][i],
        }
        for i in range(len(res["ids"][0]))
    ]


def _load_llm():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            raise RuntimeError("GROQ_API_KEY belum diisi di environment / .env")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate(messages, max_new_tokens=2000):
    client = _load_llm()
    model_name = os.environ.get("GROQ_MODEL") or LLM_MODEL
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name,
        temperature=0.3,
        max_tokens=max_new_tokens,
        top_p=0.9,
    )
    return chat_completion.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Analisis kondisi dari fitur customer (tidak diubah dari percobaan asli)
# ---------------------------------------------------------------------------
def _num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    x = float(x)
    return int(x) if x.is_integer() else x


def analyze(f):
    g = f.get
    tot, rem, att = (
        g("package_total_sessions"),
        g("remaining_package_sessions"),
        g("total_sessions_attended"),
    )
    dur, since, exp = (
        g("package_duration_days"),
        g("days_since_package_purchase"),
        g("days_until_package_expiry"),
    )
    used = tot - rem if tot is not None and rem is not None else None
    usage = round(100 * used / tot, 1) if used is not None and tot else None
    elapsed = round(100 * since / dur, 1) if since is not None and dur else None
    rc, rr = g("reminder_count_7d"), g("reminder_response_count_7d")
    rate = (
        "tidak tersedia"
        if rc is None or rr is None
        else "tidak ada reminder dalam 7 hari terakhir"
        if rc == 0
        else f"{rr} dari {rc} reminder direspons"
    )
    derived = dict(
        sessions_used=used,
        session_usage_pct=usage,
        time_elapsed_pct=elapsed,
        reminder_rate=rate,
    )

    c, last = [], g("days_since_last_visit")
    if g("has_ever_visited") == 0:
        c.append("belum_pernah_hadir")
    if last is not None and last >= CFG["lama_tidak_hadir_hari"]:
        c.append("lama_tidak_hadir")
    if g("has_ever_visited") == 1 and g("attendance_frequency_7d") == 0:
        c.append("tidak_hadir_7d")
    if (g("missed_session_count_7d") or 0) >= CFG["sesi_terlewat_min"]:
        c.append("sesi_terlewat")
    if (g("reschedule_frequency_7d") or 0) >= CFG["reschedule_min"]:
        c.append("reschedule_tinggi")
    if rc and rr == 0:
        c.append("reminder_tanpa_respons")
    if rem is not None and rem <= CFG["sesi_hampir_habis_max"]:
        c.append("sesi_hampir_habis")
    if exp is not None and 0 <= exp <= CFG["paket_hampir_berakhir_hari"]:
        c.append("paket_hampir_berakhir")
    if exp is not None and exp < 0:
        c.append("paket_sudah_berakhir")
    if (
        usage is not None
        and elapsed is not None
        and elapsed - usage >= CFG["pemakaian_lambat_selisih"]
    ):
        c.append("pemakaian_paket_lambat")

    notes = []
    missing = [
        k
        for k, v in f.items()
        if v is None and not (k == "days_since_last_visit" and g("has_ever_visited") == 0)
    ]
    if missing:
        notes.append("Data tidak tersedia: " + ", ".join(missing))
    if None not in (dur, since, exp) and abs((dur - since) - exp) > CFG["toleransi_hari"]:
        notes.append(
            f"days_until_package_expiry ({exp}) tidak sama dengan durasi paket "
            f"dikurangi hari sejak pembelian ({dur - since})"
        )
    if used is not None and att is not None and abs(used - att) > CFG["toleransi_sesi"]:
        notes.append(
            f"Sesi terpakai menurut sisa paket ({used}) berbeda dari "
            f"total_sessions_attended ({att}); penyebabnya tidak diketahui"
        )
    return derived, c, notes


COND_QUERY = {
    "belum_pernah_hadir": "follow-up customer baru yang belum pernah hadir sesi pertama",
    "lama_tidak_hadir": "follow-up customer yang lama tidak hadir latihan",
    "tidak_hadir_7d": "follow-up customer tidak hadir dalam 7 hari terakhir",
    "sesi_terlewat": "follow-up customer dengan sesi terlewat",
    "reschedule_tinggi": "follow-up customer yang sering reschedule",
    "reminder_tanpa_respons": "pendekatan customer tidak merespons reminder",
    "sesi_hampir_habis": "follow-up customer sesi hampir habis perpanjangan paket",
    "paket_hampir_berakhir": "follow-up customer paket hampir berakhir",
    "paket_sudah_berakhir": "follow-up customer paket sudah berakhir",
    "pemakaian_paket_lambat": "follow-up customer pemakaian paket lambat",
}


def retrieve_for(question, conds, per_query=2, max_total=6):
    queries = [question] + [COND_QUERY[c] for c in conds if c in COND_QUERY]
    seen, out = set(), []
    for q in queries:
        for r in retrieve(q, per_query):
            key = (r["meta"]["source_pdf"], r["meta"]["section_title"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out[:max_total]


SYSTEM_PROMPT = """Anda adalah AI Customer Retention Assistant untuk admin Gymnasium (gym dan private coaching).
Anda membantu ADMIN memahami kondisi customer dan menyusun draft pesan follow-up. Anda tidak berbicara langsung dengan customer dan tidak mengirim pesan apa pun.

SUMBER INFORMASI:
- [DATA_CUSTOMER] dan [TURUNAN]: fakta aktual tentang customer. Jadikan dasar utama.
- [HASIL_ML]: label klasifikasi dari model machine learning. Sampaikan apa adanya. JANGAN mengubah, membantah, atau menghitung ulang. JANGAN menyatakan customer pasti berhenti; label hanya indikasi.
- [KONDISI_TERDETEKSI] dan [CATATAN_DATA]: ringkasan deskriptif dari data, bukan klasifikasi.
- [REFERENSI_SOP]: panduan dari knowledge base. Ini rekomendasi awal, BUKAN kebijakan resmi Gymnasium.

ATURAN:
1. Gunakan hanya informasi dalam context. Jangan mengarang riwayat, penyebab, nama customer, nama coach, harga, promo, atau kebijakan.
2. customer_id hanya identitas, bukan indikator risiko.
3. Jelaskan kondisi customer berdasarkan nilai fitur dan sebutkan angkanya.
4. Jika sebuah nilai "tidak tersedia", katakan datanya tidak tersedia. Jangan menebak.
5. Fitur bertanda 7 hari hanya mencerminkan 7 hari terakhir.
6. Model tidak memberikan faktor penyebab. Jangan mengatakan "model menilai karena...". Gunakan "berdasarkan data, terlihat bahwa...".
7. Nama customer tidak tersedia: gunakan [Nama Customer] di draft pesan.
8. Draft pesan ramah, sopan, tidak memaksa, tidak menyalahkan. Jangan menyebut label ML, "At Risk", atau "berisiko berhenti" di dalam pesan.
9. Gunakan bahasa Indonesia. Semua draft adalah usulan yang harus diperiksa admin.
10. Jika admin bertanya hal yang tidak ada di context, jawab bahwa informasinya belum tersedia."""

FORMAT = """Jawab dengan format berikut:
A. Customer Information (customer_id, klasifikasi ML sebagai indikasi, ringkasan kondisi 2-3 kalimat)
B. Risk Indicators (tabel: indikator, nilai, makna singkat; hanya fitur yang relevan)
C. Recommended Approach (strategi, alasan terkait angka di B, referensi PDF dan bagian)
D. WhatsApp Follow-Up Draft (Greeting, Isi pesan, Closing)
E. Alternative Message (versi lebih formal dan versi lebih santai)
F. Keterbatasan Data (hal yang tidak tersedia atau tidak bisa disimpulkan)"""


def _v(x, s=""):
    return "tidak tersedia" if x is None else f"{x}{s}"


def build_customer_context(customer_id, label, features, derived, conds, notes, customer_name=None):
    f = features
    name_line = f"Nama: {customer_name}\n" if customer_name else ""
    return f"""[DATA_CUSTOMER] (fakta aktual dari database)
customer_id: {customer_id} (hanya identitas)
{name_line}Paket: {_v(f.get('package_total_sessions'))} sesi, durasi {_v(f.get('package_duration_days'))} hari
Hari sejak pembelian paket: {_v(f.get('days_since_package_purchase'))}
Pernah hadir: {'ya' if f.get('has_ever_visited')==1 else 'belum pernah' if f.get('has_ever_visited')==0 else 'tidak tersedia'}
Total sesi dihadiri (paket ini): {_v(f.get('total_sessions_attended'))}
Kehadiran 7 hari terakhir: {_v(f.get('attendance_frequency_7d'))}
Sesi terlewat 7 hari terakhir: {_v(f.get('missed_session_count_7d'))}
Reschedule 7 hari terakhir: {_v(f.get('reschedule_frequency_7d'))}
Hari sejak kunjungan terakhir: {_v(f.get('days_since_last_visit'))}
Sisa sesi paket: {_v(f.get('remaining_package_sessions'))}
Hari menuju kedaluwarsa paket: {_v(f.get('days_until_package_expiry'))}
Reminder 7 hari terakhir: {_v(f.get('reminder_count_7d'))}
Reminder direspons 7 hari terakhir: {_v(f.get('reminder_response_count_7d'))}

[TURUNAN] (dihitung kode)
Sesi terpakai: {_v(derived['sessions_used'])} ({_v(derived['session_usage_pct'],'%')} dari paket)
Waktu paket berlalu: {_v(derived['time_elapsed_pct'],'%')}
Respons reminder: {derived['reminder_rate']}

[KONDISI_TERDETEKSI] {', '.join(conds) or 'tidak ada kondisi khusus'}
[CATATAN_DATA] {'; '.join(notes) or 'tidak ada'}

[HASIL_ML] (indikasi, bukan fakta mutlak)
Klasifikasi: {label or 'tidak tersedia'}
Probabilitas dan faktor dari model: tidak tersedia (model hanya mengeluarkan label)
"""


def build_sop_context(chunks):
    return "[REFERENSI_SOP] (rekomendasi awal, belum divalidasi Gymnasium)\n" + "\n\n".join(
        f"({i+1}) Sumber: {c['meta']['source_pdf']} | {c['meta']['section_title']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )


def ask_for_customer(customer_id, label, features, question=None, customer_name=None):
    t0 = time.time()
    q = question or f"Buatkan analisis dan draft follow-up untuk customer {customer_id}."
    features = {k: _num(v) for k, v in (features or {}).items()}
    derived, conds, notes = analyze(features)
    chunks = retrieve_for(q, conds)
    user = (
        build_customer_context(customer_id, label, features, derived, conds, notes, customer_name)
        + "\n"
        + build_sop_context(chunks)
        + f"\n\n[PERTANYAAN_ADMIN]\n{q}\n\n{FORMAT}"
    )
    answer = generate(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    )
    return {
        "customer_id": customer_id,
        "label": label,
        "conditions": conds,
        "notes": notes,
        "derived": derived,
        "sources": [
            {"source_pdf": c["meta"]["source_pdf"], "section_title": c["meta"]["section_title"]}
            for c in chunks
        ],
        "summary": answer,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def ask_general(question):
    t0 = time.time()
    chunks = retrieve(question, 4)
    user = (
        build_sop_context(chunks)
        + f"\n\n[PERTANYAAN_ADMIN]\n{question}\n\n"
        "Jawab ringkas berdasarkan REFERENSI_SOP. Jika referensi tidak memuat jawabannya, katakan belum tersedia."
    )
    answer = generate(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    )
    return {
        "customer_id": None,
        "label": None,
        "conditions": [],
        "notes": [],
        "derived": {},
        "sources": [
            {"source_pdf": c["meta"]["source_pdf"], "section_title": c["meta"]["section_title"]}
            for c in chunks
        ],
        "summary": answer,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def model_status():
    return {
        "chunks_loaded": _chunks_loaded,
        "embedder_loaded": _embedder is not None,
        "llm_loaded": _groq_client is not None,
        "pdf_dir": PDF_DIR,
        "embed_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
    }