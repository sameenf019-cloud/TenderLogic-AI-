# TenderLogic AI

**Enterprise Procurement Audit Platform — automated cross-examination of tender rejection notices against RFP/tender clauses, powered by fast open-weight LLMs on Groq.**

Public procurement rejections are often issued citing grounds that are vague, inconsistent, or not actually stated in the original tender document. TenderLogic AI automates the audit: it ingests a rejection notice and the original RFP/tender document, extracts the stated grounds for rejection, retrieves the most relevant clauses from the RFP, and asks an LLM to render a structured, evidence-backed verdict on whether the rejection is **valid** or **invalid** — with confidence score, key findings, and a final conclusion.

Built for a hackathon submission (HEC GenAI Cohort track) as a fast, low-cost, and offline-capable alternative to manual procurement-compliance review.

---

## Features

- **Smart document ingestion** — parses PDFs with PyMuPDF, and automatically falls back to Tesseract OCR only on scanned pages that have no extractable text layer (keeps ingestion fast).
- **Parallel extraction** — the rejection notice and the RFP document are read concurrently via a thread pool.
- **Content-hash caching** — previously processed documents are cached by file hash and skipped on re-runs.
- **Rejection-reason extraction** — pattern-based extraction of the stated grounds for rejection from the notice text.
- **Hybrid clause retrieval** — RFP text is chunked into clauses/sections and ranked against the rejection grounds using BM25.
- **AI verdict engine** — sends the grounds + matched clauses to Groq's fast open-weight models (`openai/gpt-oss-120b`, falling back to `openai/gpt-oss-20b`) and returns a structured JSON verdict: valid/invalid, confidence %, summary, key findings, and conclusion.
- **Offline heuristic fallback** — if no Groq API key is configured or the API call fails, a rule-based lexical-overlap heuristic still produces a (lower-confidence) verdict, so the tool never hard-fails.
- **Modern web app** — a React frontend (pre-built, served as static files) talks to a FastAPI backend for uploads, audit runs, and history.
- **Three ways to run it**: a one-click Google Colab notebook (with a public Cloudflare tunnel URL), a local FastAPI + React web app, or a plain CLI audit command.

## Architecture

```
Rejection Notice PDF ─┐
                       ├─▶ pipeline.py (OCR-aware extraction, parallel + cached)
RFP / Tender PDF ──────┘
                            │
                            ▼
              ai_engine.py — extract grounds → BM25 clause retrieval
                            │
                            ▼
                 Groq LLM verdict (JSON) or heuristic fallback
                            │
                            ▼
        backend/server.py (FastAPI) ──▶ frontend/dist (React UI)
```

## Tech Stack

| Layer | Tools |
|---|---|
| Document parsing | PyMuPDF, Tesseract OCR (`pytesseract`) |
| Retrieval | `rank_bm25` (BM25 keyword search over chunked RFP clauses) |
| LLM inference | Groq API — `openai/gpt-oss-120b` / `openai/gpt-oss-20b` |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React (pre-built static bundle) |
| Tunneling (Colab) | Cloudflare Tunnel (`cloudflared`) |

## Repository Structure

```
TenderLogic-AI/
├── main.py                    # Entry point: web app / CLI audit / legacy Gradio UI
├── pipeline.py                 # PDF ingestion, OCR fallback, caching
├── ai_engine.py                 # Rejection-reason extraction, BM25 retrieval, Groq verdicts
├── backend/
│   └── server.py               # FastAPI app serving the API + the built frontend
├── frontend/
│   └── dist/                   # Pre-built React app (served by the FastAPI backend)
├── notebooks/
│   └── TenderLogic_Colab.ipynb # One-click Google Colab launcher (Cloudflare tunnel)
├── requirements.txt
└── LICENSE
```

## Getting Started

### Option 1 — Google Colab (fastest, no setup)

1. Open `notebooks/TenderLogic_Colab.ipynb` in Google Colab.
2. (Optional) Add your `GROQ_API_KEY` in Colab **Secrets** (key icon in the left sidebar), or enter it when prompted.
3. **Runtime → Run all** (`Ctrl+F9`).
4. Wait for the final cell to print a public Cloudflare URL (`https://....trycloudflare.com`) and open it.

### Option 2 — Run locally

```bash
git clone https://github.com/<your-username>/TenderLogic-AI.git
cd TenderLogic-AI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# System dependency for OCR fallback (optional but recommended):
# Debian/Ubuntu: sudo apt-get install tesseract-ocr
# macOS:         brew install tesseract

# Set your Groq API key (get one free at https://console.groq.com)
export GROQ_API_KEY=gsk_your_key_here   # Windows: set GROQ_API_KEY=gsk_...

python main.py
# App runs at http://127.0.0.1:8000  (API docs at /docs)
```

### Option 3 — CLI audit

```bash
python main.py --cli --tender path/to/rejection_notice.pdf --rfp path/to/rfp.pdf
```

## Configuration

Create a `.env` file in the project root (already git-ignored) to avoid exporting the key every session:

```
GROQ_API_KEY=gsk_your_key_here
```

## Roadmap / Ideas

- Persist audit history to a database instead of in-memory storage
- Support multi-document RFP bundles (annexes, addenda)
- Add a confidence-calibration eval set for the heuristic fallback

## License

MIT — see [LICENSE](LICENSE).
