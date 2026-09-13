"""
TenderLogic AI — FastAPI Backend Server
=========================================
Connects modern frontend to the Python RAG / Procurement Audit Engine.
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add project root to sys.path to ensure imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline import extract_documents_parallel, extract_text_smart, OCR_AVAILABLE
from ai_engine import (
    get_groq_client,
    extract_rejection_reason,
    retrieve_relevant_clauses,
    chunk_rfp_clauses,
    generate_verdict,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
)

app = FastAPI(
    title="TenderLogic AI API",
    description="Enterprise API for Automated Tender & Procurement Rejection Audits",
    version="2.0.0",
)

# Enable CORS for local development (e.g. Vite dev server on 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory audit session history
AUDIT_HISTORY: List[dict] = []

LOW_CONFIDENCE_THRESHOLD = 75


# ------------------------------------------------------------------
# Response Models
# ------------------------------------------------------------------
class ClauseMatch(BaseModel):
    label: str
    text: str
    relevance_rank: int


class AuditResponse(BaseModel):
    audit_id: str
    timestamp: str
    verdict: str  # "VALID" or "INVALID"
    is_valid: bool
    confidence: int
    ai_explanation: str
    summary: Optional[str] = ""
    key_findings: List[str] = []
    conclusion: Optional[str] = ""
    rejection_grounds: str
    matched_clauses_text: str
    matched_clauses_list: List[ClauseMatch]
    total_clauses_indexed: int
    lawyer_review_recommended: bool
    processing_time_seconds: float
    used_ai: bool
    engine_name: str
    pages_analyzed: int
    full_report: str
    tender_filename: str
    rfp_filename: str


# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
def _parse_clauses_list(matched_text: str) -> List[ClauseMatch]:
    """Parses formatted clause text into structured items for the UI cards."""
    items = []
    if not matched_text or "No clauses could be extracted" in matched_text:
        return items

    raw_blocks = matched_text.split("\n\n")
    rank = 1

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        if ":::" in block:
            label, body = block.split(":::", 1)
            label = label.strip()
            body = body.strip()
        else:
            lines = block.split("\n")
            label = lines[0].strip()
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else block

        items.append(
            ClauseMatch(
                label=label if label else f"Clause Finding #{rank}",
                text=body if body else "Please refer to the full tender document for this section.",
                relevance_rank=rank,
            )
        )
        rank += 1

    return items


def _perform_audit(tender_path: str, rfp_path: str, tender_name: str, rfp_name: str) -> AuditResponse:
    start_time = time.time()

    # Step 1: Document Ingestion (Parallel extraction)
    tender_doc, rfp_doc = extract_documents_parallel(tender_path, rfp_path)

    # Step 2: Extract Rejection Grounds
    try:
        reason_text = extract_rejection_reason(tender_doc.text)
    except Exception:
        reason_text = "Could not automatically extract an explicit rejection reason from this document."

    # Step 3: Retrieve Matching Clauses via BM25 Hybrid Retrieval
    try:
        matched_clauses, clause_count = retrieve_relevant_clauses(reason_text, rfp_doc.text, top_k=5)
    except Exception:
        matched_clauses, clause_count = "Could not retrieve matching clauses from the RFP.", 0

    # Step 4: AI Legal Compliance Evaluation
    is_valid, confidence, ai_explanation, key_findings, summary_text, conclusion_text, used_ai = generate_verdict(reason_text, matched_clauses)

    elapsed = round(time.time() - start_time, 2)
    verdict_label = "VALID" if is_valid else "INVALID"
    needs_lawyer = confidence < LOW_CONFIDENCE_THRESHOLD
    engine_name = f"Groq AI ({PRIMARY_MODEL})" if used_ai else "Rule-Based Legal Fallback"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    audit_id = str(uuid.uuid4())[:8]

    # Format structured findings into report
    findings_formatted = "\n".join([f"  [{i+1}] {f}" for i, f in enumerate(key_findings)]) if key_findings else f"  • {ai_explanation}"

    # Clean matched clauses for report presentation
    clauses_clean_lines = []
    for blk in matched_clauses.split("\n\n"):
        if ":::" in blk:
            lbl, bdy = blk.split(":::", 1)
            clauses_clean_lines.append(f"  • {lbl.strip()}:\n    {bdy.strip()}")
        else:
            clauses_clean_lines.append(f"  • {blk.strip()}")
    clauses_formatted = "\n\n".join(clauses_clean_lines)

    # Generate complete professional executive compliance report
    full_report = (
        f"================================================================================\n"
        f"                   TENDERLOGIC AI — EXECUTIVE AUDIT MEMORANDUM                  \n"
        f"================================================================================\n\n"
        f"AUDIT IDENTIFIER     : TL-{audit_id.upper()}\n"
        f"TIMESTAMP            : {timestamp_str}\n"
        f"DISQUALIFIED BID     : {tender_name}\n"
        f"GOVERNING TENDER/RFP : {rfp_name}\n\n"
        f"--------------------------------------------------------------------------------\n"
        f"I. EXECUTIVE DETERMINATION & VERDICT\n"
        f"--------------------------------------------------------------------------------\n"
        f"VERDICT              : {'VALID — REJECTION UPHELD' if is_valid else 'INVALID — REJECTION OVERTURNED'}\n"
        f"CONFIDENCE LEVEL     : {confidence}%\n"
        f"AUDIT ENGINE         : {engine_name}\n"
        f"LEGAL COUNSEL STATUS : {'Independent Legal Counsel Recommended (Confidence < 75%)' if needs_lawyer else 'Standard Verification (High Certainty)'}\n"
        f"PROCESSING TIME      : {elapsed} seconds\n\n"
        f"SUMMARY:\n"
        f"{summary_text or ai_explanation}\n\n"
        f"--------------------------------------------------------------------------------\n"
        f"II. AUDIT FINDINGS & CLAUSE VERIFICATION\n"
        f"--------------------------------------------------------------------------------\n"
        f"{findings_formatted}\n\n"
        f"--------------------------------------------------------------------------------\n"
        f"III. GOVERNING TENDER CLAUSES IDENTIFIED & REFERENCED\n"
        f"--------------------------------------------------------------------------------\n"
        f"{clauses_formatted}\n\n"
        f"--------------------------------------------------------------------------------\n"
        f"IV. CONCLUSION & ACTIONABLE RECOMMENDATION\n"
        f"--------------------------------------------------------------------------------\n"
        f"{conclusion_text or ('The disqualification is confirmed to be grounded in the mandatory provisions of the tender document.' if is_valid else 'The disqualification is not substantiated by the governing tender clauses and may be contested.')}\n\n"
        f"================================================================================\n"
        f"DISCLAIMER: This document is an automated procurement audit evaluation based on\n"
        f"computational clause cross-examination. It is intended to guide compliance decisions\n"
        f"and does not substitute for formal representation before judicial or regulatory bodies.\n"
        f"================================================================================\n"
    )

    structured_clauses = _parse_clauses_list(matched_clauses)

    result = AuditResponse(
        audit_id=f"TL-{audit_id.upper()}",
        timestamp=timestamp_str,
        verdict=verdict_label,
        is_valid=is_valid,
        confidence=confidence,
        ai_explanation=ai_explanation,
        summary=summary_text,
        key_findings=key_findings,
        conclusion=conclusion_text,
        rejection_grounds=reason_text,
        matched_clauses_text=matched_clauses,
        matched_clauses_list=structured_clauses,
        total_clauses_indexed=clause_count,
        lawyer_review_recommended=needs_lawyer,
        processing_time_seconds=elapsed,
        used_ai=used_ai,
        engine_name=engine_name,
        pages_analyzed=tender_doc.page_count + rfp_doc.page_count,
        full_report=full_report,
        tender_filename=tender_name,
        rfp_filename=rfp_name,
    )

    # Save to in-memory session history
    AUDIT_HISTORY.insert(0, result.dict())
    if len(AUDIT_HISTORY) > 20:
        AUDIT_HISTORY.pop()

    return result


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------
@app.get("/api/health")
def get_health():
    client = get_groq_client()
    return {
        "status": "healthy",
        "groq_connected": client is not None,
        "primary_model": PRIMARY_MODEL,
        "ocr_available": OCR_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }



@app.post("/api/audit", response_model=AuditResponse)
async def audit_uploaded_documents(
    tender_file: UploadFile = File(...),
    rfp_file: UploadFile = File(...),
):
    """Audits two uploaded procurement documents (PDF, DOCX, or TXT)."""
    allowed_extensions = {".pdf", ".docx", ".txt"}

    tender_ext = os.path.splitext(tender_file.filename or "")[1].lower()
    rfp_ext = os.path.splitext(rfp_file.filename or "")[1].lower()

    if tender_ext not in allowed_extensions or rfp_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Supported document formats are PDF, DOCX, and TXT.",
        )

    temp_dir = tempfile.mkdtemp(prefix="tenderlogic_upload_")
    tender_path = os.path.join(temp_dir, f"tender_{tender_file.filename}")
    rfp_path = os.path.join(temp_dir, f"rfp_{rfp_file.filename}")

    try:
        with open(tender_path, "wb") as f_out:
            shutil.copyfileobj(tender_file.file, f_out)
        with open(rfp_path, "wb") as f_out:
            shutil.copyfileobj(rfp_file.file, f_out)

        return _perform_audit(
            tender_path=tender_path,
            rfp_path=rfp_path,
            tender_name=tender_file.filename or "tender_notice.pdf",
            rfp_name=rfp_file.filename or "rfp_guidelines.pdf",
        )
    finally:
        # Clean up temporary upload files
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@app.get("/api/history")
def get_audit_history():
    """Returns the list of recent audits conducted in this session."""
    return AUDIT_HISTORY


# ------------------------------------------------------------------
# Mount Production Frontend (if built)
# ------------------------------------------------------------------
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")

if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react_app(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
