"""
TenderLogic AI — Enterprise Application Entry Point
====================================================
Automated Tender & Procurement Rejection Audit System

Usage:
  python main.py                         -> Launches FastAPI server & Modern React Frontend at http://127.0.0.1:8000
  python main.py --cli -t <rej> -r <rfp> -> Runs an automated audit on real documents in terminal
  python main.py --gradio                -> Launches the legacy Gradio UI at http://127.0.0.1:7860
"""

import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure helper modules are importable
from pipeline import extract_documents_parallel
from ai_engine import (
    get_groq_client,
    extract_rejection_reason,
    retrieve_relevant_clauses,
    generate_verdict,
)


def run_cli_audit(tender_path: str, rfp_path: str):
    """Runs a complete audit in the terminal using real user documents."""
    print("\n" + "=" * 60)
    print("  TENDERLOGIC AI - COMMAND LINE AUDIT")
    print("=" * 60)

    if not tender_path or not rfp_path:
        print("\n[ERROR] Both --tender and --rfp document paths are required for CLI audit.")
        print("Example:")
        print("  python main.py --cli --tender path/to/rejection.pdf --rfp path/to/rfp.pdf\n")
        sys.exit(1)

    if not os.path.exists(tender_path):
        print(f"\n[ERROR] Rejection document not found at: {tender_path}")
        sys.exit(1)

    if not os.path.exists(rfp_path):
        print(f"\n[ERROR] RFP document not found at: {rfp_path}")
        sys.exit(1)

    print("\n[1/3] Ingesting documents (Smart extraction & parallel parsing)...")
    print(f"  - Rejection Notice: {tender_path}")
    print(f"  - RFP Guidelines:   {rfp_path}")
    tender_doc, rfp_doc = extract_documents_parallel(tender_path, rfp_path)
    print(f"  - Rejection pages: {tender_doc.page_count} | RFP pages: {rfp_doc.page_count}")

    print("\n[2/3] Extracting rejection grounds & matching RFP clauses...")
    reason = extract_rejection_reason(tender_doc.text)
    print(f"  - Grounds Found: {reason[:120]}...")
    matched_clauses, clause_count = retrieve_relevant_clauses(reason, rfp_doc.text, top_k=5)
    print(f"  - Clauses Indexed: {clause_count} clauses found")

    print("\n[3/3] Evaluating with AI verdict engine...")
    is_valid, confidence, explanation, findings, summary, conclusion, used_ai = generate_verdict(reason, matched_clauses)

    print("\n" + "-" * 60)
    print("AUDIT RESULT:")
    print(f"  Status:      {'VALID (Rejection Upheld)' if is_valid else 'INVALID (Rejection Overturned)'}")
    print(f"  Confidence:  {confidence}%")
    print(f"  Engine:      {'Groq AI' if used_ai else 'Rule-Based Fallback'}")
    print(f"  Summary:     {summary or explanation}")
    if findings:
        print("\n  Key Findings:")
        for idx, finding in enumerate(findings, 1):
            print(f"    [{idx}] {finding}")
    if conclusion:
        print(f"\n  Conclusion:  {conclusion}")
    print("-" * 60 + "\n")


def start_legacy_gradio():
    """Starts the legacy Gradio UI."""
    from app import build_app
    print("\n" + "=" * 60)
    print("  TENDERLOGIC AI - STARTING LEGACY GRADIO UI")
    print("=" * 60)
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


def start_fastapi_server(host="127.0.0.1", port=8000):
    """Starts the modern FastAPI + React enterprise backend."""
    import uvicorn
    from backend.server import app

    print("\n" + "=" * 65)
    print("  TENDERLOGIC AI — ENTERPRISE PROCUREMENT AUDIT PLATFORM")
    print("=" * 65)

    client = get_groq_client()
    if client:
        print("  Status: [OK] Groq API client connected")
    else:
        print("  Status: [NOTE] Running in rule-based fallback mode (Add GROQ_API_KEY in .env)")

    print(f"\n  👉 Application Running At: http://{host}:{port}")
    print("  👉 API Documentation:    http://{host}:{port}/docs")
    print("=" * 65 + "\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TenderLogic AI Procurement Auditor")
    parser.add_argument("--cli", action="store_true", help="Run in command-line audit mode")
    parser.add_argument("-t", "--tender", type=str, default="", help="Path to Rejection Notice PDF")
    parser.add_argument("-r", "--rfp", type=str, default="", help="Path to RFP / Tender Guidelines PDF")
    parser.add_argument("--gradio", action="store_true", help="Run legacy Gradio UI")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind FastAPI server")
    args = parser.parse_args()

    if args.cli:
        run_cli_audit(args.tender, args.rfp)
    elif args.gradio:
        start_legacy_gradio()
    else:
        start_fastapi_server(port=args.port)