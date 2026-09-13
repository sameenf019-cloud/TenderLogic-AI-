"""
TenderLogic AI — AI Verdict & Retrieval Engine
------------------------------------------------
1. Rejection-reason extraction from the tender letter text.
2. BM25 hybrid clause retrieval over the RFP text.
3. Groq API integration using fast open-weight models:
     Primary:  openai/gpt-oss-120b
     Fallback: openai/gpt-oss-20b
4. Structured JSON verdict output with confidence and reasoning.
5. Heuristic fallback for offline use or when no key is set.
"""

import os
import re
import json
import sys
from rank_bm25 import BM25Okapi

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------
# Load API Key from .env / .evn / Environment / Colab Secrets
# ------------------------------------------------------------------
def _load_env_file():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv(".evn")
    except ImportError:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in [".env", ".evn", os.path.join(base_dir, ".env"), os.path.join(base_dir, ".evn")]:
        if os.path.exists(fname):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            except Exception:
                pass


_load_env_file()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

GENAI_AVAILABLE = False
_client = None


def get_groq_client():
    global _client, GENAI_AVAILABLE
    if _client is not None:
        return _client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if api_key:
        try:
            from groq import Groq
            _client = Groq(api_key=api_key)
            GENAI_AVAILABLE = True
            return _client
        except Exception:
            return None
    return None


# ------------------------------------------------------------------
# Rejection Reason Extraction
# ------------------------------------------------------------------
REASON_HEADERS = [
    r"\breason(?:s)? for rejection[:\s]*",
    r"\bgrounds for rejection[:\s]*",
    r"\bbasis for rejection[:\s]*",
]

CLAUSE_PATTERN = re.compile(
    r"(?:^|\n)(?=(?:(?:Section|Clause|Article|Rule)\s+[A-Z0-9\.]+|[A-Z]\.\s+[A-Z\s]{4,}|(?:\d+\.)\s+[A-Z]))",
    re.IGNORECASE,
)


def extract_rejection_reason(tender_text: str) -> str:
    text = tender_text.strip()
    lower = text.lower()

    for pattern in REASON_HEADERS:
        match = re.search(pattern, lower)
        if match:
            start = match.end()
            remainder = text[start:]
            # Capture up to the next major section or 4 paragraphs
            paragraphs = [p.strip() for p in remainder.split("\n\n") if p.strip()]
            if paragraphs:
                return "\n\n".join(paragraphs[:4])

    if len(text) < 4000:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    return "\n\n".join(paragraphs[:3]) if paragraphs else text[:600]


def chunk_rfp_clauses(rfp_text: str):
    raw_chunks = [c.strip() for c in CLAUSE_PATTERN.split(rfp_text) if c.strip()]
    if not raw_chunks or len(raw_chunks) <= 1:
        # Fallback to double newline paragraphs if regex didn't split
        paragraphs = [p.strip() for p in rfp_text.split("\n\n") if len(p.strip()) > 30]
        return [(f"Section {i+1}", p) for i, p in enumerate(paragraphs)]

    clauses = []
    for i, chunk in enumerate(raw_chunks):
        first_line = chunk.split("\n")[0].strip()
        label_match = re.match(
            r"((?:Section|Clause|Article|Rule)\s+[A-Z0-9\.]+|[A-Z]\.\s+[^\n]+|(?:\d+\.)\s+[^\n]+)",
            first_line,
            re.IGNORECASE,
        )
        label = label_match.group(1)[:60] if label_match else f"Clause {i+1}"
        clauses.append((label, chunk))
    return clauses


def retrieve_relevant_clauses(reason_text: str, rfp_text: str, top_k: int = 4):
    clauses = chunk_rfp_clauses(rfp_text)
    if not clauses:
        return "No clauses could be extracted from the RFP document.", 0

    tokenized_corpus = [c[1].lower().split() for c in clauses]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = reason_text.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(clauses, scores), key=lambda x: x[1], reverse=True)
    top_matches = [c for c, score in ranked[:top_k] if score > 0]

    if not top_matches:
        top_matches = [ranked[0][0]] if ranked else []

    formatted_parts = []
    for label, body in top_matches:
        clean_body = body.strip()
        if clean_body.startswith(label):
            clean_body = clean_body[len(label):].strip(" :\n\r-")
        formatted_parts.append(f"{label}:::{clean_body}")

    formatted = "\n\n".join(formatted_parts)
    return formatted, len(clauses)


# ------------------------------------------------------------------
# Groq Verdict Generation
# ------------------------------------------------------------------
def _call_groq(prompt: str, model: str) -> str:
    client = get_groq_client()
    if client is None:
        raise RuntimeError("No Groq client available")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def generate_verdict(reason_text: str, matched_clauses: str):
    """
    Returns (is_valid: bool, confidence: int, explanation: str, used_ai: bool)
    """
    client = get_groq_client()
    prompt = f"""You are an expert procurement auditor and legal compliance analyst.
Analyze whether the rejection of a bidder is justified based strictly on the Tender/RFP document clauses provided.

Your Task:
1. Examine each rejection ground, broken rule, or missing mandatory document cited in the Rejection Notice.
2. Cross-reference with the provided Tender/RFP Clauses to verify: Is this specific rule or mandatory document requirement actually stated in the Tender Document?
3. Determine whether the rejection is VALID (if the tender document genuinely requires it and the bidder violated it) or INVALID (if the tender document does not require it or the procuring entity applied an unstated criteria).
4. Identify and cite the EXACT clauses from the tender document that support or refute each ground.

Rejection Notice / Stated Grounds:
{reason_text}

Relevant Tender/RFP Clauses Found:
{matched_clauses}

Respond ONLY with a valid JSON object in this exact format (no markdown, no backticks, no extra text):
{{
  "is_valid": true or false,
  "confidence": <integer 0-100>,
  "summary": "<A 1-sentence executive summary of the determination>",
  "key_findings": [
    "<Concise bullet point 1: Stating rule/mandatory doc, whether it is stated in tender or not, with exact clause reference>",
    "<Concise bullet point 2: Stating rule/mandatory doc, whether it is stated in tender or not, with exact clause reference>",
    "<Concise bullet point 3: Stating rule/mandatory doc, whether it is stated in tender or not, with exact clause reference>"
  ],
  "conclusion": "<1-2 sentences stating final audit conclusion whether rejection is upheld or overturned>"
}}
"""

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            raw_text = _call_groq(prompt, model_name)
            cleaned = raw_text.strip()
            cleaned = re.sub(r"^```json|```$", "", cleaned, flags=re.MULTILINE).strip()
            data = json.loads(cleaned)
            is_valid = bool(data.get("is_valid", True))
            confidence = int(data.get("confidence", 75))
            summary = str(data.get("summary", "")).strip()
            findings = data.get("key_findings", [])
            if not isinstance(findings, list):
                findings = [str(findings)]
            conclusion = str(data.get("conclusion", "")).strip()
            
            # Combine into explanation text if needed for backward compatibility
            parts = []
            if summary:
                parts.append(summary)
            if findings:
                parts.extend([f"• {f}" for f in findings])
            if conclusion:
                parts.append(conclusion)
            explanation = "\n\n".join(parts) if parts else "Analysis complete."

            return is_valid, confidence, explanation, findings, summary, conclusion, True
        except Exception:
            continue

    is_v, conf, exp, ai = _heuristic_verdict(reason_text, matched_clauses)
    return is_v, conf, exp, [], exp, "", False


def _heuristic_verdict(reason_text: str, matched_clauses: str):
    if not matched_clauses or "No clauses could be extracted" in matched_clauses:
        return True, 55, (
            "No matching RFP clause text was found. Defaults to a cautious "
            "'valid' verdict with low confidence. Manual review recommended."
        ), False

    reason_words = set(reason_text.lower().split())
    clause_words = set(matched_clauses.lower().split())
    overlap = len(reason_words & clause_words)
    confidence = min(95, 50 + overlap * 3)

    return True, confidence, (
        "Heuristic fallback (offline mode): the rejection reason shares significant "
        "wording with the matched RFP clauses, suggesting the rejection is grounded in the document."
    ), False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        from pipeline import extract_documents_parallel
        t_doc, r_doc = extract_documents_parallel(sys.argv[1], sys.argv[2])
        reason = extract_rejection_reason(t_doc.text)
        clauses, count = retrieve_relevant_clauses(reason, r_doc.text)
        is_valid, conf, expl, findings, summary, conclusion, used_ai = generate_verdict(reason, clauses)

        print("AI Engine Test Results:")
        print(f" - Used AI:      {used_ai} ({'Groq' if used_ai else 'Heuristic'})")
        print(f" - Verdict:      {'VALID (Rejection Upheld)' if is_valid else 'INVALID (Overturned)'}")
        print(f" - Confidence:   {conf}%")
        print(f" - Explanation:  {expl}")
    else:
        print("Usage: python ai_engine.py <rejection_path> <rfp_path>")
