"""
TenderLogic AI — Document Processing Pipeline
------------------------------------------------
Speed-focused document ingestion:
  1. Smart OCR fallback — only run Tesseract on pages that actually need it
     (i.e. scanned image pages with no extractable text layer).
  2. Safe fallback when Tesseract OCR binary is not installed on Windows.
  3. Content-hash caching — re-uses previously extracted text/index instead of reprocessing.
  4. Parallel extraction — tender letter and RFP document are read concurrently.
"""

import os
import hashlib
import json
import concurrent.futures
from dataclasses import dataclass, asdict

try:
    import pymupdf as fitz
except ImportError:
    import fitz

try:
    import pytesseract
    from PIL import Image
    _OCR_IMPORTED = True
except ImportError:
    _OCR_IMPORTED = False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_BASE_DIR, ".tenderlogic_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _is_tesseract_installed() -> bool:
    if not _OCR_IMPORTED:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


OCR_AVAILABLE = _is_tesseract_installed()


@dataclass
class ExtractedDoc:
    file_name: str
    text: str
    page_count: int
    ocr_pages_used: int
    from_cache: bool


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(file_hash: str) -> str:
    return os.path.join(CACHE_DIR, f"{file_hash}.json")


def _load_from_cache(file_hash: str):
    path = _cache_path(file_hash)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["from_cache"] = True
            return ExtractedDoc(**data)
        except Exception:
            return None
    return None


def _save_to_cache(doc: ExtractedDoc):
    data = asdict(doc)
    data["from_cache"] = False
    try:
        h = _file_hash_registry.get(doc.file_name, hashlib.sha256(doc.file_name.encode()).hexdigest())
        with open(_cache_path(h), "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


_file_hash_registry = {}


def extract_text_smart(path: str) -> ExtractedDoc:
    """
    Extracts text from PDF with smart caching and OCR fallback for scanned pages.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document not found at path: {path}")

    file_hash = _file_hash(path)
    cached = _load_from_cache(file_hash)
    if cached:
        return cached

    file_name = os.path.basename(path)
    _file_hash_registry[file_name] = file_hash

    full_text_parts = []
    ocr_pages_used = 0

    doc = fitz.open(path)
    page_count = doc.page_count
    for page in doc:
        page_text = page.get_text().strip()

        if page_text:
            full_text_parts.append(page_text)
        elif OCR_AVAILABLE:
            pix = page.get_pixmap(dpi=200)
            img_path = os.path.join(CACHE_DIR, f"_tmp_{file_hash}_{page.number}.png")
            pix.save(img_path)
            try:
                ocr_text = pytesseract.image_to_string(Image.open(img_path))
                full_text_parts.append(ocr_text)
                ocr_pages_used += 1
            except Exception:
                full_text_parts.append("")
            finally:
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
        else:
            full_text_parts.append("")

    doc.close()

    result = ExtractedDoc(
        file_name=file_name,
        text="\n".join(full_text_parts),
        page_count=page_count,
        ocr_pages_used=ocr_pages_used,
        from_cache=False,
    )
    _save_to_cache(result)
    return result


def extract_documents_parallel(tender_path: str, rfp_path: str):
    """
    Extracts both documents concurrently using ThreadPoolExecutor.
    Cuts ingestion time in half for two-document audits.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_tender = executor.submit(extract_text_smart, tender_path)
        future_rfp = executor.submit(extract_text_smart, rfp_path)
        tender_doc = future_tender.result()
        rfp_doc = future_rfp.result()
    return tender_doc, rfp_doc


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        t_doc, r_doc = extract_documents_parallel(sys.argv[1], sys.argv[2])
        print("Extraction successful!")
        print(f"Doc 1 text: {len(t_doc.text)} chars, {t_doc.page_count} pages")
        print(f"Doc 2 text: {len(r_doc.text)} chars, {r_doc.page_count} pages")
    else:
        print("Usage: python pipeline.py <doc1_path> <doc2_path>")
