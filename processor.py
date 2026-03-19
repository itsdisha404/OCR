import sys
import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import statistics
from typing import Dict, List, Tuple, Any

# Ensure Unicode output works on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz          # PyMuPDF  (pip install pymupdf)

# Optional: Image preprocessing for OCR improvement
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

BASE_DIR = Path(__file__).parent

# ─── OCR Quality thresholds ───────────────────────────────────────────────────
OCR_CONFIDENCE_THRESHOLD = 0.25
MIN_BOX_COUNT = 30


# ─── ACCURACY TRACKING ────────────────────────────────────────────────────────

class AccuracyTracker:
    """Track extraction accuracy for each section."""
    
    def __init__(self):
        self.scores = {}
        self.details = {}
    
    def add_field(self, section: str, field: str, value: any, confidence: float = 1.0, 
                  is_empty: bool = False, is_pattern_match: bool = False):
        """
        Record a field extraction with confidence score.
        confidence: 0.0-1.0 (1.0 = high confidence, 0.0 = low confidence)
        is_empty: True if field is empty (0.5 penalty)
        is_pattern_match: True if matched via pattern (slight penalty)
        """
        if section not in self.scores:
            self.scores[section] = {"total": 0, "sum": 0.0, "fields": {}}
            self.details[section] = []
        
        # Adjust confidence based on conditions
        final_conf = confidence
        if is_empty:
            final_conf *= 0.5  # Penalty for empty values
        if is_pattern_match:
            final_conf *= 0.85  # Small penalty for pattern matches vs direct finds
        
        self.scores[section]["total"] += 1
        self.scores[section]["sum"] += final_conf
        self.scores[section]["fields"][field] = final_conf
        
        self.details[section].append({
            "field": field,
            "value": str(value)[:100],
            "confidence": round(final_conf, 2),
            "empty": is_empty
        })
    
    def get_section_score(self, section: str) -> float:
        """Get accuracy score for a section (0-100)."""
        if section not in self.scores or self.scores[section]["total"] == 0:
            return 0.0
        score_data = self.scores[section]
        return round((score_data["sum"] / score_data["total"]) * 100, 2)
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get accuracy scores for all sections."""
        return {section: self.get_section_score(section) 
                for section in self.scores.keys()}
    
    def get_overall_score(self) -> float:
        """Get overall accuracy score (0-100)."""
        if not self.scores:
            return 0.0
        all_scores = self.get_all_scores()
        return round(sum(all_scores.values()) / len(all_scores), 2)
    
    def to_dict(self) -> dict:
        """Export accuracy data as dictionary."""
        return {
            "section_scores": self.get_all_scores(),
            "overall_accuracy": self.get_overall_score()
        }


def preprocess_image_for_ocr(image_path: Path, aggressive: bool = False) -> Path:
    """Apply image preprocessing to improve OCR quality."""
    if not CV2_AVAILABLE:
        return image_path
    
    img = cv2.imread(str(image_path))
    if img is None:
        return image_path
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if aggressive:
        scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled, None, h=25, templateWindowSize=7, searchWindowSize=25)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=1)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel, iterations=1)
        processed = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, blockSize=39, C=13)
    else:
        denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        processed = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, blockSize=25, C=8)
    
    out_path = image_path.parent / f"{image_path.stem}_pp{image_path.suffix}"
    cv2.imwrite(str(out_path), processed)
    return out_path


def check_ocr_quality(ocr_results: list) -> dict:
    """Check and report OCR result quality."""
    if not ocr_results:
        return {"avg_conf": 0, "median_conf": 0, "box_count": 0,
                "quality_score": 0, "is_acceptable": False, 
                "recommendation": "No OCR boxes found"}
    
    confidences = [r[2] for r in ocr_results if len(r) >= 3 and isinstance(r[2], (int, float))]
    box_count = len(ocr_results)
    
    if not confidences:
        return {"avg_conf": 0, "median_conf": 0, "box_count": box_count,
                "quality_score": 0, "is_acceptable": False,
                "recommendation": f"No valid confidence scores ({box_count} boxes)"}
    
    avg_conf = statistics.mean(confidences)
    median_conf = statistics.median(confidences)
    quality_score = min(100, int(avg_conf * 100 + max(0, (box_count - MIN_BOX_COUNT) / 10)))
    is_acceptable = (avg_conf >= OCR_CONFIDENCE_THRESHOLD and box_count >= MIN_BOX_COUNT)
    
    if avg_conf < 0.25:
        recommendation = "CRITICAL: Preprocessing strongly recommended"
    elif avg_conf < OCR_CONFIDENCE_THRESHOLD:
        recommendation = "LOW: Preprocessing recommended"
    elif box_count < MIN_BOX_COUNT:
        recommendation = "FEW_BOXES: Preprocessing may help"
    else:
        recommendation = "ACCEPTABLE: OCR quality is good"
    
    return {"avg_conf": round(avg_conf, 3), "median_conf": round(median_conf, 3),
            "box_count": box_count, "quality_score": quality_score,
            "is_acceptable": is_acceptable, "recommendation": recommendation}


def pdf_to_images(pdf_path: Path, images_dir: Path, scale: float = 4.0) -> list:
    """Render each PDF page as a high-resolution PNG."""
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(scale, scale)
    paths = []
    for i, page in enumerate(doc):
        out = images_dir / f"page_{i+1:03d}.png"
        page.get_pixmap(matrix=mat).save(str(out))
        print(f"    [img] page {i+1}/{len(doc)} -> {out.name}")
        paths.append(out)
    doc.close()
    return paths


def run_ocr(image_paths: list) -> list:
    """Run EasyOCR with automatic quality checking and aggressive fallback preprocessing."""
    import easyocr
    print("  Loading EasyOCR model (first run downloads ~100 MB)...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    
    print("\n  [PASS 1] Initial OCR on original images...")
    pass1_results = []
    pass1_quality = []
    preprocessed_needed = []
    
    for i, img_path in enumerate(image_paths):
        print(f"    [ocr] page {i+1}/{len(image_paths)}  {img_path.name}")
        res = reader.readtext(str(img_path), detail=1, paragraph=False)
        pass1_results.append(res)
        
        quality = check_ocr_quality(res)
        pass1_quality.append(quality)
        
        if not quality["is_acceptable"]:
            preprocessed_needed.append(i)
            print(f"      ⚠️  Quality: {quality['quality_score']}/100 - {quality['recommendation']}")
        else:
            print(f"      ✓  Quality: {quality['quality_score']}/100 - {quality['recommendation']}")
    
    if preprocessed_needed and CV2_AVAILABLE:
        print(f"\n  [PASS 2] Aggressive preprocessing for {len(preprocessed_needed)} page(s)...")
        
        for page_idx in preprocessed_needed:
            img_path = image_paths[page_idx]
            print(f"    [prep] page {page_idx+1} - aggressive enhancement + upscaling...")
            
            pp_path = preprocess_image_for_ocr(img_path, aggressive=True)
            
            print(f"    [ocr]  page {page_idx+1} - retrying on preprocessed image...")
            res_pp = reader.readtext(str(pp_path), detail=1, paragraph=False)
            
            quality_pp = check_ocr_quality(res_pp)
            quality_orig = pass1_quality[page_idx]
            
            if quality_pp["quality_score"] > quality_orig["quality_score"]:
                improvement = quality_pp["quality_score"] - quality_orig["quality_score"]
                print(f"      ✓  Improved: {quality_orig['quality_score']} → {quality_pp['quality_score']}/100 (+{improvement})")
                pass1_results[page_idx] = res_pp
            else:
                print(f"      ℹ  Original better: {quality_orig['quality_score']} vs {quality_pp['quality_score']}/100")
    
    elif preprocessed_needed and not CV2_AVAILABLE:
        print(f"\n  [WARN] {len(preprocessed_needed)} page(s) have low OCR quality.")
        print("         cv2 not available - preprocessing disabled.")
    
    return pass1_results


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def bbox_rect(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def group_rows(ocr_page: list, y_tol: int = 14) -> list:
    """Group EasyOCR boxes into horizontal rows by y-center proximity."""
    buckets = {}
    for bbox, text, conf in ocr_page:
        x1, y1, x2, y2 = bbox_rect(bbox)
        yc = (y1 + y2) / 2
        matched = next((k for k in buckets if abs(yc - k) < y_tol), None)
        if matched is None:
            matched = yc
        buckets.setdefault(matched, []).append(
            {"text": text.strip(), "x1": x1, "x2": x2,
             "y1": y1, "y2": y2, "xc": (x1 + x2) / 2, "yc": yc, "conf": conf}
        )
    return [
        sorted(v, key=lambda d: d["x1"])
        for k, v in sorted(buckets.items())
    ]


# ─── IMPROVED EXTRACTION FUNCTIONS ────────────────────────────────────────────

def _find_value(flat: list, pattern: str, same_row: bool = True) -> Tuple[str, float]:
    """
    Find a label matching `pattern` in the flat OCR list,
    then return the first non-colon value to its right (same row) or below.
    Returns (value, confidence_score)
    """
    for i, item in enumerate(flat):
        if re.search(pattern, item["text"], re.IGNORECASE):
            label_yc = item["yc"]
            label_x2 = item["x2"]
            
            if same_row:
                candidates = [
                    x for x in flat[i + 1: i + 8]
                    if abs(x["yc"] - label_yc) < 22
                    and x["x1"] >= label_x2 - 10
                    and x["text"] not in (":", "")
                ]
                if candidates:
                    best = candidates[0]
                    # Confidence based on OCR confidence and proximity
                    conf = (best.get("conf", 0.9) + 0.9) / 2
                    return best["text"], conf
            
            if not same_row:
                for x in flat[i + 1: i + 6]:
                    if x["text"] not in (":", "", "|"):
                        conf = (x.get("conf", 0.9) + 0.8) / 2
                        return x["text"], conf
    
    return "", 0.0


def _clean_invoice_number(text: str) -> str:
    """Clean and validate invoice number."""
    # Remove common OCR errors and whitespace
    cleaned = re.sub(r'[\s\-/]', '', text.upper())
    # Keep alphanumeric only
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    return cleaned[:50]  # reasonable max length


def _clean_date(text: str) -> str:
    """Clean and normalize date format (DD/MM/YYYY)."""
    # Extract only date-like patterns
    date_match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', text)
    if date_match:
        d, m, y = date_match.groups()
        # Normalize to DD/MM/YYYY
        if len(y) == 2:
            y = "20" + y
        return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
    return text


def _clean_gstin(text: str) -> str:
    """Clean GSTIN/GST number."""
    cleaned = text.strip().upper()
    # GSTINs are typically 15 chars: 2 digits + 10 digit PAN + 3 entity + 1 check
    cleaned = re.sub(r'[\s\-]', '', cleaned)
    # Keep alphanumeric only
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    return cleaned[:15]


def _clean_pan(text: str) -> str:
    """Clean PAN number."""
    cleaned = text.strip().upper()
    # PAN is 10 characters: 5 letters + 4 numbers + 1 letter
    cleaned = re.sub(r'[\s\-]', '', cleaned)
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    return cleaned[:10]


def extract_invoice_header(page1_rows: list, tracker: AccuracyTracker) -> dict:
    """Extract invoice header - only required fields. Returns null for missing values."""
    flat = [item for row in page1_rows for item in row]
    
    # Company name / Vendor name
    company_name = None
    company_conf = 0.0
    for i in flat:
        if re.search(r"ESSEN\s*MEDICARE|CHHABRA\s*HEALTHCARE|LUPIN\s*LIMITED", i["text"], re.IGNORECASE):
            company_name = i["text"].strip()
            company_conf = i.get("conf", 0.9)
            break
    
    # Address - collect from address section
    address = None
    address_conf = 0.0
    addr_lines = []
    collecting = False
    for row in page1_rows:
        row_text = " ".join(item["text"] for item in row)
        if re.search(r"Address|Location|Office", row_text, re.IGNORECASE):
            collecting = True
            continue
        if collecting:
            if re.search(r"Product|HSN|Item|Bill|Ship", row_text, re.IGNORECASE):
                break
            addr_parts = [item["text"].strip() for item in row if item["text"].strip()]
            if addr_parts:
                addr_lines.append(" ".join(addr_parts))
    
    if addr_lines:
        address = ", ".join(addr_lines)
        address_conf = 0.85
    
    # GSTIN
    gstin_text, gstin_conf = _find_value(flat, r"GSTIN\s*No|GSTIN")
    gstin_no = _clean_gstin(gstin_text) if gstin_text else None
    
    # PAN
    supplier_pan = None
    pan_conf = 0.0
    for i, item in enumerate(flat):
        txt = item["text"].replace(" ", "")
        if re.fullmatch(r"PAN", txt, re.IGNORECASE):
            context_before = " ".join(x["text"] for x in flat[max(0, i-3):i])
            if "cust" not in context_before.lower() and "customer" not in context_before.lower():
                candidates = [
                    x for x in flat[i + 1: i + 5]
                    if abs(x["yc"] - item["yc"]) < 22
                    and x["x1"] >= item["x2"] - 10
                    and x["text"] not in (":", "")
                ]
                if candidates:
                    supplier_pan = _clean_pan(candidates[0]["text"])
                    pan_conf = candidates[0].get("conf", 0.9)
                    break
    
    # Invoice number
    invoice_text, invoice_conf = _find_value(flat, r"INVOICE\s*NO|Invoice\s*Number")
    invoice_no = _clean_invoice_number(invoice_text) if invoice_text else None
    
    # Invoice date
    invoice_date_text, invoice_date_conf = _find_value(flat, r"INVOICE\s*DATE|Invoice\s*Date")
    invoice_date = _clean_date(invoice_date_text) if invoice_date_text else None
    
    # Order number
    order_text, order_conf = _find_value(flat, r"Order\s*No|Cust\.?\s*Order|PO\s*No|Against\s*Order")
    order_no = order_text if order_text else None
    
    # Track accuracy
    tracker.add_field("invoice_header", "company_name", company_name or "NULL", company_conf, 
                     is_empty=(company_name is None))
    tracker.add_field("invoice_header", "address", address or "NULL", address_conf, 
                     is_empty=(address is None))
    tracker.add_field("invoice_header", "gstin_no", gstin_no or "NULL", gstin_conf, 
                     is_empty=(gstin_no is None))
    tracker.add_field("invoice_header", "pan", supplier_pan or "NULL", pan_conf, 
                     is_empty=(supplier_pan is None))
    tracker.add_field("invoice_header", "invoice_no", invoice_no or "NULL", invoice_conf, 
                     is_empty=(invoice_no is None))
    tracker.add_field("invoice_header", "invoice_date", invoice_date or "NULL", invoice_date_conf, 
                     is_empty=(invoice_date is None))
    tracker.add_field("invoice_header", "order_no", order_no or "NULL", order_conf, 
                     is_empty=(order_no is None))
    
    return {
        "company_name": company_name,
        "address": address,
        "gstin_no": gstin_no,
        "pan": supplier_pan,
        "invoice_no": invoice_no,
        "invoice_date": invoice_date,
        "order_no": order_no,
    }


def extract_bill_to(page1_rows: list, tracker: AccuracyTracker) -> dict:
    """Extract Bill To section with improved accuracy."""
    flat = [item for row in page1_rows for item in row]
    
    # Bill To and Ship To are side-by-side; left half of page = Bill To
    LEFT_HALF = 680
    
    addr_lines, collecting = [], False
    for row in page1_rows:
        row_text = " ".join(i["text"] for i in row)
        if re.search(r"Bill\s*To|Bill\s*Address", row_text, re.IGNORECASE):
            collecting = True
            bill_to_x = next(
                (i["x2"] for i in row if re.search(r"Bill\s*To|Bill\s*Address", i["text"], re.IGNORECASE)), 0
            )
            same_row_name = [
                i["text"] for i in row
                if i["x1"] > bill_to_x
                and i["x1"] < LEFT_HALF
                and i["text"] not in (":", "")
                and not re.search(r"Ship\s*To|Delivery\s*Address", i["text"], re.IGNORECASE)
            ]
            if same_row_name:
                addr_lines.append(" ".join(same_row_name))
            continue
        
        if collecting:
            if re.search(r"(Place\s*of\s*Supply|Description|Item|HSN|Product)", row_text, re.IGNORECASE):
                break
            left = [i["text"] for i in row if i["x1"] < LEFT_HALF]
            if left:
                addr_lines.append(" ".join(left))
    
    name = addr_lines[0] if addr_lines else ""
    address = ", ".join(addr_lines[1:]).strip(", ") if len(addr_lines) > 1 else ""
    
    cust_gstin_text, cust_gstin_conf = _find_value(flat, r"Cust\.?\s*GSTIN|Customer\s*GSTIN")
    cust_gstin = _clean_gstin(cust_gstin_text) if cust_gstin_text else ""
    
    cust_dl_text, cust_dl_conf = _find_value(flat, r"Cust\.?\s*D\.?L|Customer\s*DL")
    cust_dl = cust_dl_text if cust_dl_text else ""
    
    dl_exp_text, dl_exp_conf = _find_value(flat, r"DL\s*Exp\.?\s*Date|Expiry\s*Date")
    dl_exp_date = _clean_date(dl_exp_text) if dl_exp_text else ""
    
    po_ref_text, po_ref_conf = _find_value(flat, r"Cust\.?\s*PO\s*Ref|PO\s*Reference")
    po_ref = po_ref_text if po_ref_text else ""
    
    po_ref_date_text, po_ref_date_conf = _find_value(flat, r"PO\s*Ref\.?\s*Date|PO\s*Date")
    po_ref_date = _clean_date(po_ref_date_text) if po_ref_date_text else ""
    
    # Track accuracy
    tracker.add_field("bill_to", "name", name, 0.95 if name else 0.3, is_empty=(not name))
    tracker.add_field("bill_to", "address", address, 0.9 if address else 0.3, is_empty=(not address))
    tracker.add_field("bill_to", "cust_gstin", cust_gstin, cust_gstin_conf, is_empty=(not cust_gstin))
    tracker.add_field("bill_to", "cust_dl_no", cust_dl, cust_dl_conf, is_empty=(not cust_dl))
    tracker.add_field("bill_to", "dl_exp_date", dl_exp_date, dl_exp_conf, is_empty=(not dl_exp_date))
    tracker.add_field("bill_to", "po_ref", po_ref, po_ref_conf, is_empty=(not po_ref))
    tracker.add_field("bill_to", "po_ref_date", po_ref_date, po_ref_date_conf, is_empty=(not po_ref_date))
    
    return {
        "name": name,
        "address": address,
        "cust_gstin": cust_gstin,
        "cust_dl_no": cust_dl,
        "dl_exp_date": dl_exp_date,
        "po_ref": po_ref,
        "po_ref_date": po_ref_date,
    }


# ─── LINE ITEM EXTRACTION ─────────────────────────────────────────────────────

COL_KEYWORDS = [
    ("product_code_desc", r"Product\s*Code|Product\s*Description|Product\s*Name|Description\s*of\s*Goods|Material\s*Description|Item\s*Description|Item\s*Name|Description|Goods"),
    ("hsn_code", r"\bHSN\b|HSN\s*Code|HSN/SAC|SAC\s*Code"),
    ("div", r"\bDIV\.?\b|Division"),
    ("batch_no", r"Batch\s*No\.?|Batch"),
    ("expiry_date", r"Expiry|Exp\s*Date|Expiry\s*Date|Exp\.?|Expiration"),
    ("mrp_per_unit", r"\bMRP\b|Retail\s*Price|MRP\s*\(.*\)|Retail\s*Price\s*\(MRP\)"),
    ("qty_uom", r"\bQty\b|Quantity|Billed\s*Quantity|Sale\s*Qty|Billed\s*Qty|Qty\s*/\s*UOM"),
    ("trade_price_unit", r"Price\s*/\s*Unit|Price/?Unit|Rate\s*\(Per\s*item\)|Rate\s*Per\s*Item|Unit\s*Rate|Price\s*Per\s*Unit"),
    ("tp_ptr_value", r"\bPTR\b|Price\s*To\s*Retailer|PTS"),
    ("trade_disc_pct", r"\bDisc\b|Discount\s*%|Total\s*Disc\.?%|Cash\s*Disc"),
    ("cgst_pct", r"\bCGST\b|CGST\s*Rate"),
    ("sgst_pct", r"\bSGST\b|SGST\s*Rate"),
    ("igst_pct", r"\bIGST\b|IGST\s*Rate"),
    ("pts", r"\bPTS\b|Price\s*To\s*Stockist"),
    ("amount", r"\bAmount\b|Net\s*Amount|Total\s*Amount|Taxable\s*Value|Value"),
]

NUMERIC_COLS = {
    "mrp_per_unit", "trade_price_unit", "tp_ptr_value",
    "trade_disc_pct", "cgst_pct", "sgst_pct", "pts", "amount", "qty"
}

SKIP_PATTERNS = [
    r"GST\s*IS\s*NOT\s*PAYABLE",
    r"GRAND\s*TOTAL|TOTAL\s*AMOUNT",
    r"E\s*\.\s*&\s*O\s*\.",
    r"www\..*\.com",
    r"Regd\.?\s*Office|Registered\s*Office",
    r"Transported\s*by",
    r"LR\s*No|LR\s*Date|Vehicle\s*No",
    r"Product\s*Code\s*&\s*Description",
    r"\bHSN\s*Code\b.*\bBatch\b",
    r"Place\s*of\s*Supply",
    r"Bill\s*To|Ship\s*To",
    r"Payment\s*Terms|Payment\s*Method",
    r"Charge\s*Interest",
    r"days\s*of\s*Invoice",
    r"TERMS\s*&\s*CONDITIONS|Terms\s*and\s*Conditions",
    r"Goods\s*once\s*sold|interest\s*@|Company\s*staffs",
    r"TOTAL.*PAYABLE|AMOUNT.*WORDS|Rupees.*Only|Paise",
    r"TCS\s*AMOUNT|FOR\s*ESSEN|Regd\s*Off",
    r"disputes.*subject|juridiction",
]

FOOTER_KEYWORDS = [
    "TOTAL GST", "IN WORDS", "CONDITION", "TCS AMOUNT", "Rupees", "Paise",
    "disputes", "juridiction", "FOR ESSEN", "Regd Off", "AMOUNT", "TERMS"
]


def _is_skip(row: list) -> bool:
    """Check if row should be skipped (header/footer/noise)."""
    text = " ".join(i["text"] for i in row)
    return any(re.search(p, text, re.IGNORECASE) for p in SKIP_PATTERNS)


def _contains_footer_text(value: str) -> bool:
    """Check if value contains footer/summary keywords."""
    if not value:
        return False
    value_upper = value.upper()
    return any(keyword.upper() in value_upper for keyword in FOOTER_KEYWORDS)


def _extract_clean_field(text: str) -> str:
    """Extract only the first meaningful part before footer text appears."""
    if not text:
        return ""
    
    # Split by common footer keywords and take only the first part
    for keyword in ["TOTAL", "AMOUNT", "RUPEES", "TERMS", "CONDITION", "TCS", "FOR ESSEN", "disputes"]:
        if keyword.upper() in text.upper():
            # Get text before the keyword
            idx = text.upper().find(keyword.upper())
            if idx > 0:
                text = text[:idx].strip()
    
    return text.strip()


def _is_col_header(row: list) -> bool:
    """Detect if row is column header."""
    text = " ".join(i["text"] for i in row)
    return bool(re.search(r"\bHSN\b.{0,40}\bBatch\b|\bBatch\b.{0,60}\bCGST\b|\bQty\b.{0,40}\bRate\b", text, re.IGNORECASE))


def _detect_col_positions(all_rows: list) -> dict:
    """Scan all rows across all pages to find x-centers of column headers."""
    positions = {}
    for row in all_rows:
        for col_name, pattern in COL_KEYWORDS:
            if col_name in positions:
                continue
            for item in row:
                if re.search(pattern, item["text"], re.IGNORECASE):
                    positions[col_name] = item["xc"]
                    break
    return positions


def _assign_column(xc: float, col_positions: dict) -> str:
    """Return the column name whose x-center is closest to xc."""
    if not col_positions:
        return "unknown"
    sorted_cols = sorted(col_positions.items(), key=lambda kv: kv[1])
    best = min(sorted_cols, key=lambda kv: abs(kv[1] - xc))
    return best[0]


def _extract_numeric(text: str) -> float:
    """Extract first valid number from text."""
    nums = re.findall(r"\d+\.\d+|\d+", text)
    try:
        return float(nums[0]) if nums else 0.0
    except (ValueError, IndexError):
        return 0.0


def _finalize_item(raw: defaultdict, tracker: AccuracyTracker, item_idx: int) -> dict:
    """Collapse multi-part text per column, coerce numerics, split product code."""
    item = {"row_index": item_idx}
    field_confidence = 0.0
    field_count = 0
    
    for col, parts in raw.items():
        text = " ".join(parts).strip()
        
        # CRITICAL: Clean footer text from all fields
        if _contains_footer_text(text):
            text = _extract_clean_field(text)
        
        is_empty = (not text or text == "")
        
        # If text still contains footer keywords after cleaning, skip this field
        if _contains_footer_text(text):
            item[col] = None
            conf = 0.1
        elif col in NUMERIC_COLS:
            value = _extract_numeric(text)
            item[col] = value
            conf = 0.8 if value > 0 else 0.3
        elif col == "qty_uom":
            m = re.match(r"(\d+)\s*([A-Z]+)?", text)
            if m:
                item["qty"] = int(m.group(1))
                item["uom"] = m.group(2) or ""
                conf = 0.9
            else:
                item["qty_uom"] = text if text else None
                conf = 0.5
        elif col == "product_code_desc":
            # Split "PRODUCT NAME 404552" into description + code
            m = re.match(r"^(.*?)\s+(\d{5,7})\s*$", text.strip())
            if m:
                item["product_description"] = m.group(1).strip()
                item["product_code"] = m.group(2).strip()
                conf = 0.95
            else:
                item["product_description"] = text if text else None
                item["product_code"] = None
                conf = 0.7
        elif col == "expiry_date":
            # Validate: expiry date should be in MM/YYYY or DD/MM/YYYY format
            # NOT a large decimal number like 172.25, 164.57, etc.
            cleaned = _clean_date(text) if text else None
            
            # Check if it looks like a valid date (not a price)
            # Valid: "08/2026", "13/11/2024", "Nov-2027"
            # Invalid: "172.25", "191.39", "164.57" (these are prices/decimals)
            if cleaned and not re.match(r"^\d{1,2}/\d{4}$|^\d{1,2}/\d{1,2}/\d{4}$", cleaned):
                # Doesn't match date format, likely a price - set to null
                item[col] = None
                conf = 0.2
            else:
                item[col] = cleaned
                conf = 0.85 if cleaned else 0.3
        elif col in ["batch_no", "hsn_code"]:
            item[col] = text if text else None
            conf = 0.9 if text else 0.3
        else:
            item[col] = text if text else None
            conf = 0.85 if text else 0.3
        
        if col not in ["unknown"]:
            field_confidence += conf
            field_count += 1
            tracker.add_field(f"line_item_{item_idx}", col, text, conf, is_empty=is_empty, is_pattern_match=True)
    
    # Calculate item accuracy
    item_accuracy = field_confidence / field_count if field_count > 0 else 0.0
    item["accuracy_score"] = round(item_accuracy * 100, 2)
    
    return item


def extract_line_items(all_page_rows: list, tracker: AccuracyTracker) -> list:
    """Extract product line items with improved accuracy tracking."""
    flat_rows = [row for page in all_page_rows for row in page]
    col_pos = _detect_col_positions(flat_rows)
    print(f"  Column positions detected: {list(col_pos.keys())}")
    
    items = []
    current = defaultdict(list)
    in_table = False
    first_started = False
    item_counter = 0
    
    for page_rows in all_page_rows:
        for row in page_rows:
            if _is_col_header(row):
                in_table = True
                continue
            if _is_skip(row):
                continue
            if not in_table:
                continue
            
            # Detect new line item: any cell contains an 8-digit HSN code
            row_text_joined = " ".join(i["text"] for i in row)
            has_hsn = bool(re.search(r"\b\d{8}\b", row_text_joined))
            
            if has_hsn:
                if first_started:
                    items.append(_finalize_item(current, tracker, item_counter))
                    item_counter += 1
                current = defaultdict(list)
                first_started = True
            
            if not first_started:
                continue
            
            for cell in row:
                col = _assign_column(cell["xc"], col_pos)
                if col != "unknown":
                    current[col].append(cell["text"])
    
    if current and first_started:
        items.append(_finalize_item(current, tracker, item_counter))
    
    return items


# ─── OUTPUT HANDLING ─────────────────────────────────────────────────────────

def save_outputs(data: dict, accuracy: dict, output_dir: Path, stem: str):
    """Save JSON output with accuracy scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge accuracy data
    data["accuracy"] = accuracy
    
    # JSON
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [JSON]  {json_path}")


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def process_pdf(pdf_path):
    """Process PDF invoice with accuracy tracking."""
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return None
    
    stem = pdf_path.stem
    print(f"\n{'='*60}")
    print(f" Processing : {pdf_path.name}")
    print(f"{'='*60}")
    
    images_dir = BASE_DIR / "images" / stem
    output_dir = BASE_DIR / "output" / stem
    
    # Initialize accuracy tracker
    tracker = AccuracyTracker()
    
    # 1 ─ PDF → Images
    print("\n[1/4] PDF -> images")
    image_paths = pdf_to_images(pdf_path, images_dir)
    
    # 2 ─ OCR
    print("\n[2/4] OCR")
    all_ocr = run_ocr(image_paths)
    
    # 3 ─ Parse with accuracy tracking
    print("\n[3/4] Parsing structure")
    page_rows = [group_rows(p) for p in all_ocr]
    
    header = extract_invoice_header(page_rows[0], tracker)
    bill_to = extract_bill_to(page_rows[0], tracker)
    line_items = extract_line_items(page_rows, tracker)
    
    print(f"  Invoice No  : {header.get('invoice_no', '?')}")
    print(f"  Date        : {header.get('invoice_date', '?')}")
    print(f"  Bill To     : {bill_to.get('name', '?')}")
    print(f"  Line items  : {len(line_items)}")
    
    # Get accuracy scores
    accuracy_data = tracker.to_dict()
    
    # Print accuracy summary
    print("\n[ACCURACY SCORES]")
    print(f"  Overall Accuracy: {accuracy_data['overall_accuracy']:.2f}%")
    for section, score in accuracy_data['section_scores'].items():
        print(f"  {section}: {score:.2f}%")
    
    data = {
        "source_file": pdf_path.name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "invoice_header": header,
        "bill_to": bill_to,
        "line_items": line_items,
        "summary": {
            "total_line_items": len(line_items),
        },
    }
    
    # 4 ─ Export
    print("\n[4/4] Saving outputs")
    save_outputs(data, accuracy_data, output_dir, stem)
    
    print(f"\n[DONE] Outputs saved to: {output_dir}\n")
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python processor_improved.py <invoice.pdf>")
        sys.exit(1)
    process_pdf(sys.argv[1])