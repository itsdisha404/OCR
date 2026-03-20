import sys
import json
import re
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from urllib import request, error

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

BASE_DIR = Path(__file__).parent


def load_dotenv_file(dotenv_path: Path) -> None:
    """Load KEY=VALUE pairs from .env into process environment."""
    if not dotenv_path.exists():
        return

    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


load_dotenv_file(BASE_DIR / ".env")

#  SECTION 1: CONSTANTS & CONFIGURATION

OCR_CONFIDENCE_THRESHOLD = 0.25
MIN_BOX_COUNT = 30

# Define all required fields with their schemas
HEADER_FIELDS = [
    "company_name",
    "pan",
    "invoice_no",
    "invoice_date",
    "due_date",
]

BILL_TO_FIELDS = [
    "name",
    "address",
    "date",
]

LINE_ITEM_FIELDS = [
    "row_index",
    "product_description",
    "hsn_code",
    "batch_no",
    "expiry_date",
    "qty",
    "uom",
    "mrp",
    "ptr",
    "pts",
    "discount",
    "cgst",
    "sgst",
    "total_amount",
    "accuracy_score",
]

COLUMN_SYNONYMS = {
    "mfg_name": [
        "mfg name", "mfg.", "manufacturer", "mfg"
    ],
    "category": [
        "cat", "category"
    ],
    "product_description": [
        "product", "description", "product description", "material description",
        "item description"
    ],
    "hsn_code": [
        "hsn", "hsn code", "hsn/sac", "hsn no"
    ],
    "batch_no": [
        "batch", "batch no", "batch number"
    ],
    "expiry_date": [
        "expiry", "exp", "exp date", "expiry date", "exp dt", "expiry dt", "E.X.P"
    ],
    "qty": [
        "qty", "quantity", "billed qty", "sale qty"
    ],
    "uom": [
        "unit", "uom"
    ],
    "mrp": [
        "mrp", "m.r.p"
    ],
    "ptr": [
        "ptr", "p.t.r"
    ],
    "pts": [
        "pts"
    ],
    "discount": [
        "discount", "disc"
    ],
    "cgst": [
        "cgst"
    ],
    "sgst": [
        "sgst"
    ],
    "total_amount": [
        "total amount", "line total", "net amount", "total value", "taxable value"
    ]
}

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2: UTILITY & CLEANING FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """Central data cleaning and validation"""
    
    @staticmethod
    def clean_string(text: Optional[str]) -> Optional[str]:
        """Clean string: strip, remove extra spaces"""
        if text is None or (isinstance(text, str) and not text.strip()):
            return None
        if not isinstance(text, str):
            return None
        return text.strip() if text.strip() else None
    
    @staticmethod
    def extract_number(text: Optional[str]) -> Optional[float]:
        """Extract first number from text"""
        if not text:
            return None
        numbers = re.findall(r'\d+\.?\d*', str(text))
        try:
            return float(numbers[0]) if numbers else None
        except (ValueError, IndexError):
            return None
    
    @staticmethod
    def extract_gstin(text: Optional[str]) -> Optional[str]:
        """Extract and clean GSTIN (15 chars)"""
        if not text:
            return None
        # Remove spaces and special chars
        gstin = re.sub(r'[\s\-/]', '', str(text).upper())
        gstin = re.sub(r'[^A-Z0-9]', '', gstin)
        # GSTIN is 15 chars
        if len(gstin) >= 15:
            return gstin[:15]
        return gstin if gstin else None
    
    @staticmethod
    def extract_pan(text: Optional[str]) -> Optional[str]:
        """Extract and clean PAN (10 chars)"""
        if not text:
            return None
        # Remove leading colons/special chars
        pan = re.sub(r'^[:\s]+|[:\s]+$', '', str(text))
        pan = re.sub(r'[\s\-/:]', '', pan.upper())
        pan = re.sub(r'[^A-Z0-9]', '', pan)
        return pan if pan else None
    
    @staticmethod
    def clean_date(text: Optional[str]) -> Optional[str]:
        """Normalize date to DD/MM/YYYY format"""
        if not text:
            return None
        # Try to extract date pattern
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY or DD-MM-YY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, str(text))
            if match:
                day, month, year = match.groups()
                # Normalize year
                if len(year) == 2:
                    year = '20' + year if int(year) < 50 else '19' + year
                return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        return None
    
    @staticmethod
    def clean_invoice_no(text: Optional[str]) -> Optional[str]:
        """Clean invoice number"""
        if not text:
            return None
        # Keep alphanumeric and common separators
        cleaned = re.sub(r'[\s]', '', str(text))
        return cleaned if cleaned else None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3: ACCURACY TRACKING
# ════════════════════════════════════════════════════════════════════════════════

class AccuracyScorer:
    """Score extraction accuracy at field, section, and item level"""
    
    def __init__(self):
        self.section_scores = {}
    
    def score_item(self, item_dict: Dict[str, Any]) -> float:
        """
        Calculate accuracy for a line item with realistic validation.
        Scores each critical field 0-100, then averages them.
        Penalties for: missing fields, invalid ranges, inconsistent pricing/totals.
        """
        if not item_dict:
            return 0.0
        
        field_scores = {}
        
        # Product Description
        prod = str(item_dict.get('product_description') or '').strip()
        if prod and len(prod) >= 3 and not re.search(r'^[\d\s\.\-]+$', prod):
            field_scores['product_description'] = 100.0
        elif prod:
            field_scores['product_description'] = 50.0
        else:
            field_scores['product_description'] = 0.0
        
        # HSN Code (8 digits)
        hsn = str(item_dict.get('hsn_code') or '').strip()
        if re.fullmatch(r'\d{8}', hsn):
            field_scores['hsn_code'] = 100.0
        elif hsn:
            field_scores['hsn_code'] = 40.0
        else:
            field_scores['hsn_code'] = 0.0
        
        # Batch Number
        batch = str(item_dict.get('batch_no') or '').strip()
        if batch and len(batch) >= 3 and len(batch) <= 14:
            field_scores['batch_no'] = 100.0
        elif batch:
            field_scores['batch_no'] = 40.0
        else:
            field_scores['batch_no'] = 0.0
        
        # Expiry Date
        exp = str(item_dict.get('expiry_date') or '').strip()
        if exp and re.search(r'^\d{1,2}/\d{2,4}$', exp):
            field_scores['expiry_date'] = 100.0
        elif exp:
            field_scores['expiry_date'] = 40.0
        else:
            field_scores['expiry_date'] = 20.0
        
        # Quantity (must be positive integer)
        qty_val = item_dict.get('qty')
        if isinstance(qty_val, (int, float)) and qty_val > 0:
            field_scores['qty'] = 100.0
        else:
            field_scores['qty'] = 0.0
        
        # MRP (price range 5-5000)
        mrp_val = item_dict.get('mrp')
        if isinstance(mrp_val, (int, float)) and 5 <= mrp_val <= 5000:
            field_scores['mrp'] = 100.0
        elif isinstance(mrp_val, (int, float)):
            field_scores['mrp'] = 40.0
        else:
            field_scores['mrp'] = 0.0
        
        # PTR (price range 1-5000)
        ptr_val = item_dict.get('ptr')
        if isinstance(ptr_val, (int, float)) and 1 <= ptr_val <= 5000:
            field_scores['ptr'] = 100.0
        elif isinstance(ptr_val, (int, float)):
            field_scores['ptr'] = 40.0
        else:
            field_scores['ptr'] = 0.0
        
        # PTS (price range 1-5000)
        pts_val = item_dict.get('pts')
        if isinstance(pts_val, (int, float)) and 1 <= pts_val <= 5000:
            field_scores['pts'] = 100.0
        elif isinstance(pts_val, (int, float)):
            field_scores['pts'] = 40.0
        else:
            field_scores['pts'] = 0.0
        
        # Discount (0-100%)
        disc_val = item_dict.get('discount')
        if isinstance(disc_val, (int, float)) and 0 <= disc_val <= 100:
            field_scores['discount'] = 100.0
        elif isinstance(disc_val, (int, float)):
            field_scores['discount'] = 50.0
        else:
            field_scores['discount'] = 0.0
        
        # CGST (0-28%)
        cgst_val = item_dict.get('cgst')
        if isinstance(cgst_val, (int, float)) and 0 <= cgst_val <= 28:
            field_scores['cgst'] = 100.0
        elif isinstance(cgst_val, (int, float)):
            field_scores['cgst'] = 40.0
        else:
            field_scores['cgst'] = 0.0
        
        # SGST (0-28%)
        sgst_val = item_dict.get('sgst')
        if isinstance(sgst_val, (int, float)) and 0 <= sgst_val <= 28:
            field_scores['sgst'] = 100.0
        elif isinstance(sgst_val, (int, float)):
            field_scores['sgst'] = 40.0
        else:
            field_scores['sgst'] = 0.0
        
        # Total Amount (should be reasonable: qty*ptr <= total <= qty*mrp)
        total_val = item_dict.get('total_amount')
        
        # Check if total seems plausible
        total_score = 0.0
        if isinstance(total_val, (int, float)) and total_val > 0:
            # Estimate reasonable range: qty * ptr_80% to qty * mrp
            qty_for_calc = qty_val if isinstance(qty_val, (int, float)) and qty_val > 0 else 1
            mrp_for_calc = mrp_val if isinstance(mrp_val, (int, float)) and mrp_val > 0 else 1000
            ptr_for_calc = ptr_val if isinstance(ptr_val, (int, float)) and ptr_val > 0 else 500
            
            low_estimate = qty_for_calc * ptr_for_calc * 0.7  # Allow for discount
            high_estimate = qty_for_calc * mrp_for_calc * 1.1  # Allow for minor OCR errors
            if low_estimate <= total_val <= high_estimate:
                total_score = 100.0
            elif total_val < low_estimate * 0.5 or total_val > high_estimate * 2:
                total_score = 10.0  # Way off
            else:
                total_score = 50.0  # Somewhat plausible but dubious
        else:
            total_score = 0.0
        
        field_scores['total_amount'] = total_score
        
        # Calculate average of critical fields (product, hsn, qty, mrp, total_amount, cgst, sgst)
        critical_fields = ['product_description', 'hsn_code', 'qty', 'mrp', 'total_amount', 'cgst', 'sgst']
        
        if critical_fields:
            critical_avg = sum(field_scores.get(f, 0) for f in critical_fields) / len(critical_fields)
        else:
            critical_avg = 0.0
        
        # Penalties for unrealistic values
        hierarchy_penalty = 0.0
        if (isinstance(mrp_val, (int, float)) and isinstance(ptr_val, (int, float)) and 
            mrp_val > 0 and ptr_val > mrp_val):
            hierarchy_penalty = 5.0
        
        discount_penalty = 0.0
        if isinstance(disc_val, (int, float)) and not (0 <= disc_val <= 100):
            discount_penalty = 10.0
        
        overall_accuracy = critical_avg - hierarchy_penalty - discount_penalty
        return round(min(100, max(0, overall_accuracy)), 2)
    
    def score_section(self, section_data: Dict[str, Any]) -> float:
        """Score extraction for header or bill-to section with field-level validation"""
        if not section_data:
            return 0.0

        field_scores = {}
        
        # Identify section type and score accordingly
        if 'company_name' in section_data:
            # HEADER SECTION
            company = str(section_data.get('company_name') or '').strip()
            field_scores['company_name'] = 100.0 if (company and len(company) >= 5) else 0.0
            
            pan = str(section_data.get('pan') or '').strip()
            if re.fullmatch(r'[A-Z]{5}\d{4}[A-Z]', pan):
                field_scores['pan'] = 100.0
            elif pan:
                field_scores['pan'] = 50.0
            else:
                field_scores['pan'] = 0.0
            
            inv_no = str(section_data.get('invoice_no') or '').strip()
            field_scores['invoice_no'] = 100.0 if (inv_no and len(inv_no) >= 3) else 20.0
            
            inv_date = str(section_data.get('invoice_date') or '').strip()
            if inv_date and re.search(r'\d{1,2}/\d{1,2}/\d{4}', inv_date):
                field_scores['invoice_date'] = 100.0
            elif inv_date:
                field_scores['invoice_date'] = 50.0
            else:
                field_scores['invoice_date'] = 0.0
        
        elif 'address' in section_data:
            # BILL-TO SECTION
            name = section_data.get('name')
            if isinstance(name, list) and len(name) > 0:
                field_scores['name'] = 100.0
            elif isinstance(name, str) and len(name) > 5:
                field_scores['name'] = 100.0
            else:
                field_scores['name'] = 0.0
            
            addr = str(section_data.get('address') or '').strip()
            if addr and len(addr) >= 10:
                field_scores['address'] = 100.0
            elif addr:
                field_scores['address'] = 50.0
            else:
                field_scores['address'] = 20.0
        
        if not field_scores:
            # Fallback: simple presence check
            def is_present(value: Any) -> bool:
                if value is None:
                    return False
                if isinstance(value, str):
                    return bool(value.strip())
                if isinstance(value, (list, tuple, set, dict)):
                    return len(value) > 0
                return True

            non_null_fields = sum(1 for v in section_data.values() if is_present(v))
            total_fields = len(section_data)
            return round((non_null_fields / total_fields) * 100, 2) if total_fields > 0 else 0.0
        
        avg_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
        return round(min(100, max(0, avg_score)), 2)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: PDF EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_path: Path, images_dir: Path, scale: float = 4.0) -> List[Path]:
    """Convert PDF pages to PNG images"""
    images_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(scale, scale)
    paths = []
    
    for i, page in enumerate(doc):
        out = images_dir / f"page_{i+1:03d}.png"
        page.get_pixmap(matrix=mat).save(str(out))
        print(f"  ✓ Page {i+1}/{len(doc)}")
        paths.append(out)
    
    doc.close()
    return paths


def run_ocr(image_paths: List[Path]) -> List[List]:
    """Run OCR on images with EasyOCR"""
    try:
        import easyocr
    except ImportError:
        print("ERROR: easyocr not installed. Install with: pip install easyocr")
        sys.exit(1)
    
    print("  Loading EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    
    all_results = []
    for i, img_path in enumerate(image_paths):
        print(f"  ✓ OCR page {i+1}/{len(image_paths)}")
        results = reader.readtext(str(img_path), detail=1, paragraph=False)
        all_results.append(results)
    
    return all_results


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5: TEXT EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def group_rows(ocr_page: List, y_tol: int = 14) -> List[List[Dict]]:
    """Group OCR boxes into horizontal rows"""
    buckets = {}
    
    for bbox, text, conf in ocr_page:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        
        yc = (y1 + y2) / 2
        
        # Find matching row
        matched_y = None
        for existing_y in buckets:
            if abs(yc - existing_y) < y_tol:
                matched_y = existing_y
                break
        
        if matched_y is None:
            matched_y = yc
        
        buckets.setdefault(matched_y, []).append({
            "text": text.strip(),
            "x1": x1,
            "x2": x2,
            "xc": (x1 + x2) / 2,
            "y1": y1,
            "y2": y2,
            "yc": yc,
            "conf": conf
        })
    
    # Sort rows top-to-bottom, cells left-to-right
    rows = []
    for y in sorted(buckets.keys()):
        row = sorted(buckets[y], key=lambda d: d["x1"])
        rows.append(row)
    
    return rows


def find_value_after_label(flat_cells: List[Dict], label_pattern: str, 
                           max_distance: int = 8) -> Tuple[Optional[str], float]:
    """
    Find label matching pattern, return next non-colon value and confidence.
    Returns: (value, confidence)
    """
    for i, cell in enumerate(flat_cells):
        if re.search(label_pattern, cell["text"], re.IGNORECASE):
            label_x2 = cell["x2"]
            label_yc = cell["yc"]
            
            # Look ahead for value
            candidates = [
                c for c in flat_cells[i+1:i+max_distance]
                if abs(c["yc"] - label_yc) < 25  # Same row
                and c["x1"] >= label_x2 - 10  # To the right
                and c["text"] not in (":", "", "-")
            ]
            
            if candidates:
                best = max(candidates, key=lambda x: x["conf"])
                return best["text"], best["conf"]
    
    return None, 0.0


def flatten_cells(rows: List[List[Dict]]) -> List[Dict]:
    """Flatten all rows into single list of cells"""
    return [cell for row in rows for cell in row]


def _is_noise_label_value(text: Optional[str]) -> bool:
    """Heuristic filter for words that are typically labels, not values."""
    if not text:
        return True

    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(text).upper())
    if not cleaned:
        return True

    noise_words = {
        "PAN", "GSTIN", "DATE", "OFFICE", "INVOICE", "NO", "DUE",
        "DL", "FSSAI", "REF", "PO", "ORDER", "NAME", "ADDRESS"
    }
    return cleaned in noise_words


def _extract_labeled_value(
    page_rows: List[List[Dict]],
    label_pattern: str,
    validator: Optional[Any] = None,
) -> Tuple[Optional[str], float]:
    """Extract value to the right of a matching label with optional validation."""
    best_fallback: Tuple[Optional[str], float] = (None, 0.0)

    for row in page_rows:
        sorted_row = sorted(row, key=lambda c: c.get("x1", 0))
        for idx, cell in enumerate(sorted_row):
            label_text = cell.get("text", "")
            if not re.search(label_pattern, label_text, re.IGNORECASE):
                continue

            label_x2 = cell.get("x2", 0)
            right_cells = [
                c for c in sorted_row[idx + 1:]
                if c.get("x1", 0) >= label_x2 - 10
            ]

            for candidate in right_cells:
                candidate_text = str(candidate.get("text", "")).strip(" :-|\t")
                if not candidate_text or _is_noise_label_value(candidate_text):
                    continue

                if validator is None:
                    return candidate_text, float(candidate.get("conf", 0.0))

                try:
                    if validator(candidate_text):
                        return candidate_text, float(candidate.get("conf", 0.0))
                except Exception:
                    pass

                if float(candidate.get("conf", 0.0)) > best_fallback[1]:
                    best_fallback = (candidate_text, float(candidate.get("conf", 0.0)))

    return best_fallback


def _is_valid_pan_text(text: str) -> bool:
    candidate = re.sub(r"[^A-Za-z0-9]", "", str(text).upper())
    return bool(re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", candidate))


def _is_valid_gstin_text(text: str) -> bool:
    candidate = re.sub(r"[^A-Za-z0-9]", "", str(text).upper())
    return bool(re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", candidate))


def _is_valid_invoice_no_text(text: str) -> bool:
    candidate = str(text).strip()
    if _is_noise_label_value(candidate):
        return False
    if len(candidate) < 3 or len(candidate) > 40:
        return False
    if DataCleaner.clean_date(candidate):
        return False
    return bool(re.search(r"\d", candidate))


def _extract_expiry_from_text(text: str) -> Optional[str]:
    """Extract expiry date from row text; when multiple dates exist use the latest as expiry."""
    if not text:
        return None

    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    candidates: List[Tuple[int, int, int, str]] = []

    # DD/MM/YYYY
    for day, month, year in re.findall(r"\b([0-3]?\d)[/-](0?[1-9]|1[0-2])[/-](\d{2,4})\b", text):
        y = int(year)
        if y < 100:
            y += 2000 if y < 50 else 1900
        d = int(day)
        m = int(month)
        candidates.append((y, m, d, f"{str(d).zfill(2)}/{str(m).zfill(2)}/{y}"))

    # MM/YYYY
    for month, year in re.findall(r"\b(0[1-9]|1[0-2])[/-](\d{2,4})\b", text):
        y = int(year)
        if y < 100:
            y += 2000 if y < 50 else 1900
        m = int(month)
        candidates.append((y, m, 1, f"{str(m).zfill(2)}/{y}"))

    # MON-YYYY
    for mon, year in re.findall(r"\b([A-Za-z]{3})[-/](\d{2,4})\b", text):
        mon_num = month_map.get(mon.upper())
        if mon_num is None:
            continue
        y = int(year)
        if y < 100:
            y += 2000 if y < 50 else 1900
        candidates.append((y, mon_num, 1, f"{str(mon_num).zfill(2)}/{y}"))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def _extract_tax_rates_from_row_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Infer CGST/SGST rates from row text by preferring repeated low-percentage tokens."""
    if not text:
        return None, None

    raw_tokens = re.findall(r"\d+(?:\.\d+)?", str(text))
    values: List[float] = []
    for token in raw_tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue

    rate_candidates = [v for v in values if 0 < v <= 28]
    if not rate_candidates:
        return None, None

    rounded_counts: Dict[float, int] = defaultdict(int)
    for val in rate_candidates:
        rounded_counts[round(val, 2)] += 1

    repeated = [(rate, cnt) for rate, cnt in rounded_counts.items() if cnt >= 2]
    if repeated:
        repeated.sort(key=lambda x: (x[1], x[0]), reverse=True)
        chosen = repeated[0][0]
        return _round2(chosen), _round2(chosen)

    common_rates = [2.5, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0]
    nearest = min(common_rates, key=lambda r: min(abs(v - r) for v in rate_candidates))
    if min(abs(v - nearest) for v in rate_candidates) <= 0.6:
        return _round2(nearest), _round2(nearest)

    return None, None


def _clean_product_description(product_text: Optional[str], row_values: Dict[str, str]) -> Optional[str]:
    """Normalize product description and remove manufacturer/category prefix leakage."""
    if not product_text:
        return None

    cleaned = re.sub(r'^\s*\d+\s*', '', str(product_text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' :-|')

    mfg = str(row_values.get("mfg_name") or "").strip(" .:-|")
    cat = str(row_values.get("category") or "").strip(" .:-|")

    if mfg:
        mfg_pattern = rf'^\s*{re.escape(mfg)}\.?\s+'
        cleaned = re.sub(mfg_pattern, '', cleaned, flags=re.IGNORECASE)

    if cat:
        cat_pattern = rf'^\s*{re.escape(cat)}\s+'
        cleaned = re.sub(cat_pattern, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'^\s*[+&]+\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' :-|')
    return cleaned or None


def _extract_batch_from_text(text: str, hsn_code: Optional[str]) -> Optional[str]:
    """Extract likely batch number from row text when column mapping misses it."""
    if not text:
        return None

    tokens = re.findall(r"\b[A-Z0-9']{4,14}\b", str(text).upper())
    hsn_code = str(hsn_code or "").strip().upper()
    stop_tokens = {
        "BATCH", "QTY", "MRP", "PTR", "PTS", "CGST", "SGST", "IGST", "TOTAL", "AMOUNT"
    }

    for token in tokens:
        if token in stop_tokens:
            continue
        if hsn_code and token == hsn_code:
            continue
        if re.fullmatch(r"\d{8}", token):
            continue
        if re.fullmatch(r"\d{1,3}", token):
            continue
        if re.fullmatch(r"\d{6}", token):
            continue
        if re.search(r"[A-Z]", token) and re.search(r"\d", token):
            return token

    batch_hint = re.search(r"BATCH\s*[:\-]?\s*([A-Z0-9']{4,14})", str(text).upper())
    if batch_hint:
        hinted = batch_hint.group(1)
        if hinted != hsn_code and not re.fullmatch(r"\d{8}", hinted):
            return hinted

    return None


def _extract_pts_from_row_text(text: str) -> Optional[float]:
    """Extract likely PTS value from text when PTS column exists but mapping missed."""
    if not text:
        return None

    label_match = re.search(r"\bP\.?T\.?S\b\s*[:\-]?\s*(\d[\d,]*\.?\d*)", str(text), re.IGNORECASE)
    if label_match:
        try:
            return _round2(float(label_match.group(1).replace(",", "")))
        except ValueError:
            return None
    return None


def _extract_total_amount_from_row_text(text: str) -> Optional[float]:
    """Pick likely total amount from a line-item row based on right-most numeric token."""
    if not text:
        return None

    matches = re.findall(r"\d[\d,]*\.?\d*", str(text))
    if len(matches) < 3:
        return None

    values: List[float] = []
    for token in matches:
        try:
            values.append(float(token.replace(",", "")))
        except ValueError:
            continue

    if not values:
        return None

    for candidate in reversed(values):
        if candidate > 0:
            return _round2(candidate)

    return None


def _extract_taxable_value_from_row_text(text: str) -> Optional[float]:
    """Extract taxable value by locating the number immediately before first CGST rate token."""
    if not text:
        return None

    matches = list(re.finditer(r"\d+(?:\.\d+)?", str(text)))
    if len(matches) < 3:
        return None

    values: List[float] = []
    for match in matches:
        try:
            values.append(float(match.group(0)))
        except ValueError:
            values.append(-1.0)

    common_rates = {2.5, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0}
    for idx, val in enumerate(values):
        if round(val, 2) in common_rates and idx >= 1:
            taxable = values[idx - 1]
            if taxable > 0:
                return _round2(taxable)

    return None


def _extract_discount_from_row_text(text: str) -> Optional[float]:
    """Extract discount with OCR-noise tolerance, treating ').0' style tokens as 0.0."""
    if not text:
        return None

    compact = re.sub(r"\s+", "", str(text))
    if re.search(r"[\)\]]\.?0\b", compact):
        return 0.0

    label_match = re.search(r"\bDISC(?:OUNT)?\b\s*[:\-]?\s*([\)\]\(\[\d\.]+)", str(text), re.IGNORECASE)
    if label_match:
        token = re.sub(r"[^0-9\.]", "", label_match.group(1))
        if token == "" or token == "0" or token == "0.0" or token == "00":
            return 0.0
        try:
            val = float(token)
            if 0 <= val <= 100:
                return _round2(val)
        except ValueError:
            return None

    # Prefer discount qty token that appears just before discount amount/taxable segment.
    matches = list(re.finditer(r"\d+(?:\.\d+)?", str(text)))
    values: List[float] = []
    for match in matches:
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue

    if len(values) >= 4:
        common_rates = {2.5, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0}
        for idx, val in enumerate(values):
            if round(val, 2) in common_rates and idx >= 3:
                discount_qty = values[idx - 3]
                if 0 <= discount_qty <= 100:
                    if abs(discount_qty - round(discount_qty)) < 0.01:
                        return _round2(discount_qty)

    return None


def _normalize_text(text: str) -> str:
    """Normalize OCR text for fuzzy header matching."""
    normalized = re.sub(r'[^a-z0-9]+', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', normalized).strip()


def _match_column_from_text(text: str) -> Optional[str]:
    """Map a header cell text to canonical column key using COLUMN_SYNONYMS."""
    normalized = _normalize_text(text)
    if not normalized:
        return None

    for col_name, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            syn_norm = _normalize_text(synonym)
            if syn_norm and re.search(rf'\b{re.escape(syn_norm)}\b', normalized):
                return col_name
    return None


def _detect_column_positions(all_page_rows: List[List[List[Dict]]]) -> Dict[str, float]:
    """Detect column x-positions from table header rows using synonyms."""
    best_positions: Dict[str, float] = {}

    for page_rows in all_page_rows:
        for row in page_rows:
            row_matches: Dict[str, float] = {}
            for cell in row:
                col = _match_column_from_text(cell.get("text", ""))
                if col and col not in row_matches:
                    row_matches[col] = cell["xc"]

            # Strong header row signal
            if len(row_matches) >= 4 and ("product_description" in row_matches or "hsn_code" in row_matches):
                return row_matches

            # Weak fallback accumulation
            for col, x in row_matches.items():
                if col not in best_positions:
                    best_positions[col] = x

    return best_positions


def _assign_row_to_columns(row: List[Dict], col_positions: Dict[str, float]) -> Dict[str, str]:
    """Assign OCR cells in a row to nearest detected columns."""
    if not col_positions:
        return {}

    sorted_cols = sorted(col_positions.items(), key=lambda kv: kv[1])

    def max_distance_for_col(col_name: str) -> float:
        idx = next((i for i, (name, _) in enumerate(sorted_cols) if name == col_name), -1)
        if idx == -1:
            return 120.0

        x = sorted_cols[idx][1]
        left_gap = x - sorted_cols[idx - 1][1] if idx > 0 else None
        right_gap = sorted_cols[idx + 1][1] - x if idx < len(sorted_cols) - 1 else None
        gaps = [g for g in [left_gap, right_gap] if g is not None and g > 0]
        if not gaps:
            return 120.0
        return max(40.0, min(gaps) * 0.55)

    assigned: Dict[str, List[str]] = defaultdict(list)
    for cell in row:
        text = cell.get("text", "").strip()
        if not text:
            continue

        nearest_col, nearest_x = min(col_positions.items(), key=lambda kv: abs(cell["xc"] - kv[1]))
        if abs(cell["xc"] - nearest_x) <= max_distance_for_col(nearest_col):
            assigned[nearest_col].append(text)

    return {col: " ".join(parts).strip() for col, parts in assigned.items() if parts}


def _round2(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _collect_table_rows_text(all_page_rows: List[List[List[Dict]]]) -> List[str]:
    """Collect likely line-item table rows as plain text for LLM refinement context."""
    rows_text: List[str] = []
    in_table = False

    for page_rows in all_page_rows:
        for row in page_rows:
            row_text = " ".join(c.get("text", "") for c in row).strip()
            if not row_text:
                continue

            row_header_hits = {
                _match_column_from_text(c.get("text", ""))
                for c in row
                if _match_column_from_text(c.get("text", ""))
            }

            if len(row_header_hits) >= 3 and (
                "product_description" in row_header_hits or "hsn_code" in row_header_hits
            ):
                in_table = True
                continue

            if not in_table:
                continue

            if re.search(r'TOTAL.*AMOUNT|GRAND.*TOTAL|AMOUNT.*WORDS|TERMS.*CONDITION', row_text, re.IGNORECASE):
                return rows_text

            if re.search(r'Goods.*sold|Interest.*per|Company.*staff|disputes', row_text, re.IGNORECASE):
                continue

            if re.search(r'\d', row_text):
                rows_text.append(row_text)

    return rows_text


def refine_line_items_with_openrouter(
    line_items: List[Dict[str, Any]],
    table_rows_text: List[str],
    header: Dict[str, Any],
    bill_to: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Optionally refine OCR-extracted line items using OpenRouter LLM.

    Enabled only when OPENROUTER_API_KEY is set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not line_items:
        return line_items

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    system_prompt = (
        "You are an expert pharmaceutical invoice line-item processor. Your task is to refine and correct "
        "OCR-extracted invoice line items using the provided invoice header, bill-to details, and raw table text.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY a valid JSON array - no explanations, no markdown, no code fences.\n"
        "2. Maintain exact same item count and row_index values - do not add/remove/reorder items.\n"
        "3. NEVER hallucinate values - only correct if clearly inferable from provided data.\n"
        "4. Use null for any unknown/unrecoverable fields.\n\n"
        "FIELD GUIDELINES:\n"
        "- row_index: Keep unchanged (sequential identifier)\n"
        "- product_description: Medical/pharma product names (e.g., 'ASPIRIN 500MG', 'COUGH SYRUP')\n"
        "- hsn_code: 8-digit numeric code for pharmaceutical items\n"
        "- batch_no: Alphanumeric batch identifier (e.g., 'B2024001')\n"
        "- expiry_date: Format MM/YYYY or DD/MM/YYYY\n"
        "- qty: Integer quantity (e.g., 10, 100, 500)\n"
        "- uom: Unit of Measure (BOX, STRIP, BLISTER, PCS, PIECE, CARTON, etc.)\n"
        "- mrp: Maximum Retail Price (positive decimal, typically 10-10000)\n"
        "- ptr: Price to Retailer (positive decimal, typically 20-90% of MRP)\n"
        "- pts: Price to Stockist (positive decimal, typically 10-80% of MRP)\n"
        "- discount: Discount percentage (0-100, typically 0-50%)\n"
        "- cgst: Central GST percentage (0-28%, common: 5%, 12%, 18%)\n"
        "- sgst: State GST percentage (0-28%, same as CGST in India)\n"
        "- total_amount: Line total after discounts and taxes (positive decimal)\n"
        "- accuracy_score: Keep as provided or set to 0.0 if corrected\n\n"
        "VALIDATION RULES:\n"
        "- Quantity must be positive integer\n"
        "- Prices (mrp, ptr, pts) must be positive and mrp >= ptr >= pts (usually)\n"
        "- GST percentages must be 0-28 and equal (CGST = SGST in India)\n"
        "- Discount must be 0-100 percent\n"
        "- total_amount = (mrp * qty) - (discount/100 * mrp * qty) + taxes\n"
        "- Dates must be valid and past/present (not future)\n"
    )

    user_prompt = {
        "invoice_metadata": {
            "company_name": header.get("company_name"),
            "invoice_no": header.get("invoice_no"),
            "invoice_date": header.get("invoice_date"),
        },
        "bill_to": {
            "customer_name": bill_to.get("name"),
            "customer_address": bill_to.get("address"),
        },
        "instructions": [
            "Step 1: Review the invoice metadata (company, invoice#, date) for context.",
            "Step 2: Use bill_to customer details to validate product relevance.",
            "Step 3: Analyze table_rows_text to identify column positions and patterns.",
            "Step 4: For each current_item, correct fields only if clearly supported by table data.",
            "Step 5: Validate that prices follow: MRP >= PTR >= PTS (pharmaceutical pricing hierarchy).",
            "Step 6: Ensure GST values are valid Indian tax rates (typically 5%, 12%, 18%, 28%).",
            "Step 7: Cross-check totals: total_amount should align with (qty * price) and tax calculations.",
            "Step 8: Return corrected items maintaining original structure and count."
        ],
        "table_rows_text": table_rows_text,
        "current_items": line_items,
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"].strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()
        refined = json.loads(content)

        if not isinstance(refined, list) or len(refined) != len(line_items):
            print("  [LLM] Ignored refinement: invalid shape")
            return line_items

        validated: List[Dict[str, Any]] = []
        for i, item in enumerate(refined):
            if not isinstance(item, dict):
                return line_items

            base = dict(line_items[i])
            for key in LINE_ITEM_FIELDS:
                if key in item:
                    base[key] = item[key]

            for key in ["mrp", "ptr", "pts", "discount", "cgst", "sgst", "total_amount"]:
                if base.get(key) is not None:
                    try:
                        base[key] = _round2(float(base[key]))
                    except (ValueError, TypeError):
                        base[key] = None

            if base.get("qty") is not None:
                try:
                    base["qty"] = int(float(base["qty"]))
                except (ValueError, TypeError):
                    base["qty"] = None

            validated.append(base)

        print("  [LLM] OpenRouter refinement applied")
        return validated

    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [LLM] Refinement skipped: {exc}")
        return line_items


def _normalize_line_item_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """Apply deterministic sanity fixes to extracted line-item values."""
    normalized = dict(item)

    hsn_code = str(normalized.get("hsn_code") or "").strip()
    batch_no = str(normalized.get("batch_no") or "").strip()

    # If batch accidentally captured as HSN, drop it instead of returning wrong data.
    if hsn_code and batch_no and hsn_code == batch_no and re.fullmatch(r"\d{8}", batch_no):
        normalized["batch_no"] = None

    total_amount = normalized.get("total_amount")

    # If CGST/SGST captured as tax amount instead of rate, convert to percentage.
    for tax_key in ["cgst", "sgst"]:
        tax_val = normalized.get(tax_key)
        if tax_val is None:
            continue
        try:
            tax_val_f = float(tax_val)
        except (ValueError, TypeError):
            normalized[tax_key] = None
            continue

        if total_amount is not None:
            try:
                total_f = float(total_amount)
            except (ValueError, TypeError):
                total_f = 0.0

            if total_f > 0 and tax_val_f > 40:
                normalized[tax_key] = _round2((tax_val_f / total_f) * 100)
            else:
                normalized[tax_key] = _round2(tax_val_f)
        else:
            normalized[tax_key] = _round2(tax_val_f)

    # Keep GST rates within realistic percentage range.
    for tax_key in ["cgst", "sgst"]:
        tax_val = normalized.get(tax_key)
        if tax_val is None:
            continue
        if not (0 <= float(tax_val) <= 28):
            normalized[tax_key] = None

    for number_key in ["mrp", "ptr", "pts", "discount", "total_amount"]:
        val = normalized.get(number_key)
        if val is not None:
            try:
                normalized[number_key] = _round2(float(val))
            except (ValueError, TypeError):
                normalized[number_key] = None

    # Keep extracted taxable value; do not force qty*ptr because invoice quantity columns vary.

    # Mandatory pricing hierarchy enforcement for every invoice item:
    # MRP >= PTR >= PTS
    mrp = normalized.get("mrp")
    ptr = normalized.get("ptr")
    pts = normalized.get("pts")

    if mrp is not None and ptr is not None and pts is not None:
        ordered = sorted([float(mrp), float(ptr), float(pts)], reverse=True)
        normalized["mrp"] = _round2(ordered[0])
        normalized["ptr"] = _round2(ordered[1])
        normalized["pts"] = _round2(ordered[2])
    elif mrp is not None and ptr is not None and float(ptr) > float(mrp):
        normalized["mrp"] = _round2(float(ptr))
        normalized["ptr"] = _round2(float(mrp))
    elif ptr is not None and pts is not None and float(pts) > float(ptr):
        normalized["ptr"] = _round2(float(pts))
        normalized["pts"] = _round2(float(ptr))
    elif mrp is not None and pts is not None and float(pts) > float(mrp):
        normalized["mrp"] = _round2(float(pts))
        normalized["pts"] = _round2(float(mrp))

    # Guard against shifted-column mistakes:
    # discount often gets copied from tax rates when discount column is blank.
    discount = normalized.get("discount")
    cgst = normalized.get("cgst")
    sgst = normalized.get("sgst")
    if discount is not None and cgst is not None:
        same_as_cgst = abs(float(discount) - float(cgst)) < 0.01
        same_as_sgst = sgst is not None and abs(float(discount) - float(sgst)) < 0.01
        if same_as_cgst and (same_as_sgst or sgst is None):
            normalized["discount"] = None

    if normalized.get("discount") is None:
        normalized["discount"] = 0.0
    else:
        try:
            discount_val = float(normalized["discount"])
        except (ValueError, TypeError):
            discount_val = 0.0

        # Correct unrealistic discount leakage from neighboring columns.
        if discount_val > 60 and normalized.get("mrp") is not None and normalized.get("ptr") is not None:
            normalized["discount"] = 0.0

    # PTS should not mirror PTR/MRP exactly in shifted extraction scenarios.
    pts = normalized.get("pts")
    ptr = normalized.get("ptr")
    mrp = normalized.get("mrp")
    if pts is not None:
        if ptr is not None and abs(float(pts) - float(ptr)) < 0.01:
            normalized["pts"] = None
        elif mrp is not None and abs(float(pts) - float(mrp)) < 0.01:
            normalized["pts"] = None

    qty_val = normalized.get("qty")
    if qty_val is not None:
        try:
            normalized["qty"] = int(float(qty_val))
        except (ValueError, TypeError):
            normalized["qty"] = None

    return normalized


def normalize_line_items(line_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize all line items with deterministic quality checks."""
    return [_normalize_line_item_fields(item) for item in line_items]


def _looks_like_company_name(text: str) -> bool:
    """Check if text looks like a company name (not address, email, etc)."""
    if not text or len(text.strip()) < 5:
        return False

    candidate = re.sub(r'\s+', ' ', str(text).strip())

    exclude = re.compile(
        r'(^EMAIL|^PHONE|^MOBILE|^FAX|^WEBSITE|^ADDRESS|^WAREHOUSE|'
        r'FLOOR|BUILDING|PLOT|ROAD|STREET|CITY|STATE|PIN|VILLAGE|'
        r'^INVOICE|^BILL|^SHIP|^CUSTOMER|ATTN|C/O|DIN|PAN|GSTIN|CIN NO|TAN|'
        r'SUBJECT|REF|DATE|DOOR|HOUSE|APARTMENT|FLAT|SUITE)',
        re.IGNORECASE,
    )
    if exclude.search(candidate):
        return False

    letters = re.sub(r'[^A-Za-z]', '', candidate)
    if len(letters) < 4:
        return False

    # Avoid rows that are mostly IDs/vehicle/license style values.
    if re.search(r'\b\d{2,}[A-Z]*\b', candidate) and len(letters) < 8:
        return False

    if re.search(r'\bDL\s*NO\b|\bGSTIN\b|\bPAN\b|\bINVOICE\b|\bDATE\b|\bIRN\b|\bVEHICLE\b', candidate, re.IGNORECASE):
        return False

    return True


def _extract_company_candidates(
    rows: List[List[Dict]],
    max_rows: int = 25,
    require_legal_suffix: bool = True,
) -> List[str]:
    """Extract company-name candidates with strict or relaxed legal-suffix mode."""
    candidates: List[Tuple[float, str]] = []
    legal_suffix = re.compile(
        r'\b(PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|LIMITED|LTD\.?|LLP|LLC|INC\.?|CORP\.?|CORPORATION|'
        r'ENTERPRISES?|TRADERS?|DISTRIBUTORS?|HOLDING|MANUFACTURING|PHARMA|CHEMICALS?|'
        r'COMPANY|CO\.?|ASSOCIATES?|GROUP|SERVICES?|SOLUTIONS?|SYSTEMS?)\b',
        re.IGNORECASE,
    )

    exclude_patterns = re.compile(
        r'(^CIN\s+NO|EMAIL|ADDRESS|PHONE|MOBILE|FAX|WEBSITE|WAREHOUSE|OFFICE|'
        r'FLOOR|BUILDING|PLOT|ROAD|STREET|CITY|STATE|PIN|GSTIN|PAN|TAN|'
        r'INVOICE|BILL|SHIP|CUSTOMER\s*DETAILS?|SUBJECT|REF\.|DATE|DOOR|HOUSE|'
        r'APARTMENT|FLAT|SUITE|ATTN|C/O|DIN|@|INV NO|IRN|TRANSPORT|VEHICLE|DL\s*NO|FSSAI)',
        re.IGNORECASE,
    )

    for row_idx, row in enumerate(rows[:max_rows]):
        row_text = ' '.join(c.get('text', '') for c in row).strip()
        row_text = re.sub(r'\s+', ' ', row_text)
        if not row_text or len(row_text) < 8:
            continue

        has_legal_suffix = bool(legal_suffix.search(row_text))
        if require_legal_suffix and not has_legal_suffix:
            continue

        if exclude_patterns.search(row_text):
            continue

        if not _looks_like_company_name(row_text):
            continue

        special_count = len(re.findall(r'[^\w\s\.\-\&\']', row_text))
        if special_count > 3:
            continue

        letters = re.sub(r'[^A-Za-z]', '', row_text)
        upper_letters = re.sub(r'[^A-Z]', '', row_text)
        upper_ratio = (len(upper_letters) / len(letters)) if letters else 0.0

        score = 0.0
        score += max(0.0, 2.0 - (row_idx * 0.1))
        if 10 <= len(row_text) <= 80:
            score += 1.5
        if 0.5 < upper_ratio < 0.95:
            score += 1.0
        elif upper_ratio >= 0.8:
            score += 1.2

        if has_legal_suffix:
            score += 3.0
        elif not require_legal_suffix:
            score += 0.8

        numbers = len(re.findall(r'\d', row_text))
        if numbers > 3:
            score -= 0.5

        candidates.append((score, row_text))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)

    deduped: List[str] = []
    seen_keys = set()
    for _, text in candidates:
        key = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(text)

    return deduped


def _extract_bill_to_block_lines(page_rows: List[List[Dict]]) -> List[str]:
    """Extract only customer-details block lines from the left section of page."""
    in_block = False
    lines: List[str] = []
    left_block_limit: Optional[float] = None

    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row).strip()
        if not row_text:
            continue

        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER\s*DETAILS?', row_text, re.IGNORECASE):
            in_block = True

            right_markers = [
                c.get("x1")
                for c in row
                if re.search(r'IRN|TYPE\s*OF\s*SALE|PLACE\s*OF\s*SALE|CUST\.?\s*ORDER', c.get("text", ""), re.IGNORECASE)
            ]
            if right_markers:
                left_block_limit = min(right_markers) - 20
            continue

        if not in_block:
            continue

        if re.search(r'PRODUCT\s*DESCRIPTION|HSN|BATCH|QTY|TAXABLE|CGST|SGST|IGST', row_text, re.IGNORECASE):
            break

        if re.search(r'TOTAL\s*AMOUNT|GRAND\s*TOTAL|TERMS\s*&?\s*CONDITION', row_text, re.IGNORECASE):
            break

        if left_block_limit is not None:
            left_cells = [c for c in row if c.get("x1", 0) < left_block_limit]
        else:
            left_cells = [c for c in row if c.get("x1", 0) < 900]

        if not left_cells:
            continue

        line = " ".join(c.get("text", "") for c in left_cells).strip()
        line = re.sub(r'\s+', ' ', line)
        if not line:
            continue

        # Ignore known labels while keeping actual values.
        if re.fullmatch(r'CUSTOMER\s*DETAILS\s*:?', line, flags=re.IGNORECASE):
            continue

        lines.append(line)

    return lines


def _extract_supplier_header_left_lines(page_rows: List[List[Dict]]) -> List[str]:
    """Extract likely supplier-header lines from left side before customer-details section."""
    lines: List[str] = []
    bill_to_x1: Optional[float] = None

    for row in page_rows:
        for cell in row:
            if re.search(r'\bBILL\s*TO\b|\bSHIP\s*TO\b', cell.get("text", ""), re.IGNORECASE):
                x_val = cell.get("x1")
                if isinstance(x_val, (int, float)):
                    bill_to_x1 = x_val if bill_to_x1 is None else min(bill_to_x1, x_val)

    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row).strip()
        if not row_text:
            continue

        if re.search(r'PRODUCT\s*DESCRIPTION|HSN\s*CODE|BATCH\s*NO|TOTAL\s*AMOUNT', row_text, re.IGNORECASE):
            break

        sorted_row = sorted(row, key=lambda c: c.get("x1", 0))
        if not sorted_row:
            continue

        if bill_to_x1 is not None:
            left_cells = [c for c in sorted_row if c.get("x1", 0) < bill_to_x1 - 15]
        else:
            right_markers = [
                c.get("x1", 0)
                for c in sorted_row
                if re.search(
                    r'\bDL\s*NO\b|\bGSTIN\b|\bPAN\b|\bINVOICE\b|\bDUE\s*DATE\b|\bSTORE\s*TRN\b|\bAGAINST\s*ORDER\b|\bTYPE\s*OF\s*SALE\b',
                    c.get("text", ""),
                    re.IGNORECASE,
                )
            ]
            if right_markers:
                boundary = min(right_markers) - 20
                left_cells = [c for c in sorted_row if c.get("x1", 0) < boundary]
            else:
                left_cells = sorted_row

        if not left_cells:
            continue

        line = " ".join(c.get("text", "") for c in left_cells).strip()
        line = re.sub(r'\s+', ' ', line)
        if len(line) < 5:
            continue

        if re.search(r'ORIGINAL\s*FOR\s*BUYER|TAX\s*INVOICE', line, re.IGNORECASE):
            continue

        lines.append(line)

    return lines


def _extract_company_names_from_header(page_rows: List[List[Dict]]) -> List[str]:
    """Extract company names from header section (before bill-to section)."""
    header_rows: List[List[Dict]] = []
    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row)
        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS|DELIVERY\s*ADDRESS', row_text, re.IGNORECASE):
            break
        header_rows.append(row)

    search_rows = header_rows if header_rows else page_rows

    # Priority pass: isolate left supplier header block and extract from that only.
    header_left_lines = _extract_supplier_header_left_lines(search_rows)
    if header_left_lines:
        pseudo_rows: List[List[Dict]] = [
            [{
                "text": line,
                "x1": 0,
                "x2": 1,
                "xc": 0,
                "y1": 0,
                "y2": 1,
                "yc": 0,
                "conf": 1.0,
            }]
            for line in header_left_lines
        ]

        strict_left = _extract_company_candidates(
            pseudo_rows,
            max_rows=10,
            require_legal_suffix=True,
        )
        if strict_left:
            return strict_left[:1]

        relaxed_left = _extract_company_candidates(
            pseudo_rows,
            max_rows=10,
            require_legal_suffix=False,
        )
        if relaxed_left:
            return relaxed_left[:1]

    # Pass 1: strict legal-suffix extraction.
    strict_candidates = _extract_company_candidates(
        search_rows,
        max_rows=25,
        require_legal_suffix=True,
    )
    if strict_candidates:
        return strict_candidates[:1]

    # Pass 2: relaxed fallback (no mandatory suffix).
    relaxed_candidates = _extract_company_candidates(
        search_rows,
        max_rows=25,
        require_legal_suffix=False,
    )
    if relaxed_candidates:
        print("  [INFO] Company name extracted via relaxed fallback (no legal suffix detected)")
        return relaxed_candidates[:1]

    print("  [WARN] Company name could not be extracted from header")
    return []


def _extract_company_names_from_bill_to(page_rows: List[List[Dict]]) -> List[str]:
    block_lines = _extract_bill_to_block_lines(page_rows)
    if not block_lines:
        return []

    # Build pseudo-rows so we can reuse the existing scoring pipeline.
    pseudo_rows: List[List[Dict]] = []
    for line in block_lines:
        pseudo_rows.append([
            {
                "text": line,
                "x1": 0,
                "x2": 1,
                "xc": 0,
                "y1": 0,
                "y2": 1,
                "yc": 0,
                "conf": 1.0,
            }
        ])

    names = _extract_company_candidates(pseudo_rows, max_rows=12, require_legal_suffix=False)[:3]
    if names:
        return names

    # Fallback: first meaningful line with letters and no strong label pattern.
    for line in block_lines:
        if re.search(r'STATE\s*:?|GSTIN\s*:?|PAN\s*:?|DL\s*NO\s*:?|FSSAI\s*:?', line, re.IGNORECASE):
            continue
        if re.search(r'[A-Za-z]{4,}', line) and not re.search(r'\d{6,}', line):
            return [line]

    return []


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: HEADER EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_invoice_header(page_rows: List[List[Dict]]) -> Tuple[Dict, float]:
   #Extract invoice header fields with guaranteed structure.
    flat = flatten_cells(page_rows)
    scorer = AccuracyScorer()
    cleaner = DataCleaner()
    
    # Initialize with all fields as None
    header = {field: None for field in HEADER_FIELDS}
    header["company_name"] = None
    
    # ─── Company Name ──────────────────────────────────────────────────────
    company_names = _extract_company_names_from_header(page_rows)
    if company_names:
        header["company_name"] = cleaner.clean_string(company_names[0])
    
    # ─── PAN (Supplier) ────────────────────────────────────────────────────
    pan_text, _ = _extract_labeled_value(
        page_rows,
        r'^PAN\s*$|\bPAN\b',
        validator=_is_valid_pan_text,
    )
    if pan_text:
        header["pan"] = cleaner.extract_pan(pan_text)
    
    # ─── Invoice Number ────────────────────────────────────────────────────
    inv_no, _ = _extract_labeled_value(
        page_rows,
        r'INVOICE\s*NO|INV\.?\s*NO',
        validator=_is_valid_invoice_no_text,
    )
    if inv_no:
        header["invoice_no"] = cleaner.clean_invoice_no(inv_no)
    
    # ─── Invoice Date ────────────────────────────────────────────────────
    inv_date, _ = _extract_labeled_value(
        page_rows,
        r'INVOICE\s*DATE|\bDATE\b',
        validator=lambda t: DataCleaner.clean_date(t) is not None,
    )
    if inv_date:
        header["invoice_date"] = cleaner.clean_date(inv_date)
    
    # ─── Due Date ────────────────────────────────────────────────────────
    due_date, _ = _extract_labeled_value(
        page_rows,
        r'DUE\s*DATE',
        validator=lambda t: DataCleaner.clean_date(t) is not None,
    )
    if due_date:
        header["due_date"] = cleaner.clean_date(due_date)
    
    # Calculate accuracy
    accuracy = scorer.score_section(header)
    
    return header, accuracy


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7: BILL-TO EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_bill_to(page_rows: List[List[Dict]]) -> Tuple[Dict, float]:
    """
    Extract bill-to information.
    Always returns dict with all BILL_TO_FIELDS present (null if not found).
    Returns: (bill_to_dict, accuracy_score)
    """
    scorer = AccuracyScorer()
    cleaner = DataCleaner()
    
    # Initialize with all fields as None
    bill_to = {field: None for field in BILL_TO_FIELDS}
    bill_to["name"] = []
    
    # ─── Extract Bill-To block lines ────────────────────────────────────
    block_lines = _extract_bill_to_block_lines(page_rows)
    
    bill_names = _extract_company_names_from_bill_to(page_rows)
    bill_to["name"] = [n for n in (cleaner.clean_string(x) for x in bill_names) if n]

    if block_lines:
        filtered_lines = []
        for line in block_lines:
            if any(re.search(rf'^{re.escape(name)}$', line, re.IGNORECASE) for name in bill_to["name"]):
                continue
            if re.search(r'^(STATE|STATE\s*CODE|GSTIN\s*NO|PAN\s*NO|DL\s*NO|FSSAI\s*NO)\b', line, re.IGNORECASE):
                continue
            filtered_lines.append(line)

        if filtered_lines:
            bill_to["address"] = " ".join(filtered_lines)
    
    # ─── Bill Date (PO/Order date if available) ─────────────────────────
    bill_date, _ = _extract_labeled_value(
        page_rows,
        r'PO.*DATE|ORDER.*DATE',
        validator=lambda t: DataCleaner.clean_date(t) is not None,
    )
    if bill_date:
        bill_to["date"] = cleaner.clean_date(bill_date)
    
    # Calculate accuracy
    accuracy = scorer.score_section(bill_to)
    
    return bill_to, accuracy


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 8: LINE ITEMS EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_line_items(all_page_rows: List[List[List[Dict]]]) -> Tuple[List[Dict], List[float]]:
    """
    Extract line items with guaranteed field structure.
    Every item returned has all LINE_ITEM_FIELDS (null if not found).
    Returns: (items_list, accuracy_scores_list)
    """
    scorer = AccuracyScorer()
    cleaner = DataCleaner()

    col_positions = _detect_column_positions(all_page_rows)
    detected_cols = set(col_positions.keys())
    print(f"  Detected item columns: {sorted(col_positions.keys())}")

    items = []
    item_accuracies = []
    current_item = None
    item_index = 0
    in_table = False
    table_ended = False

    for page_rows in all_page_rows:
        for row in page_rows:
            row_text = " ".join(c["text"] for c in row)

            row_header_hits = {
                _match_column_from_text(c.get("text", ""))
                for c in row
                if _match_column_from_text(c.get("text", ""))
            }

            # ─── Detect table start ──────────────────────────────────────
            if len(row_header_hits) >= 3 and (
                "product_description" in row_header_hits or "hsn_code" in row_header_hits
            ):
                in_table = True
                continue

            if not in_table:
                continue

            # ─── Detect end of items table ─────────────────────────────
            if re.search(r'TOTAL.*AMOUNT|GRAND.*TOTAL|AMOUNT.*WORDS|TERMS.*CONDITION',
                        row_text, re.IGNORECASE):
                if current_item:
                    items.append(current_item)
                    item_accuracies.append(
                        scorer.score_item(current_item)
                    )
                    current_item = None
                table_ended = True
                break

            # ─── Skip header and noise rows ──────────────────────────────
            if re.search(r'Goods.*sold|Interest.*per|Company.*staff|disputes',
                        row_text, re.IGNORECASE):
                continue

            # ─── Map current row values to detected columns ─────────────
            row_values = _assign_row_to_columns(row, col_positions)

            # ─── Detect new item: prefer HSN column, fallback regex ──────
            hsn_candidate = row_values.get("hsn_code")
            hsn_match = re.search(r'\b(\d{8})\b', hsn_candidate or row_text)

            if hsn_match:
                # Save previous item
                if current_item:
                    items.append(current_item)
                    item_accuracies.append(scorer.score_item(current_item))

                # Create new item with ALL fields
                current_item = {field: None for field in LINE_ITEM_FIELDS}
                current_item["row_index"] = item_index
                item_index += 1

                # Set HSN immediately
                current_item["hsn_code"] = hsn_match.group(1)

                # Prefer product column; fallback to text before HSN
                product_text = row_values.get("product_description")
                if product_text:
                    product_text = re.sub(r'^\s*\d+\s*', '', product_text)
                    product_text = re.sub(r'\s+', ' ', product_text).strip(' :-|')
                if not product_text:
                    left_text = row_text.split(hsn_match.group(1), 1)[0]
                    product_text = re.sub(r'^\s*\d+\s*', '', left_text)
                    product_text = re.sub(r'\s+', ' ', product_text).strip(' :-|')
                if product_text and re.search(r'[A-Za-z]{2,}', product_text):
                    current_item["product_description"] = cleaner.clean_string(product_text)

            if current_item is None:
                continue

            # ─── Field extraction from mapped columns ───────────────────
            if not current_item["product_description"]:
                product_text = row_values.get("product_description")
                if product_text and not re.search(r'PRODUCT|DESCRIPTION', product_text, re.IGNORECASE):
                    product_text = _clean_product_description(product_text, row_values)
                    if product_text and re.search(r'[A-Za-z]{2,}', product_text):
                        current_item["product_description"] = cleaner.clean_string(product_text)

            if not current_item["batch_no"] and row_values.get("batch_no"):
                batch_text = row_values.get("batch_no")
                batch_match = re.search(r'[A-Z0-9\']{4,12}', str(batch_text).upper())
                if batch_match:
                    current_item["batch_no"] = batch_match.group(0)

            if not current_item["expiry_date"] and row_values.get("expiry_date"):
                exp_text = row_values.get("expiry_date")
                exp_match = re.search(r'(\d{1,2}/\d{2,4}|[A-Za-z]{3}-\d{4})', str(exp_text))
                if exp_match:
                    current_item["expiry_date"] = exp_match.group(1)

            if current_item["qty"] is None and row_values.get("qty"):
                qty_num = cleaner.extract_number(row_values.get("qty"))
                if qty_num is not None:
                    current_item["qty"] = int(qty_num)

            if current_item["uom"] is None and row_values.get("uom"):
                uom_text = re.sub(r'[^A-Za-z]', '', row_values.get("uom", "")).upper()
                current_item["uom"] = uom_text if uom_text else None

            numeric_field_map = {
                "mrp": "mrp",
                "ptr": "ptr",
                "pts": "pts",
                "discount": "discount",
                "cgst": "cgst",
                "sgst": "sgst",
                "total_amount": "total_amount",
            }
            for col_name, item_key in numeric_field_map.items():
                if col_name not in detected_cols:
                    continue
                if current_item[item_key] is None and row_values.get(col_name):
                    parsed = cleaner.extract_number(row_values.get(col_name))
                    if parsed is not None:
                        current_item[item_key] = _round2(parsed)

            if current_item["discount"] is None:
                parsed_discount = _extract_discount_from_row_text(row_text)
                if parsed_discount is not None:
                    current_item["discount"] = parsed_discount

            # Prefer taxable value extraction for total_amount when available.
            taxable_value = _extract_taxable_value_from_row_text(row_text)
            if taxable_value is not None:
                current_item["total_amount"] = taxable_value

            # Improve GST parsing: prefer inferred repeated rate values from row text.
            inferred_cgst, inferred_sgst = _extract_tax_rates_from_row_text(row_text)
            if inferred_cgst is not None and inferred_sgst is not None:
                current_cgst = current_item.get("cgst")
                current_sgst = current_item.get("sgst")
                need_override = (
                    current_cgst is None
                    or current_sgst is None
                    or current_cgst > 28
                    or current_sgst > 28
                    or abs(float(current_cgst) - float(current_sgst)) > 0.5
                )
                if need_override:
                    current_item["cgst"] = inferred_cgst
                    current_item["sgst"] = inferred_sgst

            # Fallbacks from raw row text for key fields when column mapping misses.
            if current_item["expiry_date"] is None:
                exp_fallback = _extract_expiry_from_text(row_text)
                if exp_fallback:
                    current_item["expiry_date"] = exp_fallback

            if current_item["expiry_date"] is not None:
                normalized_exp = _extract_expiry_from_text(str(current_item["expiry_date"]))
                if normalized_exp:
                    current_item["expiry_date"] = normalized_exp

            if current_item["batch_no"] is None:
                batch_fallback = _extract_batch_from_text(row_text, current_item.get("hsn_code"))
                if batch_fallback:
                    current_item["batch_no"] = batch_fallback

            if current_item["pts"] is None and "pts" in detected_cols:
                pts_fallback = _extract_pts_from_row_text(row_text)
                if pts_fallback is not None:
                    current_item["pts"] = pts_fallback

            total_fallback = _extract_total_amount_from_row_text(row_text)
            if total_fallback is not None:
                if current_item["total_amount"] is None:
                    current_item["total_amount"] = total_fallback
                else:
                    try:
                        mapped_total = float(current_item["total_amount"])
                    except (ValueError, TypeError):
                        mapped_total = 0.0

                    # Keep taxable-value parse; only use row-end fallback if current looks invalid.
                    if mapped_total <= 0:
                        current_item["total_amount"] = total_fallback

            # Fallback for product name from continuation row text
            if not current_item["product_description"]:
                cleaned_row = re.sub(r'\b\d+(?:\.\d+)?\b', ' ', row_text)
                cleaned_row = re.sub(r'\s+', ' ', cleaned_row).strip(' :-|')
                if re.search(r'[A-Za-z]{3,}', cleaned_row) and not re.search(
                    r'PRODUCT|HSN|BATCH|QTY|TOTAL|AMOUNT|CGST|SGST|IGST|TERMS',
                    cleaned_row,
                    re.IGNORECASE,
                ):
                    cleaned_row = _clean_product_description(cleaned_row, row_values) or cleaned_row
                    current_item["product_description"] = cleaner.clean_string(cleaned_row)

        if table_ended:
            break
    
    # Don't forget last item
    if current_item:
        items.append(current_item)
        item_accuracies.append(scorer.score_item(current_item))
    
    # Add accuracy scores to items
    for i, item in enumerate(items):
        if i < len(item_accuracies):
            item["accuracy_score"] = item_accuracies[i]
        else:
            item["accuracy_score"] = 0.0
    
    return items, item_accuracies


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 9: MAIN PROCESSING PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """Main processing pipeline"""
    pdf_path = Path(pdf_path).resolve()
    
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        return None
    
    stem = pdf_path.stem
    print(f"\n{'='*70}")
    print(f"  Processing: {pdf_path.name}")
    print(f"{'='*70}\n")
    
    images_dir = BASE_DIR / "images" / stem
    output_dir = BASE_DIR / "output" / stem
    
    # Step 1: PDF to Images
    print("[1/4] Converting PDF to images...")
    image_paths = pdf_to_images(pdf_path, images_dir)
    
    # Step 2: OCR
    print("\n[2/4] Running OCR...")
    all_ocr = run_ocr(image_paths)
    
    # Step 3: Extraction
    print("\n[3/4] Extracting invoice data...")
    page_rows = [group_rows(p) for p in all_ocr]
    
    header, header_acc = extract_invoice_header(page_rows[0])
    bill_to, bill_to_acc = extract_bill_to(page_rows[0])
    line_items, item_accs = extract_line_items(page_rows)

    table_rows_text = _collect_table_rows_text(page_rows)
    line_items = refine_line_items_with_openrouter(line_items, table_rows_text, header, bill_to)
    line_items = normalize_line_items(line_items)
    
    print(f"  ✓ Header extracted (accuracy: {header_acc}%)")
    print(f"  ✓ Bill-to extracted (accuracy: {bill_to_acc}%)")
    print(f"  ✓ {len(line_items)} line items extracted")
    
    # Step 4: Compile output
    print("\n[4/4] Compiling output...")
    
    output_data = {
        "source_file": pdf_path.name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "invoice_header": header,
        "bill_to": bill_to,
        "line_items": line_items,
        "summary": {
            "total_line_items": len(line_items),
        }
    }
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Output saved: {json_path}")
    print(f"\nSummary:")
    print(f"  Header Accuracy:  {header_acc:.2f}%")
    print(f"  Bill-To Accuracy: {bill_to_acc:.2f}%")
    print(f"  Items Accuracy:   {sum(item_accs)/len(item_accs):.2f}%" if item_accs else "  Items: None")
    all_accuracies = [header_acc, bill_to_acc] + item_accs
    overall_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0
    print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
    
    return output_data


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python processor_production_v3.py <invoice.pdf>")
        sys.exit(1)
    
    result = process_pdf(sys.argv[1])