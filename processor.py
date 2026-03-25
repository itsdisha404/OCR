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
        "expiry", "exp", "exp date", "expiry date"
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
        "amount", "total", "net amount", "taxable"
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
        Calculate accuracy for a line item.
        Score based on:
        - Non-null fields: 80%
        - Field quality: 20%
        """
        if not item_dict:
            return 0.0
        
        # Critical fields for pharmaceutical items
        critical_fields = [
            'product_description', 'hsn_code', 'batch_no', 
            'qty', 'mrp', 'pts', 'cgst', 'sgst'
        ]
        
        # Count non-null critical fields
        non_null_critical = sum(
            1 for f in critical_fields 
            if item_dict.get(f) is not None
        )
        
        critical_ratio = non_null_critical / len(critical_fields)
        
        # Quality check: verify numeric fields are actually numbers
        quality_score = 1.0
        numeric_fields = ['qty', 'mrp', 'ptr', 'pts', 'cgst', 'sgst', 'igst', 'discount']
        for field in numeric_fields:
            val = item_dict.get(field)
            if val is not None and not isinstance(val, (int, float)):
                quality_score -= 0.1
        
        overall_accuracy = (critical_ratio * 80) + (quality_score * 20)
        return round(min(100, max(0, overall_accuracy)), 2)
    
    def score_section(self, section_data: Dict[str, Any]) -> float:
        """Score extraction for header or bill-to section"""
        if not section_data:
            return 0.0

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
        
        if total_fields == 0:
            return 0.0
        
        return round((non_null_fields / total_fields) * 100, 2)


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
    """Extract expiry date in MM/YYYY-like forms from row text."""
    if not text:
        return None

    mm_yyyy = re.search(r"\b(0[1-9]|1[0-2])[/-](\d{2,4})\b", text)
    if mm_yyyy:
        month, year = mm_yyyy.groups()
        if len(year) == 2:
            year = f"20{year}"
        return f"{month}/{year}"

    mon_yyyy = re.search(r"\b([A-Za-z]{3})[-/](\d{2,4})\b", text)
    if mon_yyyy:
        mon, year = mon_yyyy.groups()
        if len(year) == 2:
            year = f"20{year}"
        return f"{mon.title()}-{year}"

    return None


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

    candidate = values[-1]
    return _round2(candidate) if candidate > 0 else None


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


def _is_valid_company_name(name: Optional[str]) -> bool:
    """
    Validate that a string is a legitimate company name and not a label/identifier.
    Rejects common label patterns like 'CIN NO_', 'DL NO_', 'GSTIN', 'PAN NO_', etc.
    """
    if not name or not isinstance(name, str):
        return False

    name = name.strip()
    if not name:
        return False

    name_upper = name.upper()

    # List of patterns that indicate this is a label, not a company name
    invalid_patterns = [
        r'^CIN\s*NO',      # CIN NO, CIN NO_
        r'^DL\s*NO',       # DL NO, DL NO_
        r'^GSTIN',         # GSTIN
        r'^PAN\s*NO',      # PAN NO
        r'^PAN:',          # PAN: xxx
        r'^TAN',           # TAN
        r'^FSSAI',         # FSSAI
        r'^IEC\s*CODE',    # IEC CODE
        r'^IEC\s*NO',      # IEC NO
        r'^REG\s*NO',      # REG NO
        r'^REG\s*CODE',    # REG CODE
        r'^LICENSE',       # LICENSE
        r'^LICENCE',       # LICENCE
        r'^AUTH',          # AUTH
        r'^CERT',          # CERT
        r'^CODE\s*NO',     # CODE NO
        r'^ID\s*NO',       # ID NO
        r'^NO\.',          # NO. (standalone)
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, name_upper):
            return False

    # Must start with a letter OR have at least 4 letters (to avoid "123ABC" type strings)
    starts_with_letter = re.match(r'^[a-zA-Z]', name)
    has_min_letters = len(re.findall(r'[a-zA-Z]', name)) >= 4

    if not (starts_with_letter or has_min_letters):
        return False

    return True


def refine_header_with_openrouter(
    header: Dict[str, Any],
    bill_to: Dict[str, Any],
    line_items: List[Dict[str, Any]],
    ocr_text_sample: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Refine OCR-extracted header and bill-to using OpenRouter LLM.
    Uses full context including line items to intelligently segregate metadata.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return header, bill_to

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # Prepare line items summary for context
    line_items_summary = []
    for i, item in enumerate(line_items[:5]):  # First 5 items for context
        summary = {
            "product": item.get("product_description"),
            "hsn": item.get("hsn_code"),
            "qty": item.get("qty"),
            "mrp": item.get("mrp"),
            "pts": item.get("pts"),
        }
        line_items_summary.append(summary)

    system_prompt = (
        "You are an expert invoice header parser. Your task is to intelligently analyze and refine "
        "OCR-extracted invoice header data by segregating supplier company name from customer company name, "
        "addresses, and other metadata.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY a valid JSON object - no explanations, markdown, or code fences.\n"
        "2. SUPPLIER COMPANY (header.company_name): Usually appears at TOP of invoice with legal suffix (PVT LTD, LIMITED, etc)\n"
        "3. CUSTOMER COMPANY (bill_to.name): Appears in 'BILL TO', 'SHIP TO', or 'CUSTOMER DETAILS' section\n"
        "4. NEVER use addresses as company names - avoid: PLOT, ROAD, STREET, AREA, GANJ, SECTOR, INDUSTRIAL, etc.\n"
        "5. NEVER use registration/license identifiers as company names - reject: CIN NO, DL NO, GSTIN, PAN NO, TAN, FSSAI, IEC, LICENSE, CERT, REG NO, etc.\n"
        "6. Use provided line_items context (products, HSN codes) to understand document domain\n"
        "7. Extract exact values from OCR text when possible\n"
        "8. Use null for any field you cannot reliably extract\n"
        "9. Return company names as simple strings (name) and address as single string (address)\n"
        "10. If you find only fragments or label patterns where company name should be, return null for that field.\n\n"
        "REQUIRED OUTPUT FORMAT:\n"
        "{\n"
        '  "header": {\n'
        '    "company_name": "SUPPLIER COMPANY NAME PVT LTD",\n'
        '    "pan": "AAFFE3923M",\n'
        '    "invoice_no": "INV/2024/12345",\n'
        '    "invoice_date": "13/11/2024",\n'
        '    "due_date": "28/11/2024"\n'
        "  },\n"
        '  "bill_to": {\n'
        '    "name": "CUSTOMER COMPANY NAME PVT LTD",\n'
        '    "address": "PLOT NO 321, SECTOR 10, DELHI 110092, PAN: AAHCC7397L, DL NO: 2B/DLIPTG/123962",\n'
        '    "date": null\n'
        "  }\n"
        "}\n"
    )

    user_prompt = {
        "instructions": [
            "Step 1: Identify the SUPPLIER (top of invoice, usually company name with legal suffix like PVT LTD/LIMITED).",
            "Step 2: Identify the CUSTOMER (from Bill To/Ship To/Customer Details section).",
            "Step 3: Separate ADDRESSES from COMPANY NAMES - addresses contain: PLOT, ROAD, STREET, ROAD, INDUSTRIAL AREA, city names, etc.",
            "Step 4: Extract and validate PAN, Invoice Number, and Dates.",
            "Step 5: Use line items context (products, domains) to validate company extraction.",
            "Step 6: Return clean, segregated JSON with company names separate from addresses.",
        ],
        "ocr_raw_text": ocr_text_sample[:2000],  # First 2000 chars of raw OCR
        "currently_extracted": {
            "header": header,
            "bill_to": bill_to,
        },
        "line_items_sample": {
            "count": len(line_items),
            "sample_items": line_items_summary,
            "has_pharmaceutical_context": any(
                any(
                    keyword in str(item.get("product_description", "")).lower()
                    for keyword in ["tablet", "capsule", "syrup", "injection", "medicine", "pharma"]
                )
                for item in line_items[:5]
            ),
        },
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
        with request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"].strip()
        # Remove markdown code blocks if present
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()
        refined = json.loads(content)

        if not isinstance(refined, dict) or "header" not in refined or "bill_to" not in refined:
            print("  [LLM] Header refinement ignored: invalid JSON format")
            return header, bill_to

        # Validate and merge refined header data
        refined_header = dict(header)
        refined_bill_to = dict(bill_to)

        # Process refined header
        if isinstance(refined.get("header"), dict):
            refined_header_data = refined["header"]

            # Company name (string or list) - with validation to reject labels
            if "company_name" in refined_header_data and refined_header_data["company_name"]:
                company_name = refined_header_data["company_name"]
                assigned = False

                if isinstance(company_name, str):
                    if _is_valid_company_name(company_name):
                        refined_header["company_name"] = company_name.strip()
                        assigned = True
                    else:
                        print(f"    [VALIDATION] Rejected invalid company_name: '{company_name}'")
                elif isinstance(company_name, list):
                    # If list, take first valid item
                    for name in company_name:
                        if isinstance(name, str) and _is_valid_company_name(name):
                            refined_header["company_name"] = name.strip()
                            assigned = True
                            break
                    if not assigned and company_name:
                        print(f"    [VALIDATION] No valid company_name found in list: {company_name}")

            # PAN
            if "pan" in refined_header_data and refined_header_data["pan"]:
                refined_header["pan"] = refined_header_data["pan"]

            # Invoice number
            if "invoice_no" in refined_header_data and refined_header_data["invoice_no"]:
                refined_header["invoice_no"] = refined_header_data["invoice_no"]

            # Invoice date
            if "invoice_date" in refined_header_data and refined_header_data["invoice_date"]:
                refined_header["invoice_date"] = refined_header_data["invoice_date"]

            # Due date
            if "due_date" in refined_header_data and refined_header_data["due_date"]:
                refined_header["due_date"] = refined_header_data["due_date"]

        # Process refined bill_to
        if isinstance(refined.get("bill_to"), dict):
            refined_bill_to_data = refined["bill_to"]

            # Customer name (string preferred, but handle list) - with validation
            if "name" in refined_bill_to_data:
                name_val = refined_bill_to_data["name"]
                assigned = False

                if isinstance(name_val, str):
                    if _is_valid_company_name(name_val):
                        refined_bill_to["name"] = name_val.strip()
                        assigned = True
                    else:
                        print(f"    [VALIDATION] Rejected invalid bill_to.name: '{name_val}'")
                        refined_bill_to["name"] = None
                elif isinstance(name_val, list) and name_val:
                    # Handle list: take first valid item
                    for name in name_val:
                        if isinstance(name, str) and _is_valid_company_name(name):
                            refined_bill_to["name"] = name.strip()
                            assigned = True
                            break
                    if not assigned:
                        refined_bill_to["name"] = None
                elif not name_val:
                    refined_bill_to["name"] = None

            # Address
            if "address" in refined_bill_to_data and refined_bill_to_data["address"]:
                refined_bill_to["address"] = refined_bill_to_data["address"]

            # Date
            if "date" in refined_bill_to_data and refined_bill_to_data["date"]:
                refined_bill_to["date"] = refined_bill_to_data["date"]

        print("  [LLM] Header and Bill-To refinement applied successfully")
        return refined_header, refined_bill_to

    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [LLM] Header refinement skipped: {exc}")
        return header, bill_to
    except json.JSONDecodeError as e:
        print(f"  [LLM] Failed to parse JSON response: {e}")
        return header, bill_to
    except Exception as e:
        print(f"  [LLM] Unexpected error during header refinement: {e}")
        return header, bill_to


def refine_line_items_with_openrouter(
    line_items: List[Dict[str, Any]],
    table_rows_text: List[str],
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
        "You are an invoice line-item correction engine. "
        "Return ONLY valid JSON array. "
        "Keep same item count and row_index values. "
        "Do not hallucinate values not inferable from provided rows. "
        "Fields allowed: row_index, product_description, hsn_code, batch_no, expiry_date, qty, uom, "
        "mrp, ptr, pts, discount, cgst, sgst, total_amount, accuracy_score."
    )

    user_prompt = {
        "instructions": [
            "Correct numeric column assignments (mrp/ptr/pts/discount/cgst/sgst/total_amount).",
            "Preserve product_description and hsn_code when already plausible.",
            "Use null for unknown fields.",
            "Keep same list length and order."
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

    for number_key in ["mrp", "ptr", "pts", "discount", "total_amount"]:
        val = normalized.get(number_key)
        if val is not None:
            try:
                normalized[number_key] = _round2(float(val))
            except (ValueError, TypeError):
                normalized[number_key] = None

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
    candidate = re.sub(r'\s+', ' ', str(text or '').strip())
    if len(candidate) < 4 or len(candidate) > 90:
        return False

    if not re.search(r'[A-Za-z]{3,}', candidate):
        return False

    blocked = re.search(
        r'INVOICE|BILL\s*TO|SHIP\s*TO|GSTIN|PAN|DATE|DUE|PO\s*REF|ORDER|HSN|QTY|AMOUNT|TOTAL|CGST|SGST|TERMS|'
        r'CUSTOMER\s*DETAILS|IRN\s*NO|TYPE\s*OF\s*SALE|PLACE\s*OF\s*SALE|EWAY|TRANSPORT',
        candidate,
        re.IGNORECASE,
    )
    if blocked:
        return False

    address_like = re.search(
        r'\b(ROAD|RD|STREET|ST\.?|LANE|AREA|NAGAR|SECTOR|PHASE|BLOCK|FLOOR|BUILDING|BLDG|'
        r'NEAR|OPP|DISTRICT|CITY|STATE|COUNTRY|PIN|PINCODE|POSTAL|INDUSTRIAL)\b',
        candidate,
        re.IGNORECASE,
    )
    if address_like:
        return False

    if re.search(r'\b\d{6}\b', candidate):
        return False

    digits = len(re.findall(r'\d', candidate))
    if digits > max(3, int(len(candidate) * 0.25)):
        return False

    return True


def _extract_company_candidates(rows: List[List[Dict]], max_rows: int = 25) -> List[str]:
    candidates: List[Tuple[float, str]] = []
    legal_suffix = re.compile(
        r'\b(PVT\.?\s*LTD\.?|PRIVATE\s+LIMITED|LIMITED|LTD\.?|LLP|LLC|INC\.?|CORP\.?|CORPORATION|ENTERPRISES?|TRADERS?|DISTRIBUTORS?)\b',
        re.IGNORECASE,
    )

    for row_idx, row in enumerate(rows[:max_rows]):
        row_text = ' '.join(c.get('text', '') for c in row).strip()
        row_text = re.sub(r'\s+', ' ', row_text)
        if not _looks_like_company_name(row_text):
            continue

        letters = re.sub(r'[^A-Za-z]', '', row_text)
        upper_letters = re.sub(r'[^A-Z]', '', row_text)
        upper_ratio = (len(upper_letters) / len(letters)) if letters else 0.0

        score = 0.0
        score += max(0.0, 1.4 - (row_idx * 0.06))
        if 8 <= len(row_text) <= 60:
            score += 0.8
        if upper_ratio > 0.75:
            score += 1.0
        if legal_suffix.search(row_text):
            score += 2.0

        candidates.append((score, row_text))

    candidates.sort(key=lambda item: item[0], reverse=True)

    deduped: List[str] = []
    seen_keys = set()
    for _, text in candidates:
        key = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(text)

    # Prefer legal-entity lines if present; avoids selecting address-like remnants.
    legal_only = [t for t in deduped if legal_suffix.search(t)]
    if legal_only:
        return legal_only

    return deduped


def _extract_company_names_from_header(page_rows: List[List[Dict]]) -> List[str]:
    header_rows: List[List[Dict]] = []
    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row)
        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS', row_text, re.IGNORECASE):
            break
        header_rows.append(row)

    return _extract_company_candidates(header_rows if header_rows else page_rows, max_rows=25)[:3]


def _extract_company_names_from_bill_to(page_rows: List[List[Dict]]) -> List[str]:
    in_bill_to = False
    bill_rows: List[List[Dict]] = []

    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row)
        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS', row_text, re.IGNORECASE):
            in_bill_to = True
            continue

        if not in_bill_to:
            continue

        if re.search(r'PRODUCT|HSN|BATCH|QTY|TOTAL|AMOUNT|INVOICE|DUE', row_text, re.IGNORECASE):
            break

        bill_rows.append(row)

    return _extract_company_candidates(bill_rows, max_rows=15)[:3]


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
    header["company_name"] = []
    
    # ─── Company Name ──────────────────────────────────────────────────────
    company_names = _extract_company_names_from_header(page_rows)
    header["company_name"] = [n for n in (cleaner.clean_string(x) for x in company_names) if n]
    
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
    
    # ─── Extract Bill-To Address Lines ──────────────────────────────────
    addr_lines = []
    in_bill_to = False
    LEFT_MARGIN = 800  # Approximate left boundary
    
    for row in page_rows:
        row_text = " ".join(c["text"] for c in row)
        
        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS', row_text, re.IGNORECASE):
            in_bill_to = True
            continue
        
        if in_bill_to:
            # Stop at table headers or summary sections
            if re.search(r'PRODUCT|HSN|BATCH|QTY|TOTAL|AMOUNT|INVOICE|DUE', 
                        row_text, re.IGNORECASE):
                break
            
            # Collect address text from left side of page
            left_cells = [c for c in row if c["x1"] < LEFT_MARGIN]
            if left_cells:
                line = " ".join(c["text"] for c in left_cells).strip()
                if line and len(line) > 3:
                    addr_lines.append(line)
    
    bill_names = _extract_company_names_from_bill_to(page_rows)
    bill_to["name"] = [n for n in (cleaner.clean_string(x) for x in bill_names) if n]

    if addr_lines:
        if len(addr_lines) > 1:
            bill_to["address"] = " ".join(addr_lines[1:])
    
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
                    product_text = re.sub(r'^\s*\d+\s*', '', product_text)
                    product_text = re.sub(r'\s+', ' ', product_text).strip(' :-|')
                    if re.search(r'[A-Za-z]{2,}', product_text):
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

            # Fallbacks from raw row text for key fields when column mapping misses.
            if current_item["expiry_date"] is None:
                exp_fallback = _extract_expiry_from_text(row_text)
                if exp_fallback:
                    current_item["expiry_date"] = exp_fallback

            if current_item["batch_no"] is None:
                batch_fallback = _extract_batch_from_text(row_text, current_item.get("hsn_code"))
                if batch_fallback:
                    current_item["batch_no"] = batch_fallback

            if current_item["pts"] is None and "pts" in detected_cols:
                pts_fallback = _extract_pts_from_row_text(row_text)
                if pts_fallback is not None:
                    current_item["pts"] = pts_fallback

            if current_item["total_amount"] is None:
                total_fallback = _extract_total_amount_from_row_text(row_text)
                if total_fallback is not None:
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

    # Extract basic header, bill_to, and line items first
    header, header_acc = extract_invoice_header(page_rows[0])
    bill_to, bill_to_acc = extract_bill_to(page_rows[0])
    line_items, item_accs = extract_line_items(page_rows)

    # Prepare OCR text sample (first page, first 3000 chars)
    ocr_text_sample = ""
    if page_rows:
        for row in page_rows[0][:50]:  # First 50 rows
            row_text = " ".join(c.get("text", "") for c in row if c.get("text"))
            if row_text.strip():
                ocr_text_sample += row_text + "\n"

    # **CRITICAL: Refine header and bill_to using LLM with full context**
    # This uses line items and OCR text to intelligently segregate metadata
    print("  [LLM] Refining header and bill-to with line items context...")
    header, bill_to = refine_header_with_openrouter(header, bill_to, line_items, ocr_text_sample)

    # Refine line items
    table_rows_text = _collect_table_rows_text(page_rows)
    line_items = refine_line_items_with_openrouter(line_items, table_rows_text)
    line_items = normalize_line_items(line_items)
    
    print(f"  ✓ Header extracted (accuracy: {header_acc}%)")
    print(f"  ✓ Bill-to extracted (accuracy: {bill_to_acc}%)")
    print(f"  ✓ {len(line_items)} line items extracted")
    
    # Step 4: Compile output
    print("\n[4/4] Compiling output...")

    # Ensure company_name is properly formatted
    if isinstance(header.get("company_name"), list):
        header["company_name"] = header["company_name"][0] if header["company_name"] else None

    # Ensure bill_to.name is properly formatted
    if isinstance(bill_to.get("name"), list):
        bill_to["name"] = bill_to["name"][0] if bill_to["name"] else None

    # Remove internal fields from final output
    for item in line_items:
        item.pop("tp_value", None)
        item.pop("scheme_value", None)

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