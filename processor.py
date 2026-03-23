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
    "tp_value",
    "scheme_value",
    "discount",
    "cgst",
    "sgst",
    "total_amount",
    "accuracy_score",
]

COLUMN_SYNONYMS = {
    "product_description": [
        "product", "description", "product description", "material description",
        "item description", "item", "material"
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
    "tp_value": [
        "tp value", "tp val", "tp amount"
    ],
    "scheme_value": [
        "scheme value", "scheme val", "scheme amount", "scheme"
    ],
    "discount": [
        "discount", "disc", "td", "trade discount", "td%", "trade disc"
    ],
    "cgst": [
        "cgst"
    ],
    "sgst": [
        "sgst"
    ],
    "total_amount": [
        "transaction value", "trans value", "transaction amount", "trans amount",
        "final amount", "net amount", "line total", "amount", "total"
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
        numeric_fields = ['qty', 'mrp', 'ptr', 'pts', 'tp_value', 'scheme_value', 'cgst', 'sgst', 'igst', 'discount']
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


def _is_valid_hsn_context(text: str, hsn_code: str) -> bool:
    """Validate that HSN code appears in a valid product context, not address/phone."""
    if not text or not hsn_code:
        return False

    text_lower = text.lower()

    # Skip if context contains clear address indicators
    address_indicators = [
        'ph no:', 'phone:', 'tel:', 'contact:',
        'email:', 'www.', '@',
        'pvt ltd', 'private limited',
        'gstin:', 'pan no:', 'dl no:',
        'industrial area', 'ganj industrial'
    ]

    for indicator in address_indicators:
        if indicator in text_lower:
            return False

    # Skip if HSN is clearly part of a phone number pattern
    phone_patterns = [
        rf'\b0\d{{2,3}}\-{hsn_code}',  # 011-42419902 pattern
        rf'{hsn_code}\/9\d{{9}}',      # Phone with HSN prefix and 9xxxxxxxx
        rf'ph\s*no[\s:]*.*{hsn_code}', # "Ph No: ..." containing HSN
    ]

    for pattern in phone_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # If we get here and there are any product-like indicators, it's valid
    product_indicators = [
        'batch', 'exp', 'qty', 'box', 'tab', 'cap', 'mg', 'ml', 'gm',
        'mrp', 'ptr', 'pts', 'pack', 'strip', 'bottle', 'vial', 't)', 'tablet'
    ]

    has_product_context = any(indicator in text_lower for indicator in product_indicators)
    if has_product_context:
        return True

    # More lenient check - if there's text before HSN that looks like a product
    # and it's not obviously an address, allow it
    pre_hsn_text = text.split(hsn_code)[0].strip()
    if pre_hsn_text:
        # Remove leading numbers and clean up
        product_text = re.sub(r'^\s*\d+\s*', '', pre_hsn_text).strip()
        # If there's meaningful text (not just punctuation/spaces), consider it valid
        if len(product_text) >= 2 and re.search(r'[A-Za-z]{2,}', product_text):
            return True

    # If HSN appears with other numbers that look like product data, allow it
    numbers_in_text = re.findall(r'\d+(?:\.\d+)?', text)
    if len(numbers_in_text) >= 3:  # HSN + at least 2 other numbers (qty, prices, etc.)
        return True

    return False


def _extract_total_amount_from_row_text(text: str) -> Optional[float]:
    """Extract transaction value from the rightmost column only.
    Avoids extracting from multiple columns."""
    if not text:
        return None

    # Find ALL complete numeric values followed by space and uppercase letter
    # Then pick the rightmost one (which is the transaction value column)
    pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s+[A-Z]'
    matches = list(re.finditer(pattern, text))

    if matches:
        # Get the last match (rightmost)
        rightmost_match = matches[-1]
        try:
            value = float(rightmost_match.group(1).replace(",", ""))
            if 50.0 <= value <= 100000:
                return _round2(value)
        except ValueError:
            pass

    # Fallback: Look for explicit transaction/total value labels
    transaction_patterns = [
        r'(?:transaction|trans)\s*(?:value|val)[\s\-:]*(\d[\d,]*\.?\d*)',
        r'(\d[\d,]*\.?\d*)\s*transaction',
        r'final\s*(?:amount|value)[\s\-:]*(\d[\d,]*\.?\d*)',
    ]

    for pattern in transaction_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                if 10.0 <= value <= 100000:
                    return _round2(value)
            except ValueError:
                continue

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


def _extract_columns_from_complex_header(text: str, cell_x: float, cell_width: float = 100.0) -> Dict[str, float]:
    """Extract multiple column matches from a complex header cell with estimated positions."""
    normalized = _normalize_text(text)
    if not normalized:
        return {}

    matches = {}

    # Find all column matches with their positions in the text
    for col_name, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            syn_norm = _normalize_text(synonym)
            if syn_norm:
                match = re.search(rf'\b{re.escape(syn_norm)}\b', normalized)
                if match:
                    # Calculate estimated x position based on match position within text
                    text_pos_ratio = (match.start() + len(match.group())) / len(normalized)
                    estimated_x = cell_x + (text_pos_ratio * cell_width * 0.7)  # 0.7 factor for text spacing
                    matches[col_name] = estimated_x
                    break  # Only take first synonym match per column

    return matches


def _detect_column_positions(all_page_rows: List[List[List[Dict]]]) -> Dict[str, float]:
    """Detect column x-positions from table header rows using synonyms."""
    best_positions: Dict[str, float] = {}
    header_debug_info = []

    for page_rows in all_page_rows:
        for row in page_rows:
            row_matches: Dict[str, float] = {}
            row_debug = []

            for cell in row:
                cell_text = cell.get("text", "")
                cell_x = cell.get("xc", 0.0)
                cell_width = cell.get("width", 100.0)

                # Try single column match first
                col = _match_column_from_text(cell_text)
                if col and col not in row_matches:
                    row_matches[col] = cell_x
                    row_debug.append(f"'{cell_text}' -> {col}")

                # Try multi-column extraction for complex headers
                multi_matches = _extract_columns_from_complex_header(cell_text, cell_x, cell_width)
                for col_name, estimated_x in multi_matches.items():
                    if col_name not in row_matches:
                        row_matches[col_name] = estimated_x
                        row_debug.append(f"'{cell_text}' -> {col_name} (multi)")

            # Debug output for header analysis
            if len(row_matches) >= 2:
                header_debug_info.append({
                    'row_text': ' '.join(c.get('text', '') for c in row),
                    'matches': dict(row_matches),
                    'debug': row_debug
                })

            # Strong header row signal
            if len(row_matches) >= 4 and ("product_description" in row_matches or "hsn_code" in row_matches):
                print(f"  📋 Found strong header row with {len(row_matches)} columns:")
                for col, x in sorted(row_matches.items(), key=lambda kv: kv[1]):
                    print(f"    {col}: x={x}")
                return row_matches

            # Weak fallback accumulation
            for col, x in row_matches.items():
                if col not in best_positions:
                    best_positions[col] = x

    # If no strong header found, use accumulated positions
    if best_positions:
        print(f"  📋 Using accumulated column positions ({len(best_positions)} columns):")
        for col, x in sorted(best_positions.items(), key=lambda kv: kv[1]):
            print(f"    {col}: x={x}")

    # Debug: show header analysis
    if header_debug_info:
        print(f"  🔍 Header analysis found {len(header_debug_info)} potential header rows")
        for i, info in enumerate(header_debug_info[:2], 1):  # Show first 2
            print(f"    Row {i}: {len(info['matches'])} columns - {info['row_text'][:50]}...")
            if 'total_amount' in info['matches']:
                print(f"      → total_amount detected at x={info['matches']['total_amount']}")

    return best_positions


def _assign_row_to_columns(row: List[Dict], col_positions: Dict[str, float]) -> Dict[str, str]:
    """Assign OCR cells in a row to nearest detected columns with improved rightmost detection."""
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

        # For total_amount (rightmost column), be more lenient to capture full values
        if col_name == "total_amount":
            return max(60.0, min(gaps) * 0.8)  # Increased tolerance for rightmost column

        return max(40.0, min(gaps) * 0.55)

    assigned: Dict[str, List[str]] = defaultdict(list)

    # Special handling for total_amount - look for rightmost numeric values
    if "total_amount" in col_positions:
        total_amount_x = col_positions["total_amount"]

        # Find cells that are near or to the right of the total_amount column
        rightmost_cells = []
        for cell in row:
            cell_x = cell.get("xc", 0)
            cell_text = cell.get("text", "").strip()

            # Look for cells with numeric values near/right of total_amount column
            if (cell_x >= total_amount_x - 100 and  # Allow some left tolerance
                re.search(r'\d[\d,]*\.?\d*', cell_text) and  # Contains numbers
                not re.search(r'^[A-Z]{2,}$|^(BATCH|QTY|BOX|TAB|CAP|MG)$', cell_text, re.IGNORECASE)):  # Not just labels
                rightmost_cells.append((cell_x, cell_text))

        # Sort by x position and pick the rightmost numeric cell
        if rightmost_cells:
            rightmost_cells.sort(key=lambda x: x[0])
            rightmost_text = rightmost_cells[-1][1]  # Rightmost cell
            assigned["total_amount"].append(rightmost_text)

    # Regular assignment for other columns
    for cell in row:
        text = cell.get("text", "").strip()
        if not text:
            continue

        # Skip if already assigned to total_amount
        if "total_amount" in assigned and text in assigned["total_amount"]:
            continue

        nearest_col, nearest_x = min(col_positions.items(), key=lambda kv: abs(cell["xc"] - kv[1]))
        if abs(cell["xc"] - nearest_x) <= max_distance_for_col(nearest_col):
            # Avoid double-assigning transaction values to other columns
            if (nearest_col != "total_amount" and
                re.match(r'^\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$', text)):  # Looks like transaction value
                try:
                    value = float(text.replace(",", ""))
                    if value > 500:  # Large amounts likely transaction values
                        continue  # Skip assigning large amounts to non-total columns
                except ValueError:
                    pass  # If can't parse as number, continue with normal assignment

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

            for key in ["mrp", "ptr", "pts", "tp_value", "scheme_value", "discount", "cgst", "sgst", "total_amount"]:
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

    for number_key in ["mrp", "ptr", "pts", "tp_value", "scheme_value", "discount", "total_amount"]:
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


def _extract_company_names_from_header(page_rows: List[List[Dict]]) -> List[str]:
    """Extract company names from header section (before bill-to section)."""
    header_rows: List[List[Dict]] = []
    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row)
        if re.search(r'BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS|DELIVERY\s*ADDRESS', row_text, re.IGNORECASE):
            break
        header_rows.append(row)

    search_rows = header_rows if header_rows else page_rows

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
    Uses batch number for duplicate detection.
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
    seen_batches = set()  # Track (product_name, batch_number) tuples to avoid duplicates


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
                    item_accuracies.append(scorer.score_item(current_item))
                    current_item = None
                table_ended = True
                break

            # ─── Skip header and noise rows ──────────────────────────────
            if re.search(r'Goods.*sold|Interest.*per|Company.*staff|disputes',
                        row_text, re.IGNORECASE):
                continue

            # ─── Map current row values to detected columns ─────────────
            row_values = _assign_row_to_columns(row, col_positions)

            # ─── Detect new item: HSN code or strong product pattern ──────
            hsn_candidate = row_values.get("hsn_code")
            hsn_match = re.search(r'\b(\d{8})\b', hsn_candidate or row_text)

            # More flexible item detection - not just HSN
            is_new_item = False
            detected_hsn = None

            if hsn_match and _is_valid_hsn_context(row_text, hsn_match.group(1)):
                is_new_item = True
                detected_hsn = hsn_match.group(1)

            # Alternative detection: Strong product pattern even without HSN
            elif (not current_item and
                  re.search(r'\b[A-Z]{3,}\b.*\b(TAB|CAP|INJ|SYR|TABLET|CAPSULE)\b', row_text, re.IGNORECASE) and
                  re.search(r'\d+', row_text)):  # Has some numbers
                is_new_item = True

            if is_new_item:
                # Extract product description and batch early to check for duplicates
                product_text = row_values.get("product_description")
                if product_text:
                    product_text = re.sub(r'^\s*\d+\s*', '', product_text)
                    product_text = re.sub(r'\s+', ' ', product_text).strip(' :-|')
                    # If it looks like a short code, try to get better description
                    if len(product_text) <= 10 and re.match(r'^[A-Z0-9]+$', product_text):
                        for cell in row:
                            cell_text = cell.get("text", "").strip()
                            if (len(cell_text) > 10 and
                                not re.match(r'^\d+$', cell_text) and
                                not re.match(r'^[A-Z0-9]{3,10}$', cell_text) and
                                re.search(r'[A-Za-z]{3,}', cell_text) and
                                not re.search(r'\d{8}', cell_text)):
                                product_text = cell_text
                                break

                if not product_text and detected_hsn:
                    left_text = row_text.split(detected_hsn, 1)[0]
                    product_text = re.sub(r'^\s*\d+\s*', '', left_text)
                    product_text = re.sub(r'\s+', ' ', product_text).strip(' :-|')

                # Check for batch-based duplicates before creating new item
                potential_batch = None

                # Try to extract batch number from this row
                if row_values.get("batch_no"):
                    batch_text = row_values.get("batch_no")
                    batch_match = re.search(r'[A-Z0-9\']{4,12}', str(batch_text).upper())
                    if batch_match:
                        potential_batch = batch_match.group(0)
                else:
                    # Fallback batch extraction from row text
                    potential_batch = _extract_batch_from_text(row_text, detected_hsn)

                # Skip only if we've seen this exact product+batch combination (true duplicate)
                # Don't skip just because batch exists - allow same product with different batches
                product_batch_key = (product_text, potential_batch) if product_text else (None, potential_batch)
                if product_batch_key in seen_batches:
                    continue

                # Save previous item
                if current_item:
                    items.append(current_item)
                    item_accuracies.append(scorer.score_item(current_item))

                # Create new item with ALL fields
                current_item = {field: None for field in LINE_ITEM_FIELDS}
                current_item["row_index"] = item_index
                item_index += 1

                # Set HSN if detected
                if detected_hsn:
                    current_item["hsn_code"] = detected_hsn

                # Record product+batch combination to prevent duplicates
                if product_batch_key not in seen_batches:
                    seen_batches.add(product_batch_key)
                    if potential_batch:
                        current_item["batch_no"] = potential_batch

                # Set product description
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

                    # Check if this looks like a proper description vs a code
                    if len(product_text) > 6 and not re.match(r'^[A-Z0-9]{3,10}$', product_text):
                        if re.search(r'[A-Za-z]{3,}', product_text):
                            current_item["product_description"] = cleaner.clean_string(product_text)
                    elif len(product_text) <= 10:
                        # Short text - might be code, look for better description in row
                        for cell in row:
                            cell_text = cell.get("text", "").strip()
                            if (len(cell_text) > 10 and
                                cell_text != product_text and
                                not re.match(r'^\d+$', cell_text) and
                                not re.match(r'^[A-Z0-9]{3,10}$', cell_text) and
                                re.search(r'[A-Za-z]{3,}', cell_text) and
                                not re.search(r'\d{8}', cell_text)):
                                current_item["product_description"] = cleaner.clean_string(cell_text)
                                break

            if not current_item["batch_no"] and row_values.get("batch_no"):
                batch_text = row_values.get("batch_no")
                batch_match = re.search(r'[A-Z0-9\']{4,12}', str(batch_text).upper())
                if batch_match:
                    batch_candidate = batch_match.group(0)
                    if batch_candidate not in seen_batches:
                        current_item["batch_no"] = batch_candidate
                        seen_batches.add(batch_candidate)

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
                "tp_value": "tp_value",
                "scheme_value": "scheme_value",
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

            # ──── IMPROVED TOTAL AMOUNT EXTRACTION ────────────────────────
            if current_item["total_amount"] is None:
                # Priority 1: Transaction Value from improved column detection
                if "total_amount" in detected_cols and row_values.get("total_amount"):
                    raw_value = row_values.get("total_amount")
                    # Handle comma-separated values properly
                    if re.match(r'^\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$', raw_value):
                        parsed = float(raw_value.replace(",", ""))
                        if parsed > 10:  # Must be reasonable transaction value
                            current_item["total_amount"] = _round2(parsed)
                    else:
                        # Try to extract number if not perfectly formatted
                        parsed = cleaner.extract_number(raw_value)
                        if parsed is not None and parsed > 10:
                            current_item["total_amount"] = _round2(parsed)

                # Priority 2: Enhanced pattern matching on row text
                if current_item["total_amount"] is None:
                    total_fallback = _extract_total_amount_from_row_text(row_text)
                    if total_fallback is not None:
                        current_item["total_amount"] = total_fallback

                # Priority 3: Look for rightmost large numeric value in row
                if current_item["total_amount"] is None:
                    # Find all substantial numeric values in the row
                    all_numbers = []
                    for cell in row:
                        cell_text = cell.get("text", "").strip()
                        # Match patterns like "1,598.40", "13,558.90", etc.
                        number_match = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', cell_text)
                        if number_match:
                            try:
                                value = float(number_match.group(1).replace(",", ""))
                                if value > 50:  # Must be substantial for transaction value
                                    all_numbers.append((cell.get("xc", 0), value))
                            except ValueError:
                                continue

                    # Pick the rightmost substantial value
                    if all_numbers:
                        all_numbers.sort(key=lambda x: x[0])  # Sort by x position
                        rightmost_value = all_numbers[-1][1]
                        current_item["total_amount"] = _round2(rightmost_value)

                # Priority 4: Calculate from available data if still missing
                if (current_item["total_amount"] is None and
                    current_item.get("qty") and current_item.get("pts")):
                    calculated_base = current_item["qty"] * current_item["pts"]
                    # Apply discount if available
                    if current_item.get("discount"):
                        calculated_base *= (1 - current_item["discount"] / 100)
                    # Add taxes if available
                    if current_item.get("sgst"):
                        calculated_base *= (1 + current_item["sgst"] / 100)
                    if current_item.get("cgst"):
                        calculated_base *= (1 + current_item["cgst"] / 100)
                    current_item["total_amount"] = _round2(calculated_base)

            # Fallbacks from raw row text for key fields when column mapping misses.
            if current_item["expiry_date"] is None:
                exp_fallback = _extract_expiry_from_text(row_text)
                if exp_fallback:
                    current_item["expiry_date"] = exp_fallback

            if current_item["batch_no"] is None:
                batch_fallback = _extract_batch_from_text(row_text, current_item.get("hsn_code"))
                if batch_fallback:
                    # Check if this product+batch combination already exists
                    product_batch_key = (current_item.get("product_description"), batch_fallback)
                    if product_batch_key not in seen_batches:
                        current_item["batch_no"] = batch_fallback
                        seen_batches.add(product_batch_key)

            if current_item["pts"] is None and "pts" in detected_cols:
                pts_fallback = _extract_pts_from_row_text(row_text)
                if pts_fallback is not None:
                    current_item["pts"] = pts_fallback

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

    print(f"  ✓ Extracted {len(items)} unique items (batch-based deduplication)")
    print(f"  ✓ Detected batches: {len(seen_batches)}")

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