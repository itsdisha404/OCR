
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

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

OCR_CONFIDENCE_THRESHOLD = 0.25
MIN_BOX_COUNT = 30

# Define all required fields with their schemas
HEADER_FIELDS = [
    "company_name",
    "pan",
    "gstin_no",
    "invoice_no",
    "invoice_date",
    "due_date",
    "fssai_lic_no",
    "dl_no",
]

BILL_TO_FIELDS = [
    "name",
    "address",
    "cust_gstin",
    "cust_dl_no",
    "dl_exp_date",
    "po_ref",
    "po_ref_date",
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
        "item description", "product name"
    ],
    "hsn_code": [
        "hsn", "hsn code", "hsn/sac", "hsn no", "hsn number"
    ],
    "batch_no": [
        "batch", "batch no", "batch number", "batch code", "lot", "lot no", "batch#"
    ],
    "expiry_date": [
        "expiry", "exp", "exp date", "expiry date", "mfg date", "exp dt"
    ],
    "qty": [
        "qty", "quantity", "billed qty", "sale qty", "ordered qty"
    ],
    "uom": [
        "unit", "uom", "unit of measure", "pack size"
    ],
    "mrp": [
        "mrp", "m.r.p", "retail price"
    ],
    "ptr": [
        "ptr", "p.t.r", "retailer price"
    ],
    "pts": [
        "pts", "p.t.s", "stockist price", "pts price"
    ],
    "discount": [
        "discount", "disc", "disc%", "trade disc", "qty disc", "discount%"
    ],
    "cgst": [
        "cgst", "cgst%", "cgst rate"
    ],
    "sgst": [
        "sgst", "sgst%", "sgst rate"
    ],
    "total_amount": [
        "amount", "total", "net amount", "taxable", "total value", "line total"
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
        Calculate accuracy for a line item (0-100).
        Scores: 60% required fields + 25% optional fields + 15% data quality.
        """
        if not item_dict:
            return 0.0

        # Required fields (must-have for valid item)
        required_fields = [
            'product_description', 'hsn_code', 'qty', 'mrp', 'ptr'
        ]
        optional_fields = ['batch_no', 'pts', 'cgst', 'sgst', 'discount']
        numeric_fields = ['qty', 'mrp', 'ptr', 'pts', 'cgst', 'sgst', 'discount']

        # Required field completeness (60% weight)
        non_null_required = sum(
            1 for f in required_fields 
            if item_dict.get(f) is not None
        )
        required_ratio = non_null_required / len(required_fields)
        required_score = required_ratio * 60

        # Optional field completeness (25% weight)
        non_null_optional = sum(
            1 for f in optional_fields
            if item_dict.get(f) is not None
        )
        optional_ratio = non_null_optional / len(optional_fields) if optional_fields else 0.0
        optional_score = optional_ratio * 25

        # Data quality validation (15% weight)
        quality_score = 1.0
        
        # Type validation
        for field in numeric_fields:
            val = item_dict.get(field)
            if val is not None and not isinstance(val, (int, float)):
                quality_score -= 0.05

        # Range validation for plausible values
        qty = item_dict.get('qty')
        if qty is not None:
            try:
                vf = float(qty)
                if not (1 <= vf <= 10000):
                    quality_score -= 0.05
            except (ValueError, TypeError):
                pass

        for tax_field in ['cgst', 'sgst']:
            val = item_dict.get(tax_field)
            if val is not None:
                try:
                    vf = float(val)
                    if not (0 <= vf <= 50):  # Realistic GST range
                        quality_score -= 0.10
                except (ValueError, TypeError):
                    pass

        for price_field in ['mrp', 'ptr', 'pts']:
            val = item_dict.get(price_field)
            if val is not None:
                try:
                    vf = float(val)
                    if vf < 0 or vf > 100000:
                        quality_score -= 0.05
                except (ValueError, TypeError):
                    pass

        quality_score = max(0.0, min(15.0, quality_score * 15))
        overall_accuracy = required_score + optional_score + quality_score
        return round(min(100, max(0, overall_accuracy)), 2)
    
    def score_section(self, section_data: Dict[str, Any]) -> float:
        """Score extraction for header or bill-to section"""
        if not section_data:
            return 0.0
        
        non_null_fields = sum(1 for v in section_data.values() if v is not None)
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


def _normalize_text(text: str) -> str:
    """Normalize OCR text for fuzzy header matching."""
    normalized = re.sub(r'[^a-z0-9]+', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', normalized).strip()


def _match_column_from_text(text: str) -> Optional[str]:
    """Map a header cell text to canonical column key using COLUMN_SYNONYMS.
    
    Uses both exact word boundary matching and substring matching for fuzzy variations.
    """
    if not text or not text.strip():
        return None
    
    normalized = _normalize_text(text)
    if not normalized:
        return None

    # Try exact word boundary matches first
    for col_name, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            syn_norm = _normalize_text(synonym)
            if syn_norm and re.search(rf'\b{re.escape(syn_norm)}\b', normalized):
                return col_name
    
    # Fallback: fuzzy substring match for partial headers (e.g., "PTS%" -> "pts")
    for col_name, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            syn_norm = _normalize_text(synonym).split()[0] if _normalize_text(synonym) else ""
            # Match at least first 3 chars of synonym, or entire synonym if shorter
            min_match_len = min(3, len(syn_norm)) if syn_norm else 0
            if min_match_len > 0 and syn_norm[:min_match_len] in normalized:
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
    """Assign OCR cells in a row to nearest detected columns using smarter overlap handling."""
    if not col_positions:
        return {}

    assigned: Dict[str, List[str]] = defaultdict(list)
    sorted_cols = sorted(col_positions.items(), key=lambda kv: kv[1])
    sorted_cells = sorted(row, key=lambda c: c["xc"])

    # Use a greedy left-to-right assignment to avoid overlapping cells
    used_cols = set()
    cell_to_col: Dict[int, str] = {}  # Map cell index to assigned column

    for cell_idx, cell in enumerate(sorted_cells):
        text = cell.get("text", "").strip()
        if not text:
            continue

        # Find nearest unused column (prefer if not yet assigned)
        best_col = None
        best_dist = float('inf')

        for col_name, col_x in sorted_cols:
            dist = abs(cell["xc"] - col_x)
            # Prefer unassigned columns; accept assigned if distance is much better
            if col_name not in used_cols and dist < best_dist:
                best_col = col_name
                best_dist = dist
            elif col_name in used_cols and dist < best_dist * 0.5:  # Significant improvement
                best_col = col_name
                best_dist = dist

        if best_col:
            used_cols.add(best_col)
            assigned[best_col].append(text)
            cell_to_col[cell_idx] = best_col

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
) -> List[Dict[str, Any]]:
    """Optionally refine OCR-extracted line items using OpenRouter LLM.

    Enabled only when OPENROUTER_API_KEY is set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or not line_items:
        return line_items

    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
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

            # Heuristic: realistic tax rate is usually <= 40; larger likely amount.
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


def _improve_batch_extraction(row: List[Dict], current_hsn: Optional[str] = None) -> Optional[str]:
    """
    Improved batch number extraction from row cells.
    Looks for alphanumeric patterns that resemble pharmaceutical batch codes.
    Avoids HSN codes (8 digits) and date patterns.
    Handles multi-part batches like "15*5" or "7's".
    """
    for cell in row:
        text = cell.get("text", "").strip()
        if not text or len(text) < 2:
            continue

        # Skip if matches HSN code (pure 8 digits)
        if re.fullmatch(r'\d{8}', text):
            continue

        # Skip pure date patterns or numeric decimals
        if re.fullmatch(r'^\d+\.\d+$|^[A-Z]{3}-\d{4}$', text):
            continue

        # Look for batch-like patterns:
        # - Alphanumeric 3-12 chars with at least one letter
        # - Or patterns like "15*5", "15'5", "ARF50025", etc.
        batch_patterns = [
            r'([A-Z][A-Z0-9]{2,11})',  # Letter first, alphanum after
            r'(\d+[*\'\-]\d+)',  # e.g. "15*5" or "15'5"
            r'([A-Z0-9]+[*\'\-][A-Z0-9]+)',  # Mixed with separator
        ]

        for pattern in batch_patterns:
            batch_match = re.search(pattern, text.upper())
            if batch_match:
                candidate = batch_match.group(1).upper()
                if len(candidate) >= 2 and not re.fullmatch(r'\d+', candidate):
                    return candidate

    return None


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: HEADER EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_invoice_header(page_rows: List[List[Dict]]) -> Tuple[Dict, float]:
    """
    Extract invoice header information.
    Always returns dict with all HEADER_FIELDS present (null if not found).
    Returns: (header_dict, accuracy_score)
    """
    flat = flatten_cells(page_rows)
    scorer = AccuracyScorer()
    cleaner = DataCleaner()
    
    # Initialize with all fields as None
    header = {field: None for field in HEADER_FIELDS}
    
    # ─── Company Name ──────────────────────────────────────────────────────
    for cell in flat:
        if re.search(r'ESSEN|CHHABRA|LUPIN|HEALTHCARE|MEDICARE', 
                     cell["text"], re.IGNORECASE):
            if len(cell["text"]) > 2:  # Avoid single chars
                header["company_name"] = cleaner.clean_string(cell["text"])
                break
    
    # ─── GSTIN (Supplier) ───────────────────────────────────────────────────
    # Look for "GSTIN" label and take next value
    gstin_text, gstin_conf = find_value_after_label(
        flat, r'GSTIN\s*NO|GSTIN\s*No(?!\..*CUST)'
    )
    if gstin_text:
        header["gstin_no"] = cleaner.extract_gstin(gstin_text)
    
    # ─── PAN (Supplier) ────────────────────────────────────────────────────
    # Get PAN that's NOT customer PAN
    for i, cell in enumerate(flat):
        if re.match(r'^PAN\s*$', cell["text"], re.IGNORECASE):
            # Check context - shouldn't be after "CUST"
            context = " ".join(c["text"] for c in flat[max(0, i-3):i])
            if "cust" not in context.lower():
                candidates = [
                    c for c in flat[i+1:i+5]
                    if c["text"] not in (":", "")
                ]
                if candidates:
                    pan_val = max(candidates, key=lambda x: x["conf"])["text"]
                    header["pan"] = cleaner.extract_pan(pan_val)
                    break
    
    # ─── Invoice Number ────────────────────────────────────────────────────
    inv_no, _ = find_value_after_label(flat, r'INVOICE\s*NO|INV\.?\s*NO')
    if inv_no:
        header["invoice_no"] = cleaner.clean_invoice_no(inv_no)
    
    # ─── Invoice Date ────────────────────────────────────────────────────
    inv_date, _ = find_value_after_label(flat, r'INVOICE\s*DATE|DATE.*[12]\d{3}')
    if inv_date:
        header["invoice_date"] = cleaner.clean_date(inv_date)
    
    # ─── Due Date ────────────────────────────────────────────────────────
    due_date, _ = find_value_after_label(flat, r'DUE\s*DATE')
    if due_date:
        header["due_date"] = cleaner.clean_date(due_date)
    
    # ─── FSSAI License ───────────────────────────────────────────────────
    # Look for "FSSAI" label
    fssai_text, _ = find_value_after_label(flat, r'FSSAI')
    if fssai_text:
        # Extract numbers only
        fssai_nums = re.findall(r'\d{14}', fssai_text)
        if fssai_nums:
            header["fssai_lic_no"] = fssai_nums[0]
    
    # ─── DL Number ──────────────────────────────────────────────────────
    dl_text, _ = find_value_after_label(flat, r'\bDL\s*NO\.?\s*[^:]|LICENSE\s*NO')
    if dl_text:
        # Clean up DL number
        dl_clean = re.sub(r'[\s]+', ' ', dl_text).strip()
        header["dl_no"] = dl_clean[:50] if dl_clean else None
    
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
    flat = flatten_cells(page_rows)
    scorer = AccuracyScorer()
    cleaner = DataCleaner()
    
    # Initialize with all fields as None
    bill_to = {field: None for field in BILL_TO_FIELDS}
    
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
    
    if addr_lines:
        bill_to["name"] = cleaner.clean_string(addr_lines[0]) if addr_lines else None
        if len(addr_lines) > 1:
            bill_to["address"] = " ".join(addr_lines[1:])
    
    # ─── Customer GSTIN ─────────────────────────────────────────────────
    cust_gstin, _ = find_value_after_label(flat, r'CUST.*GSTIN|GSTIN.*CUST')
    if cust_gstin:
        bill_to["cust_gstin"] = cleaner.extract_gstin(cust_gstin)
    
    # ─── Customer DL ───────────────────────────────────────────────────
    cust_dl, _ = find_value_after_label(flat, r'CUST.*D\.?L\.?|D\.?L\.?.*CUST')
    if cust_dl:
        bill_to["cust_dl_no"] = cleaner.clean_string(cust_dl)
    
    # ─── DL Expiry Date ───────────────────────────────────────────────
    dl_exp, _ = find_value_after_label(flat, r'DL.*EXP|EXP.*DATE')
    if dl_exp:
        bill_to["dl_exp_date"] = cleaner.clean_date(dl_exp)
    
    # ─── PO Reference ──────────────────────────────────────────────────
    po_ref, _ = find_value_after_label(flat, r'PO\s*REF|ORDER\s*NO')
    if po_ref:
        bill_to["po_ref"] = cleaner.clean_string(po_ref)
    
    # ─── PO Date ────────────────────────────────────────────────────────
    po_date, _ = find_value_after_label(flat, r'PO.*DATE|ORDER.*DATE')
    if po_date:
        bill_to["po_ref_date"] = cleaner.clean_date(po_date)
    
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
                # Clean batch text: remove trailing numbers that look like prices
                batch_text = re.sub(r'\s+\d+\.\d+\s*$', '', batch_text)
                batch_match = re.search(r'([A-Z0-9\*\'\-]{2,12})', str(batch_text).upper())
                if batch_match:
                    candidate = batch_match.group(1)
                    # Validate: not pure digits, not date-like
                    if not re.fullmatch(r'\d+', candidate) and not re.fullmatch(r'[A-Z]{3}-\d{4}', candidate):
                        current_item["batch_no"] = candidate

            # Fallback: extract batch directly from row if column mapping failed
            if not current_item["batch_no"]:
                fallback_batch = _improve_batch_extraction(row, current_item.get("hsn_code"))
                if fallback_batch:
                    current_item["batch_no"] = fallback_batch

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
                if current_item[item_key] is None and row_values.get(col_name):
                    parsed = cleaner.extract_number(row_values.get(col_name))
                    if parsed is not None:
                        # Validate: discount and tax should be < 100, prices typically < 5000
                        if col_name in ["discount", "cgst", "sgst"]:
                            if parsed > 100:
                                continue  # Likely not this field
                        current_item[item_key] = _round2(parsed)

            # Fallback extraction for missing PTS and DISCOUNT from raw row text
            if current_item["pts"] is None or current_item["discount"] is None:
                all_numbers = re.findall(r'\d+\.?\d*', row_text)
                # Extract numeric values in order: assume pricing order is roughly
                # qty, mrp, ptr, pts, discount, cgst, sgst, amount
                if len(all_numbers) >= 7 and current_item["pts"] is None:
                    try:
                        pts_candidate = float(all_numbers[3])  # Heuristic: 4th number
                        if 0 < pts_candidate <= 10000:
                            current_item["pts"] = _round2(pts_candidate)
                    except (ValueError, IndexError):
                        pass

                if len(all_numbers) >= 5 and current_item["discount"] is None:
                    try:
                        disc_candidate = float(all_numbers[4])  # Heuristic: 5th number
                        if 0 < disc_candidate <= 100:
                            current_item["discount"] = _round2(disc_candidate)
                    except (ValueError, IndexError):
                        pass

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
    
    header, header_acc = extract_invoice_header(page_rows[0])
    bill_to, bill_to_acc = extract_bill_to(page_rows[0])
    line_items, item_accs = extract_line_items(page_rows)

    table_rows_text = _collect_table_rows_text(page_rows)
    line_items = refine_line_items_with_openrouter(line_items, table_rows_text)
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