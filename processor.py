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

# GLOBAL MODEL CACHE

_EASYOCR_READER = None  # Global cache for EasyOCR model


def _get_easyocr_reader():
    """Get cached EasyOCR reader, loading it if not already loaded.
    
    This prevents reloading the 150-200MB model for every PDF.
    First PDF: ~3-5s (model load) + OCR time
    Subsequent PDFs: 0s overhead (model cached) + OCR time
    
    Saves 3-5 seconds per document in batch processing.
    """
    global _EASYOCR_READER
    
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    
    try:
        import easyocr
    except ImportError:
        print("ERROR: easyocr not installed. Install with: pip install easyocr")
        sys.exit(1)
    
    print("  Loading EasyOCR model (cached for subsequent documents)...")
    _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


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
    "company_name",    # Vendor name
    "gstin",
    "pan",
    "cin_no",
    "phone",
    "address",
    "email_id",
]



LINE_ITEM_FIELDS = [
    "row_index",
    "product_description",
    "hsn_code",
    "batch_no",
    "expiry_date",
    "free_qty",
    "billed_qty",
    "uom",
    "mrp",
    "ptr",
    "pts",
    "discount",
    "cgst",
    "sgst",
    "tax_rate",
    "igst",
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
    "free_qty": [
        "free qty", "free quantity", "bonus qty", "gift qty"
    ],
    "billed_qty": [
        "billed qty", "qty", "quantity", "sale qty"
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
    "tax_rate": [
        "tax rate", "tax %", "combined tax", "combined gst", "tax"
    ],
    "igst": [
        "igst", "igst rate", "igst %"
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
        """Extract, normalize and validate GSTIN (15 chars, format: SSPPPPPPPPPENZC)
        SS = state code (2 digits)
        PPPPPPPPPPP = PAN section (10 chars)
        E = entity number (1 digit)
        N = fixed 'N' character
        Z = fixed 'Z' character
        C = check digit (1 alphanumeric)
        """
        if not text:
            return None
        # Remove leading colons/special chars
        gstin = re.sub(r'^[:\s]+|[:\s]+$', '', str(text))
        gstin = re.sub(r'[\s\-/:]', '', gstin.upper())
        gstin = re.sub(r'[^A-Z0-9]', '', gstin)
        
        # Normalize GSTIN to correct OCR errors
        if gstin:
            gstin = DataCleaner._normalize_gstin(gstin)
        
        # Validate GSTIN format
        if gstin and DataCleaner._is_valid_gstin_format(gstin):
            return gstin
        
        return gstin if gstin else None
    
    @staticmethod
    def _normalize_gstin(text: str) -> str:
        """Normalize OCR errors in GSTIN using structural rules.
        Format: SSPPPPPPPPPENZC (15 chars)
        - SS: state code (2 digits)
        - PPPPPPPPPPP: PAN section (10 chars)
        - E: entity number (1 digit)
        - N: fixed character (must be 'N')
        - Z: fixed character (must be 'Z')
        - C: check digit
        """
        if not text:
            return text
        
        text = text.strip().upper()
        
        # Remove spaces and punctuation
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        if len(text) != 15:
            return text
        
        chars = list(text)
        
        # ---- SS (state code) → digits ----
        digit_map = {"O": "0", "I": "1", "L": "1", "Z": "7", "S": "5", "B": "8"}
        for i in [0, 1]:
            if chars[i] in digit_map:
                chars[i] = digit_map[chars[i]]
        
        # ---- PAN section (positions 2-11) ----
        # PAN = 5 letters + 4 digits + 1 letter
        for i in range(2, 7):  # positions 2-6: letters
            if chars[i].isdigit():
                letter_map = {"0": "O", "1": "I", "5": "S", "8": "B"}
                chars[i] = letter_map.get(chars[i], chars[i])
        
        for i in range(7, 11):  # positions 7-10: digits
            if chars[i].isalpha():
                chars[i] = digit_map.get(chars[i], chars[i])
        
        # ---- Entity number (position 11, digit) ----
        if chars[11].isalpha():
            chars[11] = digit_map.get(chars[11], chars[11])
        
        # ---- Fixed characters ----
        # Position 12 should be 'N'
        if chars[12] != 'N':
            chars[12] = 'N'
        
        # Position 13 should be 'Z'
        if chars[13] != 'Z':
            chars[13] = 'Z'
        
        # Position 14: check digit (keep as-is or normalize)
        
        return "".join(chars)
    
    @staticmethod
    def _is_valid_gstin_format(gstin: str) -> bool:
        """Validate GSTIN format: SSPPPPPPPPPENZC (15 chars)
        SS = 2 digits (state code)
        PPPPPPPPPPP = 10 char PAN section (5 letters + 4 digits + 1 letter)
        E = 1 digit (entity number)
        NZ = fixed characters
        C = 1 check digit (alphanumeric)
        """
        if not gstin or len(gstin) != 15:
            return False
        
        # Positions 0-1: state code (2 digits)
        if not gstin[0:2].isdigit():
            return False
        
        # Positions 2-6: first 5 chars of PAN (letters)
        if not gstin[2:7].isalpha():
            return False
        
        # Positions 7-10: 4 digits of PAN
        if not gstin[7:11].isdigit():
            return False
        
        # Position 11: entity number (digit)
        if not gstin[11].isdigit():
            return False
        
        # Position 12: fixed 'N'
        if gstin[12] != 'N':
            return False
        
        # Position 13: fixed 'Z'
        if gstin[13] != 'Z':
            return False
        
        # Position 14: check digit (alphanumeric)
        if not gstin[14].isalnum():
            return False
        
        return True
    
    @staticmethod
    def extract_pan(text: Optional[str]) -> Optional[str]:
        """Extract and clean PAN (10 chars, format: AAAPL1234C)
        Where:
        - AAA = 3 alphabetic characters (A-Z)
        - P = 1 holder type character
        - L = 1 letter (first letter of surname/entity)
        - 1234 = 4 numeric digits
        - C = 1 check character
        """
        if not text:
            return None
        # Remove leading colons/special chars
        pan = re.sub(r'^[:\s]+|[:\s]+$', '', str(text))
        pan = re.sub(r'[\s\-/:]', '', pan.upper())
        pan = re.sub(r'[^A-Z0-9]', '', pan)
        
        # Validate PAN format and extract if valid
        if pan and DataCleaner._is_valid_pan_format(pan):
            return pan if len(pan) == 10 else pan[:10]
        
        return pan if pan else None
    
    @staticmethod
    def _is_valid_pan_format(pan: str) -> bool:
        """Validate PAN format: AAAPL1234C
        AAA = 3 alphabetic, P = holder type, L = first letter,
        1234 = 4 digits, C = check digit
        """
        if not pan or len(pan) != 10:
            return False
        # Pattern: 5 letters, 4 digits, 1 letter
        # More specifically: AAAPL1234C
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z0-9]$'
        return bool(re.match(pattern, pan))
    
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
        
        # Quantity (free_qty + billed_qty - must be positive integers)
        free_qty_val = item_dict.get('free_qty')
        billed_qty_val = item_dict.get('billed_qty')
        
        # Score billed_qty (primary billing quantity)
        if isinstance(billed_qty_val, (int, float)) and billed_qty_val > 0:
            field_scores['billed_qty'] = 100.0
        else:
            field_scores['billed_qty'] = 0.0
        
        # Score free_qty (bonus/free quantity - can be 0)
        if free_qty_val is None:
            field_scores['free_qty'] = 100.0  # It's OK to not have free qty
        elif isinstance(free_qty_val, (int, float)) and free_qty_val >= 0:
            field_scores['free_qty'] = 100.0
        else:
            field_scores['free_qty'] = 40.0
        
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
            qty_for_calc = billed_qty_val if isinstance(billed_qty_val, (int, float)) and billed_qty_val > 0 else 1
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
            # HEADER SECTION (vendor-only fields)
            company = str(section_data.get('company_name') or '').strip()
            field_scores['company_name'] = 100.0 if (company and len(company) >= 5) else 0.0

            gstin = str(section_data.get('gstin') or '').strip()
            if re.fullmatch(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', gstin):
                field_scores['gstin'] = 100.0
            elif gstin:
                field_scores['gstin'] = 50.0
            else:
                field_scores['gstin'] = 0.0

            pan = str(section_data.get('pan') or '').strip()
            if re.fullmatch(r'[A-Z]{5}\d{4}[A-Z]', pan):
                field_scores['pan'] = 100.0
            elif pan:
                field_scores['pan'] = 50.0
            else:
                field_scores['pan'] = 0.0

            cin_no = str(section_data.get('cin_no') or '').strip().upper()
            if re.fullmatch(r'[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', cin_no):
                field_scores['cin_no'] = 100.0
            elif cin_no:
                field_scores['cin_no'] = 50.0
            else:
                field_scores['cin_no'] = 0.0

            phone = str(section_data.get('phone') or '').strip()
            phone_digits = re.sub(r'\D', '', phone)
            field_scores['phone'] = 100.0 if len(phone_digits) >= 10 else (40.0 if phone else 0.0)

            email = str(section_data.get('email_id') or '').strip()
            if re.fullmatch(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', email):
                field_scores['email_id'] = 100.0
            elif email:
                field_scores['email_id'] = 40.0
            else:
                field_scores['email_id'] = 0.0

            address = str(section_data.get('address') or '').strip()
            field_scores['address'] = 100.0 if len(address) >= 12 else (40.0 if address else 0.0)

        if not field_scores:
            return 0.0

        return round(sum(field_scores.values()) / len(field_scores), 2)

       

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4: PDF EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def pdf_to_images(pdf_path: Path, images_dir: Path, scale: float = 2.0) -> List[Path]:
    """Convert PDF pages to PNG images with optimized scale factor for speed and accuracy.
    
    Scale=2.0 provides:
    - 4x faster processing vs scale=4.0 (5-10s per document vs 30-60s)
    - ~100MB RAM per page vs 400-600MB (4x memory reduction)
    - Sufficient quality for pharmaceutical invoice OCR (typical text ~8-10pt)
    - Optional adaptive scaling: use scale=3.0 for documents with small fonts
    """
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
    """Run OCR on images with EasyOCR (uses cached model for speed)"""
    # Get cached reader (loaded once, reused for all subsequent PDFs)
    reader = _get_easyocr_reader()
    
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


def _extract_tax_rates_from_row_text(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Extract tax rates from row text: returns (cgst, sgst, tax_rate, igst).
    
    Logic:
    - If IGST label found → return igst value, others null
    - If TAX/COMBINED label found → return tax_rate, others null  
    - If repeated pairs of same rate found → return as cgst/sgst, others null
    - Otherwise all null
    
    This prevents mixing: only one tax representation returned per row.
    """
    if not text:
        return None, None, None, None

    text_upper = str(text).upper()
    

    # Check for explicit IGST label (cross-state sales)
    
    igst_match = re.search(r'\bIGST\s*[:\-]?\s*([\d\.]+)\s*%?', text_upper)
    if igst_match:
        try:
            igst_val = float(igst_match.group(1))
            if 0 < igst_val <= 28:
                return None, None, None, _round2(igst_val)
        except (ValueError, IndexError):
            pass
    

    # Check for explicit combined TAX label (single tax column)

    combined_patterns = [
        r'\bTAX\s*(?:%|RATE)?\s*[:\-]?\s*([\d\.]+)\s*%?',  # TAX: 18
        r'\bCOMBINED\s+(?:GST|TAX)\s*[:\-]?\s*([\d\.]+)\s*%?',  # COMBINED GST: 18
        r'\bTAX\s*%\s*[:\-]?\s*([\d\.]+)\s*%?',  # TAX %: 18
    ]
    
    for pattern in combined_patterns:
        match = re.search(pattern, text_upper)
        if match:
            try:
                tax_val = float(match.group(1))
                if 0 < tax_val <= 56:  # Allow up to 56 for combined (18+18+20 IGC)
                    return None, None, _round2(tax_val), None
            except (ValueError, IndexError):
                pass
    
    # ─────────────────────────────────────────────────────────────────────────
    # Fallback: Extract paired CGST/SGST values (old logic)
    # ─────────────────────────────────────────────────────────────────────────
    raw_tokens = re.findall(r"\d+(?:\.\d+)?", str(text))
    values: List[float] = []
    for token in raw_tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue

    rate_candidates = [v for v in values if 0 < v <= 28]
    if not rate_candidates:
        return None, None, None, None

    rounded_counts: Dict[float, int] = defaultdict(int)
    for val in rate_candidates:
        rounded_counts[round(val, 2)] += 1

    # Prefer repeated rates (indicates CGST/SGST pair like "9, 9")
    repeated = [(rate, cnt) for rate, cnt in rounded_counts.items() if cnt >= 2]
    if repeated:
        repeated.sort(key=lambda x: (x[1], x[0]), reverse=True)
        chosen = repeated[0][0]
        return _round2(chosen), _round2(chosen), None, None

    # Fall back to nearest common rate only if we have a strong match
    common_rates = [2.5, 5.0, 6.0, 9.0, 12.0, 14.0, 18.0, 28.0]
    nearest = min(common_rates, key=lambda r: min(abs(v - r) for v in rate_candidates))
    if min(abs(v - nearest) for v in rate_candidates) <= 0.6:
        # Single rate without clear label - return as pair (same CGST/SGST)
        return _round2(nearest), _round2(nearest), None, None

    return None, None, None, None


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
    """Extract PTS (Price to Stockist) value from row text with multiple patterns.
    
    PTS is typically:
    - Lower than PTR (Price to Retailer)
    - Usually 10-80% of MRP
    - Comes after PTR in pharmaceutical invoices
    """
    if not text:
        return None

    # Pattern 1: Explicit PTS label with value (P.T.S: 150.50)
    label_match = re.search(r"\bP\.?T\.?S\b\s*[:\-]?\s*(\d[\d,]*\.?\d*)", str(text), re.IGNORECASE)
    if label_match:
        try:
            return _round2(float(label_match.group(1).replace(",", "")))
        except ValueError:
            pass
    
    # Pattern 2: "PTS" or "Pts" followed by a price
    pts_label_match = re.search(r"\bPTS\b\s*(\d+[\d.,]*)", str(text), re.IGNORECASE)
    if pts_label_match:
        try:
            value_str = pts_label_match.group(1).replace(",", "")
            value = float(value_str)
            if 1 <= value <= 5000:  # Reasonable PTS range
                return _round2(value)
        except ValueError:
            pass
    
    # Pattern 3: Look for price values and identify likely PTS based on context
    # PTS typically appears in a sequence: PTR -> PTS or in "PTR/PTS" format
    all_prices = re.findall(r'\d+[\d.,]*', text)
    if len(all_prices) >= 2:
        try:
            prices = [float(p.replace(",", "")) for p in all_prices if 0 < float(p.replace(",", "")) <= 5000]
            # PTS is typically the lowest price in the row (after discount)
            if prices:
                # Try to detect the sequence by context
                pts_match = re.search(r'PTR\s*[/\s]\s*PTS\s*[:\-]?\s*(\d+[\d.,]*)', text, re.IGNORECASE)
                if pts_match:
                    try:
                        return _round2(float(pts_match.group(1).replace(",", "")))
                    except ValueError:
                        pass
        except (ValueError, IndexError):
            pass
    
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


def extract_and_normalize_header_via_llm(vendor_rows_text: List[str]) -> Dict[str, Any]:
    """
    Extract and normalize header fields (GSTIN, PAN, email, etc.) from vendor rows using LLM.
    
    This function:
    1. Sends vendor context to LLM for structured field extraction
    2. Applies normalization rules for GSTIN (structural OCR correction)
    3. Validates PAN format
    4. Normalizes email addresses
    5. Returns cleaned header dictionary
    
    Args:
        vendor_rows_text: List of raw vendor/header row strings from OCR
        
    Returns:
        Dictionary with extracted and normalized header fields:
        {
            'company_name': str,
            'gstin': str (normalized),
            'pan': str (validated),
            'email_id': str,
            'phone': str,
            'address': str,
            'cin_no': str,
            'invoice_no': str,
            'invoice_date': str
        }
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        print("  [LLM] Skipped: No API key configured")
        return {}
    
    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    
    system_prompt = (
        "You are an expert at parsing pharmaceutical invoice headers. "
        "Extract and structure the following fields from vendor/header information:\n\n"
        "FIELDS TO EXTRACT:\n"
        "1. company_name: Vendor/supplier company legal name\n"
        "2. gstin: 15-character GST ID (may have OCR errors)\n"
        "3. pan: 10-character PAN number (format: AAAPL1234C)\n"
        "4. email_id: Email address for vendor\n"
        "5. phone: Phone/mobile number (+91 format or 10 digits)\n"
        "6. address: Full vendor address\n"
        "7. cin_no: Corporate identity number if present\n"
        "8. invoice_no: Invoice/document number\n"
        "9. invoice_date: Date of invoice (DD/MM/YYYY format)\n\n"
        "RETURN FORMAT:\n"
        "Return ONLY a valid JSON object with these exact keys. "
        "Use null for missing fields. Do NOT add explanations.\n"
        "Example:\n"
        "{\n"
        '  "company_name": "ESSEN MEDICARE LLP",\n'
        '  "gstin": "OZAAFFE3923MIZA",\n'
        '  "pan": "AAFFE3923M",\n'
        '  "email_id": "founrts@sngroup.ltd",\n'
        '  "phone": "+91 8287936614",\n'
        '  "address": "B-304, First Floor, Okhla Industrial Area, New Delhi 110020",\n'
        '  "cin_no": null,\n'
        '  "invoice_no": "EML2425IGIS01520",\n'
        '  "invoice_date": "13/11/2024"\n'
        "}"
    )
    
    user_prompt = {
        "vendor_context": vendor_rows_text,
        "task": "Extract and structure all header fields from the vendor rows above"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
    }
    
    headers_req = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers_req,
            method="POST",
        )
        with request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        
        content = body["choices"][0]["message"]["content"].strip()
        # Remove markdown code fences if present
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()
        parsed = json.loads(content)
        
        if not isinstance(parsed, dict):
            print("  [LLM] Invalid response format")
            return {}
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # APPLY NORMALIZATIONS
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Normalize GSTIN using structural rules
        if parsed.get("gstin"):
            cleaner = DataCleaner()
            normalized_gstin = cleaner.extract_gstin(parsed["gstin"])
            if normalized_gstin:
                parsed["gstin"] = normalized_gstin
                print(f"    ✓ GSTIN normalized: {parsed['gstin']}")
        
        # Validate PAN format
        if parsed.get("pan"):
            cleaner = DataCleaner()
            validated_pan = cleaner.extract_pan(parsed["pan"])
            if validated_pan:
                parsed["pan"] = validated_pan
                print(f"    ✓ PAN validated: {parsed['pan']}")
            else:
                parsed["pan"] = None
                print(f"    ✗ PAN format invalid: {parsed.get('pan')}")
        
        # Normalize email (remove extra spaces, ensure lowercase)
        if parsed.get("email_id"):
            email = str(parsed["email_id"]).strip().lower()
            # Remove common OCR mistakes like space in .com
            email = re.sub(r'\s+', '', email)
            parsed["email_id"] = email if '@' in email else None
        
        # Normalize phone (keep digits + + sign)
        if parsed.get("phone"):
            phone = re.sub(r'[^\d+]', '', str(parsed["phone"]))
            parsed["phone"] = phone if phone else None
        
        print(f"  [LLM] Header extraction successful")
        print(f"    Extracted fields: {', '.join(k for k, v in parsed.items() if v)}")
        
        return parsed
        
    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [LLM] Extraction failed: {exc}")
        return {}


def refine_extraction_with_openrouter(
    header: Dict[str, Any],
    line_items: List[Dict[str, Any]],
    vendor_rows_text: List[str],
    table_rows_text: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    UNIFIED LLM REFINEMENT: Validate and refine both header and line items in a single API call.
    
    This consolidates header and line items refinement to:
    1. Reduce API calls and latency
    2. Allow cross-validation (e.g., company name from header matches invoice context)
    3. Ensure consistent business logic applied to both sections
    4. Apply anti-hallucination rules (state tracking) uniformly
    
    Returns: (refined_header, refined_line_items)
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        # No API key: return as-is
        return header, line_items

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # ═══════════════════════════════════════════════════════════════════════════════
    # STATE TRACKING FOR ANTI-HALLUCINATION
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Track which header fields were detected (non-null)
    detected_header_fields = {
        "non_null_fields": [k for k in HEADER_FIELDS if header.get(k) is not None],
        "null_fields": [k for k in HEADER_FIELDS if header.get(k) is None],
    }
    
    # Track which fields were detected per line item
    detected_fields_per_item: List[Dict[str, List[str]]] = []
    for item in line_items:
        detected = {
            "non_null_fields": [k for k in LINE_ITEM_FIELDS if item.get(k) is not None],
            "null_fields": [k for k in LINE_ITEM_FIELDS if item.get(k) is None],
        }
        detected_fields_per_item.append(detected)

    system_prompt = (
        "You are an expert pharmaceutical invoice processor. Your task is to validate and refine "
        "BOTH invoice header information AND line items in a single unified response.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY a valid JSON object with 'header' and 'line_items' keys - no explanations.\n"
        "2. NEVER hallucinate - ONLY refine fields that were DETECTED in OCR extraction.\n"
        "3. For fields marked NULL in detected_fields, KEEP THEM NULL even if plausible.\n"
        "4. Maintain exact item count and row_index values.\n"
        "5. Cross-validate: company name in header should match invoice context in line items.\n\n"
        "HEADER FIELD GUIDELINES:\n"
        "- company_name: Vendor/supplier company name (e.g., 'Fusion Health Care Private Limited')\n"
        "- gstin: 15-character GST ID in format ZZZZZZZZZZZZZZZ (2 digits + 5 letters + 4 digits + 1 letter + 1 letter + 1 digit + 1 letter)\n"
        "- pan: 10-character PAN in format AAAPL1234C (5 letters + 4 digits + 1 letter)\n"
        "- cin_no: Corporate ID (21 chars if present)\n"
        "- phone: 10-digit mobile or phone number\n"
        "- address: Vendor address (can be multiple lines)\n"
        "- email_id: Vendor email address\n\n"
        "LINE ITEM FIELD GUIDELINES:\n"
        "- product_description: Pharmaceutical product name\n"
        "- hsn_code: 8-digit HSN code\n"
        "- batch_no: Batch number (alphanumeric)\n"
        "- expiry_date: Format MM/YYYY or DD/MM/YYYY\n"
        "- free_qty: Free/bonus quantity (0 or positive integer)\n"
        "- billed_qty: Quantity billed (positive integer)\n"
        "- uom: Unit of measure (BOX, STRIP, PCS, etc.)\n"
        "- mrp: Maximum Retail Price (positive number)\n"
        "- ptr: Price to Retailer (positive, typically <= mrp)\n"
        "- pts: Price to Stockist (positive, typically <= ptr)\n"
        "- discount: Discount percentage (0-100)\n"
        "- cgst/sgst: GST components (0-28%, equal for most India invoices)\n"
        "- total_amount: Line total (positive decimal)\n\n"
        "VALIDATION RULES:\n"
        "- Prices: MRP >= PTR >= PTS\n"
        "- GST: CGST = SGST (typically) in India\n"
        "- Quantities: positive integers\n"
        "- Dates: DD/MM/YYYY or MM/YYYY format, not future dates\n"
        "- PAN format validation: AAAPL1234C (5 letters + 4 digits + 1 check char)\n"
    )

    user_prompt = {
        "header": {
            "current_values": header,
            "detected_fields": detected_header_fields,
            "vendor_context": vendor_rows_text[:15],  # Top 15 vendor lines for context
        },
        "line_items": {
            "current_items": line_items,
            "detected_fields": detected_fields_per_item,
            "table_context": table_rows_text[:20],  # Top 20 table rows for context
        },
        "instructions": [
            "1. Review header.detected_fields to identify which header fields were ACTUALLY present in OCR.",
            "2. Only refine header fields in the 'non_null_fields' list.",
            "3. Validate PAN format (if present): Should match AAAPL1234C pattern.",
            "4. Review each line item's detected_fields list.",
            "5. For EACH item, only modify fields in its 'non_null_fields' list.",
            "6. Correct OCR errors: misspelled drug names, wrong quantities, price typos.",
            "7. Validate price hierarchy: MRP >= PTR >= PTS.",
            "8. Ensure GST rates are valid (0-28%, usually CGST=SGST).",
            "9. Cross-check: Do line items match the vendor company name in header?",
            "10. Return JSON with 'header' and 'line_items' keys only."
        ],
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
    }

    headers_req = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers_req,
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"].strip()
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()
        parsed = json.loads(content)

        if not isinstance(parsed, dict) or "header" not in parsed or "line_items" not in parsed:
            print("  [LLM] Ignored: Invalid response structure")
            return header, line_items

        # ═══════════════════════════════════════════════════════════════════════════════
        # APPLY REFINEMENTS WITH ANTI-HALLUCINATION ENFORCEMENT
        # ═══════════════════════════════════════════════════════════════════════════════
        
        # Refine Header
        refined_header = dict(header)
        llm_header = parsed.get("header", {})
        if isinstance(llm_header, dict):
            for key in HEADER_FIELDS:
                if key in llm_header and key in detected_header_fields["non_null_fields"]:
                    refined_header[key] = llm_header[key]
                    # Apply post-processing for specific fields
                    if key == "pan" and llm_header[key]:
                        refined_header[key] = DataCleaner.extract_pan(str(llm_header[key]))

        # Refine Line Items
        refined_items = []
        llm_items = parsed.get("line_items", [])
        if isinstance(llm_items, list) and len(llm_items) == len(line_items):
            for i, llm_item in enumerate(llm_items):
                if not isinstance(llm_item, dict):
                    refined_items.append(line_items[i])
                    continue

                base = dict(line_items[i])
                detected = detected_fields_per_item[i]
                allowed_fields = set(detected["non_null_fields"])

                for key in LINE_ITEM_FIELDS:
                    if key in llm_item and key in allowed_fields:
                        base[key] = llm_item[key]

                # Type normalization
                for key in ["mrp", "ptr", "pts", "discount", "cgst", "sgst", "total_amount"]:
                    if base.get(key) is not None:
                        try:
                            base[key] = _round2(float(base[key]))
                        except (ValueError, TypeError):
                            base[key] = None

                for key in ["free_qty", "billed_qty"]:
                    if base.get(key) is not None:
                        try:
                            base[key] = int(float(base[key]))
                        except (ValueError, TypeError):
                            base[key] = None

                refined_items.append(base)
        else:
            refined_items = line_items

        return refined_header, refined_items

    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        print(f"  [LLM] Refinement failed: {exc}")
        return header, line_items


def refine_line_items_with_openrouter(
    line_items: List[Dict[str, Any]],
    table_rows_text: List[str],
    header: Dict[str, Any],
   
) -> List[Dict[str, Any]]:
    """Optionally refine OCR-extracted line items using OpenRouter LLM.

    Enabled when OPENROUTER_API_KEY (or fallback API_KEY) is set.
    Uses state tracking to prevent hallucination of missing fields.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key or not line_items:
        return line_items

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # ═══════════════════════════════════════════════════════════════════════════════
    # STATE TRACKING: Detect which fields were actually present in OCR extraction
    # This prevents LLM from hallucinating values for truly missing columns
    # ═══════════════════════════════════════════════════════════════════════════════
    detected_fields_per_item: List[Dict[str, List[str]]] = []
    for item in line_items:
        detected = {
            "non_null_fields": [k for k in LINE_ITEM_FIELDS if item.get(k) is not None],
            "null_fields": [k for k in LINE_ITEM_FIELDS if item.get(k) is None],
        }
        detected_fields_per_item.append(detected)

    system_prompt = (
        "You are an expert pharmaceutical invoice line-item processor. Your task is to refine and correct "
        "OCR-extracted invoice line items using the provided invoice header, and raw table text.\n\n"
        "CRITICAL RULES:\n"
        "1. Return ONLY a valid JSON array - no explanations, no markdown, no code fences.\n"
        "2. Maintain exact same item count and row_index values - do not add/remove/reorder items.\n"
        "3. NEVER hallucinate values - ONLY refine fields that were DETECTED in OCR extraction.\n"
        "4. For fields marked NULL in detected_fields, KEEP THEM NULL even if plausible values exist.\n"
        "5. Only modify fields that are present in the non_null_fields list for each item.\n"
        "6. Use null for any unknown/unrecoverable fields.\n\n"
        "WHY STATE TRACKING MATTERS:\n"
        "If a field like 'mrp' is null in detected_fields, the invoice table DOES NOT HAVE an MRP column.\n"
        "Do NOT invent pharmaceutical pricing (e.g., ₹50-500) for missing columns.\n"
        "Missing data = null, not 'to be estimated'.\n\n"
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
        
        "detected_fields": detected_fields_per_item,
        "instructions": [
            "Step 1: Review the invoice metadata (company, invoice#, date) for context.",
            "Step 3: Analyze table_rows_text to identify column positions and patterns.",
            "Step 4: For each current_item, FIRST CHECK its detected_fields.",
            "Step 5: Correct ONLY fields in the non_null_fields list - do NOT fill null_fields.",
            "Step 6: Validate that prices follow: MRP >= PTR >= PTS (pharmaceutical pricing hierarchy).",
            "Step 7: Ensure GST values are valid Indian tax rates (typically 5%, 12%, 18%, 28%).",
            "Step 8: Cross-check totals: total_amount should align with (qty * price) and tax calculations.",
            "Step 9: Return corrected items maintaining original structure and count."
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
            detected = detected_fields_per_item[i]
            allowed_fields = set(detected["non_null_fields"])

            for key in LINE_ITEM_FIELDS:
                if key in item:
                    # ──────────────────────────────────────────────────────────────────────────
                    # ANTI-HALLUCINATION: Only allow LLM to modify fields that were detected
                    # in the original OCR extraction. Null fields stay null.
                    # ──────────────────────────────────────────────────────────────────────────
                    if key in allowed_fields:
                        base[key] = item[key]
                    elif key not in allowed_fields and item[key] is not None:
                        # LLM tried to fill a null field - ignore the change
                        pass

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

    # Detect and flag pricing hierarchy violations (NOT silent fix)
    # Pharmaceutical pricing rule: MRP >= PTR >= PTS (usually)
    # But some invoices may use different schemas - flag them for review instead of corrupting data
    mrp = normalized.get("mrp")
    ptr = normalized.get("ptr")
    pts = normalized.get("pts")
    
    price_violation = False
    
    if mrp is not None and ptr is not None and float(ptr) > float(mrp):
        price_violation = True
    if ptr is not None and pts is not None and float(pts) > float(ptr):
        price_violation = True
    if mrp is not None and pts is not None and float(pts) > float(mrp):
        price_violation = True
    
    if price_violation:
        if "price_violation_detected" not in normalized:
            normalized["price_violation_detected"] = True
     
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

    qty_val_billed = normalized.get("billed_qty")
    qty_val_free = normalized.get("free_qty")
    
    # Normalize billed_qty (primary billing quantity)
    if qty_val_billed is not None:
        try:
            normalized["billed_qty"] = int(float(qty_val_billed))
        except (ValueError, TypeError):
            normalized["billed_qty"] = None
    
    # Normalize free_qty (bonus/free quantity)
    if qty_val_free is not None:
        try:
            normalized["free_qty"] = int(float(qty_val_free))
        except (ValueError, TypeError):
            normalized["free_qty"] = None
    else:
        # If free_qty not explicitly set, it's acceptable (default to None)
        normalized["free_qty"] = None

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
        r'^INVOICE|^TAX\s*INVOICE|ORIGINAL\s*FOR\s*BUYER|BUYER\s*COPY|'
        r'^BILL|^SHIP|^CUSTOMER|ATTN|C/O|DIN|PAN|GSTIN|CIN NO|TAN|'
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
        r'APARTMENT|FLAT|SUITE|ATTN|C/O|DIN|@|INV NO|IRN|TRANSPORT|VEHICLE|DL\s*NO|FSSAI|'
        r'ORIGINAL\s*FOR\s*BUYER|BUYER\s*COPY|TAX\s*INVOICE)',
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

import re
from typing import List, Dict

def _extract_company_names_from_header(page_rows: List[List[Dict]]) -> List[str]:
    """
    Extract vendor/company name from the invoice header.
    Avoids Bill To / Ship To sections and filters invoice titles.
    """

    MAX_HEADER_ROWS = 35

    IGNORE_PATTERNS = re.compile(
        r"TAX\s*INVOICE|ORIGINAL|DUPLICATE|TRIPLICATE|PROFORMA|INVOICE\s*NO",
        re.I
    )

    STOP_PATTERNS = re.compile(
        r"BILL\s*TO|SHIP\s*TO|CUSTOMER.*ADDRESS|DELIVERY\s*ADDRESS|CONSIGNEE",
        re.I
    )

    LEGAL_SUFFIX = re.compile(
        r"\b(LTD|LIMITED|LLP|PRIVATE\s+LIMITED|PVT\s+LTD|PVT\.?\s+LTD)\b",
        re.I
    )

    VENDOR_HINTS = [
        "pharmaceutical",
        "laboratories",
        "healthcare",
        "medicare",
        "remedies",
        "specialities",
        "industries",
        "pharma",
        "biotech"
    ]

    # -----------------------------
    # Step 1 — Extract header rows
    # -----------------------------
    header_rows: List[List[Dict]] = []

    for row in page_rows[:MAX_HEADER_ROWS]:

        row_text = " ".join(c.get("text", "") for c in row).strip()

        if not row_text:
            continue

        if STOP_PATTERNS.search(row_text):
            break

        if IGNORE_PATTERNS.search(row_text):
            continue

        header_rows.append(row)

    search_rows = header_rows if header_rows else page_rows[:MAX_HEADER_ROWS]

    # -------------------------------------
    # Step 2 — Convert rows to plain text
    # -------------------------------------
    text_rows = [
        " ".join(c.get("text", "") for c in row).strip()
        for row in search_rows
    ]

    # -------------------------------------
    # Step 3 — Merge adjacent rows
    # (logo row + company name row case)
    # -------------------------------------
    merged_rows = []

    for i in range(len(text_rows)):
        merged_rows.append(text_rows[i])

        if i < len(text_rows) - 1:
            merged_rows.append(text_rows[i] + " " + text_rows[i + 1])

    # -------------------------------------
    # Step 4 — Try left supplier block first
    # -------------------------------------
    header_left_lines = _extract_supplier_header_left_lines(search_rows)

    if header_left_lines:

        # strict pass
        strict_left = _extract_company_candidates(
            [[{"text": l}] for l in header_left_lines],
            max_rows=10,
            require_legal_suffix=True
        )

        if strict_left:
            return strict_left[:1]

        # relaxed pass
        relaxed_left = _extract_company_candidates(
            [[{"text": l}] for l in header_left_lines],
            max_rows=10,
            require_legal_suffix=False
        )

        if relaxed_left:
            return relaxed_left[:1]

    # -------------------------------------
    # Step 5 — Strict legal suffix detection
    # Extract just the company name part, not extra trailing info
    for text in merged_rows[:25]:
        if LEGAL_SUFFIX.search(text):
            # Extract just the company name part (before common trailing info)
            company_only = re.sub(r'\s+(Delhi|Mumbai|Bangalore|NEW DELHI|EAST\s+DELHI|Cheque|Account|Bank|Address|Pin|State|City|Road|Street|Plot|Building|Phone|Mobile|Email|GSTIN|PAN).*', '', text, flags=re.IGNORECASE)
            if company_only and len(company_only) > 5:
                return [company_only.strip()]
            return [text.strip()]

    # Step 6 — Vendor keyword detection
    # Extract company name with vendor keywords
    for text in merged_rows[:25]:
        lower = text.lower()

        if any(v in lower for v in VENDOR_HINTS):
            # Extract just the company name, not trailing address/info
            company_only = re.sub(r'\s+(Delhi|Mumbai|Bangalore|NEW\s+DELHI|EAST\s+DELHI|Cheque|Account|Bank|Address|Pin|State|City|Road|Street|Plot|Building|Phone|Mobile|Email|GSTIN|PAN).*', '', text, flags=re.IGNORECASE)
            if company_only and len(company_only) > 5:
                return [company_only.strip()]
            return [text.strip()]

    # Step 6B — Multi-word business pattern detection
    # (words that look like business names even without legal suffixes)
    business_indicators = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?',  # Title case words (Company Name Ltd)
        r'\b([A-Z]{2,}\s+)+[A-Z]{2,}',  # Acronyms
        r'\b[A-Z][a-z]+\s+(AND|&)\s+[A-Z][a-z]+',  # Company & Co
    ]
    
    for text in merged_rows[:25]:
        for pattern in business_indicators:
            if re.search(pattern, text) and len(text) > 15:
                if not STOP_PATTERNS.search(text) and not re.search(r'\d{10,}', text):
                    # Extract just company name, not trailing info
                    company_only = re.sub(r'\s+(Delhi|Mumbai|Bangalore|NEW\s+DELHI|EAST\s+DELHI|Cheque|Account|Bank|Address|Pin|State|City|Road|Street|Plot|Building|Phone|Mobile|Email|GSTIN|PAN).*', '', text, flags=re.IGNORECASE)
                    if company_only and len(company_only) > 5:
                        return [company_only.strip()]
                    return [text.strip()]

    # Step 7 — Relaxed fallback (any multi-word non-label line)
    for text in merged_rows[:25]:
        if len(text.split()) >= 2 and len(text) < 120:
            if not STOP_PATTERNS.search(text):
                # Avoid pure numbers or too many numbers
                if re.sub(r'\D', '', text) != text and len(re.sub(r'\D', '', text)) < 10:
                    # Extract company name without trailing details
                    company_only = re.sub(r'\s+(Delhi|Mumbai|Bangalore|NEW\s+DELHI|EAST\s+DELHI|Cheque|Account|Bank|Address|Pin|State|City|Road|Street|Plot|Building|Phone|Mobile|Email|GSTIN|PAN).*', '', text, flags=re.IGNORECASE)
                    if company_only and len(company_only) > 5:
                        return [company_only.strip()]
                    return [text.strip()]

    # Step 8 — Emergency fallback: first substantial line
    for text in merged_rows[:30]:
        if len(text) > 8 and not STOP_PATTERNS.search(text) and len(text.split()) >= 1:
            if not re.search(r'^\d+$', text):  # Not just numbers
                # Extract company name without trailing details
                company_only = re.sub(r'\s+(Delhi|Mumbai|Bangalore|NEW\s+DELHI|EAST\s+DELHI|Cheque|Account|Bank|Address|Pin|State|City|Road|Street|Plot|Building|Phone|Mobile|Email|GSTIN|PAN).*', '', text, flags=re.IGNORECASE)
                if company_only and len(company_only) > 5:
                    return [company_only.strip()]
                return [text.strip()]

    return []
def _extract_bill_to_block_lines(page_rows: List[List[Dict]]) -> List[str]:
    """Extract text lines from Bill To/Ship To block until next section boundary."""
    block_lines: List[str] = []
    in_block = False

    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row).strip()
        if not row_text:
            continue

        if not in_block and re.search(r'\b(BILL\s*TO|SHIP\s*TO|SOLD\s*TO|CUSTOMER\s*DETAILS?)\b', row_text, re.IGNORECASE):
            in_block = True
            continue

        if not in_block:
            continue

        if re.search(r'\b(PRODUCT|DESCRIPTION|HSN|BATCH|QTY|UOM|RATE|AMOUNT|TOTAL|TERMS|CONDITION|INVOICE\s*NO|INVOICE\s*DATE)\b', row_text, re.IGNORECASE):
            break

        cleaned = re.sub(r'\s+', ' ', row_text).strip(' :-|')
        if cleaned:
            block_lines.append(cleaned)

    return block_lines


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
# SECTION 5B: VENDOR HEADER EXTRACTION HELPERS
# ════════════════════════════════════════════════════════════════════════════════
# Specialized functions to extract ONLY vendor-specific information from header
# Strictly avoids extracting from Bill To, Ship To, or Customer Details sections
# ════════════════════════════════════════════════════════════════════════════════

def _get_vendor_section_rows(page_rows: List[List[Dict]]) -> List[List[Dict]]:
    """
    Extract only vendor header rows (before Bill To, Ship To, Customer Details).
    Returns: rows that belong to vendor section only
    """
    vendor_rows: List[List[Dict]] = []
    found_customer_section = False

    for row in page_rows:
        row_text = " ".join(c.get("text", "") for c in row).strip()

        # Stop at customer/bill/ship section markers
        if re.search(r'\b(BILL\s*TO|SHIP\s*TO|CUSTOMER\s*DETAILS?|SOLD\s*TO|DELIVERY\s*ADDRESS)\b',
                     row_text, re.IGNORECASE):
            found_customer_section = True
            break

        # Stop at table headers (items start)
        if re.search(r'\b(PRODUCT|DESCRIPTION|HSN|BATCH|QTY|UOM|RATE|AMOUNT|TOTAL)\b',
                     row_text, re.IGNORECASE):
            found_customer_section = True
            break

        vendor_rows.append(row)

    return vendor_rows


def _is_vendor_label_line(row_text: str) -> bool:
    """Check if a line contains vendor information labels (GSTIN, PAN, CIN, etc.)"""
    vendor_labels = r'\b(GSTIN|PAN|CIN|Phone|Tel|Email|Address|Website|Mobile|FAX|DL\s*NO|FSSAI)\b'
    return bool(re.search(vendor_labels, row_text, re.IGNORECASE))


def _flatten_page_rows(page_rows: List[List[Dict]]) -> List[Dict]:
    """Step 1: flatten OCR rows while preserving bounding-box geometry."""
    return [cell for row in page_rows for cell in row if str(cell.get("text", "")).strip()]


def _divide_page_into_zones(page_rows: List[List[Dict]]) -> Dict[str, Dict[str, float]]:
    """Step 2: divide page into geometric zones used for vendor-block filtering."""
    cells = _flatten_page_rows(page_rows)
    if not cells:
        return {
            "top": {"y_max": 0.0},
            "left": {"x_max": 0.0},
            "right": {"x_min": 0.0},
        }

    max_x = max(float(c.get("x2", 0.0)) for c in cells)
    max_y = max(float(c.get("y2", 0.0)) for c in cells)

    # Enhanced: Use more generous boundaries to capture all vendor header info
    # even when vendor details appear in right column or lower header area
    return {
        "top": {"y_max": max_y * 0.78},  # Increased from 0.68 to capture more header rows
        "left": {"x_max": max_x * 0.90},  # Increased from 0.78 to capture wider left area
        "right": {"x_min": max_x * 0.10},  # Expanded to capture right-side vendor info
    }


def _is_title_noise_row(text: str) -> bool:
    """Identify invoice title rows that must be ignored before company extraction."""
    if not text:
        return True

    return bool(re.search(
        r'\b(TAX\s*INVOICE|ORIGINAL\s*FOR\s*BUYER|BUYER\s*COPY|DUPLICATE\s*FOR|'
        r'TRIPLICATE\s*FOR|COPY\s*FOR\s*TRANSPORTER|CASH\s*MEMO|RETAIL\s*INVOICE|'
        r'E\-?INVOICE|EWAY\s*BILL|ACKNOWLEDGEMENT)\b',
        str(text),
        re.IGNORECASE,
    ))


def _remove_buyer_ship_sections(page_rows: List[List[Dict]]) -> List[List[Dict]]:
    """Step 6: remove buyer/ship/customer sections before vendor extraction.
    
    Enhanced to be smarter: only break if we find a clear section marker
    AND we already have vendor content captured.
    """
    filtered_rows: List[List[Dict]] = []
    stop_markers = re.compile(
        r'\b(BILL\s*TO|SHIP\s*TO|SOLD\s*TO|CONSIGNEE|CUSTOMER\s*DETAILS?|DELIVERY\s*ADDRESS)\b',
        re.IGNORECASE,
    )

    for row_idx, row in enumerate(page_rows):
        row_text = " ".join(c.get("text", "") for c in row)
        
        # Only stop at section marker if we have meaningful content already
        # (to avoid stopping at header labels before vendor info appears)
        if stop_markers.search(row_text) and len(filtered_rows) > 3:
            break
            
        filtered_rows.append(row)

    return filtered_rows if filtered_rows else page_rows


def _detect_vendor_block(page_rows: List[List[Dict]], zones: Dict[str, Dict[str, float]]) -> List[List[Dict]]:
    """Step 3: detect vendor block from upper header region.

    Enhanced 4-pass strategy to ensure we capture vendor info even in sparse documents:
    1. Strict: use zone boundaries
    2. Relaxed: 1.40x threshold, capture more rows
    3. Aggressive: include all rows until table/delivery marker
    4. Last resort: return all page rows if nothing substantive found
    """
    vendor_rows = _remove_buyer_ship_sections(page_rows)
    top_y_max = zones.get("top", {}).get("y_max", 0.0)

    # Pass 1: Strict zone-based capture
    block: List[List[Dict]] = []
    for row in vendor_rows:
        keep_cells = [
            c for c in row
            if float(c.get("yc", 0.0)) <= top_y_max
        ]
        if keep_cells:
            block.append(keep_cells)

    # Pass 2: If capture is too narrow, relax the y threshold
    if len(block) < 5 and vendor_rows:
        relaxed_block: List[List[Dict]] = []
        for row in vendor_rows:
            keep_cells = [
                c for c in row
                if float(c.get("yc", 0.0)) <= (top_y_max * 1.40)
            ]
            if keep_cells:
                relaxed_block.append(keep_cells)
        if relaxed_block:
            block = relaxed_block

    # Pass 3: Aggressive fallback - include rows up to table start
    if len(block) < 8 and vendor_rows:
        aggressive_block: List[List[Dict]] = []
        for row_idx, row in enumerate(vendor_rows):
            row_text = " ".join(c.get("text", "") for c in row).strip()
            # Stop at first table marker (HSN, BATCH, QTY, etc.) or delivery marker
            if re.search(r'\b(PRODUCT|DESCRIPTION|HSN|BATCH|QTY|UOM|MRP|RATE|AMOUNT|TOTAL|DELIVERY|SHIP|CUSTOMER)\b', row_text, re.IGNORECASE):
                # But only break if we have some content. Title rows alone don't count
                content_count = len([r for r in aggressive_block if not _is_title_noise_row(" ".join(c.get("text", "") for c in r))])
                if content_count >= 2:
                    break
            if row_text:
                aggressive_block.append(row)
        if len(aggressive_block) > len(block):
            block = aggressive_block

    # Pass 4: Last resort - if still empty or only title rows, return all vendor rows
    # This ensures we have content to work with even on sparse invoices
    if (not block or len([r for r in block if not _is_title_noise_row(" ".join(c.get("text", "") for c in r))]) == 0) and vendor_rows:
        block = vendor_rows

    return block if block else vendor_rows


def _extract_regex_candidates(vendor_block_rows: List[List[Dict]]) -> Dict[str, List[Dict[str, Any]]]:
    """Step 4: extract field candidates using regex from vendor block rows."""
    patterns = {
        "gstin": re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z]\dZ[A-Z0-9]", re.IGNORECASE),
        "pan": re.compile(r'\b[A-Z]{3}[PCHFTABLJG][A-Z]\d{4}[A-Z]\b', re.IGNORECASE),
        "cin_no": re.compile(r'\b[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b', re.IGNORECASE),
        "email_id": re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', re.IGNORECASE),
        "phone": re.compile(r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b', re.IGNORECASE),
    }

    candidates: Dict[str, List[Dict[str, Any]]] = {k: [] for k in patterns}

    for row in vendor_block_rows:
        row_text = " ".join(c.get("text", "") for c in row)
        row_y = min(float(c.get("yc", 0.0)) for c in row) if row else 0.0
        for key, pattern in patterns.items():
            for match in pattern.findall(row_text):
                candidates[key].append({
                    "value": str(match).strip(),
                    "row_text": row_text,
                    "row_y": row_y,
                })

    return candidates


def _score_vendor_candidate(
    regex_match: bool,
    x_dist: float,
    y_dist: float,
    in_vendor_section: bool,
) -> float:
    """Combined score: regex match + spatial proximity + section filtering."""
    regex_score = 50.0 if regex_match else 0.0
    proximity_score = max(0.0, 40.0 - (x_dist * 0.20) - (y_dist * 2.5))
    section_score = 10.0 if in_vendor_section else -100.0
    return regex_score + proximity_score + section_score


def _best_proximity_value(
    vendor_block_rows: List[List[Dict]],
    label_pattern: str,
    value_pattern: str,
    cleaner: Optional[Any] = None,
    validator: Optional[Any] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Step 5: combined scoring (regex + proximity + section filter) and return best value.

    Looks for the value:
    1. To the right of the label on the same row (inline format: "GSTIN: 27AABC...")
    2. On the immediately following row below the label (stacked format: "GSTIN:\\n27AABC...")
    """
    best_value: Optional[str] = None
    best_score = -1.0
    score_details: List[Dict[str, Any]] = []

    label_re = re.compile(label_pattern, re.IGNORECASE)
    value_re = re.compile(value_pattern, re.IGNORECASE)

    for row_idx, row in enumerate(vendor_block_rows):
        sorted_row = sorted(row, key=lambda c: c.get("x1", 0))
        for idx, cell in enumerate(sorted_row):
            text = str(cell.get("text", ""))
            if not label_re.search(text):
                continue

            label_x2 = float(cell.get("x2", 0.0))
            label_yc = float(cell.get("yc", 0.0))

            # --- Pass 1: same-row cells to the right ---
            same_row_candidates = sorted_row[idx + 1:]

            # --- Pass 2: first cell(s) on the next row (stacked label/value layout) ---
            next_row_candidates: List[Dict] = []
            if row_idx + 1 < len(vendor_block_rows):
                next_row = sorted(vendor_block_rows[row_idx + 1], key=lambda c: c.get("x1", 0))
                next_row_candidates = next_row

            for cand in same_row_candidates + next_row_candidates:
                cand_text = str(cand.get("text", "")).strip(" :-|\t")
                if not cand_text:
                    continue

                regex_match = bool(value_re.search(cand_text))
                if not regex_match:
                    continue

                if validator is not None:
                    try:
                        if not validator(cand_text):
                            continue
                    except Exception:
                        continue

                x_dist = max(0.0, float(cand.get("x1", 0.0)) - label_x2)
                y_dist = abs(float(cand.get("yc", 0.0)) - label_yc)
                score = _score_vendor_candidate(
                    regex_match=regex_match,
                    x_dist=x_dist,
                    y_dist=y_dist,
                    in_vendor_section=True,
                )

                score_details.append({
                    "label": text,
                    "candidate": cand_text,
                    "x_dist": round(x_dist, 2),
                    "y_dist": round(y_dist, 2),
                    "regex_match": regex_match,
                    "section_filtered": True,
                    "combined_score": round(score, 2),
                })

                if score > best_score:
                    best_score = score
                    best_value = cand_text

    if best_value is None:
        return None, score_details
    return (cleaner(best_value) if cleaner else best_value), score_details


def _extract_vendor_header_six_steps(page_rows: List[List[Dict]]) -> Dict[str, Any]:
    """Enhanced pipeline with comprehensive fallback patterns to minimize nulls.
    
    When normal extractors fail, falls back to:
    - Searching full page_rows (not just vendor_block)
    - Text mining for any matching patterns
    - First substantive text as company name
    """
    header = {field: None for field in HEADER_FIELDS}

    # Step 1 + Step 2 + Step 6
    _ = _flatten_page_rows(page_rows)
    zones = _divide_page_into_zones(page_rows)
    cleaned_rows = _remove_buyer_ship_sections(page_rows)

    # Step 3
    vendor_block = _detect_vendor_block(cleaned_rows, zones)
    
    # Keep reference to full page content for fallback
    full_page_text = " ".join(c.get("text", "") for row in page_rows for c in row)

    # Step 4 (regex candidates) — scan vendor_block first, then broaden to all
    # cleaned_rows and full page as fallback
    regex_candidates = _extract_regex_candidates(vendor_block)
    regex_candidates_broad = _extract_regex_candidates(cleaned_rows)
    regex_candidates_full = _extract_regex_candidates(page_rows)
    
    # Merge: prefer vendor_block hits, then cleaned_rows, then full page
    for key in regex_candidates:
        if not regex_candidates[key] and regex_candidates_broad[key]:
            regex_candidates[key] = regex_candidates_broad[key]
        elif not regex_candidates[key] and regex_candidates_full[key]:
            regex_candidates[key] = regex_candidates_full[key]
    
    proximity_trace: Dict[str, Any] = {}

    # Step 5 (proximity scoring) with enhanced fallbacks
    header["gstin"], proximity_trace["gstin"] = _best_proximity_value(
        vendor_block,
        r'\bGSTIN\b|\bGST\s*IN|\bGST\s*NUMBER',
        r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b',
        cleaner=DataCleaner.extract_gstin,
        validator=_is_valid_gstin_text,
    )
    if header["gstin"] is None and regex_candidates["gstin"]:
        header["gstin"] = DataCleaner.extract_gstin(regex_candidates["gstin"][0]["value"])
    # Fallback: search entire page text for GSTIN pattern
    if header["gstin"] is None:
        gstin_match = re.search(r'\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]', full_page_text)
        if gstin_match:
            header["gstin"] = DataCleaner.extract_gstin(gstin_match.group(0))

    header["pan"], proximity_trace["pan"] = _best_proximity_value(
        vendor_block,
        r'\bPAN\b|\bPAN\s*NO|\bP\.A\.N',
        r'\b[A-Z]{5}\d{4}[A-Z]\b',
        cleaner=DataCleaner.extract_pan,
        validator=_is_valid_pan_text,
    )
    if header["pan"] is None and regex_candidates["pan"]:
        header["pan"] = DataCleaner.extract_pan(regex_candidates["pan"][0]["value"])
    # Fallback: search entire page text for PAN pattern
    if header["pan"] is None:
        pan_match = re.search(r'[A-Z]{5}\d{4}[A-Z]', full_page_text)
        if pan_match:
            header["pan"] = DataCleaner.extract_pan(pan_match.group(0))

    header["cin_no"], proximity_trace["cin_no"] = _best_proximity_value(
        vendor_block,
        r'\bCIN\b|\bCIN\s*NO|\bCIN\s*NUMBER',
        r'\b[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b',
        cleaner=DataCleaner.clean_string,
    )
    if header["cin_no"] is None and regex_candidates["cin_no"]:
        header["cin_no"] = DataCleaner.clean_string(regex_candidates["cin_no"][0]["value"])
    # Fallback: search entire page for CIN pattern
    if header["cin_no"] is None:
        cin_match = re.search(r'[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}', full_page_text)
        if cin_match:
            header["cin_no"] = DataCleaner.clean_string(cin_match.group(0))

    header["email_id"], proximity_trace["email_id"] = _best_proximity_value(
        vendor_block,
        r'\b(EMAIL|E\-?MAIL)\b',
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        cleaner=DataCleaner.clean_string,
    )
    if header["email_id"] is None and regex_candidates["email_id"]:
        header["email_id"] = DataCleaner.clean_string(regex_candidates["email_id"][0]["value"])
    # Fallback: search entire page for email pattern
    if header["email_id"] is None:
        email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', full_page_text)
        if email_match:
            header["email_id"] = DataCleaner.clean_string(email_match.group(0))

    header["phone"], proximity_trace["phone"] = _best_proximity_value(
        vendor_block,
        r'\b(PHONE|TEL|MOBILE|CONTACT|PH|MOB|TELEPHONE)\b',
        r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
        cleaner=DataCleaner.clean_string,
    )
    if header["phone"] is None and regex_candidates["phone"]:
        header["phone"] = DataCleaner.clean_string(regex_candidates["phone"][0]["value"])
    # Fallback: search entire page for phone pattern
    if header["phone"] is None:
        phone_match = re.search(r'(?:\+91[-\s]?)?[6-9]\d{9}', full_page_text)
        if phone_match:
            header["phone"] = DataCleaner.clean_string(phone_match.group(0))

    # Filter out title noise rows before company extraction
    company_rows: List[List[Dict]] = []
    removed_title_rows: List[str] = []
    for row in vendor_block:
        row_text = " ".join(c.get("text", "") for c in row).strip()
        if _is_title_noise_row(row_text):
            removed_title_rows.append(row_text)
            continue
        company_rows.append(row)

    # Vendor name and address from filtered vendor block
    if company_rows:
        company_names = _extract_company_names_from_header(company_rows)
        if company_names:
            header["company_name"] = DataCleaner.clean_string(company_names[0])

    # Fallback: extract first substantial non-title line as company name
    if header["company_name"] is None and vendor_block:
        for row in vendor_block:
            row_text = " ".join(c.get("text", "") for c in row).strip()
            if not _is_title_noise_row(row_text) and len(row_text) > 10 and not re.search(r'GSTIN|PAN|EMAIL|PHONE', row_text, re.IGNORECASE):
                header["company_name"] = DataCleaner.clean_string(row_text)
                break

    # Emergency fallback: search full page for company name if still null
    if header["company_name"] is None and cleaned_rows:
        for row in cleaned_rows:
            row_text = " ".join(c.get("text", "") for c in row).strip()
            if not _is_title_noise_row(row_text) and len(row_text) > 8 and not re.search(r'GSTIN|PAN|EMAIL|PHONE|PRODUCT|DESCRIPTION|HSN|BATCH', row_text, re.IGNORECASE):
                if len(row_text.split()) >= 2:
                    header["company_name"] = DataCleaner.clean_string(row_text)
                    break

    header["address"] = _extract_vendor_address(company_rows if company_rows else vendor_block)
    
    # Fallback address extraction from cleaned rows if nothing found
    if header["address"] is None and cleaned_rows:
        header["address"] = _extract_vendor_address(cleaned_rows)

    _log_header_stage("ZONES", zones)
    _log_header_stage("VENDOR_BLOCK_ROWS", [" ".join(c.get("text", "") for c in row).strip() for row in vendor_block])
    _log_header_stage("TITLE_ROWS_REMOVED", removed_title_rows)
    _log_header_stage("REGEX_CANDIDATES", regex_candidates)
    _log_header_stage("PROXIMITY_SCORING", proximity_trace)
    _log_header_stage("PROXIMITY_OUTPUT", header)

    return header


def _extract_vendor_gstin(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """Extract GSTIN from vendor header section only"""
    gstin_text, _ = _extract_labeled_value(
        vendor_rows,
        r'\bGSTIN\b|\bGST\s*IN|\bGST\s*NUMBER',
        validator=_is_valid_gstin_text,
    )

    if gstin_text:
        return DataCleaner.extract_gstin(gstin_text)
    return None


def _extract_vendor_pan(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """Extract PAN from vendor header section only"""
    pan_text, _ = _extract_labeled_value(
        vendor_rows,
        r'\bPAN\b|\bPAN\s*NO|\bP\.A\.N',
        validator=_is_valid_pan_text,
    )

    if pan_text:
        return DataCleaner.extract_pan(pan_text)
    return None


def _extract_vendor_cin(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """
    Extract CIN (Corporate Identification Number) from vendor header.
    CIN format: U12345AB1234CIN000 (21 characters, specific pattern)
    """
    cin_text, _ = _extract_labeled_value(
        vendor_rows,
        r'\bCIN\b|\bCIN\s*NO|\bCIN\s*NUMBER',
        validator=lambda t: bool(re.search(r'[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{3}', str(t))),
    )

    if cin_text:
        # Extract CIN pattern
        match = re.search(r'[A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{3}', str(cin_text).upper())
        if match:
            return match.group(0)
    return None


def _extract_vendor_phone(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """
    Extract phone/mobile number from vendor header.
    Matches: 10-digit numbers, formats like (123) 456-7890, +91 numbers, etc.
    """
    # Try labeled extraction first (Phone, Tel, Mobile)
    phone_text, _ = _extract_labeled_value(
        vendor_rows,
        r'\b(PHONE|TEL|MOBILE|CONTACT|PH|MOB|TELEPHONE)\b',
    )

    if phone_text:
        # Extract phone number pattern from text
        phone_match = re.search(
            r'(\+91[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}|\d{10}',
            str(phone_text).replace(" ", "")
        )
        if phone_match:
            return phone_match.group(0)

    # Fallback: search for 10-digit numbers in vendor section
    vendor_text = " ".join(cell.get("text", "") for row in vendor_rows for cell in row)
    phones = re.findall(r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b', vendor_text)
    if phones:
        return phones[0]

    return None


def _extract_vendor_email(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """Extract email ID from vendor header"""
    # Try labeled extraction first (Email, Contact, etc.)
    email_text, _ = _extract_labeled_value(
        vendor_rows,
        r'\b(EMAIL|E\-?MAIL|CONTACT|MSG)\b',
    )

    if email_text:
        # Validate email pattern
        email_match = re.search(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            str(email_text)
        )
        if email_match:
            return email_match.group(0)

    # Fallback: search for email pattern in vendor section
    vendor_text = " ".join(cell.get("text", "") for row in vendor_rows for cell in row)
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', vendor_text)
    if emails:
        return emails[0]

    return None


def _extract_vendor_address(vendor_rows: List[List[Dict]]) -> Optional[str]:
    """
    Enhanced vendor address extraction with intelligent filtering.
    Excludes: labels, table headers, line items, invoice data.
    Collects only substantive address components.
    """
    address_lines: List[str] = []
    
    # Patterns for content to skip
    skip_patterns = re.compile(
        r'^\s*(GSTIN|PAN|CIN|EMAIL|PHONE|MOBILE|TEL|WEBSITE|DL\s*NO|FSSAI|'
        r'COMPANY|NAME|OFFICE|REGISTERED|STATE|COUNTRY|PIN|DOOR|FLAT|'
        r'SUITE|PLOT|ROAD|STREET|ADDRESS|BANK|ACCOUNT|IFSC|INVOICE|'
        r'HSN|BATCH|QTY|UOM|MRP|PTR|CGST|SGST|IGST|RATE|AMOUNT|'
        r'TOTAL|BILLED|DIVISION|METABOLICS|SHIPPER|WEIGHT|DISPATCH|'
        r'INSUR|POLICY|MODE|COURIER|CASES|AWB|EWAY|IRN|DUE|DATE|'
        r'PRODUCT\s*DESCRIPTION|CHEQUE).{0,30}$',
        re.IGNORECASE,
    )
    
    # Patterns for table/line item content
    table_patterns = re.compile(
        r'(Mfg\s+Name|Product\s+Description|Batch\s+No|Date|Boxes|Qty|PT\.?R|MRP|'
        r'Total\s+Value|Rate\s+of|Amount|Sub\s+Total|Disc|Taxable|CGST|SGST|IGST|'
        r'Round\s+off|Invoice\s+Total|Payment\s+Mode|Division|\d{10,})',
        re.IGNORECASE,
    )

    for row in vendor_rows:
        row_text = " ".join(c.get("text", "") for c in row).strip()

        if not row_text or len(row_text) < 5:
            continue

        # Skip rows matching skip patterns
        if skip_patterns.search(row_text):
            continue

        # Skip rows with too much table/line item content
        table_match_count = len(re.findall(table_patterns, row_text))
        if table_match_count >= 3:  # Too many table-like terms
            continue

        # Skip header rows
        if _is_title_noise_row(row_text):
            continue

        # Skip invoice heading/noise lines
        if re.search(r'\b(ORIGINAL\s*FOR\s*BUYER|TAX\s*INVOICE|BUYER\s*COPY|INVOICE\s*NO|BILL\s*TO|SHIP\s*TO|CUSTOMER|DELIVERY)\b', row_text, re.IGNORECASE):
            continue

        # Reject lines with too many numbers (likely data rows)
        digit_count = len(re.findall(r'\d', row_text))
        if digit_count > len(row_text) * 0.4:  # More than 40% digits
            continue

        # Check letter ratio to filter noise
        letter_ratio = len(re.findall(r'[A-Za-z]', row_text)) / max(len(row_text), 1)
        if letter_ratio < 0.25:  # Too few letters
            continue

        # Additional check: exclude lines that look like table headers or payment info
        if re.search(r'\b(Account|Bank|IFSC|Remarks|Terms|Condition|Cheque|Payment|Favour)\b', row_text, re.IGNORECASE):
            continue

        # Deduplicate: check if line is substantially similar to existing lines
        normalized = re.sub(r'[^A-Za-z0-9]', '', row_text.lower())
        is_duplicate = False
        for existing in address_lines:
            existing_normalized = re.sub(r'[^A-Za-z0-9]', '', existing.lower())
            if normalized and existing_normalized:
                # Check overlap ratio
                overlap = len(set(existing_normalized) & set(normalized)) / max(len(existing_normalized), len(normalized))
                if overlap > 0.8:  # High similarity
                    is_duplicate = True
                    break

        if not is_duplicate:
            address_lines.append(row_text)

    # Keep top 4 most informative lines (prefer those with city/state info)
    if len(address_lines) > 4:
        # Score lines by address-like content
        def address_score(line: str) -> float:
            score = len(line)  # Prefer longer, more complete lines
            # Boost score for location indicators
            if re.search(r'\b(Delhi|Mumbai|Bangalore|New Delhi|Road|Street|Plot|Building|Pin|Postal|Code)\b', line, re.IGNORECASE):
                score += 50
            return score
        
        address_lines.sort(key=address_score, reverse=True)
        address_lines = address_lines[:4]

    if address_lines:
        # Join with comma and space for better formatting
        combined = ", ".join(address_lines).strip()
        # Clean up excessive spacing and special chars at boundaries
        combined = re.sub(r',\s*,', ',', combined)
        combined = re.sub(r'^\s*[,:;-]+\s*', '', combined)
        combined = re.sub(r'\s*[,:;-]+\s*$', '', combined)
        combined = re.sub(r'\s+', ' ', combined)  # Normalize spacing
        return combined if len(combined) >= 8 else None

    return None


def _collect_vendor_rows_text(page_rows: List[List[Dict]]) -> List[str]:
    """Fix 4: collect larger vendor-safe header context for LLM."""
    zones = _divide_page_into_zones(page_rows)
    cleaned_rows = _remove_buyer_ship_sections(page_rows)

    top_limit = zones.get("top", {}).get("y_max", 0.0) * 1.22
    context_rows: List[str] = []

    for row in cleaned_rows:
        if not row:
            continue

        row_y = min(float(c.get("yc", 0.0)) for c in row)
        if row_y > top_limit:
            continue

        row_text = " ".join(c.get("text", "") for c in row).strip()
        row_text = re.sub(r"\s+", " ", row_text)
        if not row_text:
            continue

        # Keep title rows only as fallback context, prioritize informative rows.
        if _is_title_noise_row(row_text):
            continue

        context_rows.append(row_text)

    # If still tiny, fallback to all cleaned header rows (except buyer/ship block).
    if len(context_rows) < 4:
        context_rows = []
        for row in cleaned_rows:
            row_text = " ".join(c.get("text", "") for c in row).strip()
            row_text = re.sub(r"\s+", " ", row_text)
            if row_text and not _is_title_noise_row(row_text):
                context_rows.append(row_text)

    # Final fallback: include cleaned rows even if they are title lines to avoid empty context.
    if not context_rows:
        for row in cleaned_rows:
            row_text = " ".join(c.get("text", "") for c in row).strip()
            row_text = re.sub(r"\s+", " ", row_text)
            if row_text:
                context_rows.append(row_text)

    # Deduplicate while preserving order and cap to keep prompts bounded.
    deduped: List[str] = []
    seen = set()
    for line in context_rows:
        key = re.sub(r"\W+", "", line.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    return deduped[:40]


def _log_header_stage(stage: str, payload: Any) -> None:
    """Structured logger for header extraction/refinement stages."""
    print(f"\n[HEADER][{stage}]")
    try:
        if isinstance(payload, str):
            print(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        print(str(payload))


def refine_header_with_openrouter(
    header: Dict[str, Any],
    vendor_rows_text: List[str],
) -> Dict[str, Any]:
    """Optionally refine vendor header fields with OpenRouter and log all stages."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        _log_header_stage("LLM_INPUT", {
            "status": "skipped",
            "reason": "OPENROUTER_API_KEY/API_KEY not set",
            "current_header": header,
        })
        _log_header_stage("LLM_OUTPUT", {
            "status": "skipped",
            "output": header,
        })
        return header

    model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    detected_fields = [field for field in HEADER_FIELDS if header.get(field) is not None]
    null_fields = [field for field in HEADER_FIELDS if header.get(field) is None]

    system_prompt = (
        "You are an expert invoice header validator. Refine ONLY vendor/supplier header fields.\n"
        "Never use Bill To, Ship To, customer details, consignee, or delivery sections.\n"
        "Return ONLY a valid JSON object with exact keys: "
        "company_name, gstin, pan, cin_no, phone, address, email_id.\n"
        "Do not add extra keys.\n"
        "Do NOT hallucinate. For fields that are null and not clearly present, keep null.\n"
        "Only refine values using provided vendor_rows_text."
    )

    user_payload = {
        "allowed_to_refine": detected_fields,
        "must_stay_null_if_not_certain": null_fields,
        "vendor_rows_text": vendor_rows_text,
        "current_header": header,
    }

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0,
    }

    _log_header_stage("LLM_INPUT", request_payload)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        req = request.Request(
            endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=45) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE | re.MULTILINE).strip()
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            _log_header_stage("LLM_OUTPUT", {
                "status": "ignored",
                "reason": "LLM response is not a JSON object",
                "raw_content": content,
            })
            return header

        validated = dict(header)
        allowed_fields = set(detected_fields)

        for key in HEADER_FIELDS:
            if key not in parsed:
                continue
            value = parsed.get(key)
            if key not in allowed_fields and value is not None:
                continue
            validated[key] = value

        validated["company_name"] = DataCleaner.clean_string(validated.get("company_name"))
        validated["gstin"] = DataCleaner.extract_gstin(validated.get("gstin"))
        validated["pan"] = DataCleaner.extract_pan(validated.get("pan"))
        validated["cin_no"] = DataCleaner.clean_string(validated.get("cin_no"))
        validated["phone"] = DataCleaner.clean_string(validated.get("phone"))
        validated["address"] = DataCleaner.clean_string(validated.get("address"))
        validated["email_id"] = DataCleaner.clean_string(validated.get("email_id"))

        _log_header_stage("LLM_OUTPUT", {
            "status": "applied",
            "raw_content": content,
            "validated_header": validated,
        })
        return validated

    except (error.URLError, error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
        _log_header_stage("LLM_OUTPUT", {
            "status": "failed",
            "error": str(exc),
            "fallback_header": header,
        })
        return header


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6: HEADER EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

def extract_invoice_header(page_rows: List[List[Dict]]) -> Tuple[Dict, float]:
    """
    Extract ONLY vendor/supplier header information.
    Strictly avoids extracting from Bill To, Ship To, Customer Details, or other sections.

    Returns: (header_dict, accuracy_score)

    Vendor header fields:
    - company_name: Vendor/supplier company name
    - gstin: GST Identification Number
    - pan: Permanent Account Number
    - cin_no: Corporate Identification Number
    - phone: Vendor phone/mobile number
    - address: Vendor address
    - email_id: Vendor email address
    """
    scorer = AccuracyScorer()
    header = _extract_vendor_header_six_steps(page_rows)
    accuracy = scorer.score_section(header)

    return header, accuracy


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

            # Extract billed_qty (primary quantity for billing)
            if current_item["billed_qty"] is None and row_values.get("billed_qty"):
                billed_qty_num = cleaner.extract_number(row_values.get("billed_qty"))
                if billed_qty_num is not None:
                    current_item["billed_qty"] = int(billed_qty_num)

            # Extract free_qty (bonus/gift quantity - optional)
            if current_item["free_qty"] is None and row_values.get("free_qty"):
                free_qty_num = cleaner.extract_number(row_values.get("free_qty"))
                if free_qty_num is not None:
                    current_item["free_qty"] = int(free_qty_num)

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

            # Improve GST parsing: prefer inferred tax rates from row text.
            # Handles CGST/SGST pairs, combined TAX%, and IGST
            inferred_cgst, inferred_sgst, inferred_tax_rate, inferred_igst = _extract_tax_rates_from_row_text(row_text)
            
            # Only override if we have extracted values, and prioritize what was clearly detected
            if inferred_igst is not None:
                # IGST takes precedence (cross-state sales)
                if current_item.get("igst") is None:
                    current_item["igst"] = inferred_igst
                    # Clear CGST/SGST when IGST is used (mutually exclusive)
                    current_item["cgst"] = None
                    current_item["sgst"] = None
            elif inferred_tax_rate is not None:
                # Combined tax rate (both CGST and SGST apply with same percentage)
                if current_item.get("tax_rate") is None:
                    current_item["tax_rate"] = inferred_tax_rate
                    # When tax_rate is used, CGST/SGST stay null (per repo improvements)
                    current_item["cgst"] = None
                    current_item["sgst"] = None
            elif inferred_cgst is not None and inferred_sgst is not None:
                # Paired CGST/SGST rates
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

    header, _ = extract_invoice_header(page_rows[0])
    _log_header_stage("RAW_EXTRACTION", header)

    vendor_rows_text = _collect_vendor_rows_text(page_rows[0])
    _log_header_stage("VENDOR_ROWS_CONTEXT", vendor_rows_text)

    line_items, item_accs = extract_line_items(page_rows)

    table_rows_text = _collect_table_rows_text(page_rows)
    
    # UNIFIED LLM REFINEMENT: Single call for both header and line items
    header, line_items = refine_extraction_with_openrouter(
        header, 
        line_items, 
        vendor_rows_text, 
        table_rows_text
    )
    
    line_items = normalize_line_items(line_items)
    
    scorer = AccuracyScorer()
    header_acc = scorer.score_section(header)
    
    print(f"  ✓ Header extracted (accuracy: {header_acc}%)")
    print(f"  ✓ {len(line_items)} line items extracted")
    
    # Step 4: Compile output
    print("\n[4/4] Compiling output...")
    
    output_data = {
        "source_file": pdf_path.name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "invoice_header": header,
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

    print(f"  Items Accuracy:   {sum(item_accs)/len(item_accs):.2f}%" if item_accs else "  Items: None")
    all_accuracies = [header_acc] + item_accs
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