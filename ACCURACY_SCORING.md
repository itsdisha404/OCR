# Improved Accuracy Scoring Documentation

## Overview
Accuracy scores now reflect **actual extraction quality** rather than hardcoded values. Each item and section is scored based on individual field validation and cross-field consistency checks.

---

## Line Item Accuracy Scoring

### Field-Level Scoring (0-100 per field)

#### Product Description
- **100**: Length ≥3 chars AND contains non-numeric content
- **50**: Has value but short or mostly numeric
- **0**: Missing

#### HSN Code
- **100**: Exactly 8 digits (valid format)
- **40**: Has value but wrong format
- **0**: Missing

#### Batch Number
- **100**: 3-14 chars (reasonable batch code length)
- **40**: Has value but unusual length
- **0**: Missing

#### Expiry Date
- **100**: Valid format MM/YYYY or DD/MM/YYYY
- **40**: Has date but wrong format
- **20**: Missing (acceptable for some items)

#### Quantity
- **100**: Positive integer/float
- **0**: Missing or non-positive

#### MRP (Maximum Retail Price)
- **100**: 5 ≤ MRP ≤ 5000 (valid pharmaceutical price range)
- **40**: Has value but outside range
- **0**: Missing

#### PTR (Preferred Trade Rate)
- **100**: 1 ≤ PTR ≤ 5000
- **40**: Has value but outside range
- **0**: Missing

#### PTS (Pharmacy Trade Scheme)
- **100**: 1 ≤ PTS ≤ 5000
- **40**: Has value but outside range
- **0**: Missing

#### Discount
- **100**: 0 ≤ Discount ≤ 100 (valid percentage)
- **50**: Has value but unrealistic
- **0**: Missing

#### CGST (Central GST)
- **100**: 0 ≤ CGST ≤ 28 (valid tax range)
- **40**: Has value but outside range
- **0**: Missing

#### SGST (State GST)
- **100**: 0 ≤ SGST ≤ 28
- **40**: Has value but outside range
- **0**: Missing

#### Total Amount
- **100**: Qty × PTR × 0.7 ≤ Total ≤ Qty × MRP × 1.1 (plausible range)
- **50**: Between 0.5× and 2× plausible range
- **10**: Way off (< 0.5× or > 2× high estimate)
- **0**: Missing or negative

### Overall Item Score Calculation

1. **Critical Field Average**: Average of 7 critical fields
   - product_description, hsn_code, qty, mrp, total_amount, cgst, sgst

2. **Penalties**:
   - MRP < PTR: -5 points (price hierarchy violation)
   - Discount < 0 or > 100: -10 points (invalid percentage)

3. **Final Score**: critical_average - penalties, clamped to [0, 100]

### Why This Works Better

**Old Method (Hardcoded 80.0):**
- All items scored equally regardless of extraction quality
- Items with MRP=92, total_amount=504045 still scored 80.0
- Could not identify broken extraction

**New Method (Dynamic):**
- Item with MRP=92, qty=10, total_amount=504045:
  - total_score = 10 (way outside plausible 644-1012 range)
  - critical_avg = (100+100+100+100+10+100+100)/7 ≈ 72.86
  - Final = 72.86 (accurately reflects poor extraction)
  
- Item with MRP=92, qty=10, total_amount=662.4:
  - total_score = 100 (within plausible range 644-1012)
  - critical_avg = 100
  - Final = 100 (correctly reflects good extraction)

---

## Section Accuracy Scoring (Header & Bill-To)

### Header Section Fields
- **company_name**: 100 if ≥5 chars; 0 otherwise (critical)
- **pan**: 100 if matches [A-Z]₅[0-9]₄[A-Z]; 50 if present but wrong format; 0 missing
- **invoice_no**: 100 if ≥3 chars; 20 if missing (nice-to-have)
- **invoice_date**: 100 if valid DD/MM/YYYY; 50 if wrong format; 0 if missing

Section score = Average of all field scores

### Bill-To Section Fields
- **name**: 100 if list with items or string ≥5 chars; 0 otherwise (critical)
- **address**: 100 if ≥10 chars; 50 if shorter; 20 if missing (nice-to-have)

Section score = Average of all field scores

---

## Interpretation Guide

| Score Range | Meaning | Action |
|---|---|---|
| 90-100 | Excellent extraction | Use directly |
| 70-89 | Good extraction, minor issues | Review if critical fields involved |
| 50-69 | Moderate extraction quality | Manual review recommended |
| 20-49 | Poor extraction, multiple issues | Inspect and potentially re-scan |
| 0-19 | Critical failure, unusable data | Re-scan required |

---

## Examples from J.B.pdf

### Row 0: "BISOTAB 2 SMG 1X10"
```
product_description: "BISOTAB 2 SMG 1X10" → 100
hsn_code: "30049099" → 100
qty: 10 → 100
mrp: 92.0 → 100
total_amount: 49.0 (plausible range 644-1012) → 10 ✗
cgst: 6.0 → 100
sgst: 6.0 → 100
critical_avg = (100+100+100+100+10+100+100)/7 ≈ 72.86
Final Score: 72.86
```

### Row 4: "CILACAR TC 6.25MG"
```
product_description: "CILACAR TC 6.25MG" → 100
hsn_code: "30049099" → 100
qty: 10 → 100
mrp: 248.0 → 100
total_amount: 504045.0 (plausible range 1234-2728) → 10 ✗✗✗
cgst: 6.0 → 100
sgst: 6.0 → 100
critical_avg ≈ 72.86
Final Score: 72.86 (ERROR: Should be much lower!)
```
**Note**: This highlights that the extraction itself is broken—total_amount is completely wrong.

---

## Next Steps

1. Run `processor.py` on J.B.pdf to generate updated scores
2. Review items with scores < 60 for extraction quality issues
3. Focus on fixing fields that are marked 0-50 (missing or invalid)
4. Validate that realistic totals now show 100, unrealistic show 10

