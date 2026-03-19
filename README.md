# OCR Machine 📄

एक **PDF processing aur data extraction** ka tool jo automatically invoices, bills, aur documents se data nikaal ke tabular format mein deta hai!

## What is OCR Machine? (यह क्या है?)

OCR Machine yaani **Optical Character Recognition** machine - ek intelligent tool jo:

- 📄 PDF files ko padh leta hai
- 👀 Text aur structured data (jaise invoice details) identify karta hai  
- 💾 Data ko JSON, CSV, aur Excel formats mein save karta hai
- 🤖 Automatically file changes detect karke processing shakil karta hai

**Especially useful for** : Pharmaceutical invoices, billing documents, procurement records

---

## Features ✨

### Data Extraction
- **Header Information**: Company name, PAN, GSTIN, Invoice number, Dates
- **Bill Details**: Customer name, address, GSTIN, Delivery details
- **Line Items**: Product descriptions, pricing, taxes, quantities, discounts
- **Data Cleaning**: Automatic normalization of dates, numbers, codes

### Smart Field Recognition
- Handles multiple **column name variations** (jaise "qty", "quantity", "billed qty" - sab recognize hota hai)
- Confidence scoring for each extracted field
- Robust pattern matching for critical fields like GSTIN, PAN, dates

### Automatic Processing
- **File Watcher**: Automatically detects new PDFs in `input/` folder
- **Batch Processing**: Process multiple files continuously
- **Error Handling**: Graceful failure handling with detailed error logs

### Output Formats
- JSON (structured, API-ready)
- CSV (spreadsheet compatible)
- XLSX (Excel format with formatting)
- PNG images (one image per PDF page)

---

## Installation 🛠️

### Requirements
- Python 3.8+
- pip (Python package manager)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd OCR

# Create virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
PyMuPDF (fitz)     # PDF reading
opencv-python      # Image processing
numpy              # Numerical operations
watchdog           # File system monitoring
openpyxl           # Excel format support
```

---

## Usage 🚀

### Option 1: Automatic Watcher (सबसे आसान)

```bash
python watcher.py
```

यह command करने के baad:
1. Ek folder `input/` create hota hai
2. Folder mein PDF files daalo
3. Automatically processing hogi!
4. Results `output/` aur `images/` folders mein save honge

**Output structure** ⬇️
```
output/
  invoice_name/
    ├── invoice_name.json      # Structured data
    ├── invoice_name.csv        # Spreadsheet format
    └── invoice_name.xlsx       # Excel format

images/
  invoice_name/
    ├── page_001.png
    ├── page_002.png
    └── ...
```

### Option 2: Direct Processing (Advanced)

```python
from processor import process_pdf
from pathlib import Path

pdf_path = Path("path/to/invoice.pdf")
result = process_pdf(pdf_path)
print(result)  # Returns processed data as dict
```

---

## Configuration ⚙️

### .env File (Optional)

Create ek `.env` file project root mein:

```env
OCR_CONFIDENCE_THRESHOLD=0.25      # Minimum confidence for OCR (0-1)
MIN_BOX_COUNT=30                   # Minimum text boxes to detect
```

---

## Data Fields Extracted 📋

### Header Fields
- Company name, PAN, GSTIN
- Invoice number and date
- Due date
- FSSAI license number
- Driving license number

### Bill-To Fields
- Customer name and address
- Customer GSTIN and DL number
- PO reference details

### Line Items (वस्तु विवरण)
- Product description
- HSN code (tax classification)
- Batch number & expiry date
- Quantity and unit of measure
- Pricing: MRP, PTR, PTS
- Taxes: CGST, SGST
- Discount percentage
- Total amount
- Accuracy score

---

## Example Output 📊

### JSON Output
```json
{
  "header": {
    "company_name": "ABC Pharma Ltd",
    "pan": "AABCU1234A",
    "gstin_no": "27AABCU1234A1Z0",
    "invoice_no": "INV-2024-001",
    "invoice_date": "01/03/2024"
  },
  "bill_to": {
    "name": "XYZ Retail Store",
    "address": "123 Main St, City",
    "cust_gstin": "27AABCT5678B1Z0"
  },
  "line_items": [
    {
      "row_index": 1,
      "product_description": "Medicine A - 500mg",
      "hsn_code": "300219",
      "batch_no": "B12345",
      "expiry_date": "12/2025",
      "qty": 100,
      "uom": "Strips",
      "mrp": 50.00,
      "ptr": 40.00,
      "discount": 5.0,
      "cgst": 9.0,
      "sgst": 9.0,
      "total_amount": 3800.00,
      "accuracy_score": 0.95
    }
  ]
}
```

---

## Troubleshooting 🐛

| Problem | Solution |
|---------|----------|
| Files not detecting | Ensure `input/` folder exists and is readable |
| Import errors | Run `pip install -r requirements.txt` |
| Low accuracy scores | Check PDF quality; ensure text is clear (not image-based) |
| Missing fields | Some fields may not exist in your invoice - this is normal |
| File stuck processing | Check file permissions; restart watcher.py |

---

## Project Structure 📁

```
OCR/
├── processor.py          # Main processing logic
├── watcher.py            # File watcher for auto-processing
├── README.md             # This file
├── .gitignore            # Git configuration
├── input/                # Drop PDFs here (auto-created)
├── output/               # Extracted data (auto-created)
└── images/               # Extracted page images (auto-created)
```

---

## How It Works (तकनीकी विवरण)

```
PDF File
  ↓
Extract text using PyMuPDF (using OCR if needed)
  ↓
Identify sections (Header, Bill-To, Line Items)
  ↓
Match columns with synonyms (qty = quantity = billed qty)
  ↓
Clean & validate data (dates, numbers, codes)
  ↓
Calculate accuracy scores
  ↓
Export to JSON, CSV, XLSX
  ↓
Save extracted images as PNG
```

---

## Performance Notes ⚡

- **Processing time**: ~5-30 seconds per page (PDF size + complexity dependent)
- **RAM usage**: ~200MB average
- **Batch processing**: Can handle continuous file drops without queue buildup
- **Best results**: Clear printed invoices with standard layouts

---

## Future Enhancements 🚀

- [ ] Multi-language support (currently English-heavy)
- [ ] API endpoint for remote processing
- [ ] Database storage option
- [ ] Web dashboard for monitoring
- [ ] Model fine-tuning for specific invoice layouts

---

## License & Credits

Created for efficient invoice data extraction in pharmaceutical supply chains.

**Dependencies**: PyMuPDF, OpenCV, NumPy, watchdog, openpyxl

---

## Questions? 🤔

- Check the code comments in `processor.py` and `watcher.py`
- Review sample outputs in the `output/` folder
- Check error logs in console output

**Happy OCR-ing!** 🎉
