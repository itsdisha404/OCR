"""
core/tasks.py
─────────────
Celery task that orchestrates the full invoice processing pipeline.
"""

import os
from pathlib import Path

from celery import shared_task

from pipeline.detector import detect_pdf_type, PDFType
from pipeline.digital_extractor import extract_digital
from pipeline.validators import validate_invoice


@shared_task(bind=True, max_retries=2)
def process_invoice_task(self, job_id: int, pdf_path: str):
    """
    Main pipeline orchestrator. Runs as a Celery task.

    Steps:
        1. Detect PDF type (digital / scanned / mixed)
        2. Extract data (pdfplumber for digital, Surya for scanned)
        3. Validate (GST math, HSN, totals)
        4. Run React Agent if LOW confidence
        5. Save outputs (JSON + Excel)
    """
    # Import Django model inside task to avoid circular imports
    from core.models import InvoiceJob

    job = InvoiceJob.objects.get(id=job_id)
    job.status = "processing"
    job.save()

    try:
        path = Path(pdf_path)

        # ── Stage 1: Detect PDF type ──────────────────────────────────────
        detection = detect_pdf_type(path)
        job.pdf_type = detection.pdf_type.value
        job.save()

        invoice = {"header": {}, "line_items": [], "summary": {}}

        # ── Stage 2: Extract based on type ────────────────────────────────
        if detection.pdf_type == PDFType.DIGITAL:
            invoice = extract_digital(path, detection.digital_pages)

        elif detection.pdf_type == PDFType.SCANNED:
            from pipeline.ocr_extractor import extract_with_surya, reconstruct_tables_from_surya
            from pipeline.vendor_router import detect_vendor, get_parser

            ocr_results = extract_with_surya(path, detection.scanned_pages)
            full_text = "\n".join(r["text"] for r in ocr_results)
            tables = [reconstruct_tables_from_surya(r) for r in ocr_results]

            vendor_id = detect_vendor(full_text)
            parser = get_parser(vendor_id)
            invoice = parser.parse(full_text, [tables], str(path))
            invoice.setdefault("header", {})
            invoice["header"]["extraction_method"] = "surya_ocr"
            invoice["header"]["vendor_id"] = vendor_id
            invoice["header"]["source_file"] = path.name

        else:  # MIXED
            # Digital pages first
            digital_inv = extract_digital(path, detection.digital_pages)

            # Scanned pages via Surya
            from pipeline.ocr_extractor import extract_with_surya, reconstruct_tables_from_surya
            from pipeline.vendor_router import detect_vendor, get_parser

            ocr_results = extract_with_surya(path, detection.scanned_pages)
            for r in ocr_results:
                tables = [reconstruct_tables_from_surya(r)]
                vendor_id = detect_vendor(r["text"])
                parser = get_parser(vendor_id)
                partial = parser.parse(r["text"], [tables], str(path))
                digital_inv.setdefault("line_items", []).extend(
                    partial.get("line_items", [])
                )

            invoice = digital_inv
            invoice.setdefault("header", {})
            invoice["header"]["extraction_method"] = "mixed"

        # ── Stage 3: Validate + score confidence ──────────────────────────
        invoice = validate_invoice(invoice)

        # ── Stage 4: React Agent for LOW confidence ───────────────────────
        overall_conf = invoice.get("validation_summary", {}).get("overall_confidence")
        if overall_conf == "LOW":
            try:
                from pipeline.agent import run_error_agent
                invoice = run_error_agent(invoice)
                invoice = validate_invoice(invoice)  # Re-validate after fix
            except Exception as agent_exc:
                invoice.setdefault("validation_summary", {})
                invoice["validation_summary"].setdefault("header_flags", []).append(
                    f"AGENT_ERROR:{str(agent_exc)[:100]}"
                )

        # ── Stage 5: Save outputs ─────────────────────────────────────────
        output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))

        from outputs.json_writer import save_json
        from outputs.excel_writer import append_to_excel

        save_json(invoice, output_dir / "json")
        append_to_excel(invoice, output_dir / "invoices_master.xlsx")

        # Update job
        job.status = "done"
        job.vendor_id = invoice.get("header", {}).get("vendor_id", "")
        job.confidence = invoice.get("validation_summary", {}).get("overall_confidence", "")
        job.result_json = invoice
        job.save()

    except Exception as exc:
        job.status = "failed"
        job.error_msg = str(exc)
        job.save()
        raise self.retry(exc=exc, countdown=30)
