from django.db import models


class InvoiceJob(models.Model):
    """Tracks the processing status and result of a single invoice PDF."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    pdf_path = models.CharField(max_length=512)
    original_filename = models.CharField(max_length=256, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    pdf_type = models.CharField(max_length=20, blank=True)  # digital / scanned / mixed
    vendor_id = models.CharField(max_length=50, blank=True)
    confidence = models.CharField(max_length=10, blank=True)  # HIGH / MEDIUM / LOW
    error_msg = models.TextField(blank=True)
    result_json = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Job #{self.id} — {self.original_filename} [{self.status}]"
