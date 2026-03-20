from django.contrib import admin
from .models import InvoiceJob


@admin.register(InvoiceJob)
class InvoiceJobAdmin(admin.ModelAdmin):
    list_display = ["id", "original_filename", "status", "vendor_id", "confidence", "created_at"]
    list_filter = ["status", "confidence", "vendor_id"]
    search_fields = ["original_filename", "vendor_id"]
    readonly_fields = ["result_json"]
