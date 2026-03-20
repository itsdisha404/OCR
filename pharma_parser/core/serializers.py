from rest_framework import serializers
from .models import InvoiceJob


class InvoiceJobSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvoiceJob
        fields = [
            "id",
            "original_filename",
            "status",
            "pdf_type",
            "vendor_id",
            "confidence",
            "error_msg",
            "result_json",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields
