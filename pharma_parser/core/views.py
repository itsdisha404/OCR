import os
from pathlib import Path

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import InvoiceJob
from .serializers import InvoiceJobSerializer


class ProcessInvoiceView(APIView):
    """POST /api/process/ — Upload PDF, queue Celery task, return job_id."""

    def post(self, request):
        file = request.FILES.get("pdf")
        if not file:
            return Response({"error": "No PDF uploaded. Send as multipart with key 'pdf'."}, status=400)

        if not file.name.lower().endswith(".pdf"):
            return Response({"error": "Only PDF files are accepted."}, status=400)

        input_dir = Path(os.getenv("PDF_INPUT_DIR", "./input_pdfs"))
        input_dir.mkdir(parents=True, exist_ok=True)
        save_path = input_dir / file.name

        with open(save_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)

        job = InvoiceJob.objects.create(
            pdf_path=str(save_path),
            original_filename=file.name,
        )

        # Queue Celery task
        try:
            from .tasks import process_invoice_task
            process_invoice_task.delay(job.id, str(save_path))
        except Exception as exc:
            # Celery/Redis not available — run synchronously as fallback
            job.status = "failed"
            job.error_msg = f"Celery unavailable: {exc}. Start Redis and Celery worker."
            job.save()

        return Response({"job_id": job.id, "status": job.status}, status=201)


class ResultView(APIView):
    """GET /api/results/<job_id>/ — Get job status and extracted data."""

    def get(self, request, job_id):
        try:
            job = InvoiceJob.objects.get(id=job_id)
        except InvoiceJob.DoesNotExist:
            return Response({"error": "Job not found"}, status=404)

        serializer = InvoiceJobSerializer(job)
        return Response(serializer.data)


class JobListView(APIView):
    """GET /api/jobs/ — List all jobs."""

    def get(self, request):
        jobs = InvoiceJob.objects.all()[:100]
        serializer = InvoiceJobSerializer(jobs, many=True)
        return Response(serializer.data)


class ReviewQueueView(APIView):
    """GET /api/review-queue/ — List LOW-confidence items needing review."""

    def get(self, request):
        jobs = InvoiceJob.objects.filter(confidence="LOW", status="done")[:100]
        serializer = InvoiceJobSerializer(jobs, many=True)
        return Response(serializer.data)
