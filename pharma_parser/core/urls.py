from django.urls import path
from . import views

urlpatterns = [
    path("process/", views.ProcessInvoiceView.as_view(), name="process-invoice"),
    path("results/<int:job_id>/", views.ResultView.as_view(), name="job-result"),
    path("jobs/", views.JobListView.as_view(), name="job-list"),
    path("review-queue/", views.ReviewQueueView.as_view(), name="review-queue"),
]
