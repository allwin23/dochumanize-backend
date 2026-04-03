"""
Celery application configuration.
Broker + result backend both use Redis (single instance on Render free tier).
"""

import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "dochumanize",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,       # One task at a time per worker (memory safe)

    # Timeouts — 60-page doc can take ~15 min
    task_soft_time_limit=1_200,         # 20 min soft kill
    task_time_limit=1_500,              # 25 min hard kill

    # Result expiry — keep results 2 hours for download
    result_expires=7_200,

    # Routing
    task_default_queue="humanize"444