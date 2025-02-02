#!/bin/bash
exec celery -A text_extract_api.celery_app "$@"
