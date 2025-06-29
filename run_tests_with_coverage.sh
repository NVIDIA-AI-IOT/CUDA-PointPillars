#!/bin/bash
# Script to run tests with coverage reporting

poetry run pytest \
    --cov=tool \
    --cov-branch \
    --cov-report=term-missing:skip-covered \
    --cov-report=html:htmlcov \
    --cov-report=xml:coverage.xml \
    --cov-fail-under=80