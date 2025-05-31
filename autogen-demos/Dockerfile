# Stage 1: Python base
FROM python:3.10.12-slim AS python-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    pip install --upgrade pip && \
    pip install uv

# Stage 2: Spark setup
FROM bitnami/spark:latest

USER root

# Copy Python 3.10 from python-base stage
COPY --from=python-base /usr/local /usr/local

# Optional: clean pre-installed Python 3.8 links to avoid fallback
RUN rm -f /usr/bin/python /usr/bin/pip && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip

# Set env so uv uses correct Python
ENV PYTHON=/usr/local/bin/python3.10
ENV PATH="/usr/local/bin:$PATH"

# Force uv to use correct Python explicitly
RUN uv init --python=/usr/local/bin/python3.10 && \
    uv add jupyterlab pyspark

USER spark  # or USER spark depending on your image
