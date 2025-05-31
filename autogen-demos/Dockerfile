# Stage 1: Python environment setup
FROM python:3.10-slim AS python-base

# Install necessary packages and upgrade pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    pip install --upgrade pip

# Install the uv tool and other Python dependencies
RUN pip install uv

# Stage 2: Spark setup
FROM spark:latest

USER root

# Copy Python environment from the previous stage
COPY --from=python-base /usr/local /usr/local

# Ensure the correct Python version is used
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip

# Initialize uv and add additional packages
RUN uv init
RUN uv add jupyterlab pyspark

USER spark
