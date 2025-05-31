# Stage 1: Python 3.10 + uv environment
FROM python:3.10.12-slim AS python-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libexpat1 && \
    pip install --upgrade pip && \
    pip install uv

# Stage 2: Spark setup with Python and uv
FROM bitnami/spark:latest

USER root

# Copy Python from python-base
COPY --from=python-base /usr/local /usr/local

# Replace default python with Python 3.10
RUN rm -f /usr/bin/python /usr/bin/pip && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip

# Set environment for uv and PATH
ENV PYTHON=/usr/local/bin/python3.10
ENV PATH="/usr/local/bin:$PATH"

# Create .venv using Python 3.10 and install dependencies
RUN uv init --python=/usr/local/bin/python3.10 && \
    uv add jupyterlab pyspark

# Expose Jupyter port
EXPOSE 8888

# Run JupyterLab with .venv Python
CMD [".venv/bin/python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

USER spark  # Optional: or use `USER spark` if defined in image
