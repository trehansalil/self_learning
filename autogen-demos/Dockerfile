FROM python:3.10.12-slim AS python-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libexpat1 && \
    pip install --upgrade pip uv && \
    rm -rf /var/lib/apt/lists/*

FROM bitnami/spark:3.4.0

USER root

COPY --from=python-base /usr/local /usr/local

RUN rm -f /usr/bin/python /usr/bin/pip && \
    ln -s /usr/local/bin/python3.10 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/bin/pip

ENV PYTHON=/usr/local/bin/python3.10
ENV PATH="/usr/local/bin:$PATH"

RUN uv init --python=/usr/local/bin/python3.10 && \
    uv add jupyterlab pyspark

EXPOSE 8888

CMD [".venv/bin/python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

USER spark