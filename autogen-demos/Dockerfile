FROM bitnami/spark:3.4.0

USER root

# -------------------- Install dependencies for Python build --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Build and install Python 3.10.12 --------------------
ENV PYTHON_VERSION=3.10.12

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && make altinstall && \
    cd .. && rm -rf Python-${PYTHON_VERSION}*

# -------------------- Set Python 3.10 as default --------------------
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip

ENV PYTHON=/usr/local/bin/python3.10
ENV PATH="/usr/local/bin:$PATH"

# -------------------- Install uv and create virtual environment --------------------
RUN pip install --upgrade uv && \
    uv init
    
RUN uv add notebook jupyterlab pyspark

# -------------------- Expose port and run jupyterlab --------------------
EXPOSE 8888

CMD [".venv/bin/python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]

USER spark
