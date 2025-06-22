# Stage 1: Install UV and dependencies
FROM python:3.10-slim AS builder

# Install curl for UV installer
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install pinned UV version (replace 0.7.13 with your version)
ADD https://astral.sh/uv/0.7.13/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv sync --locked --no-editable --no-install-project
RUN uv pip install minio

# Stage 2: Final image
FROM python:3.10-slim

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install MinIO SDK and application code
WORKDIR /app
COPY . .

# Custom MinIO loader class (save as minio_loader.py)
