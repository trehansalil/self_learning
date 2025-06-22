# Stage 1: Install UV and dependencies
FROM python:3.10-slim

# Install pip dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# # Copy your custom Python classes
# COPY custom_classes /custom_classes

# # (Optional) Set PYTHONPATH so your custom classes are importable from anywhere
# ENV PYTHONPATH="/custom_classes:${PYTHONPATH}"
