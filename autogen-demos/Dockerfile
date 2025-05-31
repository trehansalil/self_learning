FROM spark:latest
USER root
RUN pip install uv
RUN uv init
RUN uv add jupyterlab pyspark
USER spark

