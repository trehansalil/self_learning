# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y wget curl openjdk-17-jdk  # Try openjdk-17-jdk if openjdk-11 is unavailable

# Install PySpark and JupyterLab
RUN pip install pyspark jupyterlab notebook

# Set environment variables for Java and Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.10/site-packages/pyspark
ENV PATH=$PATH:$SPARK_HOME/bin

# Expose the Jupyter port
EXPOSE 8888

# Define the entrypoint to start JupyterLab automatically
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
