# Base image with PyTorch + CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy source files
COPY ./inference/inference.py /app/inference.py
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y git unzip curl awscli
RUN pip install --upgrade pip && pip install -r requirements.txt

# Create model directory (will be populated from S3)
RUN mkdir -p /app/model

# Set environment variables for default path (overridden by SageMaker ENV)
ENV MODEL_PATH=/app/model

EXPOSE 8080
CMD ["python", "inference.py"]
