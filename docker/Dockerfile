# Base image with PyTorch + CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy files
COPY ./model /app/model
COPY ./inference/inference.py /app/inference.py
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8080
CMD ["python", "inference.py"]
