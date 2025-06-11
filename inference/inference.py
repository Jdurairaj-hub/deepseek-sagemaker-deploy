from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import os
import boto3

app = Flask(__name__)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path and S3 info
S3_BUCKET = os.environ.get("S3_BUCKET", "my-deepseek-models-bucket")
S3_PREFIX = os.environ.get("S3_PREFIX", "model")
MODEL_PATH = "/app/model"

# Download model from S3
def download_model_from_s3():
    s3 = boto3.client("s3")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            relative_path = os.path.relpath(s3_key, S3_PREFIX)
            local_file_path = os.path.join(MODEL_PATH, relative_path)

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(S3_BUCKET, s3_key, local_file_path)

# Only download if not already cached
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print("Downloading model from S3...")
    download_model_from_s3()
    print("Model download complete.")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto" if device.type == "cuda" else {"": device},
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.9,
            top_k=100,
            top_p=0.92,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=True,
            max_time=20.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "device": str(device)})

if __name__ == "__main__":
    print(f"Running on device: {device}")
    app.run(host="0.0.0.0", port=8080)
