from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)
MODEL_PATH = "./model"

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Set pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
