from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_dir="./",
    local_dir_use_symlinks=False
)
