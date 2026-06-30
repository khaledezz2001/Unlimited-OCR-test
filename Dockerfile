# ================================================================
# Official vLLM Unlimited-OCR image — pre-built with NGram processor
# ================================================================
FROM vllm/vllm-openai:unlimited-ocr

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# ---------------------------------------------------------------
# HF cache
# ---------------------------------------------------------------
ENV HF_HOME=/models/hf
ENV HF_HUB_CACHE=/models/hf
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
ENV TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------
# Blackwell / SM120 flags
# ---------------------------------------------------------------
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV VLLM_FLASH_ATTN_VERSION=2
ENV VLLM_ATTENTION_BACKEND=FLASHINFER

# ---------------------------------------------------------------
# System deps for PDF processing
# ---------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------
# Python deps
# ---------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------
# Download Unlimited-OCR at build time (~6GB, 3B params)
# ---------------------------------------------------------------
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="baidu/Unlimited-OCR",
    local_dir="/models/hf/baidu/Unlimited-OCR",
    local_dir_use_symlinks=False,
)
print("Unlimited-OCR downloaded successfully")
PYEOF

# ---------------------------------------------------------------
# Lock offline at runtime
# ---------------------------------------------------------------
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------
# App
# ---------------------------------------------------------------
COPY handler.py .

# vllm/vllm-openai sets ENTRYPOINT to the vllm CLI — override it so our
# handler runs with python3 directly instead of being passed as vllm args.
ENTRYPOINT ["python3", "-u", "handler.py"]
