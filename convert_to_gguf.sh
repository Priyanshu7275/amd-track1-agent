#!/usr/bin/env bash
#
# Convert the merged fp16 model to a 4-bit GGUF that llama.cpp can run
# on CPU inside the 4 GB evaluation sandbox.
#
#   merged_model/ (3.1 GB, fp16, safetensors)
#        │
#        ├── convert_hf_to_gguf.py --outtype f16
#        │        └──▶ qwen-finetuned-f16.gguf      (2.9 GB)
#        │
#        └── llama-quantize ... Q4_K_M
#                 └──▶ qwen-finetuned-q4_k_m.gguf   (941 MB)  ← shipped
#
set -euo pipefail

WORKDIR=/workspace/amd-track1-finetune

# ---- build the llama.cpp quantiser ------------------------------------------
git clone https://github.com/ggerganov/llama.cpp.git /workspace/llama.cpp
cd /workspace/llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt

cmake -B build -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j 4 --target llama-quantize

# ---- HuggingFace safetensors -> GGUF (fp16) ---------------------------------
python convert_hf_to_gguf.py "${WORKDIR}/merged_model" \
    --outfile "${WORKDIR}/qwen-finetuned-f16.gguf" \
    --outtype f16

# ---- fp16 GGUF -> Q4_K_M ----------------------------------------------------
# Q4_K_M is the sweet spot: ~1/3 the size of fp16 with minimal quality loss,
# and it comfortably fits the 4 GB RAM ceiling alongside the Python runtime.
./build/bin/llama-quantize \
    "${WORKDIR}/qwen-finetuned-f16.gguf" \
    "${WORKDIR}/qwen-finetuned-q4_k_m.gguf" \
    Q4_K_M

# ---- publish ----------------------------------------------------------------
# The Docker build pulls the model from here at image-build time, which keeps
# the repository small and the model versioned independently.
huggingface-cli upload priyanshu941/track1-qwen-finetuned \
    "${WORKDIR}/qwen-finetuned-q4_k_m.gguf" \
    qwen-finetuned-q4_k_m.gguf

echo "Published: huggingface.co/priyanshu941/track1-qwen-finetuned"
