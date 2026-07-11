FROM --platform=linux/amd64 python:3.11-slim

# Install build + runtime dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Force llama-cpp-python to build with OpenMP, portable across CPU types
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir -r requirements.txt

# Download the quantized Qwen 1.5B model (~1 GB, 4-bit)
RUN mkdir -p /app/models && \
    curl -L -o /app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Remove build tools only — keep libgomp1 for runtime
RUN apt-get purge -y --auto-remove build-essential cmake curl && \
    rm -rf /var/lib/apt/lists/*

COPY agent.py .

ENTRYPOINT ["python", "-u", "agent.py"]