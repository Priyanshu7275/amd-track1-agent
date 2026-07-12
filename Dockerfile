FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    curl -L -o /app/models/qwen-finetuned-q4_k_m.gguf \
    "https://huggingface.co/priyanshu941/track1-qwen-finetuned/resolve/main/qwen-finetuned-q4_k_m.gguf"

RUN apt-get purge -y --auto-remove build-essential cmake curl && \
    rm -rf /var/lib/apt/lists/*

COPY agent.py .

ENTRYPOINT ["python", "-u", "agent.py"]