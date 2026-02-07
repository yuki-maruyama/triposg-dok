FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git wget build-essential ninja-build cmake \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1)
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone TripoSR
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /app/triposr

# Install requirements
WORKDIR /app/triposr
# V100 GPU uses compute capability 7.0
ENV TORCH_CUDA_ARCH_LIST="7.0"

RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir onnxruntime

WORKDIR /app
RUN mkdir -p /opt/artifact

# Pre-download models during build (saves ~2min on each run)
ENV HF_HOME=/app/models
ENV U2NET_HOME=/app/models/u2net

# TripoSR model from HuggingFace (~1.6GB)
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/TripoSR', local_dir='/app/models/TripoSR')"

# rembg model (u2net.onnx ~176MB for background removal)
RUN mkdir -p /app/models/u2net && \
    wget -q -O /app/models/u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
