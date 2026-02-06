FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git wget \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone TripoSG
RUN git clone https://github.com/VAST-AI-Research/TripoSG.git /app/triposg

# Install requirements
WORKDIR /app/triposg
RUN pip3 install --no-cache-dir -r requirements.txt || \
    pip3 install --no-cache-dir \
        transformers accelerate safetensors \
        trimesh pillow requests numpy \
        huggingface_hub einops omegaconf

# Pre-download model
RUN python3 -c "\
import sys; sys.path.insert(0, '/app/triposg'); \
from triposg.pipelines import TripoSGPipeline; \
TripoSGPipeline.from_pretrained('VAST-AI/TripoSG')" || echo "Model pre-download attempted"

WORKDIR /app
RUN mkdir -p /opt/artifact

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
