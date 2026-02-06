FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git wget ninja-build \
        libgl1-mesa-glx libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Clone TripoSG
RUN git clone https://github.com/VAST-AI-Research/TripoSG.git /app/triposg

# Install requirements (from TripoSG requirements.txt)
WORKDIR /app/triposg
RUN pip3 install --no-cache-dir \
        diffusers transformers accelerate safetensors \
        trimesh pillow requests scipy \
        huggingface_hub einops omegaconf \
        opencv-python scikit-image peft jaxtyping typeguard \
        diso pymeshlab numpy==1.22.3

# Pre-download model
RUN python3 -c "\
import sys; sys.path.insert(0, '/app/triposg'); \
from triposg.pipelines import TripoSGPipeline; \
TripoSGPipeline.from_pretrained('VAST-AI/TripoSG')" || echo "Model pre-download attempted"

WORKDIR /app
RUN mkdir -p /opt/artifact

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
