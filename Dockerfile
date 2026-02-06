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
RUN pip3 install --upgrade pip setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
RUN mkdir -p /opt/artifact

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
