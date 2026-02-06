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

# Clone TripoSR (the simpler, faster model)
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /app/triposr

# Install requirements
WORKDIR /app/triposr
RUN pip3 install --upgrade setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

WORKDIR /app
RUN mkdir -p /opt/artifact

COPY generate.py /app/generate.py
ENTRYPOINT ["python3", "/app/generate.py"]
