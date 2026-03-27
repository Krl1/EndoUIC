# RTX 3080 (Ampere, sm_86) + fp16 / CUDA 12.1
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FORCE_CUDA=1
ENV BASICSR_EXT=True

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ninja-build \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip "setuptools<70" wheel

RUN pip install \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install ninja packaging einops timm "numpy>=1.21,<2.0"

RUN pip install causal-conv1d==1.2.0.post2 --no-build-isolation
RUN pip install mamba-ssm==1.2.0.post1 --no-build-isolation

WORKDIR /app

COPY BasicSR-light /app/BasicSR-light
COPY EndoUIC /app/EndoUIC

RUN grep -v "^numpy" /app/EndoUIC/requirements.txt > /tmp/req_endouic.txt \
 && pip install -r /tmp/req_endouic.txt

RUN grep -v "^numpy" /app/BasicSR-light/requirements.txt > /tmp/req_basicsr.txt \
 && pip install -r /tmp/req_basicsr.txt

RUN pip install "numpy>=1.21,<2.0"

RUN cd /app/BasicSR-light && pip install -e . --no-build-isolation
RUN cd /app/EndoUIC && pip install -e . --no-build-isolation

WORKDIR /app/EndoUIC

ENV PYTHONPATH=/app/EndoUIC

CMD ["/bin/bash"]
