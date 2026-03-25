# Używamy wersji devel z CUDA 11.8, która jest stabilnym łącznikiem między 
# wymaganiami Torch 1.7+ a nowoczesnymi procesorami GPU.
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Ustawienia środowiska
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Instalacja zależności systemowych (w tym gcc-8 wymagane przez BasicSR)
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    gcc-8 \
    g++-8 \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Ustawienie Pythona 3.9 jako domyślnego
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Aktualizacja narzędzi budowania (setuptools 69.5.1 zapobiega błędom w starych pakietach)
RUN python -m pip install --upgrade pip "setuptools<70" wheel

# Instalacja PyTorch (CUDA 11.8) - wersja 2.0 jest wstecznie kompatybilna z kodem 1.7
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install ninja packaging
RUN pip install transformers==4.39.0
RUN pip install "numpy<2.0" "pandas<2.0" einops
RUN pip install hatchling setuptools-scm

# Kopiujemy pliki wymagań, aby zainstalować zależności przed zamontowaniem kodu (optymalizacja cache)
COPY BasicSR-light/requirements.txt /tmp/req_basicsr.txt
COPY EndoUIC/requirements.txt /tmp/req_endouic.txt

# Instalacja specyficznych wersji podanych przez Ciebie (numpy < 1.21, etc.)
RUN pip install -r /tmp/req_basicsr.txt
RUN pip install -r /tmp/req_endouic.txt
RUN pip install timm

ENV TORCH_CUDA_ARCH_LIST="6.1"
ENV FORCE_CUDA=1

# Ręczna kompilacja Causal Conv1d
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d \
    && cd /tmp/causal-conv1d && git checkout v1.2.0.post2 \
    && pip install . && rm -rf /tmp/causal-conv1d
# Ręczna kompilacja Mamby za pomocą pip
RUN git clone https://github.com/state-spaces/mamba.git /tmp/mamba \
    && cd /tmp/mamba && git checkout v1.1.3 \
    && pip install . && rm -rf /tmp/mamba

# Ścieżka bazowa projektu
WORKDIR /home/kkulawiec/Documents/low-light-image-enhancement/externals/EndoUIC

# Kopiowanie plików projektowych (opcjonalne jeśli montujemy wolumen, ale dobre dla stabilności obrazu)
COPY . .

# Ponowna instalacja paczek z wnętrza obrazu dla pewności
RUN cd BasicSR-light && pip install --no-build-isolation -e .
RUN cd EndoUIC && pip install --no-build-isolation -e .

# Ustawienia pod BasicSR (wymuszenie gcc-8 dla kompilacji C++/CUDA)
ENV CC=gcc-8
ENV CXX=g++-8
ENV BASICSR_EXT=True
ENV BASICSR_JIT=True