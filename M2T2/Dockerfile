FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.8 python3-pip python3.8-venv \
    git wget zip unzip ffmpeg libsm6 libxext6 \
    build-essential cmake ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip && \
    pip install torch==2.0.1 torchvision==0.15.2

COPY setup.py ./setup.py
COPY requirements.txt ./requirements.txt
COPY config.yaml ./config.yaml
COPY m2t2 ./m2t2
COPY pointnet2_ops ./pointnet2_ops
COPY train.py demo.py demo_rlbench.py watch_and_infer.py ./
COPY m2t2.pth ./m2t2.pth

RUN pip install -r requirements.txt && \
    pip install .

ENV TORCH_CUDA_ARCH_LIST="8.6"

RUN pip install ./pointnet2_ops/

CMD ["python3", "watch_and_infer.py"]
