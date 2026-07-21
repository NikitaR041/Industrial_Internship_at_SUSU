FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libgomp1 \
    git nano \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/airockchip/ultralytics_yolo11.git /opt/ultralytics

# torch РОВНО 2.4.0 CPU (иначе pip тянет CUDA-сборку на гигабайты)
RUN pip install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    onnx==1.16.1

RUN pip install --no-cache-dir rknn-toolkit2==2.3.2
RUN pip install --no-cache-dir -e /opt/ultralytics

# Скрипты конвейера запекаются в образ и доступны как команды
COPY scripts/convert /usr/local/bin/convert
RUN chmod +x /usr/local/bin/convert

WORKDIR /workspace
CMD ["/bin/bash"]