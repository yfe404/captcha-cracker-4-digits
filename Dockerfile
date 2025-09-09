FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-onnx.txt .
RUN pip install --upgrade pip && pip install -r requirements-onnx.txt

COPY server_onnx.py /app/server_onnx.py
# Put your ONNX model under ./exports on the host, we’ll mount it to /models
# or bake it in:
# COPY exports/resnet34_4head.onnx /models/resnet34_4head.onnx

EXPOSE 8080
ENV MODEL_PATH=/models/resnet34_4head.onnx \
    DEVICE=cpu \
    IMG_SIZE=320 \
    TTA_ANGLES="-4,0,4"

CMD ["uvicorn", "server_onnx:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

