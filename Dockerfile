FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /app/cache

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制模型下载脚本并执行
COPY download_model.py .
RUN python download_model.py

# 复制项目源代码
COPY . .

# 暴露端口并启动
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "600", "--worker-class", "gthread", "--preload", "main:app"]
