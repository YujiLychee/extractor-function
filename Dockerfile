FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# 设置工作目录
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

# 复制源代码
COPY . .

# 创建数据库文件
RUN touch property_translations.db

# 暴露端口
EXPOSE 8080

# 启动命令
CMD exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --worker-class gthread \
    --preload \
    main:app
