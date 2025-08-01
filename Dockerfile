FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /app/cache

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建模型下载脚本
RUN echo 'import os\n\
print("开始下载BERT模型...")\n\
try:\n\
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline\n\
    model_name = "ckiplab/bert-base-chinese-ner"\n\
    cache_dir = "/app/cache"\n\
    print(f"下载模型: {model_name}")\n\
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)\n\
    print("Tokenizer下载完成")\n\
    model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)\n\
    print("Model下载完成")\n\
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, cache_dir=cache_dir, device=-1)\n\
    print("Pipeline测试成功，模型预下载完成!")\n\
except Exception as e:\n\
    print(f"模型下载失败: {e}")\n\
    print("将在运行时使用规则模式")' > download_model.py

# 执行模型下载
RUN python download_model.py

# 复制应用代码
COPY . .

EXPOSE 8080

CMD exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --worker-class gthread \
    --preload \
    main:app
