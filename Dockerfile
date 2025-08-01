FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/cache

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "
import os
print('开始下载BERT模型...')
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
model_name = 'ckiplab/bert-base-chinese-ner'
print(f'下载模型: {model_name}')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/app/cache')
print('Tokenizer下载完成')
model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir='/app/cache')
print('Model下载完成')
# 测试pipeline创建
pipeline('ner', model=model_name, tokenizer=model_name, cache_dir='/app/cache', device=-1)
print('Pipeline测试成功，模型预下载完成!')
"
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
