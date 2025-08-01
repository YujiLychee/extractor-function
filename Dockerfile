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

# 分步下载BERT模型，避免多行字符串问题
RUN python -c "print('开始下载BERT模型...')"
RUN python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ckiplab/bert-base-chinese-ner', cache_dir='/app/cache'); print('Tokenizer下载完成')"
RUN python -c "from transformers import AutoModelForTokenClassification; model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ner', cache_dir='/app/cache'); print('Model下载完成')"
RUN python -c "from transformers import pipeline; pipeline('ner', model='ckiplab/bert-base-chinese-ner', cache_dir='/app/cache', device=-1); print('Pipeline测试成功，模型预下载完成!')"

COPY . .

EXPOSE 8080

CMD exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --worker-class gthread \
    --preload \
    main:appFROM python:3.11-slim

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

RUN python -c "print('开始下载BERT模型...')"
RUN python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('ckiplab/bert-base-chinese-ner', cache_dir='/app/cache'); print('Tokenizer下载完成')"
RUN python -c "from transformers import AutoModelForTokenClassification; model = AutoModelForTokenClassification.from_pretrained('ckiplab/bert-base-chinese-ner', cache_dir='/app/cache'); print('Model下载完成')"
RUN python -c "from transformers import pipeline; pipeline('ner', model='ckiplab/bert-base-chinese-ner', cache_dir='/app/cache', device=-1); print('Pipeline测试成功，模型预下载完成!')"

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
