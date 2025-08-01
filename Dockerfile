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

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 创建并测试模型下载脚本
# （移除了 pipeline 的 cache_dir 参数）
RUN cat > download_model.py << 'EOF'
import os
print("开始下载 BERT 模型...")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    model_name = "ckiplab/bert-base-chinese-ner"
    cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", None))

    print(f"下载 tokenizer 和 model 到 {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer 下载完成")
    model     = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model 下载完成")

    # 不再传 cache_dir 给 pipeline
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1
    )
    print("Pipeline 测试成功，模型预下载完成!")
except Exception as e:
    print(f"模型下载失败: {e}")
    print("将在运行时使用规则模式")
EOF

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
