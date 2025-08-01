# Dockerfile

FROM python:3.11-slim

# 环境变量：让 transformers 缓存到 /app/cache
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

WORKDIR /app

# 安装系统依赖
RUN apt-get update \
 && apt-get install -y gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /app/cache

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 在 build 阶段直接通过 stdin 给 python，下载并测试模型
RUN python3 - << 'EOF'
import os
print("开始下载 BERT 模型…")
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    model_name = "ckiplab/bert-base-chinese-ner"
    cache_dir  = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", None))
    print(f"缓存目录: {cache_dir}")

    # 下载 tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer 下载完成")
    model     = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model 下载完成")

    # 用已载入的 model/tokenizer 初始化 pipeline（不传 cache_dir）
    ner = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1
    )
    print("Pipeline 测试成功，模型预下载完成！")
except Exception as e:
    print(f"模型下载/测试失败: {e}")
    print("将在运行时切换到规则模式")
EOF

# 复制项目源代码
COPY . .

# 暴露端口并启动
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "600", "--worker-class", "gthread", "--preload", "main:app"]
