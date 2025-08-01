# Dockerfile

FROM python:3.11-slim

# 设置环境变量，让 transformers 读取 /app/cache 作为模型缓存
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

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 生成并执行下载模型脚本（注意：所有 Python 代码都放在 RUN … << 'EOF' 块里）
RUN cat << 'EOF' > download_model.py
import os
print("开始下载 BERT 模型...")
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    model_name = "ckiplab/bert-base-chinese-ner"
    cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", None))
    print(f"下载 tokenizer 和 model 到 {cache_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer 下载完成")
    model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model 下载完成")
    # 用已经加载好的 model/tokenizer，避免 pipeline 再次下载
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

# 在镜像里运行这个脚本，确保缓存里确实有模型
RUN python download_model.py

# 复制应用代码
COPY . .

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
