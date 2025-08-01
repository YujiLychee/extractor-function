# Dockerfile

# 1. 基础镜像
FROM python:3.11-slim

# 2. 环境变量，让 transformers 缓存到 /app/cache
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# 3. 工作目录
WORKDIR /app

# 4. 安装系统依赖
RUN apt-get update \
 && apt-get install -y gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# 5. 创建缓存目录
RUN mkdir -p /app/cache

# 6. 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 7. 在镜像构建时生成下载脚本并执行，确保模型已缓存且 pipeline 初始化不报 cache_dir 错误
RUN cat << 'EOF' > download_model.py
#!/usr/bin/env python3
import os
print("开始下载 BERT 模型...")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    model_name = "ckiplab/bert-base-chinese-ner"
    cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", None))
    print(f"从 {cache_dir} 载入/tokenizer 和 model…")

    # 先下载 tokenizer & model 到 cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer 下载完成")
    model     = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model 下载完成")

    # 再用已加载好的 model/tokenizer 初始化 pipeline（不传 cache_dir）
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
    print("将在运行时切换到规则模式")
EOF

# 执行下载脚本
RUN python download_model.py

# 8. 复制项目源代码
COPY . .

# 9. 暴露端口并启动
EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "600", "--worker-class", "gthread", "--preload", "main:app"]
