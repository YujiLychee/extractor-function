#!/usr/bin/env python3
import os
print("开始下载 BERT 模型...")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    
    model_name = "ckiplab/bert-base-chinese-ner"
    cache_dir = os.getenv("HF_HOME", os.getenv("TRANSFORMERS_CACHE", None))
    print(f"使用缓存目录: {cache_dir}")
    
    # 下载 tokenizer 和 model 到指定缓存目录
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer 下载完成")
    
    model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
    print("Model 下载完成")
    
    # 使用已加载的 model 和 tokenizer 创建 pipeline
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