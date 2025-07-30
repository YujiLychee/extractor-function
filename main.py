from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import os
import threading
import time

# 创建 Flask 应用
app = Flask(__name__)

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
extractor = None
_initialization_lock = threading.Lock()
_initialization_status = "pending"  # pending, loading, ready, failed

def init_extractor_background():
    """后台异步初始化提取器"""
    global extractor, _initialization_status
    
    with _initialization_lock:
        if _initialization_status != "pending":
            return
        
        _initialization_status = "loading"
        logger.info("🚀 开始后台初始化 BERT Extractor...")
        
        try:
            # 动态导入
            from extract import SmartNewsExtractor
            
            # 检查数据库文件
            db_path = "property_translations.db"
            if not os.path.exists(db_path):
                logger.warning(f"数据库文件 {db_path} 不存在，使用默认配置")
            
            # 启动时间记录
            start_time = time.time()
            
            # 使用轻量级 BERT 模型加快启动
            extractor = SmartNewsExtractor(
                use_bert=True, 
                preload_db=db_path,
                model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 更小更快的模型
            )
            
            load_time = time.time() - start_time
            logger.info(f" BERT Extractor 初始化完成 (耗时: {load_time:.2f}秒)")
            _initialization_status = "ready"
            
        except Exception as e:
            logger.error(f" BERT Extractor 初始化失败: {e}", exc_info=True)
            extractor = None
            _initialization_status = "failed"

def get_extractor_status():
    """获取提取器状态"""
    return {
        "status": _initialization_status,
        "ready": _initialization_status == "ready",
        "extractor_available": extractor is not None
    }

# 健康检查路由
@app.route('/')
@app.route('/health')
def health_check():
    status = get_extractor_status()
    return jsonify({
        "status": "healthy", 
        "service": "extractor-api",
        "bert_status": status["status"],
        "ready": status["ready"]
    }), 200

# 状态检查路由
@app.route('/status')
def status_check():
    return jsonify(get_extractor_status()), 200

# 主要的提取路由
@app.route('/extract', methods=['POST', 'OPTIONS'])
def extract_handler():
    # 处理 CORS 预检请求
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    # 检查初始化状态
    if _initialization_status == "pending":
        return jsonify({
            "error": "服务正在启动中，请稍后重试",
            "status": "initializing"
        }), 503

    elif _initialization_status == "loading":
        return jsonify({
            "error": "BERT 模型正在加载中，请稍后重试",
            "status": "loading",
            "estimated_wait": "30-60 seconds"
        }), 503

    elif _initialization_status == "failed":
        return jsonify({
            "error": "BERT 模型加载失败",
            "status": "failed"
        }), 500

    elif not extractor:
        return jsonify({
            "error": "Extractor 不可用",
            "status": "unavailable"
        }), 500

    # 验证请求
    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return jsonify({"error": "请求体必须是包含 'content' 键的 JSON"}), 400

    try:
        # 执行提取
        logger.info("开始执行 BERT 提取...")
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        
        # 转换为字典
        result_dict = asdict(extraction_result)
        logger.info("BERT 提取完成")
        return jsonify(result_dict), 200
        
    except Exception as e:
        logger.error(f"BERT 提取过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部提取错误: {str(e)}"}), 500

# 启动后台初始化
def start_background_initialization():
    """启动后台初始化线程"""
    init_thread = threading.Thread(target=init_extractor_background)
    init_thread.daemon = True
    init_thread.start()
    logger.info(" 已启动 BERT 后台初始化线程")

# 应用启动时开始后台初始化
start_background_initialization()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
