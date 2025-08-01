import os
from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import threading
import time

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
extractor = None
_initialization_lock = threading.Lock()
_initialization_status = "pending"

def init_extractor_background():
    """后台初始化提取器"""
    global extractor, _initialization_status
    
    with _initialization_lock:
        if _initialization_status != "pending":
            return
            
        _initialization_status = "loading"
        logger.info("开始后台初始化...")
        
        try:
            # 快速启动模式 - 禁用BERT确保快速启动
            from extract import SmartNewsExtractor
            
            extractor = SmartNewsExtractor(
                use_bert=False,  # 快速启动
                preload_db="property_translations.db"
            )
            
            _initialization_status = "ready"
            logger.info("初始化完成（规则模式）")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            _initialization_status = "failed"

@app.route('/')
@app.route('/health')
def health_check():
    """健康检查端点 - 必须快速响应"""
    return jsonify({
        "status": "healthy",
        "service": "extractor-api",
        "ready": True,
        "mode": _initialization_status
    }), 200

@app.route('/status')
def status_check():
    """状态检查"""
    return jsonify({
        "status": _initialization_status,
        "ready": extractor is not None,
        "extractor_available": extractor is not None
    }), 200

@app.route('/extract', methods=['POST', 'OPTIONS'])
def extract_handler():
    """提取处理端点"""
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    if not extractor:
        return jsonify({
            "error": "服务正在初始化",
            "status": _initialization_status
        }), 503

    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return jsonify({"error": "请求体必须是包含 'content' 键的 JSON"}), 400

    try:
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        result_dict = asdict(extraction_result)
        return jsonify(result_dict), 200

    except Exception as e:
        logger.error(f"提取错误: {e}", exc_info=True)
        return jsonify({"error": f"提取错误: {str(e)}"}), 500

# 启动后台初始化
threading.Thread(target=init_extractor_background, daemon=True).start()

if __name__ == '__main__':
    # 从环境变量获取端口，确保监听正确端口
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"启动应用，监听端口: {port}")
    logger.info(f"监听所有网络接口: 0.0.0.0")
    
    # 确保监听 0.0.0.0 而不是 127.0.0.1
    app.run(
        host='0.0.0.0',  # 必须是 0.0.0.0
        port=port,
        debug=False,
        threaded=True
    )
