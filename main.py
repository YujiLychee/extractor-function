from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import os
import threading
import time

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

extractor = None
_initialization_lock = threading.Lock()
_initialization_status = "pending"
_initialization_start_time = None

def init_extractor_background():
    global extractor, _initialization_status, _initialization_start_time
    
    with _initialization_lock:
        if _initialization_status != "pending":
            return
        
        _initialization_status = "loading"
        _initialization_start_time = time.time()
        logger.info("开始后台初始化...")
        
        try:
            logger.info(f"当前目录: {os.getcwd()}")
            logger.info(f"文件列表: {os.listdir('.')}")
            
            logger.info("导入 SmartNewsExtractor...")
            from extract import SmartNewsExtractor
            logger.info("导入成功")
            
            db_path = "property_translations.db"
            if os.path.exists(db_path):
                logger.info(f"数据库文件存在: {db_path}")
                logger.info(f"数据库大小: {os.path.getsize(db_path)} bytes")
            else:
                logger.warning(f"数据库文件不存在: {db_path}")
            
            logger.info("初始化 (use_bert=True)...")
            extractor = SmartNewsExtractor(
                use_bert=True,
                preload_db=db_path
            )
            
            load_time = time.time() - _initialization_start_time
            logger.info(f"初始化完成 (耗时: {load_time:.2f}秒)")
            _initialization_status = "ready"
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            extractor = None
            _initialization_status = "failed"

def get_extractor_status():
    global _initialization_start_time
    
    status_info = {
        "status": _initialization_status,
        "ready": _initialization_status == "ready",
        "extractor_available": extractor is not None
    }
    
    if _initialization_start_time:
        elapsed = time.time() - _initialization_start_time
        status_info["loading_time"] = f"{elapsed:.1f}s"
    
    return status_info

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

@app.route('/status')
def status_check():
    return jsonify(get_extractor_status()), 200

@app.route('/extract', methods=['POST', 'OPTIONS'])
def extract_handler():
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    headers = {'Access-Control-Allow-Origin': '*'}

    if _initialization_status == "pending":
        return jsonify({
            "error": "服务正在启动中，请稍后重试",
            "status": "initializing"
        }), 503

    elif _initialization_status == "loading":
        return jsonify({
            "error": "模型正在加载中，请稍后重试",
            "status": "loading",
            "estimated_wait": "30-60 seconds"
        }), 503

    elif _initialization_status == "failed":
        return jsonify({
            "error": "模型加载失败",
            "status": "failed"
        }), 500

    elif not extractor:
        return jsonify({
            "error": "Extractor 不可用",
            "status": "unavailable"
        }), 500

    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return jsonify({"error": "请求体必须是包含 'content' 键的 JSON"}), 400

    try:
        logger.info("开始执行提取...")
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        
        result_dict = asdict(extraction_result)
        logger.info(f"提取完成: {len(extraction_result.property_candidates)} 个房产候选")
        return jsonify(result_dict), 200
        
    except Exception as e:
        logger.error(f"提取过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部提取错误: {str(e)}"}), 500

def start_background_initialization():
    init_thread = threading.Thread(target=init_extractor_background)
    init_thread.daemon = True
    init_thread.start()
    logger.info("已启动后台初始化线程")

start_background_initialization()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
