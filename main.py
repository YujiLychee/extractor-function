from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import os
import threading
import time

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
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
            # 快速启动模式 - 先用规则模式，避免启动超时
            logger.info("使用快速启动模式（规则模式）")
            from extract import SmartNewsExtractor
            
            extractor = SmartNewsExtractor(
                use_bert=False,  # 先禁用BERT，确保快速启动
                preload_db="property_translations.db"
            )
            
            load_time = time.time() - _initialization_start_time
            logger.info(f"快速启动完成 (耗时: {load_time:.2f}秒)")
            _initialization_status = "ready_fast"
            
            # 后台异步加载BERT（可选）
            threading.Thread(target=load_bert_async, daemon=True).start()
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            _initialization_status = "failed"
            extractor = None

def load_bert_async():
    """后台异步加载BERT模型"""
    global extractor, _initialization_status
    
    try:
        logger.info("开始后台加载BERT模型...")
        time.sleep(5)  # 给系统一些时间稳定
        
        from extract import SmartNewsExtractor
        bert_extractor = SmartNewsExtractor(
            use_bert=True,
            preload_db="property_translations.db"
        )
        
        # 如果BERT加载成功，替换提取器
        with _initialization_lock:
            global extractor
            extractor = bert_extractor
            _initialization_status = "ready_bert"
            
        logger.info("BERT模型后台加载完成")
        
    except Exception as e:
        logger.warning(f"BERT后台加载失败，继续使用规则模式: {e}")

@app.route('/')
@app.route('/health')
def health_check():
    """健康检查端点 - 必须快速响应"""
    return jsonify({
        "status": "healthy",
        "service": "extractor-api",
        "ready": True,  # 总是返回ready，即使BERT未加载
        "mode": _initialization_status
    }), 200

@app.route('/status')
def status_check():
    return jsonify({
        "status": _initialization_status,
        "ready": extractor is not None,
        "extractor_available": extractor is not None
    }), 200

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

    if not extractor:
        return jsonify({
            "error": "服务暂时不可用",
            "status": _initialization_status
        }), 503

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

# 立即启动后台初始化
start_background_initialization()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"启动Flask应用，监听端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
