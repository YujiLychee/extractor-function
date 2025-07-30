from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import os

# 导入您的核心提取器
from extract import SmartNewsExtractor

# 创建 Flask 应用
app = Flask(__name__)

# --- 全局初始化 ---
logging.basicConfig(level=logging.INFO)
extractor = None

def init_extractor():
    global extractor
    try:
        logging.info("Extractor API: 正在进行冷启动初始化...")
        # 检查数据库文件是否存在
        db_path = "property_translations.db"
        if not os.path.exists(db_path):
            logging.warning(f"数据库文件 {db_path} 不存在，使用默认配置")
            # 创建一个空的数据库或使用默认配置
        
        extractor = SmartNewsExtractor(use_bert=True, preload_db=db_path)
        logging.info(" Extractor API: 初始化完成。")
    except Exception as e:
        logging.error(f" Extractor API: 初始化时发生严重错误: {e}")
        extractor = None

# 健康检查路由
@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "extractor-api"}), 200

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

    if not extractor:
        init_extractor()  # 重试初始化
        
    if not extractor:
        return jsonify({"error": "Extractor 未能成功初始化。"}), 500

    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return jsonify({"error": "请求体必须是包含 'content' 键的 JSON。"}), 400

    try:
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        # 将 dataclass 结果转换为字典以便返回
        result_dict = asdict(extraction_result)
        return jsonify(result_dict), 200
    except Exception as e:
        logging.error(f"提取过程中发生错误: {e}", exc_info=True)
        return jsonify({"error": "服务器内部提取错误。"}), 500

# 初始化提取器
init_extractor()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
