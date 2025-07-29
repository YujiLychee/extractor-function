from flask import jsonify
from dataclasses import asdict
import logging

# 導入您的核心提取器
from extract import SmartNewsExtractor

# --- 全局初始化 ---
# 這段代碼只在雲函數實例啟動時運行一次，避免重複加載模型
logging.basicConfig(level=logging.INFO)
try:
    logging.info("Extractor API: 正在進行冷啟動初始化...")
    # 注意：這裡我們仍然需要 db 文件來初始化關鍵詞列表
    extractor = SmartNewsExtractor(use_bert=True, preload_db="property_translations.db")
    logging.info("✅ Extractor API: 初始化完成。")
except Exception as e:
    logging.error(f"❌ Extractor API: 初始化時發生嚴重錯誤: {e}")
    extractor = None

# --- Google Cloud Function 入口函數 ---
def extract_handler(request):
    # 處理 CORS 預檢請求
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
        return (jsonify({"error": "Extractor 未能成功初始化。"}), 500, headers)

    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return (jsonify({"error": "請求體必須是包含 'content' 鍵的 JSON。"}), 400, headers)

    try:
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        # 將 dataclass 結果轉換為字典以便返回
        result_dict = asdict(extraction_result)
        return (jsonify(result_dict), 200, headers)
    except Exception as e:
        logging.error(f"提取過程中發生錯誤: {e}", exc_info=True)
        return (jsonify({"error": "服務器內部提取錯誤。"}), 500, headers)