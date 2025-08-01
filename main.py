# import os
# from flask import Flask, request, jsonify
# from dataclasses import asdict
# import logging
# import threading
# import time

# app = Flask(__name__)

# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # 全局变量
# extractor = None
# _initialization_lock = threading.Lock()
# _initialization_status = "pending"

# def init_extractor_background():
#     """后台初始化提取器"""
#     global extractor, _initialization_status
    
#     with _initialization_lock:
#         if _initialization_status != "pending":
#             return
            
#         _initialization_status = "loading"
#         logger.info("开始后台初始化...")
        
#         try:
#             # 快速启动模式 - 禁用BERT确保快速启动
#             from extract import SmartNewsExtractor
            
#             extractor = SmartNewsExtractor(
#                 use_bert=False,  # 快速启动
#                 preload_db="property_translations.db"
#             )
            
#             _initialization_status = "ready"
#             logger.info("初始化完成（规则模式）")
            
#         except Exception as e:
#             logger.error(f"初始化失败: {e}", exc_info=True)
#             _initialization_status = "failed"

# @app.route('/')
# @app.route('/health')
# def health_check():
#     """健康检查端点 - 必须快速响应"""
#     return jsonify({
#         "status": "healthy",
#         "service": "extractor-api",
#         "ready": True,
#         "mode": _initialization_status
#     }), 200

# @app.route('/status')
# def status_check():
#     """状态检查"""
#     return jsonify({
#         "status": _initialization_status,
#         "ready": extractor is not None,
#         "extractor_available": extractor is not None
#     }), 200

# @app.route('/extract', methods=['POST', 'OPTIONS'])
# def extract_handler():
#     """提取处理端点"""
#     if request.method == 'OPTIONS':
#         headers = {
#             'Access-Control-Allow-Origin': '*',
#             'Access-Control-Allow-Methods': 'POST',
#             'Access-Control-Allow-Headers': 'Content-Type',
#             'Access-Control-Max-Age': '3600'
#         }
#         return ('', 204, headers)

#     if not extractor:
#         return jsonify({
#             "error": "服务正在初始化",
#             "status": _initialization_status
#         }), 503

#     request_json = request.get_json(silent=True)
#     if not request_json or 'content' not in request_json:
#         return jsonify({"error": "请求体必须是包含 'content' 键的 JSON"}), 400

#     try:
#         extraction_result = extractor.extract_candidates(
#             news_content=request_json['content'],
#             title=request_json.get('title', '')
#         )
#         result_dict = asdict(extraction_result)
#         return jsonify(result_dict), 200

#     except Exception as e:
#         logger.error(f"提取错误: {e}", exc_info=True)
#         return jsonify({"error": f"提取错误: {str(e)}"}), 500

# # 启动后台初始化
# threading.Thread(target=init_extractor_background, daemon=True).start()

# if __name__ == '__main__':
#     # 从环境变量获取端口，确保监听正确端口
#     port = int(os.environ.get('PORT', 8080))
    
#     logger.info(f"启动应用，监听端口: {port}")
#     logger.info(f"监听所有网络接口: 0.0.0.0")
    
#     # 确保监听 0.0.0.0 而不是 127.0.0.1
#     app.run(
#         host='0.0.0.0',  # 必须是 0.0.0.0
#         port=port,
#         debug=False,
#         threaded=True
#     )
import os
import time
from flask import Flask, request, jsonify
from dataclasses import asdict
import logging
import threading
import signal

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
extractor = None
_initialization_lock = threading.Lock()
_initialization_status = "pending"
_initialization_start_time = None

class TimeoutError(Exception):
    pass

def create_minimal_extractor():
    """创建最小配置的提取器，使用数据库关键词"""
    logger.info("创建最小配置提取器，加载数据库关键词...")
    
    # 最简化的提取器类
    class MinimalExtractor:
        def __init__(self):
            self.db_keywords = self._load_db_keywords()
            self.protected_locations = self._load_protected_locations()
            logger.info(f"加载了 {len(self.db_keywords)} 个数据库关键词")
            logger.info(f"加载了 {len(self.protected_locations)} 个保护地名")
        
        def _load_db_keywords(self):
            """从数据库加载关键词"""
            try:
                import sqlite3
                db_path = "property_translations.db"
                if not os.path.exists(db_path):
                    logger.warning(f"数据库文件不存在: {db_path}")
                    return []
                
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT chinese_name FROM verified_translations")
                keywords = [row[0] for row in cur.fetchall()]
                conn.close()
                return keywords
            except Exception as e:
                logger.error(f"加载数据库关键词失败: {e}")
                return []
        
        def _load_protected_locations(self):
            """加载保护的地名"""
            return [
                # 香港区域
                '中西區', '灣仔', '東區', '南區', '深水埗', '油尖旺', '九龍城',
                '黃大仙', '觀塘', '荃灣', '屯門', '元朗', '北區', '大埔',
                '沙田', '西貢', '離島',
                
                # 主要地区
                '中環', '金鐘', '灣仔', '銅鑼灣', '天后', '炮台山', '北角',
                '鰂魚涌', '太古', '西營盤', '上環', '堅尼地城', '薄扶林',
                '香港仔', '鴨脷洲', '赤柱', '尖沙咀', '佐敦', '油麻地',
                '旺角', '太子', '深水埗', '長沙灣', '荔枝角', '美孚',
                '九龍塘', '何文田', '紅磡', '土瓜灣', '馬頭角', '沙田',
                '大圍', '火炭', '馬鞍山', '大埔', '粉嶺', '上水', '元朗',
                '天水圍', '屯門', '荃灣', '葵涌', '青衣', '將軍澳', '西貢',
                
                # 港鐵站
                '中環站', '金鐘站', '灣仔站', '銅鑼灣站', '天后站', '炮台山站', 
                '北角站', '鰂魚涌站', '太古站', '西灣河站', '筲箕灣站', 
                '杏花邨站', '柴灣站', '上環站', '西營盤站', '香港大學站', 
                '堅尼地城站', '海怡半島站', '利東站', '黃竹坑站', '海洋公園站',
                '尖沙咀站', '佐敦站', '油麻地站', '旺角站', '太子站', 
                '深水埗站', '長沙灣站', '荔枝角站', '美孚站', '荃灣西站', 
                '荃灣站', '大窩口站', '葵興站', '葵芳站', '荔景站', '青衣站',
                '欣澳站', '東涌站', '機場站', '九龍塘站', '樂富站', 
                '黃大仙站', '鑽石山站', '彩虹站', '九龍灣站', '牛頭角站', 
                '觀塘站', '藍田站', '油塘站', '調景嶺站', '將軍澳站', 
                '坑口站', '寶琳站', '康城站', '何文田站', '土瓜灣站', 
                '宋皇臺站', '啟德站', '顯徑站', '大圍站', '車公廟站', 
                '沙田圍站', '第一城站', '石門站', '大水坑站', '恆安站', 
                '馬鞍山站', '烏溪沙站', '大學站', '火炭站', '馬場站', 
                '沙田站', '大埔墟站', '太和站', '粉嶺站', '上水站', 
                '羅湖站', '落馬洲站', '元朗站', '朗屏站', '天水圍站',
                '兆康站', '屯門站', '紅磡站', '旺角東站', '錦上路站', '奧運站'
            ]
        
        def extract_candidates(self, news_content, title=""):
            from dataclasses import dataclass
            from typing import List
            
            @dataclass
            class CandidateKeyword:
                text: str
                keyword_type: str
                confidence: float
                position: int
                context: str
                extraction_method: str
            
            @dataclass
            class NewsExtractionResult:
                property_candidates: List[CandidateKeyword]
                location_candidates: List[CandidateKeyword]
                entity_candidates: List[CandidateKeyword]
                remaining_content: str
                extraction_summary: dict
            
            property_candidates = []
            location_candidates = []
            entity_candidates = []
            
            # 1. 数据库关键词匹配（主要是楼盘名）
            for keyword in self.db_keywords:
                if keyword in news_content:
                    pos = news_content.find(keyword)
                    context = self._get_context(news_content, pos, keyword)
                    
                    property_candidates.append(CandidateKeyword(
                        text=keyword,
                        keyword_type="property",
                        confidence=0.95,
                        position=pos,
                        context=context,
                        extraction_method="db_keyword_match"
                    ))
            
            # 2. 保护地名匹配
            for location in self.protected_locations:
                if location in news_content:
                    pos = news_content.find(location)
                    context = self._get_context(news_content, pos, location)
                    
                    location_candidates.append(CandidateKeyword(
                        text=location,
                        keyword_type="location",
                        confidence=0.9,
                        position=pos,
                        context=context,
                        extraction_method="protected_location_match"
                    ))
            
            # 3. 基本开发商匹配
            developers = ['新鸿基', '恒基', '长实', '会德丰', '新世界', '太古', 
                         '嘉里', '南丰', '信和', '远东', '九建', '华润', '中海']
            
            for dev in developers:
                if dev in news_content:
                    pos = news_content.find(dev)
                    context = self._get_context(news_content, pos, dev)
                    
                    entity_candidates.append(CandidateKeyword(
                        text=dev,
                        keyword_type="developer",
                        confidence=0.85,
                        position=pos,
                        context=context,
                        extraction_method="developer_match"
                    ))
            
            # 4. 旗下模式匹配
            if "旗下" in news_content:
                qixia_pos = news_content.find("旗下")
                
                # 前面的开发商
                before_text = news_content[max(0, qixia_pos-20):qixia_pos]
                for dev in developers:
                    if dev in before_text:
                        if dev not in [c.text for c in entity_candidates]:
                            context = self._get_context(news_content, qixia_pos-10, dev)
                            entity_candidates.append(CandidateKeyword(
                                text=dev,
                                keyword_type="developer",
                                confidence=0.9,
                                position=before_text.find(dev) + max(0, qixia_pos-20),
                                context=context,
                                extraction_method="qixia_pattern_developer"
                            ))
                
                # 后面的地点
                after_text = news_content[qixia_pos+2:qixia_pos+50]
                for location in self.protected_locations:
                    if location in after_text:
                        if location not in [c.text for c in location_candidates]:
                            context = self._get_context(news_content, qixia_pos+2, location)
                            location_candidates.append(CandidateKeyword(
                                text=location,
                                keyword_type="location",
                                confidence=0.9,
                                position=qixia_pos+2+after_text.find(location),
                                context=context,
                                extraction_method="qixia_pattern_location"
                            ))
            
            # 去重
            property_candidates = self._deduplicate_candidates(property_candidates)
            location_candidates = self._deduplicate_candidates(location_candidates)
            entity_candidates = self._deduplicate_candidates(entity_candidates)
            
            # 按置信度排序
            property_candidates.sort(key=lambda x: x.confidence, reverse=True)
            location_candidates.sort(key=lambda x: x.confidence, reverse=True)
            entity_candidates.sort(key=lambda x: x.confidence, reverse=True)
            
            return NewsExtractionResult(
                property_candidates=property_candidates[:10],  # 取前10个
                location_candidates=location_candidates[:10],
                entity_candidates=entity_candidates[:5],
                remaining_content=self._generate_remaining_content(
                    news_content, property_candidates, location_candidates, entity_candidates
                ),
                extraction_summary={
                    "mode": "minimal_db_enhanced",
                    "total_property_candidates": len(property_candidates),
                    "total_location_candidates": len(location_candidates),
                    "total_entity_candidates": len(entity_candidates),
                    "db_keywords_loaded": len(self.db_keywords),
                    "extraction_methods": list(set([c.extraction_method for c in 
                                                  property_candidates + location_candidates + entity_candidates]))
                }
            )
        
        def _get_context(self, content, position, word, window=30):
            """获取词汇的上下文"""
            start = max(0, position - window)
            end = min(len(content), position + len(word) + window)
            return content[start:end]
        
        def _deduplicate_candidates(self, candidates):
            """去重候选词"""
            seen = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.text not in seen:
                    seen.add(candidate.text)
                    unique_candidates.append(candidate)
            return unique_candidates
        
        def _generate_remaining_content(self, original_content, *candidate_lists):
            """生成移除关键词后的剩余内容"""
            remaining = original_content
            all_keywords = []
            for candidates in candidate_lists:
                all_keywords.extend([c.text for c in candidates])
            
            for keyword in all_keywords:
                remaining = remaining.replace(keyword, '[REMOVED]')
            
            import re
            remaining = re.sub(r'\[REMOVED\]\s*', ' ', remaining)
            remaining = re.sub(r'\s+', ' ', remaining).strip()
            return remaining
    
    return MinimalExtractor()

def init_extractor_background():
    """快速初始化提取器"""
    global extractor, _initialization_status, _initialization_start_time
    
    with _initialization_lock:
        if _initialization_status != "pending":
            return
            
        _initialization_status = "loading"
        _initialization_start_time = time.time()
        logger.info("开始快速初始化...")
        
        try:
            # 设置初始化超时
            def timeout_handler(signum, frame):
                raise TimeoutError("初始化超时")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30秒超时
            
            try:
                from extract import SmartNewsExtractor
                
                # 使用最简化配置
                logger.info("尝试完整提取器初始化...")
                extractor = SmartNewsExtractor(
                    use_bert=False,  # 禁用BERT
                    preload_db="property_translations.db"
                )
                
                signal.alarm(0)  # 取消超时
                
                elapsed = time.time() - _initialization_start_time
                _initialization_status = "ready"
                logger.info(f"完整提取器初始化完成！耗时: {elapsed:.2f}秒")
                
            except:
                logger.warning("完整提取器初始化失败，使用最小配置")
                signal.alarm(0)  # 取消超时
                
                # 使用最小配置重试
                extractor = create_minimal_extractor()
                _initialization_status = "ready_minimal"
                elapsed = time.time() - _initialization_start_time
                logger.info(f"最小配置初始化完成！耗时: {elapsed:.2f}秒")
                
        except TimeoutError:
            logger.error("初始化超时，使用最小配置")
            try:
                # 使用最小配置重试
                extractor = create_minimal_extractor()
                _initialization_status = "ready_minimal"
                elapsed = time.time() - _initialization_start_time
                logger.info(f"超时后最小配置初始化完成！耗时: {elapsed:.2f}秒")
            except Exception as e:
                logger.error(f"最小配置也失败: {e}")
                _initialization_status = "failed"
                
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            try:
                # fallback到最小配置
                extractor = create_minimal_extractor()
                _initialization_status = "ready_minimal"
                elapsed = time.time() - _initialization_start_time
                logger.info(f"异常后最小配置初始化完成！耗时: {elapsed:.2f}秒")
            except:
                _initialization_status = "failed"

@app.route('/')
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "extractor-api",
        "ready": True,
        "mode": _initialization_status
    }), 200

@app.route('/status')
def status_check():
    elapsed = ""
    if _initialization_start_time:
        elapsed = f"{time.time() - _initialization_start_time:.1f}s"
    
    return jsonify({
        "status": _initialization_status,
        "ready": extractor is not None,
        "extractor_available": extractor is not None,
        "loading_time": elapsed
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

    # 即使在loading状态也尝试处理请求
    if not extractor:
        return jsonify({
            "error": "服务正在初始化，请稍后重试",
            "status": _initialization_status,
            "estimated_wait": "30-60秒"
        }), 503

    request_json = request.get_json(silent=True)
    if not request_json or 'content' not in request_json:
        return jsonify({"error": "请求体必须是包含 'content' 键的 JSON"}), 400

    try:
        logger.info("开始处理提取请求...")
        extraction_result = extractor.extract_candidates(
            news_content=request_json['content'],
            title=request_json.get('title', '')
        )
        
        result_dict = asdict(extraction_result)
        logger.info("提取完成")
        return jsonify(result_dict), 200

    except Exception as e:
        logger.error(f"提取错误: {e}", exc_info=True)
        return jsonify({"error": f"提取错误: {str(e)}"}), 500

# 启动初始化
logger.info("启动后台初始化...")
threading.Thread(target=init_extractor_background, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"启动Flask应用，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
