import unicodedata, re
import jieba
import jieba.posseg as pseg
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import logging
from keyword_loader import load_predefined_keywords 

@dataclass
class CandidateKeyword:
    text: str
    keyword_type: str  # 'property', 'location', 'general'
    confidence: float
    position: int  # 在文本中的位置
    context: str   # 上下文
    extraction_method: str

@dataclass
class NewsExtractionResult:
    property_candidates: List[CandidateKeyword]
    location_candidates: List[CandidateKeyword]
    entity_candidates: List[CandidateKeyword] 
    remaining_content: str
    extraction_summary: Dict

class SmartNewsExtractor:
    def __init__(self, use_bert: bool = True,
                 preload_db: str = "property_translations.db"):
        # 初始化 jieba 預設字典
        self._init_jieba_dict()
        self.slang_terms = self._load_slang_terms(preload_db)
        # 額外載入資料庫詞
        self.preloaded_keywords = load_predefined_keywords(preload_db)
        for kw in self.preloaded_keywords:
            # tag 設成 nr 只是為了讓 jieba 不拆分；freq 給高一點
            jieba.add_word(kw, freq=2000, tag='nr')

        self.protected_locations = [
            # 香港區域
            '中西區', '灣仔', '東區', '南區', '深水埗', '油尖旺', '九龍城',
            '黃大仙', '觀塘', '荃灣', '屯門', '元朗', '北區', '大埔',
            '沙田', '西貢', '離島',
            
            # 主要地區
            '中環', '金鐘', '灣仔', '銅鑼灣', '天后', '炮台山', '北角',
            '鰂魚涌', '太古', '西營盤', '上環', '堅尼地城', '薄扶林',
            '香港仔', '鴨脷洲', '赤柱', '尖沙咀', '佐敦', '油麻地',
            '旺角', '太子', '深水埗', '長沙灣', '荔枝角', '美孚',
            '九龍塘', '何文田', '紅磡', '土瓜灣', '馬頭角', '沙田',
            '大圍', '火炭', '馬鞍山', '大埔', '粉嶺', '上水', '元朗',
            '天水圍', '屯門', '荃灣', '葵涌', '青衣', '將軍澳', '西貢',
            
            # 港鐵站
            '中環站', '金鐘站', '灣仔站', '銅鑼灣站', '天后站', '炮台山站', '北角站', '鰂魚涌站',
            '太古站', '西灣河站', '筲箕灣站', '杏花邨站', '柴灣站', '上環站', '西營盤站',
            '香港大學站', '堅尼地城站', '海怡半島站', '利東站', '黃竹坑站', '海洋公園站',
            '尖沙咀站', '佐敦站', '油麻地站', '旺角站', '太子站', '深水埗站',
            '長沙灣站', '荔枝角站', '美孚站', '荃灣西站', '荃灣站', '大窩口站', '葵興站',
            '葵芳站', '荔景站', '青衣站', '欣澳站', '東涌站', '機場站', '九龍塘站',
            '樂富站', '黃大仙站', '鑽石山站', '彩虹站', '九龍灣站', '牛頭角站', '觀塘站',
            '藍田站', '油塘站', '調景嶺站', '將軍澳站', '坑口站', '寶琳站', '康城站',
            '何文田站', '土瓜灣站', '宋皇臺站', '啟德站', '顯徑站', '大圍站',
            '車公廟站', '沙田圍站', '第一城站', '石門站', '大水坑站', '恆安站', '馬鞍山站',
            '烏溪沙站', '大學站', '火炭站', '馬場站', '沙田站', '大埔墟站', '太和站',
            '粉嶺站', '上水站', '羅湖站', '落馬洲站', '元朗站', '朗屏站', '天水圍站',
            '兆康站', '屯門站', '紅磡站', '旺角東站', '錦上路站','奧運站'
        ]

        self.protected_keywords = (
            set(self.protected_locations) |
            set(self.slang_terms)
        )

    # BERT模型初始化
        self.use_bert = use_bert
        if use_bert:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
                # 使用中文NER預訓練模型
                model_name = "ckiplab/bert-base-chinese-ner"
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    aggregation_strategy="simple",
                    device=-1  # 使用CPU，如需GPU改為0
                )
                print("BERT NER模型載入成功")
            except Exception as e:
                print(f"BERT模型載入失敗，將使用規則方法: {e}")
                self.use_bert = False
                self.ner_pipeline = None
        
        # 專有名詞類型映射
        self.entity_type_mapping = {
            'PER': 'person',           # 人名（可能包含开发商人名）
            'ORG': 'organization',     # 组织机构（开发商公司）
            'LOC': 'location',         # 地点
            'GPE': 'location',         # 地缘政治实体
            'FAC': 'facility',         # 设施（建筑物等）
            'MISC': 'miscellaneous'    # 其他专有名词
        }
        
        # 開發商相關模式
        self.developer_patterns = {
            'company_suffixes': [
                '地產', '發展', '集團', '控股', '有限公司', '置業', '建設',
                '投資', '置地', '物業', '房地產', '地產發展', '國際','建築'
            ],
            'known_developers': [
                '新鴻基', '恒基', '長實', '會德豐', '新世界', '太古', '嘉里',
                '南豐', '信和', '遠東', '九建', '建業', '富力', '碧桂園',
                '華潤', '中海', '保利', '萬科', '融創', '綠地'
            ]
        }
        # 房地產關鍵字模式
        self.property_patterns = {
            'building_suffixes': [
                '花園', '苑', '庭', '閣', '廈', '座', '樓', '中心', '廣場', '城', '峰', 
                '村', '台', '軒', '居', '邸', '府', '館', '宮', '殿', '豪庭',
                '名門', '雅苑', '翠園', '金苑', '銀苑', '海景', '山景', '湖景'
            ],
            'luxury_prefixes': [
                '豪', '御', '尊', '貴', '雅', '翠', '金', '銀', '珀', '璧',
                '瑞', '祥', '富', '盛', '華', '皇', '帝', '王', '龍', '鳳',
                '麗', '美', '君', '名', '星', '月', '日', '天', '海', '山'
            ],
            'development_terms': [
                '新盤', '樓盤', '項目', '發展', '地產', '物業', '住宅', 
                '商業', '辦公', '零售', '期數', '座數'
            ]
        }
        
        # 地理位置模式
        self.location_patterns = {
            'hk_districts': [
                '中西區', '灣仔', '東區', '南區', '深水埗', '油尖旺', '九龍城',
                '黃大仙', '觀塘', '荃灣', '屯門', '元朗', '北區', '大埔',
                '沙田', '西貢', '離島'
            ],
            'major_areas': [
                '中環', '金鐘', '灣仔', '銅鑼灣', '天后', '炮台山', '北角',
                '鰂魚涌', '太古', '西營盤', '上環', '堅尼地城', '薄扶林',
                '香港仔', '鴨脷洲', '赤柱', '尖沙咀', '佐敦', '油麻地',
                '旺角', '太子', '深水埗', '長沙灣', '荔枝角', '美孚',
                '九龍塘', '何文田', '紅磡', '土瓜灣', '馬頭角', '沙田',
                '大圍', '火炭', '馬鞍山', '大埔', '粉嶺', '上水', '元朗',
                '天水圍', '屯門', '荃灣', '葵涌', '青衣', '將軍澳', '西貢'
            ],
            'mtr_stations': [
            '中環站', '金鐘站', '灣仔站', '銅鑼灣站', '天后站', '炮台山站', '北角站', '鰂魚涌站',
            '太古站', '西灣河站', '筲箕灣站', '杏花邨站', '柴灣站', '上環站', '西營盤站',
            '香港大學站', '堅尼地城站', '海怡半島站', '利東站', '黃竹坑站', '海洋公園站',
            '尖沙咀站', '佐敦站', '油麻地站', '旺角站', '太子站', '深水埗站',
            '長沙灣站', '荔枝角站', '美孚站', '荃灣西站', '荃灣站', '大窩口站', '葵興站',
            '葵芳站', '荔景站', '青衣站', '欣澳站', '東涌站', '機場站', '九龍塘站',
            '樂富站', '黃大仙站', '鑽石山站', '彩虹站', '九龍灣站', '牛頭角站', '觀塘站',
            '藍田站', '油塘站', '調景嶺站', '將軍澳站', '坑口站', '寶琳站', '康城站',
            '何文田站', '土瓜灣站', '宋皇臺站', '啟德站', '顯徑站', '大圍站',
            '車公廟站', '沙田圍站', '第一城站', '石門站', '大水坑站', '恆安站', '馬鞍山站',
            '烏溪沙站', '大學站', '火炭站', '馬場站', '沙田站', '大埔墟站', '太和站',
            '粉嶺站', '上水站', '羅湖站', '落馬洲站', '元朗站', '朗屏站', '天水圍站',
            '兆康站', '屯門站', '紅磡站', '旺角東站', '錦上路站','奧運站'
            ],
            'street_indicators': ['街', '道', '路', '徑', '里', '坊', '巷', '圍'],
            'location_indicators': ['港島', '九龍', '新界', '位於', '坐落', '鄰近']
        }
        
        # 上下文權重
        self.context_weights = {
            'first_paragraph': 1.5,   # 第一段落權重更高
            'second_paragraph': 1.2,  # 第二段落
            'title_proximity': 2.0,   # 靠近標題
            'punctuation_boundary': 1.3,  # 句號、逗號邊界
            'development_context': 1.4    # 開發相關語境
        }
        self.title_boost_factor = 1.3  # 標題中出現的詞彙置信度加成
        self.title_qixia_boost_factor = 1.5  #標題+旗下模式的額外加成
        self.type_thresholds = {
            'property': 0.6,
            'location': 0.6, 
            'developer': 0.65
        }

    def _load_slang_terms(self, db_path) -> List[str]:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT chinese_name FROM verified_translations")
        terms = [r[0] for r in cur.fetchall()]
        conn.close()
        return terms

    #初始化jieba自訂字典。下一步是與已有的詞庫結合。
    def _init_jieba_dict(self):
        # 添加房地產專業詞彙
        property_vocab = [
            '新地', '恒地', '長實', '會德豐', '新世界', '太古地產', '嘉里建設',
            '南豐', '信和', '遠東發展', '九建', '建業', '富力', '碧桂園',
            '天鑄', '日出康城', '海怡半島', '太古城', '置富花園', '美孚新邨',
            '又一城', '朗豪坊', '太古廣場', 'IFC', 'ICC', '環球貿易廣場'
        ]
        
        for word in property_vocab:
            jieba.add_word(word, freq=1000, tag='nr') #  nr = 人名/機構名
    #根據jieba詞庫預處理
    def _extract_preloaded_keywords(self, content: str) -> List[CandidateKeyword]:
        found = []
        for kw in self.preloaded_keywords:
            pos = content.find(kw)
            if pos != -1:
                found.append(CandidateKeyword(
                    text=kw,
                    keyword_type='property',          # 默認為estate name,但後續也會調整
                    confidence=0.95,
                    position=pos,
                    context=self._get_context(content, pos, kw),
                    extraction_method="preloaded_dict"
                ))
        return found

    #        #分析'旗下'模式：提取開發商、地點、樓盤名稱。
    # 我們注意到很多新聞報道都含有“旗下“這個詞語，且在前後包含了開發商名稱和地名。主要根據這個提取潛在的短語。
    def _analyze_qixia_pattern(self, segmented_words: List[Tuple[str, str, int]], qixia_index: int, content: str) -> Dict[str, List[CandidateKeyword]]:

        result = {'developers': [], 'properties': [], 'locations': []}
        
        # 1. 向前查找開發商（在"旗下"之前）
        developer = self._extract_developer_before_qixia(segmented_words, qixia_index, content)
        if developer:
            result['developers'].append(developer)
        
        # 2. 向後查找地點和樓盤（在"旗下"之後）
        locations, properties = self._extract_location_property_after_qixia(segmented_words, qixia_index, content)
        result['locations'].extend(locations)
        result['properties'].extend(properties)
        
        return result

    #  分析'旗下'模式，結合標題資訊
    def _analyze_qixia_pattern_with_title(self, segmented_words: List[Tuple[str, str, int]], 
                                        qixia_index: int, content: str, 
                                        title_candidates: List[Dict]) -> Dict[str, List[CandidateKeyword]]:
        result = {'developers': [], 'properties': [], 'locations': []}
        
        # 向前查找開發商
        developer = self._extract_developer_before_qixia(segmented_words, qixia_index, content)
        if developer:
            result['developers'].append(developer)
        
        # 向後查找地點和樓盤，結合標題資訊
        locations, properties = self._extract_location_property_after_qixia_with_title(
            segmented_words, qixia_index, content, title_candidates
        )
        result['locations'].extend(locations)
        result['properties'].extend(properties)
        
        return result

    #將內容分割為段落
    def _split_into_paragraphs(self, content: str) -> List[str]:
        # 按多種分隔符號分段
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|。\s*(?=[A-Z\u4e00-\u9fff])', content)
        # 清理空段落
        return [p.strip() for p in paragraphs if p.strip()]
    
    #獲取重點關注的內容（前幾段）
    def _get_focus_content(self, paragraphs: List[str], focus_count: int) -> str:
        focus_paragraphs = paragraphs[:focus_count]
        return ' '.join(focus_paragraphs)
    
    def _protect_priority_terms(self, text: str) -> str:
        protected_text = text
        # 長度由長到短，避免『凸海戶』→『海戶』被部分覆寫
        for term in sorted(self.protected_keywords, key=len, reverse=True):
            if term in protected_text:
                placeholder = f"<<PROTECTED_{len(term)}_{term}>>"
                protected_text = protected_text.replace(term, placeholder)
        return protected_text

    def _restore_protected_names(self, text: str) -> str:
    #恢復被保護的地名
        return re.sub(r'<<PROTECTED_\d+_(.+?)>>', r'\1', text)


    ## 智慧分詞，返回(詞彙, 詞性, 位置)，特別處理連續英文單詞
    def _smart_segmentation(self, content: str) -> List[Tuple[str, str, int]]:
        words_with_pos = []
        protected_content = self._protect_priority_terms(content)
        # 先預處理連續英文單詞
        processed_content = self._merge_consecutive_english_words(protected_content)
        
        # 使用jieba進行詞性標注
        seg_list = pseg.cut(processed_content)
        
        position = 0
        original_position = 0  # 在原始文本中的位置
        
        for word_pair in seg_list:
            # 兼容不同版本的jieba
            if hasattr(word_pair, 'word') and hasattr(word_pair, 'flag'):
                word = word_pair.word
                pos = word_pair.flag
            else:
                try:
                    word, pos = word_pair
                except (TypeError, ValueError):
                    word = str(word_pair)
                    pos = 'n'
            
            if len(word.strip()) > 0:
                # 恢復保護的地名
                restored_word = self._restore_protected_names(word)
                
                # 跳過已知占位符
                if restored_word in {"KNOWN", "[KNOWN]", "¤¤¤K¤¤¤"}:
                    original_position += len(restored_word)
                    continue
                
                words_with_pos.append((restored_word.strip(), pos, original_position))
                original_position += len(restored_word)
        
        return words_with_pos

    # 合併連續的英文單詞為一個token
    def _merge_consecutive_english_words(self, content: str) -> str:

        patterns = [
            # 英文單詞 + 中文期數 (如: NOVO LAND第3A期)
            r'([A-Z][A-Za-z\s]+)(?=第?\d*[ABC]?期)',
            # 純英文樓盤名 (如: ONE STANLEY)
            r'\b([A-Z][A-Z\s]{2,})\b(?=[^a-zA-Z]|$)',
            # 英文+數位組合 (如: NOVA 1)
            r'\b([A-Z][A-Za-z]*\s+\d+[A-Za-z]*)\b'
        ]
        
        processed = content
        english_tokens = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, processed)
            for match in matches:
                original = match.group(1).strip()
                if len(original) >= 3:  # 至少3個字元
                    # 用底線替換空格，便於jieba識別
                    token = original.replace(' ', '_')
                    processed = processed.replace(original, token)
                    english_tokens.append(token)
        
        # 將所有英文token添加到jieba詞典
        for token in english_tokens:
            jieba.add_word(token, freq=2000, tag='nr')  # 高頻率，確保不被拆分
        
        return processed

    # 驗證BERT識別的實體是否有效
    def _validate_bert_entity(self, entity: Dict, content: str) -> bool:
        text = entity['word']
        # 長度過濾
        if len(text) < 2 or len(text) > 15:
            return False       
        # 置信度過濾
        if entity['score'] < 0.5:
            return False       
        # 排除一些常見的誤識別
        false_positives = [
            '記者', '報道', '消息', '表示', '指出', '認為', '預計', '估計',
            '今日', '昨日', '明日', '上午', '下午', '晚上', '時間', '現時'
        ]
        
        if text in false_positives:
            return False
        # 排除純數字或標點
        if text.isdigit() or not any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in text):
            return False
        
        return True
    

        # True=從樓盤名稱清單中剔除。
        # 覆蓋兩大類：
        # A. 僅為期數 / 座號；
        # B. 通用設施詞：天台、車位、平台、停車場、會所、泳池 ……
    def _is_noise_property_term(self, name: str) -> bool:
        if not name:
            return False
        s = unicodedata.normalize("NFKC", name)
        s = re.sub(r"\s+", "", s)           # 去空白
        s_lower = s.lower()
            # 檢查是否為單獨的期數（如"II期"、"第3期"等）
        period_patterns = [
            r"^第?\d+[期座幢棟层樓]$",  # 第3期、3期、3座等
            r"^[IVX]+期$",            # II期、III期等羅馬數字
            r"^第?[一二三四五六七八九十]+期$",  # 第三期、三期等
        ]
        
        for pattern in period_patterns:
            if re.match(pattern, s, re.IGNORECASE):
                return True  
            
        # 太長就不可能只是編號
        if len(s_lower) <= 4:
            zh2num = "零一二三四五六七八九十"
            for i, ch in enumerate(zh2num):
                s_lower = s_lower.replace(ch, str(i))

            pattern = r"^(block|tower)?(第)?\d+[期座幢栋棟层樓]$"
            if re.match(pattern, s_lower, re.IGNORECASE):
                return True
            
            # 檢查"XX樓"模式 - 只有>=3個字才可能是樓盤
        if name.endswith("樓") and len(name) < 3:
            return True  # "現樓"、"新樓"等短詞視為噪音
        
        generic_facilities = [
            "天台", "平台", "車位", "停車場", "車庫",
            "會所", "泳池", "花園", "球場", "天橋",
            "連廊", "地下", "頂層", "裙樓","山景"
        ]
        return name.strip() in generic_facilities

    #識別房地產候選關鍵字（整合BERT結果）
    def _identify_property_candidates(self, segmented_words: List[Tuple[str, str, int]], 
                                content: str, bert_entities: List[Dict] = None) -> List[CandidateKeyword]:
        candidates = []
        bert_entities = bert_entities or []
        
        # 從BERT結果中提取房地產相關實體
        for entity in bert_entities:
            if entity['mapped_type'] in ['facility', 'organization']:
                 # 進一步驗證是否真的是房地產項目
                if self._is_property_related_entity(entity['text'], content):
                    candidates.append(CandidateKeyword(
                        text=entity['text'],
                        keyword_type='property',
                        confidence=entity['confidence'] * 0.9,
                        position=entity['start'],
                        context=self._get_context(content, entity['start'], entity['text']),
                        extraction_method=f"bert_{entity['label']}_property"
                    ))
        
        # 規則匹配房地產關鍵字
        for i, (word, pos, position) in enumerate(segmented_words):
            # 檢查是否已被BERT識別
            if any(entity['text'] == word for entity in bert_entities):
                continue
            
            confidence = 0.0
            extraction_method = "pattern_matching"

            if '_' in word:
            # 這是合併的英文詞，恢復原始名稱
                original_name = word.replace('_', ' ')
                if self._is_english_property_name_enhanced(original_name):
                    confidence += 0.85
                    extraction_method = "english_property_name"
                    
                    # 檢查是否與期數組合
                    combined_candidate = self._check_english_property_with_phase(
                        segmented_words, i, content, original_name
                    )
                    if combined_candidate:
                        candidates.append(combined_candidate)
                        continue
                    else:
                        # 單獨的英文樓盤名
                        candidates.append(CandidateKeyword(
                            text=original_name,
                            keyword_type='property',
                            confidence=confidence,
                            position=position,
                            context=self._get_context(content, position, original_name),
                            extraction_method=extraction_method
                        ))
                        continue
            
            # 尾碼匹配（如：XX花園、XX苑）
            for suffix in self.property_patterns['building_suffixes']:
                if word.endswith(suffix) and len(word) > len(suffix):
                    # 特殊處理"樓"尾碼 - 至少要3個字
                    if suffix == "樓" and len(word) < 3:
                        continue
                    
                    confidence += 0.7
                    extraction_method = f"suffix_match_{suffix}"
                    break
            
            # 首碼+尾碼組合
            for prefix in self.property_patterns['luxury_prefixes']:
                for suffix in self.property_patterns['building_suffixes']:
                    if prefix in word and word.endswith(suffix):
                        confidence += 0.8
                        extraction_method = f"prefix_suffix_match"
                        break
            
            # 詞性分析（專有名詞）
            if pos in ['nr', 'nz', 'nt'] and len(word) >= 2:
                confidence += 0.4
                extraction_method += "_proper_noun"
            
            # 上下文增強
            context = self._get_context(content, position, word)
            context_boost = self._calculate_context_boost(context, 'property')
            confidence += context_boost
            
            # 組合詞檢測（如：XX花園第X期）
            combined_candidate = self._check_combined_property(
                segmented_words, i, content
            )
            if combined_candidate:
                candidates.append(combined_candidate)
                continue
            
            #  如果信心度足夠高，添加為候選
            if confidence >= 0.5:
                candidates.append(CandidateKeyword(
                    text=word,
                    keyword_type='property',
                    confidence=min(confidence, 0.95),
                    position=position,
                    context=context,
                    extraction_method=extraction_method
                ))
        
        # 去重和排序
        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates[:10]  # 返回前10個最可能的候選

    #識別地理位置候選關鍵字，改進邊界識別
    def _identify_location_candidates(self, segmented_words: List[Tuple[str, str, int]], 
                                    content: str, bert_entities: List[Dict] = None) -> List[CandidateKeyword]:
        candidates = []
        bert_entities = bert_entities or []
        # 從BERT結果中提取地理位置，但要驗證邊界
        for entity in bert_entities:
            if entity['mapped_type'] == 'location':
                # 驗證這個地名是否合理

                if self._validate_location_boundary(entity['text'], content, entity['start']):
                    candidates.append(CandidateKeyword(
                        text=entity['text'],
                        keyword_type='location',
                        confidence=entity['confidence'],
                        position=entity['start'],
                        context=self._get_context(content, entity['start'], entity['text']),
                        extraction_method=f"bert_{entity['label']}_validated"
                    ))

        for i, (word, pos, position) in enumerate(segmented_words):
            # 檢查是否已被BERT識別（且通過驗證）
            if any(entity['text'] == word for entity in bert_entities if entity['mapped_type'] == 'location'):
                continue
            
            confidence = 0.0
            extraction_method = "location_matching"      
            # 改進的地名識別邏輯
            location_result = self._analyze_location_context(word, segmented_words, i, content)
            
            if location_result:
                confidence = location_result['confidence']
                extraction_method = location_result['method']            
                # 如果識別為地名，檢查是否與樓盤名衝突
                if confidence >= 0.5:
                    # 檢查前後詞彙，避免把樓盤名誤識別為地名
                    context_check = self._check_location_property_conflict(word, segmented_words, i)
                    if context_check['is_property']:
                        continue  # 跳過，這可能是樓盤名的一部分
                    
                    candidates.append(CandidateKeyword(
                        text=word,
                        keyword_type='location',
                        confidence=min(confidence, 0.95),
                        position=position,
                        context=self._get_context(content, position, word),
                        extraction_method=extraction_method
                    ))    
        # 去重和排序
        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates[:10]


    #驗證地名邊界是否合理
    def _validate_location_boundary(self, location_text: str, content: str, position: int) -> bool:
        # 獲取上下文
        context = self._get_context(content, position, location_text, context_window=20)       
        # 檢查是否是已知地名
        if location_text in (self.location_patterns['hk_districts'] + 
                            self.location_patterns['major_areas']+ 
                            self.location_patterns['mtr_stations']
                            ):
            return True
        
        # 檢查是否是樓盤名的一部分
        # 如果前面有已知地名，且當前詞後面跟著建築尾碼，可能是樓盤名
        before_context = content[max(0, position-10):position]
        after_context = content[position+len(location_text):position+len(location_text)+10]
        
        # 檢查前面是否有完整的地名
        for known_location in (self.location_patterns['hk_districts'] + 
                            self.location_patterns['major_areas']+ 
                            self.location_patterns['mtr_stations']
                            ):
            if known_location in before_context:
                # 檢查後面是否有建築尾碼
                for suffix in self.property_patterns['building_suffixes']:
                    if suffix in after_context:
                        return False  # 這可能是樓盤名，不是地名
        
        return len(location_text) >= 2

    #分析詞彙的地理位置上下文
    def _analyze_location_context(self, word: str, segmented_words: List[Tuple[str, str, int]], 
                                current_index: int, content: str) -> Dict:
        confidence = 0.0
        extraction_method = "location_matching"
        
        # 獲取當前詞的詞性
        current_word_info = segmented_words[current_index]
        pos = current_word_info[1] if len(current_word_info) > 1 else 'n'
        
        # 方法1：直接匹配已知地名
        if word in self.location_patterns['hk_districts']:
            confidence += 0.9
            extraction_method = "district_match"
        elif word in self.location_patterns['major_areas']:
            confidence += 0.85
            extraction_method = "area_match"
        
        # 方法2：街道模式匹配
        for indicator in self.location_patterns['street_indicators']:
            if word.endswith(indicator) and len(word) > len(indicator):
                # 檢查是否是港鐵站名的一部分
                if self._should_skip_station_part(word, content, current_index, segmented_words):
                    continue  # 跳過這個匹配
                
                confidence += 0.75
                extraction_method = f"street_match_{indicator}_geo_noun"
                break
        
        # 方法3：詞性分析 - 但要檢查上下文
        if pos in ['ns', 'nt'] and len(word) >= 2:
            # 檢查是否在"旗下"後面緊跟著
            if (current_index > 0 and 
                segmented_words[current_index-1][0] == "旗下"):
                confidence += 0.6
                extraction_method += "_after_qixia"
            else:
                confidence += 0.4
                extraction_method += "_geo_noun"
        
        # 方法4：網站模式匹配
        station_patterns = ['站', '港鐵站', '地鐵站', '巴士站', '火車站', '輕鐵站']
        for pattern in station_patterns:
            if word.endswith(pattern) and len(word) > len(pattern):
                station_name = word[:-len(pattern)]
                if (station_name in self.location_patterns['major_areas'] or
                    len(station_name) >= 2 and self._is_valid_station_name(station_name)):
                    confidence += 0.8
                    extraction_method = f"station_match_{pattern}"
                    break
        
        # 方法5：地標尾碼模式
        landmark_patterns = ['商場', '購物中心', '廣場', '大廈', '中心', '碼頭', '公園', '球場']
        for pattern in landmark_patterns:
            if word.endswith(pattern) and len(word) > len(pattern):
                landmark_name = word[:-len(pattern)]
                if len(landmark_name) >= 2:
                    confidence += 0.7
                    extraction_method = f"landmark_match_{pattern}"
                    break
        
        # 方法6：區域指示詞模式
        region_patterns = ['區', '灣', '島', '村', '鄉', '新城', '新市鎮']
        for pattern in region_patterns:
            if word.endswith(pattern) and len(word) > len(pattern):
                region_name = word[:-len(pattern)]
                if len(region_name) >= 2:
                    confidence += 0.75
                    extraction_method = f"region_match_{pattern}"
                    break
        
        # 方法7：上下文分析
        context = self._get_context(content, segmented_words[current_index][2], word)
        context_boost = self._calculate_context_boost(context, 'location')
        confidence += context_boost
        
        # 方法8：交通相關上下文增強
        transport_context_boost = self._calculate_transport_context_boost(context, word)
        confidence += transport_context_boost
        
        return {'confidence': confidence, 'method': extraction_method} if confidence > 0 else None

    #        檢查是否應該跳過某個路街匹配（避免提取港鐵站名的一部分）
    def _should_skip_station_part(self, word: str, content: str, current_index: int, 
                             segmented_words: List[Tuple[str, str, int]]) -> bool:
        # 檢查是否是常見站名的一部分
        station_exclusions = {
            "上路": ["錦上路站", "錦上路"],
            "大學": ["大學站"],
            "大圍": ["大圍站"],
            # 可以繼續添加其他容易被誤識別的站名部分
        }
        
        if word in station_exclusions:
            # 檢查前後文是否包含完整站名
            context_start = max(0, current_index - 2)
            context_end = min(len(segmented_words), current_index + 3)
            
            context_words = []
            for i in range(context_start, context_end):
                if i < len(segmented_words):
                    context_words.append(segmented_words[i][0])
            
            context_text = ''.join(context_words)
            
            for full_station in station_exclusions[word]:
                if full_station in context_text:
                    return True  # 跳過，這是站名的一部分
        
        return False
        
    #檢查地名是否與樓盤名沖
    def _check_location_property_conflict(self, word: str, segmented_words: List[Tuple[str, str, int]], 
                                        current_index: int) -> Dict:
        # 檢查前一個詞是否是已知地名
        if current_index > 0:
            prev_word = segmented_words[current_index-1][0]
            if prev_word in (self.location_patterns['hk_districts'] + 
                            self.location_patterns['major_areas']+ 
                            self.location_patterns['mtr_stations']
                            ):
                # 檢查當前詞+後續詞是否構成樓盤名
                potential_property = word
                next_index = current_index + 1             
                # 向後看幾個詞
                while (next_index < len(segmented_words) and 
                    next_index < current_index + 3):
                    next_word = segmented_words[next_index][0]
                    potential_property += next_word
                    
                    # 檢查是否包含建築尾碼
                    for suffix in self.property_patterns['building_suffixes']:
                        if potential_property.endswith(suffix):
                            return {'is_property': True, 'property_name': potential_property}
                    
                    next_index += 1
        
        return {'is_property': False, 'property_name': None}

    #識別專有名詞候選（主要是開發商等）
    def _identify_entity_candidates(self, segmented_words: List[Tuple[str, str, int]], 
                                content: str, bert_entities: List[Dict] = None) -> List[CandidateKeyword]:
        candidates = []
        bert_entities = bert_entities or []
        
        # 從BERT結果中提取組織機構
        for entity in bert_entities:
            if entity['mapped_type'] in ['organization', 'person']:
                # 進一步驗證是否為房地產相關實體
                if self._is_property_related_entity(entity['text'], content):
                    candidates.append(CandidateKeyword(
                        text=entity['text'],
                        keyword_type='developer',
                        confidence=entity['confidence'] * 0.9,  # 稍微降低置信度
                        position=entity['start'],
                        context=self._get_context(content, entity['start'], entity['text']),
                        extraction_method=f"bert_{entity['label']}_property_related"
                    ))
        # 規則匹配開發商
        for i, (word, pos, position) in enumerate(segmented_words):
            # 檢查是否已被BERT識別
            if any(entity['text'] == word for entity in bert_entities):
                continue
            
            confidence = 0.0
            extraction_method = "developer_matching"
            
            # 方法1：已知開發商直接匹配
            if word in self.developer_patterns['known_developers']:
                confidence += 0.9
                extraction_method = "known_developer"        
            # 方法2：公司尾碼匹配
            for suffix in self.developer_patterns['company_suffixes']:
                if word.endswith(suffix) and len(word) > len(suffix):
                    confidence += 0.7
                    extraction_method = f"company_suffix_{suffix}"
                    break
            # 方法3：詞性分析（機構名）
            if pos in ['nr', 'nt'] and len(word) >= 2:
                confidence += 0.4
                extraction_method += "_proper_noun"
            # 方法4：上下文分析
            context = self._get_context(content, position, word)
            if self._is_property_related_entity(word, context):
                confidence += 0.3
                extraction_method += "_property_context"
            
            if confidence >= 0.5:
                candidates.append(CandidateKeyword(
                    text=word,
                    keyword_type='developer',
                    confidence=min(confidence, 0.95),
                    position=position,
                    context=context,
                    extraction_method=extraction_method
                ))
        
        # 去重和排序
        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return candidates[:5]  # 返回前5個開發商候選

    #根據交通相關上下文計算額外加成
    def _calculate_transport_context_boost(self, context: str, word: str) -> float:
        boost = 0.0
        
        # 交通相關關鍵字
        transport_indicators = [
            '港鐵', '地鐵', '巴士', '小巴', '的士', '步行', '車程', '分鐘',
            '交通', '便利', '直達', '轉車', '轉乘', '月台', '出口',
            '鄰近', '附近', '距離', '前往'
        ]
        
        # 時間和距離指示詞
        time_distance_patterns = [
            r'\d+分鐘', r'\d+米', r'\d+公里', r'\d+站',
            r'步行\d+', r'車程\d+', r'約\d+'
        ]
        
        for indicator in transport_indicators:
            if indicator in context:
                boost += 0.1
        
        for pattern in time_distance_patterns:
            if re.search(pattern, context):
                boost += 0.15
                break
        
        # 如果詞彙本身包含"站"字，且上下文提到交通，給予更高加成
        if '站' in word and any(t_word in context for t_word in ['港鐵', '地鐵', '巴士', '交通']):
            boost += 0.2
        
        return min(boost, 0.4)  # 最大加成0.4

    #獲取詞彙的上下文
    def _get_context(self, content: str, position: int, word: str, 
                    context_window: int = 30) -> str:
        start = max(0, position - context_window)
        end = min(len(content), position + len(word) + context_window)
        return content[start:end]
    
    #根據上下文計算信心度加成
    def _calculate_context_boost(self, context: str, keyword_type: str) -> float:
        boost = 0.0
        
        if keyword_type == 'property':
            # 房地產相關上下文
            property_indicators = ['新盤', '樓盤', '項目', '發展商', '開售', '推出', '單位', '實用面積']
            for indicator in property_indicators:
                if indicator in context:
                    boost += 0.1
        
        elif keyword_type == 'location':
            # 地理位置相關上下文
            location_indicators = ['位於', '坐落', '鄰近', '交通', '港鐵', '巴士', '步行']
            for indicator in location_indicators:
                if indicator in context:
                    boost += 0.1   
        return min(boost, 0.3)  # 最大加成0.3

    #檢測組合房地產名稱（如：XX花園第X期）
    def _check_combined_property(self, segmented_words: List[Tuple[str, str, int]], 
                               current_index: int, content: str) -> CandidateKeyword:
        if current_index >= len(segmented_words) - 2:
            return None
        
        current_word = segmented_words[current_index][0]
        next_word = segmented_words[current_index + 1][0] if current_index + 1 < len(segmented_words) else ""
        next_next_word = segmented_words[current_index + 2][0] if current_index + 2 < len(segmented_words) else ""
        
        # 檢測"XX花園第X期"模式
        if (any(current_word.endswith(suffix) for suffix in self.property_patterns['building_suffixes']) and
            len(current_word) >= 3 and  # 確保不是單純的"樓"
            next_word in ['第'] and
            re.match(r'[一二三四五六七八九十\d]+期', next_next_word)):
            
            combined = f"{current_word}{next_word}{next_next_word}"
            position = segmented_words[current_index][2]
            context = self._get_context(content, position, combined)
            
            return CandidateKeyword(
                text=combined,
                keyword_type='property',
                confidence=0.9,
                position=position,
                context=context,
                extraction_method="combined_property_phase"
            )
        
        # 檢測英文樓盤名 + 期數組合（如："NOVO LAND 第3期"）
        if (self._is_english_property_name_enhanced(current_word) and
            next_word in ['第'] and
            re.match(r'[一二三四五六七八九十\d]+期', next_next_word)):
            
            combined = f"{current_word} {next_word}{next_next_word}"
            position = segmented_words[current_index][2]
            context = self._get_context(content, position, combined)
            
            return CandidateKeyword(
                text=combined,
                keyword_type='property',
                confidence=0.95,
                position=position,
                context=context,
                extraction_method="english_property_with_phase"
            )
        # 檢測"XX座"模式
        if (re.match(r'[一二三四五六七八九十ABCD\d]+', current_word) and
            next_word in ['座', '棟', '期']):
            
            combined = f"{current_word}{next_word}"
            position = segmented_words[current_index][2]
            context = self._get_context(content, position, combined)
            
            return CandidateKeyword(
                text=combined,
                keyword_type='property',
                confidence=0.8,
                position=position,
                context=context,
                extraction_method="combined_block_number"
            )
        
        return None
    
    #檢測組合位址（如：XX街XX號）
    def _check_combined_location(self, segmented_words: List[Tuple[str, str, int]], 
                               current_index: int, content: str) -> CandidateKeyword:
        if current_index >= len(segmented_words) - 1:
            return None
        
        current_word = segmented_words[current_index][0]
        next_word = segmented_words[current_index + 1][0] if current_index + 1 < len(segmented_words) else ""
        # 檢測"XX街XX號"模式
        if (any(current_word.endswith(indicator) for indicator in self.location_patterns['street_indicators']) and
            re.match(r'[一二三四五六七八九十\d]+.*號', next_word)):
            
            combined = f"{current_word}{next_word}"
            position = segmented_words[current_index][2]
            context = self._get_context(content, position, combined)
            
            return CandidateKeyword(
                text=combined,
                keyword_type='location',
                confidence=0.85,
                position=position,
                context=context,
                extraction_method="combined_street_number"
            )
        
        return None

    #檢查英文樓盤名是否與期數組合
    def _check_english_property_with_phase(self, segmented_words: List[Tuple[str, str, int]], 
                                        current_index: int, content: str, 
                                        english_name: str) -> CandidateKeyword:
        if current_index >= len(segmented_words) - 1:
            return None
        
        next_word = segmented_words[current_index + 1][0] if current_index + 1 < len(segmented_words) else ""  
        # 檢查下一個詞是否是期數標識
        if self._is_phase_indicator(next_word):
            combined = f"{english_name}{next_word}"
            position = segmented_words[current_index][2]
            context = self._get_context(content, position, combined)
            
            return CandidateKeyword(
                text=combined,
                keyword_type='property',
                confidence=0.9,  # 英文名+期數組合，高置信度
                position=position,
                context=context,
                extraction_method="english_property_with_phase"
            )
        
        return None

    #去除重複的候選關鍵字
    def _deduplicate_candidates(self, candidates: List[CandidateKeyword]) -> List[CandidateKeyword]:
        seen = set()
        unique_candidates = []
        
        for candidate in candidates:
            if candidate.text not in seen:
                seen.add(candidate.text)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    #生成移除關鍵字後的剩餘內容
    def _generate_remaining_content(self, original_content: str, 
                                property_candidates: List[CandidateKeyword],
                                location_candidates: List[CandidateKeyword],
                                entity_candidates: List[CandidateKeyword] = None) -> str:
        remaining = original_content
        
        # 移除所有識別出的關鍵字
        all_keywords = [c.text for c in property_candidates + location_candidates]
        if entity_candidates:
            all_keywords.extend([c.text for c in entity_candidates])
        
        for keyword in all_keywords:
            remaining = remaining.replace(keyword, '[KEYWORD_REMOVED]')
        
        # 清理連續的標記
        remaining = re.sub(r'\[KEYWORD_REMOVED\]\s*', ' ', remaining)
        remaining = re.sub(r'\s+', ' ', remaining).strip()
        
        return remaining

    #生成提取摘要
    def _generate_summary(self, property_candidates: List[CandidateKeyword],
                        location_candidates: List[CandidateKeyword],
                        entity_candidates: List[CandidateKeyword],
                        paragraphs: List[str]) -> Dict:

        all_candidates = property_candidates + location_candidates + entity_candidates
        
        return {
            'total_property_candidates': len(property_candidates),
            'total_location_candidates': len(location_candidates),
            'total_entity_candidates': len(entity_candidates),
            'avg_property_confidence': sum(c.confidence for c in property_candidates) / len(property_candidates) if property_candidates else 0,
            'avg_location_confidence': sum(c.confidence for c in location_candidates) / len(location_candidates) if location_candidates else 0,
            'avg_entity_confidence': sum(c.confidence for c in entity_candidates) / len(entity_candidates) if entity_candidates else 0,
            'paragraphs_analyzed': len(paragraphs),
            'extraction_methods_used': list(set([c.extraction_method for c in all_candidates])),
            'high_confidence_count': len([c for c in all_candidates if c.confidence >= 0.8]),
            'bert_enabled': self.use_bert
        }

    #識別結構化模式，結合標題資訊
    def _extract_structured_patterns(self, content: str, segmented_words: List[Tuple[str, str, int]], 
                                    title_candidates: List[Dict] = None) -> Dict[str, List[CandidateKeyword]]:
        patterns_found = {
            'developers': [],
            'properties': [],
            'locations': []
        }
        
        title_candidates = title_candidates or []
        # 查找"旗下"模式
        qixia_positions = []
        for i, (word, pos, position) in enumerate(segmented_words):
            if word == "旗下":
                qixia_positions.append(i)
        
        # 分析每個"旗下"周圍的上下文
        for qixia_index in qixia_positions:
            pattern_result = self._analyze_qixia_pattern_with_title(
                segmented_words, qixia_index, content, title_candidates
            )
            
            # 合併結果
            for key in patterns_found:
                patterns_found[key].extend(pattern_result.get(key, []))
        
        return patterns_found

    #在'旗下'之前提取開發商#
    def _extract_developer_before_qixia(self, segmented_words: List[Tuple[str, str, int]], qixia_index: int, content: str) -> CandidateKeyword:
        # 向前查找，跳過日期
        i = qixia_index - 1
        potential_developer_parts = []
        
        while i >= 0:
            word, pos, position = segmented_words[i]
            
            # 跳過日期
            if self._is_date_word(word):
                i -= 1
                continue
            
            # 如果遇到明顯的語義邊界，停止
            if self._is_strong_boundary(word):
                break
            
            # 如果包含"地產"、"發展"、"集團"等公司尾碼
            if any(suffix in word for suffix in self.developer_patterns['company_suffixes']):
                potential_developer_parts.insert(0, (word, position))
                
                # 繼續向前查找公司名首碼
                j = i - 1
                while j >= 0:
                    prev_word, prev_pos, prev_position = segmented_words[j]
                    
                    # 如果是日期或強邊界，停止
                    if self._is_date_word(prev_word) or self._is_strong_boundary(prev_word):
                        break
                    
                    # 如果看起來是公司名的一部分
                    if self._is_likely_company_prefix(prev_word, word):
                        potential_developer_parts.insert(0, (prev_word, prev_position))
                        j -= 1
                    else:
                        break
                break
            
            i -= 1
        
        # 構建開發商名稱
        if potential_developer_parts:
            developer_text = ''.join([part[0] for part in potential_developer_parts])
            first_position = potential_developer_parts[0][1]
            
            return CandidateKeyword(
                text=developer_text,
                keyword_type='developer',
                confidence=0.85,
                position=first_position,
                context=self._get_context(content, first_position, developer_text),
                extraction_method="qixia_pattern_developer"
            )
        
        return None

    #在'旗下'之後提取地點和樓盤
    def _extract_location_property_after_qixia(self, segmented_words: List[Tuple[str, str, int]], qixia_index: int, content: str) -> Tuple[List[CandidateKeyword], List[CandidateKeyword]]:
        locations = []
        properties = []
        
        i = qixia_index + 1
        found_location = False
        
        while i < len(segmented_words):
            word, pos, position = segmented_words[i]
            
            # 如果遇到強邊界，停止
            if self._is_strong_boundary(word):
                break
            
            # 檢查是否是已知地點
            if not found_location and self._is_known_location(word):
                locations.append(CandidateKeyword(
                    text=word,
                    keyword_type='location',
                    confidence=0.9,
                    position=position,
                    context=self._get_context(content, position, word),
                    extraction_method="qixia_pattern_location"
                ))
                found_location = True
            
            # 在找到地點後，查找樓盤名稱
            elif found_location:
                # 檢查是否是樓盤名稱
                property_candidate = self._extract_property_name_after_location(segmented_words, i, content)
                if property_candidate:
                    properties.append(property_candidate)
                    break  # 找到樓盤名稱後停止
            
            i += 1
        
        return locations, properties

    #在'旗下'之後提取地點和樓盤，結合標題資訊
    def _extract_location_property_after_qixia_with_title(self, segmented_words: List[Tuple[str, str, int]], 
                                                        qixia_index: int, content: str,
                                                        title_candidates: List[Dict]) -> Tuple[List[CandidateKeyword], List[CandidateKeyword]]:
        locations = []
        properties = []
        
        i = qixia_index + 1
        found_location = False
        
        while i < len(segmented_words):
            word, pos, position = segmented_words[i]
            
            if self._is_strong_boundary(word):
                break
            
            # 檢查是否是已知地點
            if not found_location and self._is_known_location(word):
                locations.append(CandidateKeyword(
                    text=word,
                    keyword_type='location',
                    confidence=0.9,
                    position=position,
                    context=self._get_context(content, position, word),
                    extraction_method="qixia_pattern_location"
                ))
                found_location = True
            
            # 在找到地點後，查找樓盤名稱
            elif found_location:
                # 檢查是否在標題中出現過
                title_match = self._find_title_match(word, title_candidates)
                
                if title_match or self._is_potential_property_word(word):
                    # 嘗試提取完整的樓盤名稱
                    property_candidate = self._extract_property_name_after_location_with_title(
                        segmented_words, i, content, title_candidates
                    )
                    
                    if property_candidate:
                        # 如果在標題中找到匹配，提高置信度
                        if title_match:
                            property_candidate.confidence = min(
                                property_candidate.confidence * self.title_qixia_boost_factor, 
                                0.95
                            )
                            property_candidate.extraction_method += "_title_confirmed"
                        
                        properties.append(property_candidate)
                        break
            
            i += 1
        
        return locations, properties

    #在地點後提取樓盤名稱
    def _extract_property_name_after_location(self, segmented_words: List[Tuple[str, str, int]], start_index: int, content: str) -> CandidateKeyword:

        property_parts = []
        i = start_index
        
        while i < len(segmented_words) and len(property_parts) < 4:  # 限制樓盤名長度
            word, pos, position = segmented_words[i]
            
            # 如果遇到強邊界，停止
            if self._is_strong_boundary(word):
                break
            
            # 如果是英文名稱或包含建築尾碼或期數標識
            if (self._is_english_property_name(word) or 
                any(suffix in word for suffix in self.property_patterns['building_suffixes']) or
                self._is_phase_indicator(word)):
                
                property_parts.append((word, position))
                
                # 檢查下一個詞是否也是樓盤名的一部分
                if i + 1 < len(segmented_words):
                    next_word = segmented_words[i + 1][0]
                    if self._is_phase_indicator(next_word):  # 如果下一個是期數
                        property_parts.append((next_word, segmented_words[i + 1][2]))
                        i += 1
                break
            
            i += 1
        
        if property_parts:
            property_text = ''.join([part[0] for part in property_parts])
            first_position = property_parts[0][1]
            
            return CandidateKeyword(
                text=property_text,
                keyword_type='property',
                confidence=0.8,
                position=first_position,
                context=self._get_context(content, first_position, property_text),
                extraction_method="qixia_pattern_property"
            )
        
        return None



    def _extract_property_name_after_location_with_title(self, segmented_words: List[Tuple[str, str, int]], 
                                                    start_index: int, content: str,
                                                    title_candidates: List[Dict]) -> CandidateKeyword:
        """在地点后提取楼盘名称，考虑标题匹配"""
        property_parts = []
        i = start_index
        title_matched = False
        
        while i < len(segmented_words) and len(property_parts) < 4:
            word, pos, position = segmented_words[i]
            
            if self._is_strong_boundary(word):
                break
            
            # 检查是否与标题匹配
            if self._find_title_match(word, title_candidates):
                title_matched = True
                property_parts.append((word, position))
                
                # 如果标题匹配，继续查找相邻的词汇组成完整名称
                next_i = i + 1
                while next_i < len(segmented_words) and next_i < i + 3:
                    next_word = segmented_words[next_i][0]
                    if (self._is_potential_property_word(next_word) or 
                        self._is_phase_indicator(next_word)):
                        property_parts.append((next_word, segmented_words[next_i][2]))
                        next_i += 1
                    else:
                        break
                break
            
            # 原有的楼盘识别逻辑
            elif (self._is_english_property_name(word) or '_' in word or
                any(suffix in word for suffix in self.property_patterns['building_suffixes']) or
                self._is_phase_indicator(word)):
                
                property_parts.append((word, position))
                
                if i + 1 < len(segmented_words):
                    next_word = segmented_words[i + 1][0]
                    if self._is_phase_indicator(next_word):
                        property_parts.append((next_word, segmented_words[i + 1][2]))
                        i += 1
                break
            
            i += 1
        
        if property_parts:
            # 处理英文名称的下划线
            processed_parts = []
            for part, pos in property_parts:
                if '_' in part:
                    processed_parts.append((part.replace('_', ' '), pos))
                else:
                    processed_parts.append((part, pos))
            
            property_text = ''.join([part[0] for part in processed_parts])
            first_position = processed_parts[0][1]
            
            # 基础置信度
            base_confidence = 0.8
            
            # 如果与标题匹配，提高置信度
            if title_matched:
                base_confidence = min(base_confidence * self.title_qixia_boost_factor, 0.95)
            
            return CandidateKeyword(
                text=property_text,
                keyword_type='property',
                confidence=base_confidence,
                position=first_position,
                context=self._get_context(content, first_position, property_text),
                extraction_method="qixia_pattern_property_title_enhanced"
            )
        
        return None

    def _expand_organization_entity(self, words, entity_index):
        current_word = words[entity_index][0]
        expanded_words = [current_word]
        start_index = entity_index
        end_index = entity_index
        
        i = entity_index - 1
        while i >= 0:
            prev_word = words[i][0]
            
            if self._is_semantic_boundary(prev_word):
                break
                
            if self._is_company_name_part(prev_word, current_word):
                expanded_words.insert(0, prev_word)
                start_index = i
                i -= 1
            else:
                break
        
        i = entity_index + 1
        while i < len(words):
            next_word = words[i][0]
            
            if self._is_semantic_boundary(next_word):
                break
                
            if next_word in self.developer_patterns['company_suffixes']:
                expanded_words.append(next_word)
                end_index = i
                break 
            else:
                break
        
        expanded_text = ''.join(expanded_words)
        start_pos = sum(len(words[j][0]) for j in range(start_index))
        end_pos = start_pos + len(expanded_text)
        
        return expanded_text, start_pos, end_pos


     # 提取候選關鍵字。
     # 0.  先以本地專有名詞庫（self.preloaded_keywords）掃描全文
     #     → 命中項用 `extraction_method = "preloaded_dict"`，置信度固定 0.95
     #     → 同時把命中的詞以占位符 [KNOWN] 移除，避免重複識別
     # 1.  對「已移除已知詞的內容」進行後續分詞 / 規則 / BERT 處理
     # 2.  合併本地命中 + 規則 / BERT / 結構化結果
     # 3.  依置信度閾值過濾、生成剩餘內容與摘要
    def extract_candidates(
            self,
            news_content: str,
            title: str = "",
            focus_first_paragraphs: int = 2
        ) -> NewsExtractionResult:
            # -1- 固定詞（行政區／車站／俗語）先掃描
        fixed_hits = []
        for term in self.protected_keywords:
            pos = news_content.find(term)
            if pos != -1:
                cat = 'location' if term in self.protected_locations else 'slang'
                fixed_hits.append(
                    CandidateKeyword(
                        text=term,
                        keyword_type=cat,
                        confidence=0.99,
                        position=pos,
                        context=self._get_context(news_content, pos, term),
                        extraction_method="fixed_dict"
                    )
                )
        # 佔位
        cleaned_content = news_content
        for hit in fixed_hits:
            cleaned_content = re.sub(re.escape(hit.text), "[¤¤¤F¤¤¤]", cleaned_content)
            # 第0步：先匹配本地詞庫的已知關鍵字
            preload_hits: List[CandidateKeyword] = self._extract_preloaded_keywords(news_content)

            # 用占位符替換，避免之後重複被分詞或再次命中
            cleaned_content = news_content
            for hit in preload_hits:
                # 保險起見使用 re.escape，避免特殊字元
                cleaned_content = re.sub(re.escape(hit.text), "[¤¤¤K¤¤¤]", cleaned_content)
            # 第1步：段落分析與焦點提取（對清理後內容進行）
            paragraphs     = self._split_into_paragraphs(cleaned_content)
            focus_content  = self._get_focus_content(paragraphs, focus_first_paragraphs)

            # 第2步：智慧分詞 + 詞性標註
            segmented_words = self._smart_segmentation(focus_content)

            # 第3步：標題分析
            title_candidates = self._extract_title_candidates(title) if title else []

            # 第4步：BERT 專有名詞抽取
            bert_entities = self._extract_entities_with_bert(focus_content) if self.use_bert else []

            # 第5步：結構化模式（含「旗下」等）偵測
            structured_patterns = self._extract_structured_patterns(
                focus_content, segmented_words, title_candidates
            )

            # 第6步：候選詞整合, 先把 preload_hits 加入，再疊加其他識別結果
            property_candidates = preload_hits[:]  # 深拷貝，避免影響原列表
            property_candidates.extend(
                self._identify_property_candidates(segmented_words, focus_content, bert_entities)
            )
            property_candidates = [
                c for c in property_candidates
                if not self._is_noise_property_term(c.text)
            ]

            location_candidates = self._identify_location_candidates(
                segmented_words, focus_content, bert_entities
            )
            entity_candidates   = self._identify_entity_candidates(
                segmented_words, focus_content, bert_entities
            )

            # 第7步：合併結構化模式輸出
            property_candidates.extend(structured_patterns["properties"])
            location_candidates.extend(structured_patterns["locations"])
            entity_candidates.extend(structured_patterns["developers"])

            # 去重（同字串僅保留首個置信度最高者）
            property_candidates = self._deduplicate_candidates(property_candidates)
            location_candidates = self._deduplicate_candidates(location_candidates)
            entity_candidates   = self._deduplicate_candidates(entity_candidates)

            # 第8步：依類型置信度閾值過濾
            property_candidates = [
                c for c in property_candidates
                if c.confidence >= self.type_thresholds["property"]
            ]

            property_candidates = [
                c for c in property_candidates
                if not self._is_noise_property_term(c.text)
            ]

            location_candidates = [
                c for c in location_candidates
                if c.confidence >= self.type_thresholds["location"]
            ]
            entity_candidates = [
                c for c in entity_candidates
                if c.confidence >= self.type_thresholds["developer"]
            ]

            # 第9步：生成剩餘內容（此處使用「原始全文」而非 cleaned_content）
            remaining_content = self._generate_remaining_content(
                news_content, property_candidates, location_candidates, entity_candidates
            )

            # 第10步：生成提取摘要
            extraction_summary = self._generate_summary(
                property_candidates, location_candidates, entity_candidates, paragraphs
            )

            # 回傳結果物件
            return NewsExtractionResult(
                property_candidates=property_candidates,
                location_candidates=location_candidates,
                entity_candidates=entity_candidates,
                remaining_content=remaining_content,
                extraction_summary=extraction_summary,
            )


    def _extract_location_property_after_qixia(self, segmented_words: List[Tuple[str, str, int]], qixia_index: int, content: str) -> Tuple[List[CandidateKeyword], List[CandidateKeyword]]:
        locations = []
        properties = []
        
        i = qixia_index + 1
        found_location = False
        
        while i < len(segmented_words):
            word, pos, position = segmented_words[i]
            
            # 如果遇到强边界，停止
            if self._is_strong_boundary(word):
                break
            
            if not found_location and self._is_known_location(word):
                locations.append(CandidateKeyword(
                    text=word,
                    keyword_type='location',
                    confidence=0.9,
                    position=position,
                    context=self._get_context(content, position, word),
                    extraction_method="qixia_pattern_location"
                ))
                found_location = True
            
            # 在找到地点后，查找楼盘名称
            elif found_location:
                property_candidate = self._extract_property_name_after_location(segmented_words, i, content)
                if property_candidate:
                    properties.append(property_candidate)
                    break  
            
            i += 1
        
        return locations, properties

#在地點後提取樓盤名稱
    def _extract_property_name_after_location(self, segmented_words: List[Tuple[str, str, int]], start_index: int, content: str) -> CandidateKeyword:

        property_parts = []
        i = start_index
        
        while i < len(segmented_words) and len(property_parts) < 4:  # 限制樓盤名長度
            word, pos, position = segmented_words[i]
            
            # 如果遇到強邊界，停止
            if self._is_strong_boundary(word):
                break
            
            # 如果是英文名稱或包含建築尾碼或期數標識
            if (self._is_english_property_name(word) or 
                any(suffix in word for suffix in self.property_patterns['building_suffixes']) or
                self._is_phase_indicator(word)):
                
                property_parts.append((word, position))
                
                # 檢查下一個詞是否也是樓盤名的一部分
                if i + 1 < len(segmented_words):
                    next_word = segmented_words[i + 1][0]
                    if self._is_phase_indicator(next_word):  # 如果下一個是期數
                        property_parts.append((next_word, segmented_words[i + 1][2]))
                        i += 1
                break
            
            i += 1
        
        if property_parts:
            property_text = ''.join([part[0] for part in property_parts])
            first_position = property_parts[0][1]
            
            return CandidateKeyword(
                text=property_text,
                keyword_type='property',
                confidence=0.8,
                position=first_position,
                context=self._get_context(content, first_position, property_text),
                extraction_method="qixia_pattern_property"
            )
        
        return None

    #基於語義邊界和已知詞彙擴展BERT識別的實體
    def _expand_entity_boundaries(self, entity: Dict, content: str) -> Dict:
   
        original_text = entity['word']
        start_pos = entity['start']
        end_pos = entity['end']
        
        try:
            # 獲取更大範圍的上下文進行分析
            context_start = max(0, start_pos - 20)
            context_end = min(len(content), end_pos + 20)
            context = content[context_start:context_end]
            
            # 對上下文進行分詞
            import jieba.posseg as pseg
            seg_list = list(pseg.cut(context))
            
            # 轉換為統一格式
            words_in_context = []
            for word_pair in seg_list:
                if hasattr(word_pair, 'word') and hasattr(word_pair, 'flag'):
                    words_in_context.append((word_pair.word, word_pair.flag))
                else:
                    try:
                        word, pos = word_pair
                        words_in_context.append((word, pos))
                    except:
                        words_in_context.append((str(word_pair), 'n'))
            
            # 找到當前實體在分詞結果中的位置
            current_pos = start_pos - context_start
            entity_word_index = self._find_entity_in_segmented_words(words_in_context, original_text, current_pos)
            
            if entity_word_index == -1:
                return self._create_entity_dict(entity, original_text, start_pos, end_pos)
            
            # 根據實體類型進行智慧擴展
            expanded_text, new_start, new_end = self._smart_expand_entity(
                words_in_context, entity_word_index, entity, context_start
            )
            
            return self._create_entity_dict(
                entity, expanded_text, 
                start_pos + new_start - current_pos, 
                start_pos + new_end - current_pos
            )
            
        except Exception as e:
            logging.warning(f"邊界擴展失敗: {e}, 使用原始實體")
            return self._create_entity_dict(entity, original_text, start_pos, end_pos)


    #擴展地理位置實體
    def _expand_location_entity(self, words, entity_index):
        current_word = words[entity_index][0]
        
        # 檢查當前詞是否是已知地名的一部分
        # 如果當前識別的是"日峰"，檢查前面是否有"朗"
        expanded_words = [current_word]
        start_index = entity_index
        
        # 向前檢查，但要避免與已知地名衝突
        i = entity_index - 1
        while i >= 0:
            prev_word = words[i][0]
            
            # 如果前一個詞是已知地名，不要擴展
            if (prev_word in self.location_patterns['hk_districts'] or 
                prev_word in self.location_patterns['major_areas'] or 
                prev_word in self.location_patterns['mtr_stations']):
                break
                
            # 如果遇到語義邊界，停止
            if self._is_semantic_boundary(prev_word):
                break
                
            # 如果看起來是地名的一部分
            if self._is_location_name_part(prev_word, current_word):
                expanded_words.insert(0, prev_word)
                start_index = i
                i -= 1
            else:
                break
        
        expanded_text = ''.join(expanded_words)
        start_pos = sum(len(words[j][0]) for j in range(start_index))
        end_pos = start_pos + len(expanded_text)
        
        return expanded_text, start_pos, end_pos



    #從標題中提取潛在的樓盤名候選
    def _extract_title_candidates(self, title: str) -> List[Dict]:
        if not title:
            return []
        
        candidates = []
        title_words = self._smart_segmentation(title)
        
        for i, (word, pos, position) in enumerate(title_words):
            candidate_info = {
                'text': word,
                'position_in_title': i,
                'is_potential_property': False,
                'confidence_boost': 0.0
            }
            
            # 判斷是否可能是樓盤名
            if self._is_potential_property_name_in_title(word, title_words, i):
                candidate_info['is_potential_property'] = True
                candidate_info['confidence_boost'] = self.title_boost_factor
            
            candidates.append(candidate_info)
        
        return candidates

    #判斷標題中的詞是否可能是樓盤名
    def _is_potential_property_name_in_title(self, word: str, title_words: List[Tuple[str, str, int]], current_index: int) -> bool:
        # 方法1：包含建築尾碼
        for suffix in self.property_patterns['building_suffixes']:
            if word.endswith(suffix) and len(word) > len(suffix):
                return True
        
        # 方法2：英文樓盤名
        if self._is_english_property_name(word) or '_' in word:
            return True
        
        # 方法3：專有名詞且不是常見詞彙mtr_stations
        if (len(word) >= 3 and 
            word not in self.location_patterns['hk_districts'] and
            word not in self.location_patterns['major_areas'] and
            word not in self.location_patterns['mtr_stations'] and
            not self._is_common_news_word(word)):
            return True
        
        return False

    #判斷是否是常見的新聞詞彙
    def _is_common_news_word(self, word: str) -> bool:
        common_words = [
            '上載', '樓書', '設', '夥', '主打', '房戶', '開售', '推出', '銷售',
            '準備', '籌備', '密鑼緊鼓', '萬', '億', '元', '價格', '呎價',
            '實用面積', '建築面積', '單位', '戶型', '新盤', '項目'
        ]
        return word in common_words

    #擴展設施實體（樓盤名稱）
    def _expand_facility_entity(self, words, entity_index):
        current_word = words[entity_index][0]
        expanded_words = [current_word]
        start_index = entity_index
        end_index = entity_index
        
        # 向前擴展樓盤名稱
        i = entity_index - 1
        while i >= 0:
            prev_word = words[i][0]
            
            # 如果遇到已知地名，停止擴展mtr_stations
            if (prev_word in self.location_patterns['hk_districts'] or 
                prev_word in self.location_patterns['major_areas'] or
                prev_word in self.location_patterns['mtr_stations']):
                break
                
            # 如果遇到語義邊界，停止
            if self._is_semantic_boundary(prev_word):
                break
                
            # 如果看起來是樓盤名的一部分
            if self._is_property_name_part(prev_word, current_word):
                expanded_words.insert(0, prev_word)
                start_index = i
                i -= 1
            else:
                break
        
        expanded_text = ''.join(expanded_words)
        start_pos = sum(len(words[j][0]) for j in range(start_index))
        end_pos = start_pos + len(expanded_text)
        
        return expanded_text, start_pos, end_pos


    #使用BERT模型提取專有名詞
    def _extract_entities_with_bert(self, content: str) -> List[Dict]:
        try:
            # 使用BERT NER pipeline
            entities = self.ner_pipeline(content)
            
            processed_entities = []
            for entity in entities:
                try:
                    # 確保entity是字典格式
                    if not isinstance(entity, dict):
                        continue
                        
                    # 檢查必要的欄位
                    required_fields = ['word', 'entity_group', 'score', 'start', 'end']
                    if not all(field in entity for field in required_fields):
                        continue
                    
                    # 過濾和驗證實體
                    if self._validate_bert_entity(entity, content):
                        # 檢查是否需要擴展實體邊界
                        expanded_entity = self._expand_entity_boundaries(entity, content)
                        processed_entities.append(expanded_entity)
                        
                except Exception as e:
                    logging.warning(f"處理單個實體時出錯: {e}, 實體: {entity}")
                    continue
            
            return processed_entities
                
        except Exception as e:
            logging.error(f"BERT實體提取失敗: {e}")
            return []

    #在分詞結果中找到實體的位置
    def _find_entity_in_segmented_words(self, words, entity_text, approximate_pos):
        pos = 0
        for i, (word, _) in enumerate(words):
            if entity_text in word or word in entity_text:
                if abs(pos - approximate_pos) < 5:  # 允許小誤差
                    return i
            pos += len(word)
        return -1

    #檢查詞彙是否在標題候選中
    def _find_title_match(self, word: str, title_candidates: List[Dict]) -> bool:
        for candidate in title_candidates:
            if (word == candidate['text'] or 
                word in candidate['text'] or 
                candidate['text'] in word):
                return True
        return False

    #智慧擴展實體邊界
    def _smart_expand_entity(self, words_in_context, entity_index, entity, context_start):
        entity_type = entity['entity_group']
        current_word = words_in_context[entity_index][0]
        
        if entity_type == 'ORG':  # 組織機構（開發商）
            return self._expand_organization_entity(words_in_context, entity_index)
        elif entity_type in ['LOC', 'GPE']:  # 地理位置
            return self._expand_location_entity(words_in_context, entity_index)
        elif entity_type == 'FAC':  # 設施（樓盤）
            return self._expand_facility_entity(words_in_context, entity_index)
        else:
            # 默認不擴展
            word = words_in_context[entity_index][0]
            return word, 0, len(word)

    #判斷是否是日期詞
    def _is_date_word(self, word: str) -> bool:
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{1,2}月\d{1,2}日',
            r'周[一二三四五六日]',
            r'星期[一二三四五六日]'
        ]
        return any(re.match(pattern, word) for pattern in date_patterns)

    #判斷是否是強語義邊界
    def _is_strong_boundary(self, word: str) -> bool:
        boundaries = ['，', '。', '；', '：', '（', '）', 'Written', 'by', 'More', 'in', '傳', '收']
        return word in boundaries or len(word) > 10

    #判斷是否是已知地點
    def _is_known_location(self, word: str) -> bool:
        return (word in self.location_patterns['hk_districts'] or 
                word in self.location_patterns['major_areas'] or
                word in self.location_patterns['mtr_stations'])

    #判斷是否可能是公司名首碼
    def _is_likely_company_prefix(self, word: str, company_suffix_word: str) -> bool:
        # 如果公司尾碼包含"地產"，首碼通常是2-4個字的中文
        if '地產' in company_suffix_word:
            return 1 <= len(word) <= 4 and word.isalpha() and not word.isdigit()
        return False

    #判斷是否是英文樓盤名（基礎版本）
    def _is_english_property_name(self, word: str) -> bool:
        return bool(re.match(r'^[A-Z][A-Z\s]*$', word)) and len(word) >= 2

    #增強的英文樓盤名判斷
    def _is_english_property_name_enhanced(self, name: str) -> bool:
        # 移除常見的非樓盤英文詞
        excluded_words = [
            'AND', 'THE', 'OF', 'IN', 'ON', 'AT', 'BY', 'FOR', 'WITH', 'FROM',
            'WRITTEN', 'MORE', 'NEWS', 'PROPERTY', 'ESTATE'
        ]
        
        words = name.upper().split()
        if any(word in excluded_words for word in words):
            return False
        
        # 樓盤名特徵
        return (
            len(name) >= 3 and 
            len(words) <= 4 and  # 通常不超過4個單詞
            all(word.isalpha() or word.isalnum() for word in words) and
            any(len(word) >= 2 for word in words)  # 至少有一個單詞長度>=2
        )


    #判斷是否是期數指示詞
    def _is_phase_indicator(self, word: str) -> bool:
        phase_patterns = [
            r'第\d+期',           # 第3期
            r'\d+期',             # 3期
            r'第\d+[ABC]期',      # 第3A期
            r'\d+[ABC]期',        # 3A期
            r'第[一二三四五六七八九十]+期',  # 第三期
            r'[一二三四五六七八九十]+期'     # 三期
        ]
        return any(re.match(pattern, word) for pattern in phase_patterns)
    
    #判斷是否是語義邊界（如日期、標點符號等）
    def _is_semantic_boundary(self, word):
        # 日期模式
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # 2025-06-30
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{1,2}月\d{1,2}日',
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, word):
                return True
        
        # 標點符號和連接詞
        boundaries = ['，', '。', '、', '；', '：', '（', '）', '[', ']', '旗下', '位於', '坐落']
        return word in boundaries

    #判斷前一個詞是否是公司名的一部分
    def _is_company_name_part(self, prev_word, current_word):
        # 如果當前詞包含"地"，前一個詞可能是公司名首碼
        if '地' in current_word:
            return len(prev_word) <= 3 and prev_word.isalpha()
        return False

    #判斷是否是地名的一部分
    def _is_location_name_part(self, prev_word, current_word):
        # 避免與已知地名衝突的情況下，判斷是否應該合併
        combined = prev_word + current_word
        
        # 如果合併後看起來像樓盤名（有建築尾碼），則認為是樓盤而非地名
        for suffix in self.property_patterns['building_suffixes']:
            if combined.endswith(suffix):
                return True
        
        return False

    #判斷是否是樓盤名的一部分
    def _is_property_name_part(self, prev_word, current_word):
        # 如果當前詞是建築尾碼，前面的詞可能是樓盤名首碼
        for suffix in self.property_patterns['building_suffixes']:
            if current_word.endswith(suffix):
                return len(prev_word) <= 4 and not prev_word.isdigit()
        
        return False

    #判斷實體是否與房地產相關
    def _is_property_related_entity(self, entity_text: str, context: str) -> bool:
        property_indicators = [
            '發展商', '開發商', '地產商', '項目', '樓盤', '新盤', '推出',
            '開售', '發展', '建設', '投資', '置業', '物業'
        ]
        
        # 檢查上下文是否包含房地產相關詞彙
        context_related = any(indicator in context for indicator in property_indicators)
        
        # 檢查實體本身是否包含房地產相關詞彙
        entity_related = any(indicator in entity_text for indicator in ['地產', '發展', '置業', '建設'])
        
        return context_related or entity_related

    #長度優先去重：如某候選詞被另一候選詞完整包含，僅保留較長且高信心者。
    def _dedup_by_length(self, cands: List[CandidateKeyword]) -> List[CandidateKeyword]:
        cands = sorted(cands, key=lambda c: (-len(c.text), -c.confidence))
        kept: List[CandidateKeyword] = []
        for cand in cands:
            if any(cand.text in k.text for k in kept):
                continue
            kept.append(cand)
        return kept


    def _is_valid_station_name(self, station_name: str) -> bool:
        # 常見的香港地鐵站名模式
        valid_patterns = [
            # 包含方向的站名
            r'.*[東西南北中上下].*',
            # 包含地區特色詞彙
            r'.*[灣角塘湧埗坑頭尾山海港].*',
            # 常見站名組合
            r'.*(大學|學院|醫院|公園|廣場|中心|城|新城).*',
            # 雙字元以上的合理站名
            r'^[\u4e00-\u9fff]{2,}$'
        ]
        
        for pattern in valid_patterns:
            if re.match(pattern, station_name):
                return True
        
        # 檢查是否在已知地名列表中mtr_stations
        all_known_locations = (
            self.location_patterns['hk_districts'] + 
            self.location_patterns['major_areas'] + 
            self.location_patterns['mtr_stations']
        )
        
        return any(station_name in location for location in all_known_locations)

    def _is_potential_property_word(self, word: str) -> bool:
        return (len(word) >= 2 and 
                not self._is_common_news_word(word) and
                not word in self.location_patterns['hk_districts'] and
                not word in self.location_patterns['major_areas'] and
                not word in self.location_patterns['mtr_stations'])

    def _create_entity_dict(self, original_entity, text, start, end):
        return {
            'text': text,
            'label': original_entity['entity_group'],
            'confidence': original_entity['score'] if text == original_entity['word'] else original_entity['score'] * 0.9,
            'start': start,
            'end': end,
            'mapped_type': self.entity_type_mapping.get(original_entity['entity_group'], 'unknown'),
            'expansion_applied': text != original_entity['word']
        }

#處理提取出的候選關鍵字
# class CandidateProcessor:
    
#     def __init__(self, translation_system):
#         self.translation_system = translation_system
    
#     def process_candidates(self, extraction_result: NewsExtractionResult) -> Dict:

#         processed_properties = []
#         processed_locations = []
        
#         # 處理房地產候選詞
#         for candidate in extraction_result.property_candidates:
#             try:
#                 # 使用瀑布式翻譯系統
#                 translation_result = self.translation_system.translate(candidate.text)
                
#                 processed_properties.append({
#                     'original': candidate.text,
#                     'translation': translation_result.english_name,
#                     'extraction_confidence': candidate.confidence,
#                     'translation_confidence': translation_result.confidence,
#                     'extraction_method': candidate.extraction_method,
#                     'translation_method': translation_result.method,
#                     'translation_layer': translation_result.layer,
#                     'context': candidate.context,
#                     'overall_confidence': (candidate.confidence + translation_result.confidence) / 2
#                 })
                
#             except Exception as e:
#                 logging.error(f"處理房地產候選詞 '{candidate.text}' 失敗: {e}")
#                 processed_properties.append({
#                     'original': candidate.text,
#                     'translation': candidate.text,  # 保留原文
#                     'extraction_confidence': candidate.confidence,
#                     'translation_confidence': 0.3,
#                     'extraction_method': candidate.extraction_method,
#                     'translation_method': 'failed',
#                     'translation_layer': 0,
#                     'context': candidate.context,
#                     'overall_confidence': candidate.confidence * 0.5
#                 })
        
#         # 處理地理位置候選詞
#         for candidate in extraction_result.location_candidates:
#             try:
#                 translation_result = self.translation_system.translate(candidate.text)
                
#                 processed_locations.append({
#                     'original': candidate.text,
#                     'translation': translation_result.english_name,
#                     'extraction_confidence': candidate.confidence,
#                     'translation_confidence': translation_result.confidence,
#                     'extraction_method': candidate.extraction_method,
#                     'translation_method': translation_result.method,
#                     'translation_layer': translation_result.layer,
#                     'context': candidate.context,
#                     'overall_confidence': (candidate.confidence + translation_result.confidence) / 2
#                 })
                
#             except Exception as e:
#                 logging.error(f"處理地理位置候選詞 '{candidate.text}' 失敗: {e}")
#                 processed_locations.append({
#                     'original': candidate.text,
#                     'translation': candidate.text,
#                     'extraction_confidence': candidate.confidence,
#                     'translation_confidence': 0.3,
#                     'extraction_method': candidate.extraction_method,
#                     'translation_method': 'failed',
#                     'translation_layer': 0,
#                     'context': candidate.context,
#                     'overall_confidence': candidate.confidence * 0.5
#                 })
        
#         return {
#             'processed_properties': processed_properties,
#             'processed_locations': processed_locations,
#             'remaining_content': extraction_result.remaining_content,
#             'extraction_summary': extraction_result.extraction_summary,
#             'processing_stats': {
#                 'successful_property_translations': len([p for p in processed_properties if p['translation_method'] != 'failed']),
#                 'successful_location_translations': len([l for l in processed_locations if l['translation_method'] != 'failed']),
#                 'high_confidence_results': len([item for sublist in [processed_properties, processed_locations] for item in sublist if item['overall_confidence'] >= 0.7])
#             }
#         }

print("extract.py 已載入完成")

if __name__ == "__main__":
    print("Loading...")

