import sqlite3, jieba
from pathlib import Path, PurePath
from typing import List, Set

#   讀取 verified_translations 表，將 chinese_name 欄位返回為 set。
def load_predefined_keywords(db_path: str = "property_translations.db") -> Set[str]:
    p = Path(db_path).expanduser().resolve()      # ★ 新增
    print("[DEBUG] 將讀取的檔案:", p)              
    if not Path(db_path).exists():
        print(f"[warn] 未找到詞庫 {db_path}，跳過載入")
        return set()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT chinese_name FROM verified_translations")
        names = {row[0].strip() for row in cur.fetchall() if row[0]}
    print(f" 已從 {db_path} 載入 {len(names):,} 條專有名詞")
    return names

