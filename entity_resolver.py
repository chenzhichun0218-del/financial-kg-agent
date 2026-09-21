"""
================================================================================
 实体消歧与对齐模块
 四层递进：名称标准化 → 别名词典 → 模糊匹配 → LLM 兜底
================================================================================
"""
import re
import json
import os
from typing import Optional
from collections import defaultdict

import pandas as pd

from graph_core import (
    Entity, make_entity_id, normalize_name, map_entity_type,
    TRANSPARENT_TYPES, GraphStore, HoldingEdge
)

# ============================================================================
# 第一层：名称标准化
# ============================================================================

def clean_entity_name(name: str, holder_category: int = 0) -> str:
    """
    深度清洗实体名称，输出 canonical_name
    结合 normalize_name + 特定规则
    """
    name = normalize_name(name)
    if not name:
        return name

    # 去掉公司后缀的冗余空格
    name = re.sub(r"\s*\(\s*", "(", name)
    name = re.sub(r"\s*\)\s*", ")", name)

    # 去掉前导/尾随的特殊字符
    name = name.strip("-–—*")

    # 如果以 "-" 开头（如 "-分红-个人分红"），这是产品子账户，保留原名
    # 不特殊处理

    return name


def is_fund_product_name(name: str) -> bool:
    """
    判断名称是否为基金/资管产品名
    特征：含"组合"、"基金"、"理财产品"、"金"、数字编号
    """
    fund_patterns = [
        r"组合$", r"集合", r"资管计划", r"理财产品",
        r"私募", r"信托计划", r"专户",
        r"社保基金\d", r"基本养老保险", r"企业年金",
    ]
    for pat in fund_patterns:
        if re.search(pat, name):
            return True
    return False


def extract_fund_manager(name: str, aname: str) -> Optional[str]:
    """
    从基金产品名中提取管理人名称
    如 "乾德价值成长7号私募证券投资基金" → aname "上海乾德投资管理有限公司-乾德价值成长7号…"
    管理人通常是 aname 中 "-" 之前的部分
    """
    if pd.isna(aname) or not isinstance(aname, str):
        return None
    if "-" in aname:
        return aname.split("-")[0].strip()
    return None


# ============================================================================
# 第二层：别名词典
# ============================================================================

class AliasDict:
    """
    别名词典
    维护 canonical_name ↔ aliases 的双向映射
    支持手工添加 + 从数据自动发现
    """

    def __init__(self):
        # canonical_name → set of aliases
        self.canonical_to_aliases: dict[str, set[str]] = defaultdict(set)
        # alias → canonical_name (反向索引)
        self.alias_to_canonical: dict[str, str] = {}

    def add(self, canonical: str, alias: str):
        """添加一对映射"""
        canon_key = canonical.lower().strip()
        alias_key = alias.lower().strip()
        if canon_key == alias_key:
            return
        self.canonical_to_aliases[canon_key].add(alias)
        self.alias_to_canonical[alias_key] = canonical

    def add_from_data(self, name: str, aname: str):
        """
        从数据的 s_holder_name 和 s_holder_aname 自动发现别名关系
        规则：
        - aname 通常更完整（含管理人前缀），aname 是 canonical
        - 如果 aname 包含 name，则 aname → canonical, name → alias
        """
        if pd.isna(name) or pd.isna(aname):
            return
        name = str(name).strip()
        aname = str(aname).strip()
        if not name or not aname or name == aname:
            return

        # aname 通常包含 name 或者更完整
        canonical = aname
        alias = name
        self.add(canonical, alias)

    def resolve(self, name: str) -> Optional[str]:
        """给定一个名称，返回它的 canonical_name"""
        key = name.lower().strip()
        if key in self.alias_to_canonical:
            return self.alias_to_canonical[key]
        # 检查是否自身就是 canonical
        if key in self.canonical_to_aliases:
            return name
        return None

    def get_aliases(self, canonical: str) -> set[str]:
        """获取某个实体所有的别名"""
        return self.canonical_to_aliases.get(canonical.lower().strip(), set())

    def size(self) -> int:
        return len(self.canonical_to_aliases)

    def save(self, path: str):
        data = {k: list(v) for k, v in self.canonical_to_aliases.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "AliasDict":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls()
        for canonical, aliases in data.items():
            for alias in aliases:
                obj.add(canonical, alias)
        return obj


# ============================================================================
# 第三层：模糊匹配
# ============================================================================

class FuzzyMatcher:
    """
    基于 rapidfuzz 的模糊实体匹配
    如果 rapidfuzz 不可用，回退到 difflib
    """

    def __init__(self, candidates: list[str], threshold: float = 0.85):
        """
        candidates: 所有候选实体名称列表
        threshold: 相似度阈值 (0-1)
        """
        self.candidates = candidates
        self.threshold = threshold
        self._index: dict[str, list[str]] = self._build_index()
        self._use_rapidfuzz = self._check_rapidfuzz()

    def _check_rapidfuzz(self) -> bool:
        try:
            import rapidfuzz
            return True
        except ImportError:
            return False

    def _build_index(self) -> dict[str, list[str]]:
        """构建倒排索引：按前2个字符分组"""
        index = defaultdict(list)
        for name in self.candidates:
            key = name[:2].lower() if len(name) >= 2 else name.lower()
            index[key].append(name)
        return index

    def _jaccard_tokens(self, a: str, b: str) -> float:
        """基于字符 bigram 的 Jaccard 相似度（对中文更友好）"""
        def bigrams(s):
            return {s[i:i+2] for i in range(len(s)-1)} if len(s) >= 2 else {s}
        sa = bigrams(a)
        sb = bigrams(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def find(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        模糊查找与 query 最相似的实体
        返回 [(name, score), ...]，按分数降序
        """
        query_clean = normalize_name(query).lower()
        if not query_clean:
            return []

        # 先用倒排索引缩小候选范围
        prefix = query_clean[:2]
        candidate_pool = self._index.get(prefix, [])
        if len(candidate_pool) < 10:
            # 扩大候选池
            candidate_pool = list(self._index.values())[:50]
            candidate_pool = [n for sub in candidate_pool for n in sub]

        results = []
        if self._use_rapidfuzz:
            from rapidfuzz import fuzz
            for cand in candidate_pool[:500]:  # 最多比较 500 个
                score = fuzz.token_sort_ratio(query_clean, cand.lower()) / 100.0
                if score >= self.threshold:
                    results.append((cand, score))
        else:
            from difflib import SequenceMatcher
            for cand in candidate_pool[:500]:
                score = SequenceMatcher(None, query_clean, cand.lower()).ratio()
                if score >= self.threshold:
                    results.append((cand, score))

        # 如果 fuzzy 结果少，补充 Jaccard 相似度
        if len(results) < 3:
            for cand in candidate_pool[:500]:
                jac = self._jaccard_tokens(query_clean, cand.lower())
                if jac >= self.threshold * 0.9 and (cand, jac) not in results:
                    results.append((cand, jac))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]


# ============================================================================
# 统一实体解析器
# ============================================================================

class EntityResolver:
    """
    统一实体解析器，集成三层匹配
    使用方式：
        resolver = EntityResolver()
        resolver.build_index(all_canonical_names)
        entity_id = resolver.resolve("查询名称")
    """

    def __init__(self):
        self.alias_dict = AliasDict()
        self.fuzzy_matcher: Optional[FuzzyMatcher] = None
        self._candidates: list[str] = []
        self.graph_store: Optional[GraphStore] = None

    def build_index(self, candidates: list[str], graph_store: Optional[GraphStore] = None):
        """构建模糊匹配索引"""
        self._candidates = candidates
        self.fuzzy_matcher = FuzzyMatcher(candidates, threshold=0.85)
        self.graph_store = graph_store

    def add_alias(self, canonical: str, alias: str):
        self.alias_dict.add(canonical, alias)

    def resolve(self, query: str, use_fuzzy: bool = True) -> Optional[tuple[str, float]]:
        """
        解析实体名称
        返回 (canonical_name, confidence) 或 None
        confidence: 1.0=精确匹配, 0.85-0.99=模糊匹配
        """
        if not query or not query.strip():
            return None

        query = normalize_name(query)

        # 层1: 精确匹配 (别名词典 → normalized name)
        resolved = self.alias_dict.resolve(query)
        if resolved:
            return (resolved, 1.0)

        # 层2: 在 graph_store 中精确查找
        if self.graph_store:
            eid = self.graph_store.find_by_name(query)
            if eid:
                ent = self.graph_store.get_entity(eid)
                if ent:
                    return (ent.canonical_name, 1.0)

        # 层3: 模糊匹配
        if use_fuzzy and self.fuzzy_matcher:
            results = self.fuzzy_matcher.find(query, top_k=5)
            if results:
                return results[0]  # (name, score)

        return None

    def resolve_to_entity(self, query: str) -> Optional[Entity]:
        """解析为 Entity 对象"""
        result = self.resolve(query)
        if result and self.graph_store:
            eid = self.graph_store.find_by_name(result[0])
            if eid:
                return self.graph_store.get_entity(eid)
        return None


# ============================================================================
# 从原始股东数据构建 GraphStore
# ============================================================================

def build_graph_from_shareholders(
    shareholder_df: pd.DataFrame,
    alias_dict: Optional[AliasDict] = None,
) -> tuple[GraphStore, AliasDict]:
    """
    从股东持股数据构建完整的 GraphStore
    处理流程：
    1. 遍历所有行，创建 Entity
    2. 创建 HoldingEdge
    3. 构建别名词典
    """
    store = GraphStore()
    if alias_dict is None:
        alias_dict = AliasDict()

    df = shareholder_df.copy()

    # 需要的关键列
    required_cols = ['s_holder_name', 's_holder_pct', 's_holder_holdercategory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 先收集所有实体
    entity_map: dict[str, Entity] = {}  # canonical_name → Entity
    aname_to_name: dict[str, str] = {}  # aname → canonical_name

    print("[GraphBuilder] 开始处理实体...")

    # 第一遍：处理 s_holder_aname（完整名，优先作为 canonical）
    if 's_holder_aname' in df.columns:
        for _, row in df.iterrows():
            aname = str(row['s_holder_aname']).strip() if pd.notna(row['s_holder_aname']) else ""
            if not aname:
                continue
            cleaned = clean_entity_name(aname)
            if cleaned and cleaned not in entity_map:
                nat = row.get('s_holder_nat', '')
                entity_type = map_entity_type(nat)
                entity = Entity(
                    entity_id=make_entity_id(cleaned),
                    canonical_name=cleaned,
                    aliases={cleaned},
                    entity_type=entity_type,
                    is_transparent=entity_type in TRANSPARENT_TYPES,
                )
                entity_map[cleaned] = entity
                aname_to_name[cleaned] = cleaned

    # 第二遍：处理 s_holder_name，发现 name→aname 的别名关系
    for _, row in df.iterrows():
        name = str(row['s_holder_name']).strip() if pd.notna(row['s_holder_name']) else ""
        aname = str(row['s_holder_aname']).strip() if pd.notna(row['s_holder_aname']) else ""

        if not name:
            continue

        cleaned_name = clean_entity_name(name)
        cleaned_aname = clean_entity_name(aname) if aname else ""

        # 确定 canonical name（优先用 aname）
        canonical = cleaned_aname if cleaned_aname else cleaned_name

        if canonical not in entity_map:
            nat = row.get('s_holder_nat', '')
            entity_type = map_entity_type(nat)
            entity = Entity(
                entity_id=make_entity_id(canonical),
                canonical_name=canonical,
                aliases={canonical},
                entity_type=entity_type,
                is_transparent=entity_type in TRANSPARENT_TYPES,
            )
            entity_map[canonical] = entity
        else:
            entity = entity_map[canonical]

        # 添加别名
        if cleaned_name and cleaned_name != canonical:
            entity.aliases.add(cleaned_name)
            alias_dict.add(canonical, cleaned_name)

        if cleaned_aname and cleaned_aname != canonical:
            entity.aliases.add(cleaned_aname)
            alias_dict.add(canonical, cleaned_aname)

    # 第三遍：处理被持股公司（target side）
    for _, row in df.iterrows():
        stock_code = str(row['s_info_windcode']).strip() if pd.notna(row['s_info_windcode']) else ""
        if not stock_code:
            continue
        # 通过 stock_code 查找是否已有 stock entity
        stock_entity_id = None
        for e in entity_map.values():
            if e.stock_code and stock_code in e.stock_code:
                stock_entity_id = e.entity_id
                break
        if stock_entity_id is None:
            # 创建上市公司实体
            stock_entity = Entity(
                entity_id=make_entity_id(stock_code),
                canonical_name=stock_code,
                aliases={stock_code},
                entity_type="民营企业",  # 大多数上市公司是民企，有国企的后续修正
                is_listed=True,
                stock_code=stock_code,
            )
            entity_map[stock_code] = stock_entity

    print(f"[GraphBuilder] 共发现 {len(entity_map)} 个实体")

    # 添加到图
    for entity in entity_map.values():
        store.add_entity(entity)

    # 添加边：直接用 store 的索引查找，不再依赖 entity_map
    print("[GraphBuilder] 开始构建持股关系...")
    skipped = 0
    edge_count = 0
    for _, row in df.iterrows():
        try:
            holder_name = str(row['s_holder_name']).strip() if pd.notna(row['s_holder_name']) else ""
            holder_aname = str(row['s_holder_aname']).strip() if pd.notna(row['s_holder_aname']) else ""
            target_code = str(row['s_info_windcode']).strip() if pd.notna(row['s_info_windcode']) else ""
            pct = float(row['s_holder_pct']) if pd.notna(row['s_holder_pct']) else 0

            if not holder_name or not target_code or pct <= 0:
                skipped += 1
                continue

            # 用 store 的索引查找股东 — 先查 aname 再查 name
            holder_eid = None
            if holder_aname:
                holder_eid = store.find_by_name(clean_entity_name(holder_aname))
            if not holder_eid:
                holder_eid = store.find_by_name(clean_entity_name(holder_name))

            # 用 store 的索引查找目标公司
            target_eid = store.find_by_stock_code(target_code)
            if not target_eid:
                target_eid = store.find_by_name(target_code)

            if not holder_eid or not target_eid:
                skipped += 1
                continue

            # 日期处理
            ann_date = None
            if 'ann_dt' in df.columns:
                try:
                    ann_date_val = pd.to_datetime(row['ann_dt'], format='%Y%m%d', errors='coerce')
                    if ann_date_val is not pd.NaT:
                        ann_date = ann_date_val.to_pydatetime()
                except Exception:
                    pass

            edge = HoldingEdge(
                source_id=holder_eid,
                target_id=target_eid,
                pct=pct,
                start_date=ann_date,
                end_date=None,
                is_direct=True,
            )
            store.add_holding(edge)
            edge_count += 1
        except Exception:
            skipped += 1

    print(f"[GraphBuilder] 构建完成: {store.entity_count()} 实体, {store.edge_count()} 边, 跳过 {skipped} 行")
    print(f"[GraphBuilder] 别名词典: {alias_dict.size()} 对映射")

    return store, alias_dict


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    print("=== 名称标准化测试 ===")
    tests = [
        "国泰证券股份有限公司客户信用交易担保证券账户",
        "贵州茅台酒股份有限公司",
        "XX投资（有限合伙）",
        "社保基金17052组合",
    ]
    for t in tests:
        print(f"  [{t}] → [{normalize_name(t)}]")

    print("\n=== 实体类型映射测试 ===")
    for t in ["境内自然人", "国有法人", "基金、理财产品等", "有限合伙企业(私募基金)"]:
        print(f"  [{t}] → [{map_entity_type(t)}] (需穿透: {map_entity_type(t) in TRANSPARENT_TYPES})")
