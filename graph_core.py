"""
================================================================================
 图核心模块 — 实体、关系、图存储
 替代原有的 defaultdict(list) 简单图结构
 基于 NetworkX 构建带属性的有向图，支持时间戳边和实体消歧
================================================================================
"""
import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np


# ============================================================================
# 实体类型规范映射：100+ 种 s_holder_nat → 8 类
# ============================================================================

CANONICAL_TYPE_MAP = {
    # ── 自然人 ──
    "境内自然人": "自然人", "境外自然人": "自然人", "自然人": "自然人",
    "自然人股东": "自然人", "中国香港籍自然人": "自然人", "台湾籍自然人": "自然人",
    "中国境内自然人": "自然人", "境内自然人股东": "自然人", "境内个人": "自然人",
    "境内自然人境内自然人": "自然人", "境内自然人（已故）": "自然人",
    "境内白然人": "自然人",                               # 原始数据中的笔误
    "副总经理": "自然人", "财务负责人": "自然人",           # 高管以个人身份持股
    "副总经理、董事会秘书": "自然人", "副董事长": "自然人",
    "董事长、总经理": "自然人", "实际控制人、自然人股东": "自然人",

    # ── 民营企业 ──
    "境内非国有法人": "民营企业", "境内一般法人": "民营企业",
    "境内法人": "民营企业", "民营法人": "民营企业",
    "有限责任公司": "民营企业", "股份有限公司": "民营企业",
    "境内法人企业": "民营企业", "境内企业法人": "民营企业",
    "境内企业": "民营企业", "境内有限公司": "民营企业",
    "境内上市公司": "民营企业", "企业法人": "民营企业",
    "法人企业": "民营企业", "法人": "民营企业", "法人主体": "民营企业",
    "公司法人": "民营企业", "境内法人股东": "民营企业",
    "境内非国有法人股": "民营企业", "境内非国有法人股东": "民营企业",
    "境内非国有法人持股": "民营企业", "境内非国有法人境内非国有法人": "民营企业",
    "非国有法人": "民营企业", "国内非国有法人": "民营企业",
    "境内非国有人": "民营企业", "境内非国法人": "民营企业",
    "境内非国有": "民营企业", "境内非国有法": "民营企业",
    "境内非国人法人": "民营企业", "内非国有法人": "民营企业",
    "股东性质境内非国有法人": "民营企业", "非国有法人境内": "民营企业",
    "中国境内法人": "民营企业", "其他法人": "民营企业",
    "法人股东": "民营企业", "机构股东": "民营企业",
    "境内机构股东": "民营企业", "机构投资者": "民营企业",
    "机构": "民营企业", "外部投资者": "民营企业",
    "股份有限公司（中外合资、上市）": "民营企业",
    "股份有限公司(中外合资、上市)": "民营企业",
    "控股股东、法人股东": "民营企业", "控股股东、有限责任公司": "民营企业",
    "控股股东、自然人股东": "民营企业",   # 实际控制人是自然人，但身份类别是控股股东
    "境内自然人/一般法人": "民营企业",
    "非法人股东": "民营企业", "其他非自然人": "民营企业",
    "不详": "未知", "其它": "未知",
    "法人国有": "国有企业",               # 笔误，应为国有法人

    # ── 国有企业 ──
    "国有法人": "国有企业", "境内国有法人": "国有企业",
    "国有": "国有企业", "国家": "国有企业", "国家股": "国有企业",
    "国家机关": "政府机构", "国有股东": "国有企业",
    "境内国有法人股": "国有企业", "国有法人股": "国有企业",
    "国有法人国有法人": "国有企业", "国有法人境内": "国有企业",
    "境内国有法人股": "国有企业", "暂未分类的国有法人": "国有企业",
    "股东性质国有法人": "国有企业", "国法人": "国有企业",

    # ── 境外实体 ──
    "境外法人": "境外实体", "境外非国有法人": "境外实体",
    "境外法人股东": "境外实体", "境外法人股": "境外实体",
    "外资": "境外实体", "外资股东": "境外实体",
    "境外合伙企业": "境外实体", "境外法": "境外实体",
    "QFII": "境外实体", "合格境外投资者": "境外实体",
    "H股": "境外实体", "A股": "境外实体",

    # ── 基金/资管产品 ──
    "基金、理财产品等": "基金/资管",
    "基金,理财产品": "基金/资管", "基金、理财产品": "基金/资管",
    "基金理财产品": "基金/资管", "基金、理财产": "基金/资管",
    "金,理财产品": "基金/资管", "基金、": "基金/资管",
    "证券投资基金": "基金/资管", "其他募证券投资基金": "基金/资管",
    "私募股权基金": "基金/资管", "私募基金": "基金/资管",
    "其他(基金,理财产品)": "基金/资管",
    "基金": "基金/资管", "基金，理财产品等": "基金/资管",

    # ── 合伙企业 ──
    "境内合伙企业": "合伙企业", "有限合伙企业": "合伙企业",
    "合伙企业": "合伙企业", "境内有限合伙企业": "合伙企业",
    "有限合伙": "合伙企业", "合伙企业股东": "合伙企业",
    "合伙企业法人": "合伙企业", "有限合伙企业（私募基金）": "合伙企业",
    "有限合伙企业(私募基金)": "合伙企业", "有限合伙企业（持股平台）": "合伙企业",
    "有限合伙企业(持股平台)": "合伙企业", "员工持股平台": "合伙企业",
    "持股平台合伙企业": "合伙企业",
    "有限合伙企业、员工持股平台": "合伙企业",
    "非国有法人、有限合伙企业": "合伙企业",
    "公司法人（持股平台）": "合伙企业", "公司法人(持股平台)": "合伙企业",
    "中国境内合伙企业": "合伙企业",
}

# 剩下的统一归为 "未知"
FALLBACK_TYPE = "未知"

# 需要强制穿透的实体类型（这些实体本身不是最终控制人）
TRANSPARENT_TYPES = {"基金/资管", "合伙企业"}


def map_entity_type(nat_label: str) -> str:
    """将原始的 s_holder_nat 标签映射到 8 类规范类型"""
    if pd.isna(nat_label) or not isinstance(nat_label, str):
        return FALLBACK_TYPE
    nat_label = nat_label.strip()
    return CANONICAL_TYPE_MAP.get(nat_label, FALLBACK_TYPE)


# ============================================================================
# 实体 Schema
# ============================================================================

@dataclass
class Entity:
    """知识图谱中的实体节点"""
    entity_id: str                    # 标准化名称的 MD5 hash (稳定且可复现)
    canonical_name: str               # 标准化后的主名称
    aliases: set = field(default_factory=set)     # 所有已知别名
    entity_type: str = "未知"         # 8 类规范类型之一
    is_transparent: bool = False      # 是否需要穿透（基金/合伙/壳 → True）
    shell_score: float = 0.0          # 壳公司概率评分 0-1（规则+LLM）
    is_listed: bool = False           # 是否为上市公司
    stock_code: Optional[str] = None  # 股票代码（仅上市公司有）
    metadata: dict = field(default_factory=dict)  # 扩展信息

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.entity_id == other.entity_id
        return False

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "entity_type": self.entity_type,
            "is_transparent": self.is_transparent,
            "shell_score": self.shell_score,
            "is_listed": self.is_listed,
            "stock_code": self.stock_code,
            "metadata": self.metadata,
        }


# ============================================================================
# 持股关系边 Schema
# ============================================================================

@dataclass
class HoldingEdge:
    """持股关系边"""
    source_id: str                    # 股东 entity_id
    target_id: str                    # 被持股公司 entity_id
    pct: float                        # 持股比例 (%)
    start_date: Optional[datetime]    # 持股开始日期
    end_date: Optional[datetime]      # 持股结束日期 (None = 当前仍持有)
    is_direct: bool = True            # 直接持股 / 间接持股
    relationship_type: str = "直接持股"  # "直接持股" | "一致行动人" | "实际控制"
    metadata: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        """判断当前是否仍持有"""
        return self.end_date is None

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "pct": self.pct,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "is_direct": self.is_direct,
            "relationship_type": self.relationship_type,
        }


# ============================================================================
# 图存储 — NetworkX 属性图
# ============================================================================

class GraphStore:
    """
    股权知识图谱存储层
    基于 NetworkX 有向图，用 entity_id 作为节点 key
    支持名称索引、股票代码索引、模糊查找
    """

    def __init__(self):
        self.G = nx.DiGraph()
        # 索引
        self.name_index: dict[str, str] = {}       # 标准化名 → entity_id
        self.alias_index: dict[str, str] = {}       # 别名 → entity_id
        self.stock_index: dict[str, str] = {}       # 股票代码 → entity_id
        self.entities: dict[str, Entity] = {}       # entity_id → Entity
        # 统计
        self._edge_count = 0

    # ── 节点操作 ──

    def add_entity(self, entity: Entity):
        """添加或更新实体节点"""
        self.entities[entity.entity_id] = entity
        self.G.add_node(entity.entity_id, **entity.to_dict())
        # 更新名称索引
        key = entity.canonical_name.lower().strip()
        self.name_index[key] = entity.entity_id
        # 更新别名索引
        for alias in entity.aliases:
            self.alias_index[alias.lower().strip()] = entity.entity_id
        # 更新股票代码索引
        if entity.stock_code:
            self.stock_index[entity.stock_code] = entity.entity_id
            # 也加入不带后缀的代码
            code_short = entity.stock_code.replace('.SH', '').replace('.SZ', '').replace('.BJ', '')
            self.stock_index[code_short] = entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def entity_count(self) -> int:
        return len(self.entities)

    # ── 边操作 ──

    def add_holding(self, edge: HoldingEdge):
        """添加持股关系边"""
        self.G.add_edge(
            edge.source_id, edge.target_id,
            pct=edge.pct,
            start_date=edge.start_date,
            end_date=edge.end_date,
            is_direct=edge.is_direct,
            relationship_type=edge.relationship_type,
            metadata=edge.metadata,
        )
        self._edge_count += 1

    def edge_count(self) -> int:
        return self._edge_count

    # ── 查找操作 ──

    def find_by_name(self, name: str) -> Optional[str]:
        """
        精确查找实体
        返回 entity_id 或 None
        """
        key = name.lower().strip()
        # 尝试主名称
        if key in self.name_index:
            return self.name_index[key]
        # 尝试别名
        if key in self.alias_index:
            return self.alias_index[key]
        # 尝试匹配股票代码
        if key in self.stock_index:
            return self.stock_index[key]
        return None

    def find_by_stock_code(self, code: str) -> Optional[str]:
        """按股票代码查找"""
        code = code.strip()
        if code in self.stock_index:
            return self.stock_index[code]
        # 尝试加后缀
        for suffix in ['.SH', '.SZ', '.BJ']:
            if code + suffix in self.stock_index:
                return self.stock_index[code + suffix]
        # 模糊匹配
        for k, v in self.stock_index.items():
            if code in k:
                return v
        return None

    def get_neighbors(self, entity_id: str, direction: str = "out") -> list[dict]:
        """
        获取实体的邻居（持股关系）
        direction: "out"=作为股东持有谁, "in"=被谁持有, "both"
        """
        if direction == "out":
            edges = list(self.G.out_edges(entity_id, data=True))
            result = []
            for _, target, data in edges:
                target_entity = self.entities.get(target, None)
                result.append({
                    "target_id": target,
                    "target_name": target_entity.canonical_name if target_entity else target,
                    "target_type": target_entity.entity_type if target_entity else "未知",
                    "pct": data.get("pct", 0),
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date"),
                    "relationship_type": data.get("relationship_type", "直接持股"),
                })
            return result
        elif direction == "in":
            edges = list(self.G.in_edges(entity_id, data=True))
            result = []
            for source, _, data in edges:
                source_entity = self.entities.get(source, None)
                result.append({
                    "source_id": source,
                    "source_name": source_entity.canonical_name if source_entity else source,
                    "source_type": source_entity.entity_type if source_entity else "未知",
                    "pct": data.get("pct", 0),
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date"),
                })
            return result
        else:
            return {
                "out": self.get_neighbors(entity_id, "out"),
                "in": self.get_neighbors(entity_id, "in"),
            }

    def get_holding_pct(self, source_id: str, target_id: str) -> float:
        """获取两个实体之间的持股比例"""
        edge = self.G.get_edge_data(source_id, target_id)
        if edge:
            return edge.get("pct", 0)
        return 0.0

    # ── 子图导出 ──

    def get_subgraph(self, entity_id: str, depth: int = 3) -> nx.DiGraph:
        """获取某个实体为中心的子图（向上穿透 depth 层）"""
        nodes = {entity_id}
        frontier = {entity_id}
        for _ in range(depth):
            next_frontier = set()
            for n in frontier:
                predecessors = set(self.G.predecessors(n))
                next_frontier.update(predecessors)
            frontier = next_frontier - nodes
            nodes.update(frontier)
            if not frontier:
                break
        return self.G.subgraph(nodes)

    def subgraph_to_text(self, entity_id: str, depth: int = 3) -> str:
        """
        将子图序列化为 LLM 可读的文本（用于 Graph-to-Text 推理）
        """
        sub = self.get_subgraph(entity_id, depth)
        lines = []
        lines.append(f"实体: {self.entities.get(entity_id, Entity(entity_id, entity_id)).canonical_name}")
        lines.append("")

        for src, tgt, data in sub.edges(data=True):
            src_ent = self.entities.get(src)
            tgt_ent = self.entities.get(tgt)
            src_name = src_ent.canonical_name if src_ent else src
            tgt_name = tgt_ent.canonical_name if tgt_ent else tgt
            pct = data.get("pct", 0)
            src_type = src_ent.entity_type if src_ent else "未知"
            lines.append(f"  {src_name}[{src_type}] ──{pct}%──→ {tgt_name}")

        return "\n".join(lines)

    # ── 统计 ──

    def stats(self) -> dict:
        type_counts = defaultdict(int)
        for e in self.entities.values():
            type_counts[e.entity_type] += 1
        return {
            "total_entities": len(self.entities),
            "total_edges": self._edge_count,
            "type_distribution": dict(type_counts),
            "listed_companies": len(self.stock_index),
        }

    def save(self, path: str):
        """将图持久化（节点用 JSON，边用 CSV）"""
        import os
        os.makedirs(path, exist_ok=True)
        # 保存实体
        entity_list = [e.to_dict() for e in self.entities.values()]
        with open(os.path.join(path, "entities.json"), "w", encoding="utf-8") as f:
            json.dump(entity_list, f, ensure_ascii=False, indent=2)
        # 保存边（用 NetworkX 的 node_link_data）
        graph_data = nx.node_link_data(self.G)
        with open(os.path.join(path, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        print(f"[GraphStore] 已保存 {len(self.entities)} 个实体, {self._edge_count} 条边到 {path}")

    @classmethod
    def load(cls, path: str) -> "GraphStore":
        """从文件加载图"""
        import os
        store = cls()
        # 加载实体
        with open(os.path.join(path, "entities.json"), "r", encoding="utf-8") as f:
            entities_data = json.load(f)
        for ed in entities_data:
            entity = Entity(
                entity_id=ed["entity_id"],
                canonical_name=ed["canonical_name"],
                aliases=set(ed.get("aliases", [])),
                entity_type=ed.get("entity_type", "未知"),
                is_transparent=ed.get("is_transparent", False),
                shell_score=ed.get("shell_score", 0.0),
                is_listed=ed.get("is_listed", False),
                stock_code=ed.get("stock_code"),
                metadata=ed.get("metadata", {}),
            )
            store.add_entity(entity)
        # 加载边
        with open(os.path.join(path, "graph.json"), "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        store.G = nx.node_link_graph(graph_data)
        # 重建边计数
        store._edge_count = store.G.number_of_edges()
        print(f"[GraphStore] 已加载 {store.entity_count()} 个实体, {store.edge_count()} 条边")
        return store


# ============================================================================
# 实体 ID 工具
# ============================================================================

def make_entity_id(name: str) -> str:
    """从名称生成稳定的 entity_id"""
    return "ent_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def normalize_name(name: str) -> str:
    """
    名称标准化
    - 统一全角/半角括号
    - 去多余空格
    - 去掉常见的证券账户后缀
    """
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # 全角 → 半角
    name = name.replace("（", "(").replace("）", ")")
    name = name.replace("，", ",").replace("；", ";")
    name = name.replace("０", "0").replace("１", "1").replace("２", "2")
    name = name.replace("３", "3").replace("４", "4").replace("５", "5")
    name = name.replace("６", "6").replace("７", "7").replace("８", "8").replace("９", "9")
    name = name.replace("　", "").replace("\t", "").replace("\n", "")
    # 去多余空格
    name = " ".join(name.split())
    # 去掉常见证券账户后缀（这些不是真实实体）
    suffixes_to_strip = [
        "客户信用交易担保证券账户",
        "转融通担保证券账户",
        "约定购回专用账户",
    ]
    for suffix in suffixes_to_strip:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


# ============================================================================
# 快速测试入口
# ============================================================================

if __name__ == "__main__":
    # 测试实体类型映射
    print("=== 实体类型映射测试 ===")
    tests = ["境内自然人", "国有法人", "境内非国有法人", "基金、理财产品等",
             "有限合伙企业", "境外法人", "未知", None, "乱七八糟"]
    for t in tests:
        result = map_entity_type(t)
        print(f"  {t!r} → {result}")

    # 测试图构建
    print("\n=== 图构建测试 ===")
    store = GraphStore()

    # 添加实体
    e1 = Entity(
        entity_id=make_entity_id("张三"),
        canonical_name="张三",
        aliases={"张三"},
        entity_type="自然人",
    )
    e2 = Entity(
        entity_id=make_entity_id("XX投资管理有限公司"),
        canonical_name="XX投资管理有限公司",
        aliases={"XX投资管理有限公司", "XX投资"},
        entity_type="民营企业",
    )
    e3 = Entity(
        entity_id=make_entity_id("603439"),
        canonical_name="603439",
        aliases={"603439", "603439.SH"},
        entity_type="民营企业",
        is_listed=True,
        stock_code="603439.SH",
    )

    store.add_entity(e1)
    store.add_entity(e2)
    store.add_entity(e3)

    # 添加边
    store.add_holding(HoldingEdge(e1.entity_id, e2.entity_id, pct=80.0, start_date=None, end_date=None))
    store.add_holding(HoldingEdge(e2.entity_id, e3.entity_id, pct=30.0, start_date=None, end_date=None))

    # 查询
    found = store.find_by_name("张三")
    print(f"查找 '张三': {found}")
    found_stock = store.find_by_stock_code("603439")
    print(f"查找股票 '603439': {found_stock}")

    # 子图文本
    print("\n子图文本:")
    print(store.subgraph_to_text(e3.entity_id, depth=3))

    # 统计
    print("\n统计:", store.stats())
