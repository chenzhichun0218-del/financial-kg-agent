"""
================================================================================
 股权穿透引擎
 多跳 BFS/DFS 穿透、壳公司检测、路径排序、有效持股累乘
================================================================================
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
import re

from graph_core import (
    GraphStore, Entity, HoldingEdge,
    TRANSPARENT_TYPES
)


# ============================================================================
# 穿透路径数据结构
# ============================================================================

@dataclass
class PenetrationPath:
    """一条穿透路径"""
    node_ids: list[str]                        # entity_id 序列 (从最终控制人 → 目标公司)
    node_names: list[str]                      # 对应的实体名称
    node_types: list[str]                      # 对应的实体类型
    holding_pcts: list[float]                  # 每跳持股比例
    effective_pct: float                       # 有效持股比例 (累乘)
    depth: int                                 # 穿透深度
    terminals_at: str                          # 停止原因
    confidence: float = 1.0                    # 路径置信度

    def summary(self) -> str:
        """人类可读的路径摘要"""
        parts = []
        for i, (name, pct) in enumerate(zip(self.node_names[1:], self.holding_pcts)):
            parts.append(f"{name}({pct}%)")
        chain = " → ".join(self.node_names)
        return (
            f"[深度{self.depth}|有效{self.effective_pct:.2f}%] {chain}\n"
            f"  停止原因: {self.terminals_at}"
        )

    def to_dict(self) -> dict:
        return {
            "path": self.node_names,
            "path_ids": self.node_ids,
            "holding_pcts": self.holding_pcts,
            "effective_pct": round(self.effective_pct, 4),
            "depth": self.depth,
            "terminals_at": self.terminals_at,
            "confidence": round(self.confidence, 3),
        }


# ============================================================================
# 壳公司检测器
# ============================================================================

class ShellDetector:
    """
    壳公司/通道实体检测器
    基于规则打分，无需 LLM 即可识别大部分壳/通道
    """

    # 壳判定阈值
    SHELL_THRESHOLD = 0.4

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store
        # 缓存：哪些 entity_id 是被持股方
        self._target_entities: set[str] = self._compute_targets()

    def _compute_targets(self) -> set[str]:
        """找出所有作为被持股方出现的实体（即不是纯粹通道）"""
        targets = set()
        for _, tgt in self.store.G.edges():
            targets.add(tgt)
        return targets

    def score(self, entity_id: str) -> float:
        """
        计算壳公司概率评分 (0-1)
        分数越高越像是壳/通道
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return 0.0

        score = 0.0
        name = entity.canonical_name

        # 规则1: 实体类型决定基础分
        if entity.entity_type in TRANSPARENT_TYPES:
            if entity.entity_type == "基金/资管":
                score += 0.6
            elif entity.entity_type == "合伙企业":
                score += 0.4
        entity.is_transparent = entity.entity_type in TRANSPARENT_TYPES

        # 规则2: 名称关键词
        investment_keywords = [
            "投资管理", "投资咨询", "股权投资", "投资控股",
            "资产管理", "资本管理", "私募基金", "创业投资",
            "投资发展", "投资集团",
        ]
        for kw in investment_keywords:
            if kw in name:
                score += 0.2
                break

        # 规则3: 仅为持股方（从未作为被持股方出现）
        if entity_id not in self._target_entities:
            score += 0.15

        # 规则4: 持股高度分散（持有 10+ 家公司）
        out_degree = self.store.G.out_degree(entity_id)
        if out_degree >= 10:
            # 检查是否每家都很少
            holdings = self.store.get_neighbors(entity_id, "out")
            small_stakes = sum(
                1 for h in holdings if h.get("pct", 0) < 5.0
            )
            if small_stakes >= 10 or (holdings and small_stakes / len(holdings) >= 0.8):
                score += 0.2

        # 规则5: 注册资本/规模类（从 metadata 中读取，如果有的话）
        if entity.metadata.get("registered_capital", float("inf")) < 100_0000:  # <100万
            score += 0.1

        # 规则6: 名称含"有限合伙"（典型的 SPV 结构）
        if "有限合伙" in name:
            score += 0.15

        return min(score, 1.0)

    def is_shell(self, entity_id: str) -> bool:
        """判断是否为壳/通道实体"""
        return self.score(entity_id) >= self.SHELL_THRESHOLD

    def should_penetrate_through(self, entity_id: str) -> bool:
        """
        判断是否应该穿透该实体继续向上查找
        条件：
        1. 强制穿透类型（基金/资管/合伙企业）
        2. 壳评分 >= 阈值
        3. 如果是自然人 → 不穿透（已到底）
        4. 如果是政府机构 → 不穿透（已到底）
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return False

        # 自然人 → 终止（找到实际控制人）
        if entity.entity_type == "自然人":
            return False

        # 政府机构/国有企业 → 终止（最终控制人是国家）
        if entity.entity_type in ("政府机构", "国有企业"):
            return False

        # 强制穿透类型
        if entity.entity_type in TRANSPARENT_TYPES:
            return True

        # 壳评分判断
        return self.is_shell(entity_id)


# ============================================================================
# 股权穿透引擎
# ============================================================================

class PenetrationEngine:
    """
    股权穿透引擎
    支持 BFS/DFS 多跳穿透、路径排序、LLM 辅助决策
    """

    def __init__(self, graph_store: GraphStore):
        self.store = graph_store
        self.shell_detector = ShellDetector(graph_store)

    # ── 多跳穿透的关键：实体名 → 股票代码解析 ──

    def _resolve_to_listed_entity(self, entity_id: str) -> str | None:
        """
        尝试将一个实体解析为对应的上市公司实体ID。
        这是多跳穿透生效的关键步骤。

        场景：BFS 遇到企业股东"贝达药业股份有限公司"，
        它作为股东时的 entity_id 是 s_holder_name 生成的，
        但它的股东数据存储在 s_info_windcode 对应的 entity_id 下。

        需要根据名称找到对应的股票代码实体ID。
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return None

        # 1) 如果本身已经是上市公司，直接返回
        if entity.is_listed and entity.stock_code:
            stock_eid = self.store.find_by_stock_code(entity.stock_code)
            if stock_eid:
                return stock_eid
            return entity_id  # 没找到就用自己

        name = entity.canonical_name

        # 2) 按股票代码搜索（精确匹配）
        stock_eid = self.store.find_by_stock_code(name)
        if stock_eid and stock_eid != entity_id:
            return stock_eid

        # 3) 在 stock_index 中模糊匹配（名称标准化后比对）
        for stock_code, stock_eid in self.store.stock_index.items():
            if stock_eid == entity_id:
                continue
            stock_ent = self.store.get_entity(stock_eid)
            if stock_ent and stock_ent.canonical_name:
                # 去后缀比对
                a = name.replace("股份有限公司", "").replace("有限责任公司", "").replace("有限公司", "").strip()
                b = stock_ent.canonical_name.replace("股份有限公司", "").replace("有限责任公司", "").replace("有限公司", "").strip()
                if a and b and (a == b or a in b or b in a):
                    return stock_eid

        # 4) 按 name_index 查找
        found = self.store.find_by_name(name)
        if found and found != entity_id:
            found_ent = self.store.get_entity(found)
            if found_ent and found_ent.is_listed:
                return found

        return None

    def penetrate_upward(
        self,
        entity_id: str,
        max_depth: int = 5,
        min_effective_pct: float = 0.01,
        use_shell_detection: bool = True,
    ) -> list[PenetrationPath]:
        """
        向上穿透：从被持股方找最终控制人
        BFS 算法，用 (entity_id, path) 队列

        Args:
            entity_id: 起始实体 ID
            max_depth: 最大穿透深度
            min_effective_pct: 最小有效持股比例过滤 (默认 0.01%)
            use_shell_detection: 是否启用壳检测（True=自动穿透壳）
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return []

        completed_paths: list[PenetrationPath] = []
        # visited: 每个节点允许被访问多次（不同路径），但同一路径不循环
        # 用 path_visited 控制每条路径的循环检测

        # BFS 队列: (current_node_id, node_ids_path, holding_pcts_path)
        queue = deque()
        queue.append((entity_id, [entity_id], []))

        while queue:
            current_id, node_path, pct_path = queue.popleft()
            current_depth = len(node_path) - 1

            if current_depth >= max_depth:
                # 达到最大深度，记录路径并停止
                completed_paths.append(self._build_path(node_path, pct_path, "达到最大穿透深度"))
                continue

            # 获取当前节点的所有股东（in-edges）
            shareholders = self.store.get_neighbors(current_id, "in")

            if not shareholders:
                # ── 关键修复：尝试将企业名解析为股票代码 ──
                # 场景："贝达药业"作为股东出现，但它的股东数据存储在
                # 对应的股票代码实体下。通过名称解析找到那个实体ID。
                resolved_id = self._resolve_to_listed_entity(current_id)
                if resolved_id and resolved_id != current_id:
                    resolved_shareholders = self.store.get_neighbors(resolved_id, "in")
                    if resolved_shareholders:
                        # 用解析后的实体ID替换当前节点，继续穿透
                        shareholders = resolved_shareholders
                        # 更新路径中的最后一个节点为解析后的实体
                        if node_path and node_path[-1] == current_id:
                            node_path[-1] = resolved_id
                        current_id = resolved_id

                if not shareholders:
                    # 确实没有股东，已到顶
                    completed_paths.append(self._build_path(node_path, pct_path, "无更上层股东"))
                    continue

            has_continued = False
            for sh in shareholders:
                sh_id = sh["source_id"]
                sh_pct = sh.get("pct", 0)
                sh_type = sh.get("source_type", "未知")

                # 循环检测
                if sh_id in node_path:
                    continue

                # 有效持股过滤（在中间跳也过滤）
                current_effective = self._calc_effective(pct_path + [sh_pct])
                if current_effective < min_effective_pct and len(pct_path) > 1:
                    continue

                new_node_path = node_path + [sh_id]
                new_pct_path = pct_path + [sh_pct]

                # 判断是否应该在此停止
                should_stop = False
                stop_reason = ""

                if sh_type == "自然人":
                    should_stop = True
                    stop_reason = "到达自然人（实际控制人）"
                elif sh_type in ("政府机构", "国有企业"):
                    should_stop = True
                    stop_reason = f"到达{sh_type}（最终控制方）"
                elif use_shell_detection and self.shell_detector.should_penetrate_through(sh_id):
                    # 是壳/通道，继续穿透
                    should_stop = False
                elif not use_shell_detection:
                    # 不用壳检测，全部继续
                    pass
                else:
                    # 不是壳也不是自然人 → 可能是个真实运营公司，但仍有上层股东
                    # 继续但不强制
                    should_stop = False

                if should_stop:
                    completed_paths.append(self._build_path(new_node_path, new_pct_path, stop_reason))
                else:
                    queue.append((sh_id, new_node_path, new_pct_path))
                    has_continued = True

            # 如果所有股东都被标记为停止但没有实际停止（因为进入了队列）
            # 不需要特殊处理

        # 去重 + 排序
        completed_paths = self._dedup_paths(completed_paths)
        completed_paths.sort(key=lambda p: (-p.effective_pct, p.depth))

        return completed_paths

    def penetrate_downward(
        self,
        entity_id: str,
        max_depth: int = 5,
        min_effective_pct: float = 0.1,
    ) -> list[PenetrationPath]:
        """
        向下穿透：从股东方找其控股的所有公司
        用于回答"XX控制了哪些公司"这类问题
        """
        entity = self.store.get_entity(entity_id)
        if entity is None:
            return []

        completed_paths = []
        queue = deque()
        queue.append((entity_id, [entity_id], []))

        while queue:
            current_id, node_path, pct_path = queue.popleft()
            current_depth = len(node_path) - 1

            if current_depth >= max_depth:
                completed_paths.append(self._build_path(node_path, pct_path, "达到最大深度"))
                continue

            # 获取当前节点持股的公司（out-edges）
            holdings = self.store.get_neighbors(current_id, "out")

            if not holdings:
                if len(node_path) > 1:  # 不是自身
                    completed_paths.append(self._build_path(node_path, pct_path, "无更下层持股"))
                continue

            for h in holdings:
                target_id = h["target_id"]
                target_pct = h.get("pct", 0)

                if target_id in node_path:
                    continue

                current_effective = self._calc_effective(pct_path + [target_pct])
                if current_effective < min_effective_pct and len(pct_path) > 1:
                    continue

                new_node_path = node_path + [target_id]
                new_pct_path = pct_path + [target_pct]

                # 对于上市公司，通常停下来
                target_entity = self.store.get_entity(target_id)
                if target_entity and target_entity.is_listed:
                    completed_paths.append(self._build_path(
                        new_node_path, new_pct_path, f"到达上市公司({target_entity.stock_code})"
                    ))
                else:
                    queue.append((target_id, new_node_path, new_pct_path))

        completed_paths = self._dedup_paths(completed_paths)
        completed_paths.sort(key=lambda p: (-p.effective_pct, p.depth))
        return completed_paths

    def get_ultimate_controllers(
        self,
        entity_id: str,
        max_depth: int = 5,
        top_k: int = 10,
    ) -> list[dict]:
        """
        获取最终控制人（自然人或国资）
        这是最常用的对外接口
        """
        paths = self.penetrate_upward(entity_id, max_depth=max_depth)

        controllers = []
        for path in paths:
            if path.node_types[-1] in ("自然人", "政府机构", "国有企业"):
                controllers.append({
                    "controller_name": path.node_names[-1],
                    "controller_type": path.node_types[-1],
                    "chain": path.node_names,
                    "chain_pcts": path.holding_pcts,
                    "effective_pct": path.effective_pct,
                    "depth": path.depth,
                    "confidence": path.confidence,
                })

        # 去重（同一控制人通过不同路径控制）
        seen = set()
        unique = []
        for c in controllers:
            key = c["controller_name"]
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique[:top_k]

    # ── 辅助方法 ──

    def _build_path(self, node_ids: list[str], pcts: list[float], terminals_at: str) -> PenetrationPath:
        """从 node_ids 构建 PenetrationPath"""
        names = []
        types = []
        for nid in node_ids:
            ent = self.store.get_entity(nid)
            if ent:
                names.append(ent.canonical_name)
                types.append(ent.entity_type)
            else:
                names.append(nid)
                types.append("未知")

        return PenetrationPath(
            node_ids=node_ids,
            node_names=names,
            node_types=types,
            holding_pcts=pcts,
            effective_pct=self._calc_effective(pcts),
            depth=len(node_ids) - 1,
            terminals_at=terminals_at,
        )

    @staticmethod
    def _calc_effective(pcts: list[float]) -> float:
        """累乘计算有效持股比例"""
        result = 1.0
        for p in pcts:
            result *= p / 100.0
        return round(result * 100, 4)

    @staticmethod
    def _dedup_paths(paths: list[PenetrationPath]) -> list[PenetrationPath]:
        """去重：相同的 (node_ids 序列, effective_pct) 只保留一个"""
        seen = set()
        unique = []
        for p in paths:
            key = (tuple(p.node_ids), round(p.effective_pct, 4))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ============================================================================
# 查询入口（面向 Agent 的高级接口）
# ============================================================================

class EquityQueryService:
    """
    股权查询服务
    封装 PenetrationEngine + EntityResolver，提供 Agent 友好的查询接口
    """

    def __init__(self, graph_store: GraphStore, entity_resolver=None):
        self.store = graph_store
        self.engine = PenetrationEngine(graph_store)
        self.resolver = entity_resolver

    def query(self, entity_name: str, max_depth: int = 5) -> dict:
        """
        通用查询接口

        返回:
        {
            "query": "原始查询",
            "resolved_entity": "解析后的实体名",
            "entity_type": "实体类型",
            "top_shareholders": [...],
            "penetration_chains": [...],
            "ultimate_controllers": [...],
            "downward_chains": [...],
        }
        """
        # 解析实体
        entity_id = None
        resolved_name = entity_name

        if self.resolver:
            result = self.resolver.resolve(entity_name)
            if result:
                resolved_name = result[0]
                entity_id = self.store.find_by_name(resolved_name)

        if entity_id is None:
            entity_id = self.store.find_by_name(entity_name)

        if entity_id is None:
            # 尝试股票代码
            entity_id = self.store.find_by_stock_code(entity_name)
            if entity_id:
                resolved_name = self.store.get_entity(entity_id).canonical_name

        if entity_id is None:
            return {"query": entity_name, "error": "未找到该实体", "resolved_entity": None}

        entity = self.store.get_entity(entity_id)

        # 前十大股东
        top_holders = self.store.get_neighbors(entity_id, "in")
        top_holders.sort(key=lambda x: x.get("pct", 0), reverse=True)
        top_holders = top_holders[:10]

        # 向上穿透
        penetration = self.engine.penetrate_upward(entity_id, max_depth=max_depth)
        controllers = self.engine.get_ultimate_controllers(entity_id, max_depth=max_depth)

        # 向下穿透
        downward = self.engine.penetrate_downward(entity_id, max_depth=max_depth)

        return {
            "query": entity_name,
            "resolved_entity": resolved_name,
            "entity_type": entity.entity_type if entity else "未知",
            "is_listed": entity.is_listed if entity else False,
            "stock_code": entity.stock_code if entity else None,
            "top_shareholders": [
                {
                    "name": h.get("source_name", "未知"),
                    "pct": h.get("pct", 0),
                    "type": h.get("source_type", "未知"),
                }
                for h in top_holders
            ],
            "penetration_chains": [p.to_dict() for p in penetration[:10]],
            "ultimate_controllers": controllers,
            "downward_chains": [p.to_dict() for p in downward[:10]],
        }


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    from graph_core import make_entity_id
    from entity_resolver import build_graph_from_shareholders

    print("=== 穿透引擎测试（使用真实数据）===")
    import pandas as pd

    # 加载股东数据
    try:
        df = pd.read_excel("data_raw/2.股东持股-股权穿透/clean.xlsx")
        print(f"加载了 {len(df)} 条股东数据")
    except Exception as e:
        print(f"未找到数据文件: {e}")
        print("使用模拟数据测试...")
        df = pd.DataFrame({
            "s_holder_name": ["张三", "XX投资", "YY控股", "国家电网"],
            "s_holder_aname": ["张三", "XX投资管理有限公司", "YY控股有限公司", "国家电网有限公司"],
            "s_holder_pct": [42.0, 30.0, 25.0, 51.0],
            "s_holder_holdercategory": [1, 2, 2, 2],
            "s_holder_nat": ["境内自然人", "境内非国有法人", "境内非国有法人", "国有法人"],
            "s_info_windcode": ["603439.SH", "603439.SH", "603439.SH", "603439.SH"],
        })

    # 只取前 50000 行做快速测试（完整图需要更长时间）
    sample = df.head(50000) if len(df) > 50000 else df

    store, aliases = build_graph_from_shareholders(sample)

    # 测试穿透
    engine = PenetrationEngine(store)
    svc = EquityQueryService(store)

    # 随便找一只股票测试
    test_stocks = list(store.stock_index.keys())[:3]
    for code in test_stocks:
        print(f"\n--- 查询: {code} ---")
        result = svc.query(code, max_depth=3)
        print(f"  解析为: {result['resolved_entity']}")
        print(f"  实体类型: {result['entity_type']}")
        print(f"  前5大股东:")
        for h in result.get("top_shareholders", [])[:5]:
            print(f"    {h['name']} ({h['type']}): {h['pct']}%")
        print(f"  穿透链路: {len(result['penetration_chains'])} 条")
        for chain in result.get("penetration_chains", [])[:3]:
            print(f"    链: {chain['path']}")
            print(f"    有效持股: {chain['effective_pct']}%, 深度: {chain['depth']}")
        print(f"  最终控制人: {result['ultimate_controllers']}")

    # 测试壳检测
    print(f"\n--- 壳检测 ---")
    shell_det = ShellDetector(store)
    shell_count = 0
    for eid in list(store.entities.keys())[:1000]:
        if shell_det.is_shell(eid):
            shell_count += 1
    print(f"  前1000个实体中壳/通道占比: {shell_count/10:.1f}%")
