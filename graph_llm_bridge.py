"""
================================================================================
 LLM-图推理桥
 Graph-First, LLM-Guided 融合模式：
 图做重体力（索引/BFS/路径枚举/累乘），LLM做判断（壳识别/路径评估/解释）
================================================================================
"""
from typing import Optional, Callable
import json

from graph_core import GraphStore, Entity
from penetration_engine import PenetrationEngine, ShellDetector, PenetrationPath


# ============================================================================
# 子图 → 自然语言序列化
# ============================================================================

def entity_to_text(entity: Entity) -> str:
    """单个实体 → 文本描述"""
    parts = [f"实体名称: {entity.canonical_name}"]
    if entity.entity_type != "未知":
        parts.append(f"类型: {entity.entity_type}")
    if entity.is_listed:
        parts.append(f"上市公司 ({entity.stock_code})")
    if entity.is_transparent:
        parts.append("标记为通道/壳实体")
    if entity.shell_score > 0.3:
        parts.append(f"壳概率评分: {entity.shell_score:.2f}")
    if entity.metadata:
        for k, v in entity.metadata.items():
            if v and k not in ("aliases",):
                parts.append(f"{k}: {v}")
    return " | ".join(parts)


def holding_to_text(src_name: str, tgt_name: str, pct: float, depth: int = 1) -> str:
    """持股关系 → 文本"""
    return f"{'  ' * (depth - 1)}{src_name} 持有 {tgt_name} {pct}% 股份"


def subgraph_to_context(
    store: GraphStore,
    entity_id: str,
    depth: int = 3,
    direction: str = "up",
) -> str:
    """
    将子图转为 LLM 可读的上下文文本

    Args:
        store: 图存储
        entity_id: 起始实体
        depth: 深度
        direction: "up"(找控制人) | "down"(找控股公司) | "both"
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        return f"未找到实体: {entity_id}"

    lines = [f"## 实体: {entity.canonical_name}", f"类型: {entity.entity_type}", ""]

    if direction in ("up", "both"):
        lines.append("### 股东（向上）:")
        shareholders = store.get_neighbors(entity_id, "in")
        if shareholders:
            for sh in shareholders[:20]:
                name = sh.get("source_name", "未知")
                pct = sh.get("pct", 0)
                stype = sh.get("source_type", "")
                lines.append(f"  - {name} [{stype}]: {pct}%")
        else:
            lines.append("  (无股东数据)")

    if direction in ("down", "both"):
        lines.append("\n### 持股公司（向下）:")
        holdings = store.get_neighbors(entity_id, "out")
        if holdings:
            for h in holdings[:20]:
                name = h.get("target_name", "未知")
                pct = h.get("pct", 0)
                lines.append(f"  - {name}: {pct}%")
        else:
            lines.append("  (无持股数据)")

    return "\n".join(lines)


def path_to_narrative(path: PenetrationPath) -> str:
    """穿透路径 → 自然语言叙述"""
    if path.depth == 0:
        return f"{path.node_names[0]} 没有更上层的持股信息"

    parts = []
    for i in range(len(path.node_names) - 1):
        src = path.node_names[i + 1] if i + 1 < len(path.node_names) else "?"
        tgt = path.node_names[i]
        pct = path.holding_pcts[i] if i < len(path.holding_pcts) else 0
        parts.append(f"{src} → {tgt}({pct}%)")

    return (
        f"穿透链: {' → '.join(path.node_names)}\n"
        f"持股路径: {' → '.join(parts)}\n"
        f"有效持股比例: {path.effective_pct}%\n"
        f"穿透深度: {path.depth} 层\n"
        f"终止原因: {path.terminals_at}"
    )


# ============================================================================
# LLM 穿透决策
# ============================================================================

def llm_shell_judge(
    entity_id: str,
    store: GraphStore,
    shell_detector: ShellDetector,
    chat_fn: Optional[Callable] = None,
) -> dict:
    """
    用 LLM 判断一个中间实体是否应该被穿透。
    结合规则评分 + LLM 定性判断。

    Returns:
        {
            "should_penetrate": bool,
            "shell_score": float,
            "reasoning": str,
            "source": "rule" | "llm" | "hybrid"
        }
    """
    entity = store.get_entity(entity_id)
    if entity is None:
        return {"should_penetrate": False, "shell_score": 0, "reasoning": "实体不存在", "source": "rule"}

    # 规则评分
    rule_score = shell_detector.score(entity_id)
    rule_decision = shell_detector.should_penetrate_through(entity_id)

    # 强制类型：直接返回规则结果
    if entity.entity_type in ("自然人", "政府机构", "国有企业"):
        return {
            "should_penetrate": False,
            "shell_score": rule_score,
            "reasoning": f"实体类型为'{entity.entity_type}'，不需要继续穿透",
            "source": "rule",
        }

    if entity.entity_type in ("基金/资管", "合伙企业"):
        return {
            "should_penetrate": True,
            "shell_score": rule_score,
            "reasoning": f"实体类型为'{entity.entity_type}'，强制穿透",
            "source": "rule",
        }

    # 边界情况：规则不明确时调用 LLM
    if 0.2 <= rule_score <= 0.6 and chat_fn:
        try:
            context = subgraph_to_context(store, entity_id, depth=2, direction="both")
            prompt = (
                "你是一个股权穿透分析专家。请判断以下实体是否应该被视为"
                "'通道/壳公司'并继续向上穿透查找实际控制人。\n\n"
                f"{context}\n\n"
                f"当前规则对该实体的壳评分: {rule_score:.2f}\n\n"
                "请回答JSON格式: {\"should_penetrate\": true/false, \"reasoning\": \"理由\"}"
            )

            response = chat_fn([
                {"role": "system", "content": "你是一个股权穿透专家。只输出JSON，不要额外解释。"},
                {"role": "user", "content": prompt},
            ])

            # 尝试解析 LLM 响应
            if isinstance(response, str):
                response = json.loads(response)
            llm_decision = response.get("should_penetrate", rule_decision)
            llm_reasoning = response.get("reasoning", "")

            return {
                "should_penetrate": llm_decision,
                "shell_score": (rule_score + (0.7 if llm_decision else 0.2)) / 2,
                "reasoning": llm_reasoning or "LLM辅助判断",
                "source": "hybrid",
            }
        except Exception as e:
            pass  # LLM 失败，回退到规则

    # 纯规则判断
    return {
        "should_penetrate": rule_decision,
        "shell_score": rule_score,
        "reasoning": f"规则判断，壳评分={rule_score:.2f}",
        "source": "rule",
    }


# ============================================================================
# LLM 路径综合评估
# ============================================================================

def llm_evaluate_paths(
    paths: list[PenetrationPath],
    store: GraphStore,
    chat_fn: Optional[Callable] = None,
) -> list[dict]:
    """
    用 LLM 对多条穿透路径做综合评估，输出带置信度的路径排名

    Returns:
        [{path_dict, confidence, llm_judgment, risk_flags}, ...]
    """
    if not chat_fn:
        # 无 LLM，纯规则排序
        return [
            {**p.to_dict(), "confidence": 0.9, "llm_judgment": "", "risk_flags": []}
            for p in paths[:10]
        ]

    # 序列化所有路径
    paths_text = "\n\n".join(
        f"路径{i+1}: {path_to_narrative(p)}" for i, p in enumerate(paths[:10])
    )

    try:
        prompt = (
            "你是股权穿透分析专家。以下是同一目标公司的多条穿透路径，"
            "请评估每条路径的可信度并识别风险点。\n\n"
            f"{paths_text}\n\n"
            "请输出JSON数组，每条路径包含: "
            "{\"path_index\": 路径编号, \"confidence\": 置信度0-1, "
            "\"judgment\": \"简短评价\", \"risk_flags\": [\"风险标签\"]}"
        )

        response = chat_fn([
            {"role": "system", "content": "你是一个股权分析专家。只输出JSON数组，不要额外解释。"},
            {"role": "user", "content": prompt},
        ])

        if isinstance(response, str):
            response = json.loads(response)

        # 合并
        results = []
        for item in response[:10]:
            idx = item.get("path_index", 0) - 1
            if 0 <= idx < len(paths):
                results.append({
                    **paths[idx].to_dict(),
                    "confidence": item.get("confidence", 0.8),
                    "llm_judgment": item.get("judgment", ""),
                    "risk_flags": item.get("risk_flags", []),
                })
        return results
    except Exception:
        return [
            {**p.to_dict(), "confidence": 0.85, "llm_judgment": "", "risk_flags": []}
            for p in paths[:10]
        ]


# ============================================================================
# 综合推理器（Agent 可调用的高级接口）
# ============================================================================

class GraphLLMBridge:
    """
    LLM-图融合推理桥
    封装了图→文本、LLM穿透决策、路径评估的全部逻辑
    Agent 直接调用这个类的方法
    """

    def __init__(self, store: GraphStore, penetration_engine: PenetrationEngine):
        self.store = store
        self.engine = penetration_engine
        self.shell_detector = penetration_engine.shell_detector

    def deep_penetrate(
        self,
        entity_id: str,
        max_depth: int = 5,
        chat_fn: Optional[Callable] = None,
    ) -> dict:
        """
        深度融合穿透：图计算路径 + LLM评估可靠度

        Returns:
            {
                "entity": {...},
                "chains": [...],         # 带置信度和LLM判断的路径
                "controllers": [...],    # 最终控制人
                "llm_summary": "..."     # LLM 综合摘要（如果可用）
            }
        """
        entity = self.store.get_entity(entity_id)

        # 图计算：穿透路径
        paths = self.engine.penetrate_upward(entity_id, max_depth=max_depth)

        # LLM 路径评估
        evaluated = llm_evaluate_paths(paths, self.store, chat_fn)

        # 最终控制人
        controllers = self.engine.get_ultimate_controllers(entity_id, max_depth=max_depth)

        # LLM 综合摘要
        llm_summary = ""
        if chat_fn and evaluated:
            try:
                context = subgraph_to_context(self.store, entity_id, depth=3)
                paths_summary = "\n".join(
                    f"- {p.get('path', [])} (有效持股{p.get('effective_pct', 0)}%)"
                    for p in evaluated[:5]
                )
                prompt = (
                    f"基于以下股权穿透分析结果，用3-5句话总结这家公司的实际控制情况。\n\n"
                    f"{context}\n\n"
                    f"穿透路径:\n{paths_summary}\n\n"
                    f"最终控制人: {controllers}"
                )
                llm_summary = chat_fn([
                    {"role": "system", "content": "你是金融分析专家。用简洁中文总结。"},
                    {"role": "user", "content": prompt},
                ])
                if not isinstance(llm_summary, str):
                    llm_summary = str(llm_summary)
            except Exception:
                llm_summary = ""

        return {
            "entity": entity.to_dict() if entity else None,
            "chains": evaluated,
            "controllers": controllers,
            "llm_summary": llm_summary,
        }

    def ask_graph(
        self,
        question: str,
        entity_id: str,
        chat_fn: Optional[Callable] = None,
    ) -> str:
        """
        基于图谱回答自然语言问题
        这是把图谱作为 LLM 的"知识库"来使用

        Args:
            question: 用户的自然语言问题
            entity_id: 相关实体
            chat_fn: LLM 调用函数
        """
        if chat_fn is None:
            # 无 LLM，返回结构化数据
            result = self.deep_penetrate(entity_id)
            return json.dumps(result, ensure_ascii=False, indent=2)

        # 从图谱提取上下文
        context = subgraph_to_context(self.store, entity_id, depth=3)
        paths = self.engine.penetrate_upward(entity_id, max_depth=3)
        paths_text = "\n".join(
            f"穿透路径{i+1}: {path_to_narrative(p)}" for i, p in enumerate(paths[:5])
        )

        prompt = (
            f"你是一个金融知识图谱问答系统。请基于以下图谱数据回答用户问题。\n\n"
            f"## 图谱上下文\n{context}\n\n"
            f"## 穿透分析\n{paths_text}\n\n"
            f"## 用户问题\n{question}\n\n"
            f"请基于图谱数据给出准确、简洁的回答。如果数据不足以回答，请明确指出。"
        )

        try:
            response = chat_fn([
                {"role": "system", "content": "你是一个基于知识图谱的金融问答系统。"},
                {"role": "user", "content": prompt},
            ])
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"图谱查询结果（LLM不可用）:\n{paths_text}"


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    from graph_core import make_entity_id
    from entity_resolver import build_graph_from_shareholders
    import pandas as pd

    print("=== LLM-图推理桥测试 ===\n")

    try:
        df = pd.read_excel("data_raw/2.股东持股-股权穿透/clean.xlsx")
        sample = df.head(5000)
        store, aliases = build_graph_from_shareholders(sample)
        engine = PenetrationEngine(store)
        bridge = GraphLLMBridge(store, engine)

        # 测试一只股票
        codes = list(store.stock_index.keys())[:2]
        for code in codes:
            eid = store.find_by_stock_code(code)
            print(f"\n--- {code} ---")
            print(subgraph_to_context(store, eid, depth=2))

            # 不调用 LLM 的穿透
            result = bridge.deep_penetrate(eid, max_depth=3)
            print(f"\n穿透链 ({len(result['chains'])} 条):")
            for c in result['chains'][:3]:
                print(f"  {c['path']} => {c['effective_pct']}%")

    except FileNotFoundError:
        print("数据文件未找到，跳过测试")
