"""
================================================================================
 Agent 工具层 — Function Calling 工具定义
 将图查询、穿透、事件时间线封装为 Agent 可调用的标准工具
================================================================================
"""
import json
from typing import Optional, Callable
from datetime import datetime

from graph_core import GraphStore
from penetration_engine import PenetrationEngine, EquityQueryService
from event_pipeline import EventPipeline, EventTimeline
from graph_llm_bridge import GraphLLMBridge


# ============================================================================
# OpenAI Function Calling 工具定义
# ============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_equity_penetration",
            "description": "查询某个公司或自然人的股权穿透链路，找出多层持股关系、实际控制人和有效持股比例。支持向上穿透（找控制人）和向下穿透（找控股公司）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "实体名称（如'贵州茅台'、'张三'）或股票代码（如'603439'、'600519.SH'）",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "最大穿透深度，默认5层",
                        "default": 5,
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "both"],
                        "description": "穿透方向：up=向上找控制人, down=向下找控股公司, both=双向",
                        "default": "up",
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_timeline",
            "description": "获取某只股票的事件时间线，包括股权变更事件和舆情事件（如监管处罚、资产重组等），按时间轴对齐。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码，如'603439'、'600519.SH'",
                    },
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "过滤事件类型，可选：股权变更、监管处罚、资产重组、年报发布、高管变动",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "起始日期，格式YYYY-MM-DD",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "结束日期，格式YYYY-MM-DD",
                    },
                },
                "required": ["stock_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_related_entities",
            "description": "查找与某个实体存在关联的其他实体，包括持股关系、被持股关系、一致行动人等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "实体名称或股票代码",
                    },
                    "relation_type": {
                        "type": "string",
                        "enum": ["shareholder", "subsidiary", "all"],
                        "description": "关系类型：shareholder=谁持有它, subsidiary=它持有谁, all=全部",
                        "default": "all",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回前K个结果",
                        "default": 10,
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_shareholders",
            "description": "获取某只股票的前N大股东及其持股比例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_code": {
                        "type": "string",
                        "description": "股票代码",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "返回前N名股东",
                        "default": 10,
                    },
                },
                "required": ["stock_code"],
            },
        },
    },
]


# ============================================================================
# 工具执行器
# ============================================================================

class GraphToolExecutor:
    """
    工具执行器：接收 Function Calling 的 tool_call，执行对应的图操作
    """

    def __init__(
        self,
        graph_store: GraphStore,
        event_pipeline: Optional[EventPipeline] = None,
        chat_fn: Optional[Callable] = None,
    ):
        self.store = graph_store
        self.pipeline = event_pipeline
        self.chat_fn = chat_fn

        self.engine = PenetrationEngine(graph_store)
        self.query_service = EquityQueryService(graph_store)
        self.bridge = GraphLLMBridge(graph_store, self.engine)

    def execute(self, tool_name: str, arguments: dict) -> str:
        """
        执行工具调用，返回 JSON 字符串结果

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            JSON 格式的结果字符串
        """
        if tool_name == "query_equity_penetration":
            return self._query_penetration(arguments)
        elif tool_name == "get_event_timeline":
            return self._get_timeline(arguments)
        elif tool_name == "find_related_entities":
            return self._find_related(arguments)
        elif tool_name == "get_top_shareholders":
            return self._get_top_shareholders(arguments)
        else:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

    def _query_penetration(self, args: dict) -> str:
        entity = args.get("entity", "")
        max_depth = args.get("max_depth", 5)
        direction = args.get("direction", "up")

        result = self.query_service.query(entity, max_depth=max_depth)

        if direction == "up":
            result.pop("downward_chains", None)
        elif direction == "down":
            result.pop("penetration_chains", None)
            result.pop("ultimate_controllers", None)
            result["chains"] = result.pop("downward_chains", [])

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _get_timeline(self, args: dict) -> str:
        stock_code = args.get("stock_code", "")
        date_from = args.get("date_from")
        date_to = args.get("date_to")

        if self.pipeline is None:
            return json.dumps({
                "error": "事件管道未初始化。请先加载公告数据。",
                "stock_code": stock_code,
            }, ensure_ascii=False)

        try:
            timeline = self.pipeline.get_timeline(stock_code)
            events = timeline.build()

            # 过滤
            if args.get("event_types"):
                types = set(args["event_types"])
                events = [e for e in events if e["type"] in types]

            if date_from:
                from_dt = datetime.fromisoformat(date_from)
                events = [e for e in events if e["date"] and e["date"] >= from_dt]
            if date_to:
                to_dt = datetime.fromisoformat(date_to)
                events = [e for e in events if e["date"] and e["date"] <= to_dt]

            return json.dumps({
                "stock_code": stock_code,
                "total_events": len(events),
                "events": [
                    {
                        "date": e["date"].isoformat() if e["date"] else None,
                        "type": e["type"],
                        "description": e["description"],
                        "cluster_label": e.get("cluster_label", ""),
                    }
                    for e in events
                ],
                "timeline_text": timeline.to_text(),
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _find_related(self, args: dict) -> str:
        entity_name = args.get("entity", "")
        relation_type = args.get("relation_type", "all")
        top_k = args.get("top_k", 10)

        # 查找实体
        entity_id = self.store.find_by_name(entity_name)
        if not entity_id:
            entity_id = self.store.find_by_stock_code(entity_name)

        if not entity_id:
            return json.dumps({"error": f"未找到实体: {entity_name}"}, ensure_ascii=False)

        entity = self.store.get_entity(entity_id)
        result = {
            "entity": entity.to_dict() if entity else None,
            "related": [],
        }

        if relation_type in ("shareholder", "all"):
            shareholders = self.store.get_neighbors(entity_id, "in")
            for sh in shareholders[:top_k]:
                result["related"].append({
                    "relation": "被持股",
                    "entity": sh.get("source_name", ""),
                    "entity_type": sh.get("source_type", ""),
                    "pct": sh.get("pct", 0),
                })

        if relation_type in ("subsidiary", "all"):
            holdings = self.store.get_neighbors(entity_id, "out")
            for h in holdings[:top_k]:
                result["related"].append({
                    "relation": "持股",
                    "entity": h.get("target_name", ""),
                    "entity_type": h.get("target_type", ""),
                    "pct": h.get("pct", 0),
                })

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _get_top_shareholders(self, args: dict) -> str:
        stock_code = args.get("stock_code", "")
        top_n = args.get("top_n", 10)

        entity_id = self.store.find_by_stock_code(stock_code)
        if not entity_id:
            entity_id = self.store.find_by_name(stock_code)

        if not entity_id:
            return json.dumps({"error": f"未找到股票: {stock_code}"}, ensure_ascii=False)

        entity = self.store.get_entity(entity_id)
        shareholders = self.store.get_neighbors(entity_id, "in")
        shareholders.sort(key=lambda x: x.get("pct", 0), reverse=True)

        return json.dumps({
            "stock_code": stock_code,
            "entity_name": entity.canonical_name if entity else stock_code,
            "total_shareholders": len(shareholders),
            "top_shareholders": [
                {
                    "name": sh.get("source_name", "未知"),
                    "type": sh.get("source_type", "未知"),
                    "pct": sh.get("pct", 0),
                }
                for sh in shareholders[:top_n]
            ],
        }, ensure_ascii=False, indent=2)


# ============================================================================
# 获取工具定义（用于注册到 Agent）
# ============================================================================

def get_tool_definitions() -> list[dict]:
    return TOOL_DEFINITIONS


def get_tools_for_llm() -> list[dict]:
    """返回符合 OpenAI Function Calling 格式的工具列表"""
    return [t["function"] for t in TOOL_DEFINITIONS]


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    print("=== 工具定义验证 ===\n")
    for tool in TOOL_DEFINITIONS:
        name = tool["function"]["name"]
        params = list(tool["function"]["parameters"]["properties"].keys())
        required = tool["function"]["parameters"].get("required", [])
        print(f"  {name}: 参数={params}, 必填={required}")

    print(f"\n共 {len(TOOL_DEFINITIONS)} 个工具定义")
