---
tags: [Tool, Agent, 图谱]
---

# 图谱穿透 Tool

> 封装 [[Neo4j图数据库]] 查询

## Schema

```json
{
  "name": "query_equity_penetration",
  "parameters": {
    "query_type": "stock | person",
    "identifier": "股票代码或人名",
    "max_depth": 5
  }
}
```

## 输出

```json
{
  "chains": [{
    "path": ["自然人A", "壳公司B", "上市公司C"],
    "edges": [{"from": "A", "to": "B", "pct": 80.0}],
    "total_effective_pct": 40.08,
    "depth": 2
  }]
}
```

## 关联

- [[意图识别路由]]
- [[Neo4j图数据库]]
