---
tags: [Tool, Agent, 研报]
---

# 研报检索 Tool

> 封装 [[券商研报数据]] 语义检索

## Schema

```json
{
  "name": "search_research_reports",
  "parameters": {
    "query": "查询内容",
    "stock_code": "可选",
    "top_k": 5
  }
}
```

## 关联

- [[意图识别路由]]
- [[券商研报数据]]
