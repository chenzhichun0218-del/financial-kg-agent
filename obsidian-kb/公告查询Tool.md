---
tags: [Tool, Agent, 公告]
---

# 公告查询 Tool

> 封装 [[公司公告数据]] 查询

## Schema

```json
{
  "name": "search_announcements",
  "parameters": {
    "stock_code": "股票代码",
    "info_type": "公告类型",
    "keyword": "关键词"
  }
}
```

## 关联

- [[意图识别路由]]
- [[公司公告数据]]
