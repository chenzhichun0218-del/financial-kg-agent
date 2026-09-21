---
tags: [Tool, Agent, 财务]
---

# 财务排雷 Tool

> 封装 [[02-财报反欺诈引擎]]

## Schema

```json
{
  "name": "analyze_financial_health",
  "parameters": {
    "stock_code": "股票代码",
    "period": "报告期",
    "rule_ids": [1,2,3]
  }
}
```

## 输出

```json
{
  "risk_score": 72,
  "risk_level": "high",
  "alerts": [{
    "rule_name": "经营现金流/净利润倒挂",
    "severity": "high",
    "analysis": "..."
  }],
  "report": "【预警点】...\n【数据对比】...\n【风险提示】..."
}
```

## 关联

- [[意图识别路由]]
- [[财务排雷规则]]
