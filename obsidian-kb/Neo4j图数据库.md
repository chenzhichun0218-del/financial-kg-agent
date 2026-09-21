---
tags: [技术, 图数据库]
---

# Neo4j 图数据库

> 用于 [[03-股权穿透与知识图谱]]

## Schema

```
节点：Person | Company | Fund | Event
边：  HOLDS | AFFILIATED_WITH | LEGAL_REP_OF | INVOLVED_IN
```

## 关键查询

**多跳穿透**：
```cypher
MATCH path = (start:Person)-[:HOLDS*1..5]->(end:Company)
WHERE end.stock_code = '000001.SZ'
RETURN path
```

**持股比例累乘**：
```cypher
WITH path, REDUCE(pct=1.0, r IN RELATIONSHIPS(path) | pct * r.pct) AS total_pct
```

## 关联

- [[03-股权穿透与知识图谱]]
- [[图谱穿透Tool]]
- [[股东持股数据]]
