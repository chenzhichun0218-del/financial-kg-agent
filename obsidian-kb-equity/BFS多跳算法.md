---
tags: [技术, 算法]
---

# BFS多跳算法

## 为什么BFS

BFS逐层展开 → 先看最近的控制人 → 再看背后的控制人。
DFS一条路走到黑不适合。

## 伪代码

```
queue = [(起始股票, 路径, 持股链)]
visited = set()

while queue:
    当前, 路径, 持股链 = queue.pop()
    if 深度 >= 4: continue
    if 当前 in visited: continue  # 去重防环
    visited.add(当前)

    for 股东, 比例 in 反向图[当前]:
        记录: (新路径, 深度, 有效持股)
        if 股东是"企业": 继续追溯
        else: 到达终点(个人)
```

## 去重

交叉持股A→B→C→A会形成环。`visited` 集合记录已访问节点, 遇到直接跳过。

## 关联
- [[图数据结构]]
- [[有效持股计算]]
- [[名称匹配]]
