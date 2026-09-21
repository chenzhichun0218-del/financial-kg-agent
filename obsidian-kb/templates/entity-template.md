---
entity_id: {{entity_id}}
name: "{{canonical_name}}"
type: {{entity_type}}
listed: {{is_listed}}
stock_code: {{stock_code}}
tags: [{{type_tag}}, #实体]
created: {{date}}
---

# {{icon}} {{canonical_name}}

## 📋 基本信息

| 属性 | 值 |
|------|-----|
| 实体类型 | {{entity_type}} |
{{#if is_listed}}
| 上市状态 | 🏢 上市公司 ({{stock_code}}) |
{{/if}}
{{#if is_transparent}}
| 穿透标记 | ⚠️ 需穿透（{{entity_type}}通常为通道/壳） |
{{/if}}

## ⬆️ 股东（谁持有我）

| # | 股东 | 类型 | 持股比例 |
|---|------|------|----------|
<!-- 自动填充 -->

## 🎯 穿透分析：最终控制人

<!-- 自动填充 -->

## 🔗 图谱关联

```dataview
TABLE type, stock_code
FROM [[{{filename}}]]
WHERE file.name != this.file.name
SORT type ASC
```
