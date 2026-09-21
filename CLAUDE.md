# CLAUDE.md

## 项目路径

`C:\Users\asus\知识图谱项目\` — 唯一工作目录，所有操作在此进行

## 目录结构

```
知识图谱项目/
├── data_raw/               ← 原始比赛数据
│   ├── 1/clean.xlsx        # 测试问答集
│   ├── 2/clean.xlsx        # 股东持股
│   ├── 3/clean.xlsx        # 公司公告
│   ├── 4/*.csv             # 三大财务报表
│   └── 5/*.csv             # 券商研报
├── data_processed/         ← 清洗后数据 (.pkl)
├── obsidian-kb/            ← Obsidian 知识图谱笔记
├── main.py                 ← 完整项目主程序
├── data_processor.py       ← 数据预处理模块
├── llm_client.py           ← LLM API 客户端
├── .env                    ← API Key + Base URL
└── 赛题说明.docx / 项目步骤规划.docx / 测试样例.md
```

## 重要规则

- 所有新文件保存到 `C:\Users\asus\知识图谱项目\`
- 不碰 OneDrive 桌面
- 数据路径用 `data_raw/` 和 `data_processed/`
