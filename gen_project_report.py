"""
=============================================================================
 生成项目完整技术报告 Word 文档
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from datetime import datetime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "项目技术报告.docx")


def set_cell_shading(cell, color):
    shading = cell._element.get_or_add_tcPr()
    shd = shading.makeelement(qn('w:shd'), {qn('w:fill'): color, qn('w:val'): 'clear'})
    shading.append(shd)


def h(doc, text, level=1):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x47, 0x8a)
    return heading


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p


def table_with_header(doc, headers, rows):
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.style = 'Light Grid Accent 1'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h_text in enumerate(headers):
        cell = tbl.rows[0].cells[i]
        cell.text = h_text
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
        set_cell_shading(cell, "1a478a")
    for r_idx, row_data in enumerate(rows, 1):
        for c_idx, cell_text in enumerate(row_data):
            tbl.rows[r_idx].cells[c_idx].text = str(cell_text)
    return tbl


def build():
    doc = Document()
    sect = doc.sections[0]
    sect.page_width = Cm(21)
    sect.page_height = Cm(29.7)
    sect.left_margin = Cm(2.5)
    sect.right_margin = Cm(2.5)

    # ── 封面 ──
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.space_after = Pt(6)
    run = title.add_run("基于 Agentic AI 的金融长上下文推理、\n图谱穿透与财报反欺诈智能问答系统\n\n技术报告")
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x0d, 0x2b, 0x5e)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.space_before = Pt(30)
    run = info.add_run(f"生成日期: {datetime.now().strftime('%Y年%m月%d日')}\n"
                       "第五届中国研究生金融科技创新大赛")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_page_break()

    # ── 目录 ──
    h(doc, "目录", 1)
    toc = [
        "一、项目概述",
        "二、数据处理",
        "    2.1 数据来源与规模",
        "    2.2 五类数据清洗流程",
        "    2.3 实体类型标准化（100+ → 8类）",
        "三、知识图谱定义与构建",
        "    3.1 图谱Schema定义",
        "    3.2 四层实体解析管线",
        "    3.3 构建流程（646K行 → 80K实体 → 646K边）",
        "四、多跳股权穿透",
        "    4.1 穿透引擎架构",
        "    4.2 壳公司6维评分模型",
        "    4.3 Graph-First, LLM-Guided混合决策",
        "    4.4 全量穿透评测结果",
        "五、财报反欺诈引擎",
        "    5.1 12条检测规则",
        "    5.2 行业感知 + 模式组合 + 事件信号",
        "六、舆情事件聚类与召回",
        "    6.1 13类事件分类体系",
        "    6.2 聚类与LLM标签生成",
        "    6.3 时间线与股权-事件交叉分析",
        "七、Agent编排与RAG系统",
        "    7.1 三层记忆架构（L1/L2/L3）",
        "    7.2 语义RAG检索（Embedding + 余弦相似度）",
        "    7.3 自纠错闭环",
        "八、LLM幻觉应对机制",
        "九、全量评测结果",
        "十、专利价值分析",
    ]
    for item in toc:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ════════════════════════════════════
    # 一、项目概述
    # ════════════════════════════════════
    h(doc, "一、项目概述", 1)
    doc.add_paragraph(
        "本项目构建了一套基于 Agentic AI 的金融智能问答系统，面向A股投资者提供三大核心能力："
        "财报反欺诈分析、多层股权穿透查询、智能对话问答。系统基于约65万条股东数据、"
        "7千余条公司公告、三张财务报表及5万余篇券商研报，构建了包含80,433个实体和"
        "646,449条持股关系的金融知识图谱，并通过LLM-Graph混合推理架构实现可解释的穿透分析。"
    )

    h(doc, "核心架构", 2)
    layers = [
        ("数据层", "646K股东 + 7K公告 + 三表 + 55K研报 → 五类数据清洗标准化"),
        ("图谱层", "NetworkX有向图: 80K实体节点 + 646K边 + 四层实体解析 + 11K别名映射"),
        ("穿透层", "BFS多跳穿透 + 6维壳评分 + Graph-First/LLM-Guided混合决策"),
        ("事件层", "13类事件分类 + 35个事件簇 + 时间轴对齐 + 股权-事件交叉分析"),
        ("Agent层", "意图识别→工具编排→自纠错 + 三层记忆(L1/L2/L3) + 语义RAG"),
        ("可视层", "Obsidian知识图谱视图 + Mermaid流程图 + Dataview动态查询"),
    ]
    for name, desc in layers:
        p = doc.add_paragraph()
        run = p.add_run(f"【{name}】")
        run.font.bold = True
        p.add_run(f" {desc}")

    # ════════════════════════════════════
    # 二、数据处理
    # ════════════════════════════════════
    h(doc, "二、数据处理", 1)

    h(doc, "2.1 数据来源与规模", 2)
    table_with_header(doc,
        ["数据集", "原始格式", "规模", "产出"],
        [
            ["测试问答集", "Excel", "35个会话", "qa_test.pkl"],
            ["股东持股", "Excel", "646,449行", "shareholders.pkl"],
            ["公司公告", "Excel", "7,311条", "announcements.pkl"],
            ["三大财务报表", "CSV×3", "6,713只股票", "financials_merged.pkl"],
            ["券商研报", "CSV", "55,214篇", "reports.pkl"],
        ])

    h(doc, "2.2 五类数据清洗流程", 2)

    h(doc, "股东持股数据", 3)
    doc.add_paragraph(
        "原始数据包含 s_holder_name（股东名）、s_holder_aname（股东全名）、"
        "s_holder_pct（持股比例）、s_holder_holdercategory（1=个人/2=企业）、"
        "s_holder_nat（100+种Wind实体类型编码）、s_info_windcode（被持股的上市公司代码）等字段。"
    )
    steps = [
        "日期转换: ann_dt (int如20240102) → ann_date (datetime)",
        "去重: 同一股东+同一股票+同一天，保留最新公告",
        "异常值处理: 持股比例 > 100% → 截断为100%",
        "实体类型标准化: 100+种 s_holder_nat 原始编码 → 8类规范类型",
        "金融中介标记: 79,224个银行/基金/券商/陆股通实体被标记但不删除（保留完整数据）",
    ]
    for s in steps:
        doc.add_paragraph(s, style='List Bullet')

    h(doc, "财务报表数据", 3)
    doc.add_paragraph(
        "资产负债表（22个核心字段）、利润表（12个核心字段）、现金流量表（5个核心字段）"
        "三表按 stock_code + report_date 合并为一张宽表，计算6个衍生指标："
    )
    indicators = [
        "debt_ratio = 总负债/总资产",
        "inventory_to_revenue = 存货/营收",
        "receivable_to_revenue = 应收账款/营收",
        "cashflow_to_profit = 经营现金流/净利润",
        "goodwill_to_equity = 商誉/净资产",
        "gross_margin = 毛利/营收",
    ]
    for ind in indicators:
        code_block(doc, ind)

    h(doc, "公司公告数据", 3)
    doc.add_paragraph(
        "解析公告类型码（n_info_fcode，|分隔的多值字段），提取主类型码。"
        "基于10个风险关键词（处罚、监管、立案、调查、违规、整改、处分、留置、问询）"
        "自动标记风险事件（is_risk_event）。"
    )

    h(doc, "2.3 实体类型标准化（100+ → 8类）", 2)
    doc.add_paragraph(
        "Wind数据中 s_holder_nat 字段包含100+种原始实体类型编码（如'境内非国有法人'、"
        "'基金、理财产品等'、'有限合伙企业'、'国有法人'等），"
        "部分还存在笔误（如'境内白然人'）。系统通过 CANONICAL_TYPE_MAP 字典"
        "将所有原始编码映射到8类规范类型。"
    )
    table_with_header(doc,
        ["规范类型", "原始编码示例", "穿透策略"],
        [
            ["自然人", "境内自然人、境外自然人、中国香港籍自然人...", "终止（找到实控人）"],
            ["民营企业", "境内非国有法人、境内一般法人、有限责任公司...", "继续穿透"],
            ["国有企业", "国有法人、境内国有法人、国家股、国家机关...", "终止（国资委/财政部）"],
            ["境外实体", "境外法人、QFII、外资股东...", "继续穿透"],
            ["基金/资管", "基金理财产品、证券投资基金、私募基金...", "强制穿透"],
            ["合伙企业", "有限合伙企业、员工持股平台...", "强制穿透"],
            ["政府机构", "国家机关", "终止"],
            ["未知", "不详、其它、空值", "尝试穿透"],
        ])

    # ════════════════════════════════════
    # 三、知识图谱
    # ════════════════════════════════════
    h(doc, "三、知识图谱定义与构建", 1)

    h(doc, "3.1 图谱Schema定义", 2)
    doc.add_paragraph("实体节点（Entity）包含以下核心属性：")
    code_block(doc,
        "entity_id: str        # MD5(canonical_name)[:12] → 稳定可复现\n"
        "canonical_name: str   # 标准化主名称（优先用s_holder_aname全名）\n"
        "entity_type: str      # 8类规范类型之一\n"
        "is_transparent: bool  # 是否强制穿透（基金/合伙企业=True）\n"
        "shell_score: float    # 壳概率评分 0-1\n"
        "is_listed: bool       # 是否为上市公司\n"
        "stock_code: str       # 股票代码（仅上市公司）\n"
        "aliases: set[str]     # 所有已知别名"
    )
    doc.add_paragraph("持股关系边（HoldingEdge）包含：")
    code_block(doc,
        "source_id: str       # 股东 entity_id\n"
        "target_id: str       # 被持股公司 entity_id\n"
        "pct: float           # 持股比例 (%)\n"
        "start_date: datetime # 持股开始日期\n"
        "end_date: datetime   # 持股结束日期 (None=当前仍持有)\n"
        "is_direct: bool      # 直接/间接持股"
    )
    doc.add_paragraph("图存储层（GraphStore）基于 NetworkX 有向图，额外维护三个索引：")
    code_block(doc,
        "name_index:  dict[name → entity_id]     # 标准化名称索引\n"
        "alias_index: dict[alias → entity_id]    # 别名索引\n"
        "stock_index: dict[code → entity_id]     # 股票代码索引（含/不含后缀）"
    )

    h(doc, "3.2 四层实体解析管线", 2)
    table_with_header(doc,
        ["层级", "方法", "技术", "置信度"],
        [
            ["Layer 1", "别名词典 (AliasDict)", "11,089对自动发现的 name↔aname 映射", "confidence=1.0"],
            ["Layer 2", "GraphStore精确查找", "name_index / alias_index / stock_index", "confidence=1.0"],
            ["Layer 3", "模糊匹配 (FuzzyMatcher)", "rapidfuzz + Jaccard bigram + 倒排索引加速", "0.85~0.99"],
            ["Layer 4", "LLM兜底", "上层调用方可自行处理", "取决于LLM"],
        ])
    doc.add_paragraph(
        "其中 Layer 1 的别名自动发现是核心工程亮点：利用 s_holder_name（短名）和 "
        "s_holder_aname（全名）两个字段的天然对照关系，从646K行数据中自动提取11,089对别名映射，"
        "解决了'深圳腾讯' vs '深圳市腾讯计算机系统有限公司'这类中文金融实体名称不统一的问题。"
    )

    h(doc, "3.3 构建流程", 2)
    steps = [
        "第一遍扫描: 处理 s_holder_aname（全名），优先作为 canonical_name，生成 entity_id",
        "第二遍扫描: 处理 s_holder_name（短名），发现短名→全名的别名关系，自动构建别名词典",
        "第三遍: 构建持股关系边，对每行数据建立 shareholder_entity_id → stock_entity_id 的有向边",
        "第四遍: 构建 name_index、alias_index、stock_index 三个快速查找索引",
    ]
    for s in steps:
        doc.add_paragraph(s, style='List Number')

    table_with_header(doc,
        ["构建指标", "数值"],
        [
            ["输入数据行数", "646,449"],
            ["金融中介过滤", "79,224个实体被标记"],
            ["最终实体数", "80,433"],
            ["持股关系边数", "646,449"],
            ["上市公司数", "6,161"],
            ["别名映射对数", "11,089"],
        ])

    # ════════════════════════════════════
    # 四、多跳股权穿透
    # ════════════════════════════════════
    h(doc, "四、多跳股权穿透", 1)

    h(doc, "4.1 穿透引擎架构", 2)
    doc.add_paragraph(
        "穿透引擎（PenetrationEngine）基于BFS广度优先搜索逐层展开股权关系链。"
        "核心流程：从目标实体出发 → 查找所有入边（股东）→ 对每个股东判断是否继续穿透 → "
        "壳/通道实体继续向上 → 自然人或政府机构终止 → 有效持股比例累乘。"
        "关键改进：当企业股东在GraphStore中无直接上层股东时，自动通过名称解析找到"
        "其对应的上市公司实体ID（_resolve_to_listed_entity），实现真正的多跳穿透。"
    )

    h(doc, "4.2 壳公司6维评分模型", 2)
    doc.add_paragraph(
        "壳检测器（ShellDetector）从6个维度对每个实体进行壳概率评分（0-1）："
    )
    table_with_header(doc,
        ["维度", "评分规则", "分值"],
        [
            ["R1 实体类型", "基金/资管+0.6, 合伙企业+0.4", "+0.4~0.6"],
            ["R2 名称关键词", "含'投资管理/股权投资/私募基金/创业投资'等", "+0.2"],
            ["R3 持股角色", "仅为持股方，从未作为被持股方出现", "+0.15"],
            ["R4 持股分散度", "持有10+家公司且80%持股<5%", "+0.2"],
            ["R5 注册资本", "注册资本<100万元（metadata中读取）", "+0.1"],
            ["R6 名称特征", "名称含'有限合伙'（典型SPV结构）", "+0.15"],
        ])

    h(doc, "4.3 Graph-First, LLM-Guided 混合决策", 2)
    doc.add_paragraph(
        "核心创新：将壳判定分为三个决策区间——\n"
        "  • 确定穿透区间（评分>0.6 或 基金/合伙企业）：规则直接判定穿透\n"
        "  • 确定终止区间（评分<0.2 或 自然人/政府/国企）：规则直接判定终止\n"
        "  • 模糊区间（评分0.2~0.6）：触发LLM辅助判断，将子图序列化为文本上下文，"
        "由LLM基于图结构数据做定性判断，而非凭空猜测。\n"
        "该设计使得约90%的实体由规则快速处理（毫秒级），仅约10%的模糊案例调用LLM。"
    )

    h(doc, "4.4 全量穿透评测结果", 2)
    table_with_header(doc,
        ["指标", "数值"],
        [
            ["全量上市公司", "6,161只"],
            ["1层深度（直接持股）", "5,263只 (85.4%)"],
            ["2层深度", "310只 (5.0%)"],
            ["3层深度", "588只 (9.5%)"],
            ["链路有效性", "99.7%"],
            ["平均查询延迟", "3ms"],
            ["P95延迟", "9ms"],
        ])

    # ════════════════════════════════════
    # 五、财报反欺诈
    # ════════════════════════════════════
    h(doc, "五、财报反欺诈引擎", 1)

    h(doc, "5.1 12条检测规则", 2)
    doc.add_paragraph("包含6条单期异常检测规则和6条多期趋势异常检测规则：")

    h(doc, "单期规则", 3)
    table_with_header(doc,
        ["规则", "检测逻辑", "触发条件", "分值"],
        [
            ["存货积压", "存货/营收比", ">1.0", "15"],
            ["现金流缺口", "经营现金流/净利润比", "<0.5", "20"],
            ["应收账款激增", "应收账款/营收比", ">0.5", "15"],
            ["债务危机", "资产负债率", ">0.8", "15"],
            ["商誉炸弹", "商誉/净资产比", ">0.3", "15"],
            ["利润质量", "经营现金流 vs 净利润符号", "一正一负", "20"],
        ])

    h(doc, "趋势规则", 3)
    table_with_header(doc,
        ["规则", "检测逻辑"],
        [
            ["存货vs营收趋势", "存货增速 > 营收增速 × 2"],
            ["应收vs营收趋势", "应收账款增速 > 营收增速"],
            ["利润vs现金流趋势", "利润持续增长但现金流持续下降"],
            ["现金流连续为负", "连续3期以上经营现金流为负"],
            ["利润突增", "净利润环比增长 > 200%"],
            ["毛利率崩盘", "毛利率较上期下降 > 10个百分点"],
        ])

    h(doc, "5.2 行业感知 + 模式组合 + 事件信号", 2)
    doc.add_paragraph(
        "V2版本在此基础上增加了三项改进：\n"
        "1. 行业相对阈值：不再使用绝对阈值（如'负债率>80%即为高风险'），"
        "而是比较该指标在同行业中的分位数，仅当其超过行业P90时才触发预警。"
        "利用研报数据中的industry_l1字段构建了31个行业的基准分布。\n"
        "2. 模式组合评分：多信号叠加时风险指数增长而非线性相加，"
        "4+信号×2.5，3信号×2.0，2信号×1.5。\n"
        "3. 外部事件信号：利用事件聚类管道中提取的立案调查/行政处罚/ST风险等作为独立权重。"
    )

    # ════════════════════════════════════
    # 六、舆情事件
    # ════════════════════════════════════
    h(doc, "六、舆情事件聚类与召回", 1)

    h(doc, "6.1 13类事件分类体系", 2)
    doc.add_paragraph(
        "基于12组关键词规则对每条公告标题进行自动分类，覆盖："
        "行政处罚、立案调查、监管警示、通报批评、监管问询、ST风险、"
        "高管违规、整改公告、股权风险、财务问题、重组重整、其他风险。"
        "每条公告最多分配2个事件类型标签。"
    )

    h(doc, "6.2 聚类与LLM标签生成", 2)
    doc.add_paragraph(
        "聚类策略：首先按主事件类型分组，对于超过200条的大型类别进一步按年份拆分为子簇。"
        "每个簇自动提取TF-IDF关键词（字符级bigram）、统计风险事件占比、"
        "记录时间跨度和涉及股票范围。可选用LLM为每个簇生成12字以内的语义标签。"
    )

    h(doc, "6.3 时间线与股权-事件交叉分析", 2)
    doc.add_paragraph(
        "为每只股票生成完整的事件时间线，将股权变更节点与舆情事件在时间轴上对齐。"
        "交叉分析自动检测同日或相近日期发生的股权变动与风险事件，"
        "为投资者提供因果推断的线索（如'实控人减持←→同日业绩预亏公告'）。"
    )

    table_with_header(doc,
        ["评测指标", "数值"],
        [
            ["公告总数", "7,311条"],
            ["风险事件", "6,429条 (87.9%)"],
            ["事件簇数", "35个"],
            ["聚类覆盖率", "100.0%"],
            ["各类型平均召回率", "99.9%"],
            ["时间线质量（Top100股票）", "99/100"],
            ["综合召回得分", "98.8%"],
        ])

    # ════════════════════════════════════
    # 七、Agent编排与RAG
    # ════════════════════════════════════
    h(doc, "七、Agent编排与RAG系统", 1)

    h(doc, "7.1 三层记忆架构", 2)
    table_with_header(doc,
        ["层级", "存储策略", "容量", "检索方式"],
        [
            ["L1 短期记忆", "最近10轮对话原文（滑动窗口）", "固定窗口", "总是注入上下文"],
            ["L2 中期记忆", "每10轮生成一次压缩摘要", "可扩展", "按需检索相关片段"],
            ["L3 长期记忆", "实体库 + 关键词索引 + Embedding向量索引", "支持0.5M+ Token", "语义RAG检索"],
        ])

    h(doc, "7.2 语义RAG检索", 2)
    doc.add_paragraph(
        "L3记忆采用Embedding → 余弦相似度 → Top-K的语义检索方案：\n"
        "1. 每条用户消息在存储时自动生成Embedding向量（优先使用LLM API embedding，"
        "不可用时回退到bigram哈希向量）\n"
        "2. 新query同样生成Embedding向量\n"
        "3. 与历史所有消息的向量计算余弦相似度\n"
        "4. 额外加入字符重叠度boost（共享字符越多分数越高，对中文金融术语友好）\n"
        "5. 返回Top-K超过阈值的结果\n"
        "当Embedding不可用时，自动降级到55个金融关键词的倒排索引检索，确保系统在任何环境下可用。"
    )

    h(doc, "7.3 自纠错闭环", 2)
    doc.add_paragraph("Agent执行工具调用时内置4级自纠错机制：")
    table_with_header(doc,
        ["纠错级别", "触发条件", "处理方式"],
        [
            ["纠错1", "股票代码缺少交易所后缀", "自动补全 .SZ/.SH/.BJ"],
            ["纠错2", "股票代码有后缀但查不到", "去掉后缀重试"],
            ["纠错3", "研报搜索返回空结果", "返回最新研报代替"],
            ["纠错4", "用户没给代码但记忆中有", "自动注入最近关注的股票代码"],
        ])

    # ════════════════════════════════════
    # 八、LLM幻觉应对
    # ════════════════════════════════════
    h(doc, "八、LLM幻觉应对机制", 1)
    doc.add_paragraph(
        "系统未显式使用'幻觉检测'模块，而是通过架构设计从源头减少幻觉产生空间："
    )

    mechanisms = [
        ("结构化输出约束",
         "LLM关键决策点（壳判定、路径评估）均要求输出JSON格式。"
         "解析失败时自动回退到规则结果，不使用LLM的不可靠输出。"),
        ("工具调用作为事实锚点（Graph-First模式）",
         "Agent先通过工具从图谱/财报中查询硬数据，再将数据注入prompt，"
         "LLM仅负责翻译和总结。这使得LLM的生成空间被限定在真实数据范围内。"),
        ("多层降级链",
         "LLM API embedding不可用 → bigram哈希 → 关键词匹配\n"
         "LLM壳判定不可用 → 规则壳评分\n"
         "LLM事件标签不可用 → TF-IDF关键词\n"
         "LLM回答生成不可用 → 直接返回结构化工具查询结果\n"
         "系统在任何一层LLM失效时都不会崩溃或输出虚假信息。"),
        ("自纠错 + 规则兜底",
         "工具调用失败时不将空结果交给LLM，而是自动修正后重试。"
         "用户最终看到的输出要么有数据支撑，要么明确告知'数据不足以回答'。"),
    ]
    for title, desc in mechanisms:
        p = doc.add_paragraph()
        run = p.add_run(f"【{title}】")
        run.font.bold = True
        p.add_run(f" {desc}")

    # ════════════════════════════════════
    # 九、全量评测
    # ════════════════════════════════════
    h(doc, "九、全量评测结果", 1)

    table_with_header(doc,
        ["指标", "全量实测", "目标", "判定"],
        [
            ["4. 股权穿透链路准确率", "99.7%", "≥85%", "✅ PASS"],
            ["5. 舆情事件簇召回率", "98.8%", "≥85%", "✅ PASS"],
            ["6. 图查询响应延迟", "P95=9ms (avg=3ms)", "≤5s", "✅ PASS"],
            ["7. 财报欺诈F1-Score", "33.9% (无监督基线)", "≥85%", "⚠️ 待标注数据"],
            ["8. 排雷报告逻辑优秀率", "100.0% (460份)", "≥80%", "✅ PASS"],
        ])

    doc.add_paragraph(
        "通过: 4/5项。指标7（财报欺诈F1）因缺乏真实监管标注数据，"
        "当前使用基于监管事件（立案调查/行政处罚）的半监督伪标签评测，"
        "F1=33.9%为无监督基线的真实水平。"
    )

    h(doc, "全量数据扫描统计", 2)
    table_with_header(doc,
        ["维度", "全量数值"],
        [
            ["股权穿透 — 上市公司全量扫描", "6,161只"],
            ["股权穿透 — 深度分布", "1层:5,263 / 2层:310 / 3层:588"],
            ["事件聚类 — 公告总数", "7,311条"],
            ["事件聚类 — 事件簇数", "35个"],
            ["财报分析 — 有效标签样本", "2,282只 (正532/负1,750)"],
            ["图查询 — 延迟测试次数", "2,000次"],
            ["排雷报告 — 生成份数", "460份"],
        ])

    # ════════════════════════════════════
    # 十、专利价值分析
    # ════════════════════════════════════
    h(doc, "十、专利价值分析", 1)

    h(doc, "10.1 推荐申请方向", 2)
    table_with_header(doc,
        ["优先级", "方向", "项目实现度", "专利可行性"],
        [
            ["🥇 第一", "时序动态图谱与财务先验规则的对齐推理",
             "已有完整实现+全量评测数据支撑", "⭐⭐⭐⭐ 较高"],
            ["🥈 第二", "Graph-First/LLM-Guided混合股权穿透方法",
             "6维壳评分+三区决策+子图序列化推理桥", "⭐⭐⭐ 可申请"],
            ["🥉 第三", "面向金融知识图谱的多源事件与股权变动对齐",
             "事件聚类+时间轴对齐+交叉分析", "⭐⭐⭐ 可申请"],
        ])

    h(doc, "10.2 不推荐申请的方向", 2)
    table_with_header(doc,
        ["方向", "原因"],
        [
            ["动态记忆检索+工具调用协同", "三层记忆+工具编排的组合已被大量公开Agent框架覆盖"],
            ["逻辑链蒸馏+评分校准", "项目未实现蒸馏（无训练循环/loss/模型保存），无法申请"],
            ["检索增强微调(RAF)", "项目完全没有微调代码，不能申请未实现的技术"],
        ])

    h(doc, "10.3 专利保护核心技术特征", 2)
    features = [
        "混合决策边界：规则处理确定区间 + LLM处理模糊区间的双层架构",
        "6维壳评分模型：实体类型、名称关键词、持股角色、持股分散度、注册资本、名称特征",
        "子图→文本序列化：将NetworkX局部有向图自动转为LLM可理解的结构化文本",
        "LLM路径置信度评估：对多条穿透链进行综合排序和自然语言解释",
        "公告聚类与股权时间轴对齐：13类事件分类 + 35个事件簇 + 与股权变更节点的交叉分析",
        "四层实体解析管线：别名词典→精确查找→模糊匹配→LLM兜底，含11K自动别名发现",
    ]
    for f in features:
        doc.add_paragraph(f, style='List Bullet')

    doc.add_paragraph()
    p = doc.add_paragraph(
        "免责声明：本报告为项目技术总结，不构成法律意见。"
        "专利申请应委托具有资质的专利代理机构进行。"
    )
    p.runs[0].font.size = Pt(8)
    p.runs[0].font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    p.runs[0].font.italic = True

    doc.save(OUT)
    print(f"已保存: {OUT}")
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"文件: {path}")
