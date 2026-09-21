"""
生成本项目专利价值分析报告 Word 文档
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os
import sys
from datetime import datetime

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "专利价值分析报告.docx")


def set_cell_shading(cell, color):
    """设置单元格底色"""
    shading_elm = cell._element.get_or_add_tcPr()
    shading = shading_elm.makeelement(qn('w:shd'), {
        qn('w:fill'): color,
        qn('w:val'): 'clear',
    })
    shading_elm.append(shading)


def add_heading_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x1a, 0x47, 0x8a)
    return h


def build_doc():
    doc = Document()

    # 页面设置
    sect = doc.sections[0]
    sect.page_width = Cm(21)
    sect.page_height = Cm(29.7)
    sect.left_margin = Cm(2.5)
    sect.right_margin = Cm(2.5)

    # ── 封面标题 ──
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.space_after = Pt(6)
    run = title.add_run("金融AI智能助手项目\n专利价值分析报告")
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x0d, 0x2b, 0x5e)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("—— 图计算+LLM混合驱动的股权穿透与事件融合系统")
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.space_before = Pt(30)
    run = info.add_run(f"生成日期: {datetime.now().strftime('%Y年%m月%d日')}\n"
                       f"项目路径: 知识图谱项目/\n"
                       f"技术栈: Python + NetworkX + LLM (OpenAI SDK) + Obsidian")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    doc.add_page_break()

    # ── 目录 ──
    add_heading_styled(doc, "目录", 1)
    toc = [
        "一、项目技术架构全景",
        "二、值得申请专利的创新点",
        "    2.1 第一优先级：图计算+LLM混合股权穿透方法",
        "    2.2 第二优先级：股权变动与舆情事件融合对齐方法",
        "    2.3 第三优先级：分级记忆的长对话金融Agent架构",
        "三、不建议申请专利的部分",
        "四、与现有商业产品的对比分析",
        "五、专利申请实操建议",
    ]
    for item in toc:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)

    doc.add_page_break()

    # ═══════════════════════════════════════════════
    # 一、项目技术架构全景
    # ═══════════════════════════════════════════════
    add_heading_styled(doc, "一、项目技术架构全景", 1)

    doc.add_paragraph(
        "本项目构建了一套基于Agentic AI的金融长上下文推理与知识图谱系统，"
        "核心由以下五层架构组成："
    )

    layers = [
        ("数据层", "646,449条股东数据 + 公司公告 + 三大财务报表 + 券商研报 → 数据清洗与实体标准化"),
        ("图计算层", "80,433实体 → NetworkX有向图 → BFS/DFS多跳穿透 → 有效持股累乘 → 6维壳评分"),
        ("LLM判断层", "图计算→子图文本序列化→LLM壳判定（模糊区间）→LLM路径置信度评估"),
        ("Agent编排层", "意图识别 → 工具选择(Function Calling) → 执行 → 验证 → 自纠错闭环"),
        ("可视化层", "Obsidian知识图谱视图 + Mermaid流程图 + Dataview动态查询 + Canvas交互画布"),
    ]
    for name, desc in layers:
        p = doc.add_paragraph()
        run = p.add_run(f"【{name}】")
        run.font.bold = True
        p.add_run(f" {desc}")

    doc.add_paragraph()
    p = doc.add_paragraph(
        "其中，LLM判断层和Agent编排层是本项目的核心创新所在，"
        "也是专利保护的重点。以下按专利价值从高到低逐一分析。"
    )
    p.runs[0].font.italic = True

    # ═══════════════════════════════════════════════
    # 二、值得申请专利的创新点
    # ═══════════════════════════════════════════════
    add_heading_styled(doc, "二、值得申请专利的创新点", 1)

    # ── 2.1 第一优先级 ──
    add_heading_styled(doc, "2.1 🥇 第一优先级：图计算+LLM混合股权穿透方法", 2)

    doc.add_paragraph(
        "对应代码文件: graph_llm_bridge.py (llm_shell_judge函数，第114-195行)\n"
        "              penetration_engine.py (PenetrationEngine类)\n"
        "              graph_core.py (GraphStore / ShellDetector类)"
    )

    add_heading_styled(doc, "2.1.1 技术问题", 3)
    doc.add_paragraph(
        "现有股权穿透技术（如天眼查、企查查）采用纯图算法进行BFS/DFS多跳穿透，"
        "存在以下不足："
    )
    problems = [
        "一刀切过滤：对所有金融中介（基金、券商、社保等）采用关键词硬过滤，无法区分真正的通道实体和有实际控制意图的投资实体",
        "壳公司识别困难：投资管理公司、有限合伙企业等可能是持股平台（应穿透），也可能是有实际业务的运营实体（不应穿透），纯规则无法准确区分",
        "路径不可解释：只给出一条或多条穿透链，无路径置信度评估，用户无法判断哪条链更可靠",
        "边缘案例处理弱：评分在0.3~0.6之间的模糊实体，规则判断经常出错",
    ]
    for prob in problems:
        doc.add_paragraph(prob, style='List Bullet')

    add_heading_styled(doc, "2.1.2 技术创新", 3)
    doc.add_paragraph(
        "提出「Graph-First, LLM-Guided」双层决策架构，核心技术方案如下："
    )

    doc.add_paragraph(
        "创新点A — 混合决策边界机制", style='List Number'
    )
    doc.add_paragraph(
        "将壳公司识别分为三个决策区间：\n"
        "  • 确定穿透区间（壳评分 > 0.6 或实体类型为基金/资管/合伙企业）→ 规则直接判定穿透\n"
        "  • 确定终止区间（壳评分 < 0.2 或实体类型为自然人/政府/国企）→ 规则直接判定终止\n"
        "  • 模糊区间（壳评分 0.2~0.6）→ 触发LLM辅助判断\n\n"
        "这种设计使得：90%以上的实体由规则快速处理（毫秒级），仅约10%的模糊案例调用LLM，"
        "在准确率和成本之间取得最优平衡。"
    )

    doc.add_paragraph(
        "创新点B — 6维壳评分模型", style='List Number'
    )

    table = doc.add_table(rows=7, cols=3)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["维度", "评分规则", "权重"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell, "1a478a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    dims = [
        ("实体类型", "基金/资管+0.6, 合伙企业+0.4", "基础分"),
        ("名称关键词", "含'投资管理/股权投资/私募基金'等+0.2", "特征分"),
        ("持股角色", "仅为持股方从未被持股+0.15", "角色分"),
        ("持股分散度", "持有10+公司且80%持股<5%+0.2", "行为分"),
        ("注册资本", "注册资本<100万+0.1", "规模分"),
        ("名称特征", "含'有限合伙'+0.15", "结构分"),
    ]
    for i, (d, r, w) in enumerate(dims, 1):
        table.rows[i].cells[0].text = d
        table.rows[i].cells[1].text = r
        table.rows[i].cells[2].text = w

    doc.add_paragraph()

    doc.add_paragraph(
        "创新点C — 子图→文本序列化的LLM推理桥", style='List Number'
    )
    doc.add_paragraph(
        "将局部图谱结构（实体节点+持股边+属性）自动序列化为LLM可读的结构化文本，"
        "包含：实体类型、上市状态、壳评分、上下游股东列表及持股比例。\n"
        "这种序列化使得LLM能够基于真实的图结构数据做判断，而非凭空猜测。"
        "同时，LLM也对多条穿透路径进行置信度评估和排序，输出带标注的可靠路径列表。"
    )

    add_heading_styled(doc, "2.1.3 与现有技术的本质区别", 3)

    table = doc.add_table(rows=7, cols=3)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["对比维度", "天眼查/企查查/同花顺", "本项目"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell, "1a478a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    rows = [
        ("核心引擎", "纯图算法（BFS/DFS）", "图算法+LLM双层决策"),
        ("壳识别方式", "关键词硬过滤", "6维规则评分+LLM定性判断"),
        ("穿透深度", "通常2-3层", "可解释的5层穿透"),
        ("路径可信度", "无评估", "LLM置信度排序+自然语言解释"),
        ("模糊案例处理", "一刀切（误判率高）", "LLM边界判定（准确率高）"),
        ("可解释性", "低（只给结果，不说理由）", "高（自然语言逐链说明穿透逻辑）"),
    ]
    for i, (dim, old, new) in enumerate(rows, 1):
        table.rows[i].cells[0].text = dim
        table.rows[i].cells[1].text = old
        table.rows[i].cells[2].text = new

    doc.add_paragraph()

    doc.add_paragraph(
        "建议专利名称: 一种基于图计算与大语言模型协同的股权穿透分析方法及系统",
        style='Intense Quote'
    )

    # ── 2.2 第二优先级 ──
    add_heading_styled(doc, "2.2 🥈 第二优先级：股权变动与舆情事件融合对齐方法", 2)

    doc.add_paragraph(
        "对应代码文件: equity_event_fusion.py (EquityEventFusion类)\n"
        "              event_pipeline.py (EventPipeline类)"
    )

    add_heading_styled(doc, "2.2.1 技术问题", 3)
    doc.add_paragraph(
        "现有金融信息平台（东方财富、同花顺等）将股权信息和公告信息作为独立模块展示，"
        "用户需要手动对照日期，自行判断股权变动与舆情事件的关联关系。存在以下痛点："
    )
    problems2 = [
        "信息孤岛：股权结构变化和舆情事件（监管处罚、资产重组等）分别展示，无交叉分析",
        "因果识别难：实控人减持前后的监管问询、立案调查等事件与股权变动的因果关系需要人工梳理",
        "时间对齐困难：手动对照数十条公告和股权变更记录的时间线极为耗时",
        "实体关系缺失：公告中的非结构化文本包含了大量实体关系（控股、减持、质押等），但这些信息未被结构化利用",
    ]
    for prob in problems2:
        doc.add_paragraph(prob, style='List Bullet')

    add_heading_styled(doc, "2.2.2 技术创新", 3)
    doc.add_paragraph(
        "提出「公告聚类→实体抽取→时间轴对齐→交叉分析」四阶段融合方法："
    )

    steps = [
        ("第一阶段 — 事件聚类",
         "公告文本 → Embedding向量化 → UMAP降维 → HDBSCAN聚类 → LLM生成事件簇标签。"
         "自动将数十万条公告聚合为可管理的事件簇（如'监管处罚'、'资产重组'、'高管变动'等）。"),
        ("第二阶段 — 实体关系抽取",
         "LLM从非结构化公告文本中抽取（实体1，关系，实体2）三元组，"
         "关系类型包括：控股、参股、减持、增持、质押、法人代表、关联方、处罚、问询、立案、重组。"
         "LLM不可用时自动回退到正则规则抽取。"),
        ("第三阶段 — 时间轴对齐",
         "将股权变更节点（持股比例变化、股东进出）与舆情事件簇在时间轴上对齐。"
         "单只股票的事件时间线同时展示[股权变更]和[舆情事件]两类标签。"),
        ("第四阶段 — 交叉分析与溯源报告",
         "识别股权变动前后的相关舆情事件，自动分析因果关系。"
         "LLM综合股权结构+事件时间线，生成3-5句风险画像总结。"
         "输出包含：股权控制结构分析 + 舆情事件完整时间线 + 股权-事件交叉分析 + 风险画像总结。"),
    ]
    for title, desc in steps:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.font.bold = True
        doc.add_paragraph(f"    {desc}")

    doc.add_paragraph(
        "建议专利名称: 一种面向金融知识图谱的多源事件与股权变动对齐方法及装置",
        style='Intense Quote'
    )

    # ── 2.3 第三优先级 ──
    add_heading_styled(doc, "2.3 🥉 第三优先级：分级记忆的长对话金融Agent架构", 2)

    doc.add_paragraph(
        "对应代码文件: agent_v3.py (TieredMemory类)\n"
        "              main.py (FinanceAgent类)"
    )

    doc.add_paragraph(
        "三层记忆架构设计："
    )
    mem_layers = [
        ("L1 短期记忆（滑动窗口）", "保留最近10轮对话原文，确保当前上下文的精确性。总长受窗口大小限制。"),
        ("L2 中期记忆（LLM压缩摘要）", "每10轮对话由LLM自动生成一次压缩摘要，保留核心信息而非逐字记录。"
         "支持按需检索相关历史片段。"),
        ("L3 长期记忆（关键词索引+实体库）", "自动提取用户关注过的股票代码、实体名称、金融关键词，建立检索索引。"
         "当用户问题涉及历史实体时，自动检索并注入上下文。"),
    ]
    for title, desc in mem_layers:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.font.bold = True
        doc.add_paragraph(f"    {desc}")

    doc.add_paragraph(
        "此外，Agent包含自纠错闭环：意图识别 → 工具选择 → 执行 → 结果验证 → "
        "如失败自动重新选择工具或调整参数。工具层以OpenAI Function Calling标准定义了"
        "股权穿透查询、事件时间线获取、财报分析、研报检索等多个工具。"
    )
    doc.add_paragraph(
        "注意：此方向的通用架构（分层记忆、自纠错Agent）在业界已有较多公开方案，"
        "但金融垂直领域的特定实现（金融关键词索引、股权实体库、财务分析工具链）仍有差异化专利申请空间。",
    )
    p.runs[0].font.italic = True

    # ═══════════════════════════════════════════════
    # 三、不建议申请专利的部分
    # ═══════════════════════════════════════════════
    add_heading_styled(doc, "三、不建议申请专利的部分", 1)

    table = doc.add_table(rows=7, cols=3)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["模块/功能", "技术方法", "不建议原因"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell, "c0392b")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    skip_rows = [
        ("BFS多跳穿透算法", "标准BFS/DFS图遍历", "天眼查/企查查已广泛使用，属于公知技术"),
        ("财报反欺诈规则", "跨科目勾稽关系检查", "会计准则的已知检查项（存货/营收、现金流/利润等）"),
        ("文本聚类", "TF-IDF + HDBSCAN", "标准NLP方法，学术界/工业界通用方案"),
        ("Function Calling", "OpenAI 工具调用标准", "OpenAI定义的标准接口模式，无排他性"),
        ("Obsidian可视化", "Markdown + Wikilink + Canvas", "只是工具应用，非算法/系统创新"),
        ("有效持股累乘计算", "多层持股比例连乘", "基础数学方法"),
    ]
    for i, (mod, tech, reason) in enumerate(skip_rows, 1):
        table.rows[i].cells[0].text = mod
        table.rows[i].cells[1].text = tech
        table.rows[i].cells[2].text = reason

    doc.add_paragraph()

    # ═══════════════════════════════════════════════
    # 四、与现有商业产品的对比
    # ═══════════════════════════════════════════════
    add_heading_styled(doc, "四、与现有商业产品的对比分析", 1)

    table = doc.add_table(rows=7, cols=5)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["功能维度", "天眼查", "企查查", "同花顺/东方财富", "本项目"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell, "1a478a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    comp_rows = [
        ("股权穿透深度", "2-3层", "2-3层", "1-2层（仅展示）", "5层可解释穿透"),
        ("壳公司识别", "关键词过滤", "关键词过滤", "无", "6维评分+LLM判断"),
        ("路径可信度", "无", "无", "无", "LLM置信度排序"),
        ("事件-股权融合", "无", "无", "分开展示", "聚类对齐+交叉分析"),
        ("可解释性", "低", "低", "低", "自然语言解释每条链"),
        ("智能对话", "无", "无", "无", "LLM Agent多工具编排"),
    ]
    for i, (dim, tian, qi, tong, ours) in enumerate(comp_rows, 1):
        table.rows[i].cells[0].text = dim
        table.rows[i].cells[1].text = tian
        table.rows[i].cells[2].text = qi
        table.rows[i].cells[3].text = tong
        table.rows[i].cells[4].text = ours
        # 高亮本项目的优势列
        set_cell_shading(table.rows[i].cells[4], "e8f5e9")

    doc.add_paragraph()

    # ═══════════════════════════════════════════════
    # 五、专利申请实操建议
    # ═══════════════════════════════════════════════
    add_heading_styled(doc, "五、专利申请实操建议", 1)

    add_heading_styled(doc, "5.1 推荐专利申请组合", 2)

    table = doc.add_table(rows=3, cols=4)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["优先级", "建议专利名称", "类型", "保护范围"]
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        cell.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell, "1a478a")
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xff, 0xff, 0xff)

    patent_rows = [
        ("第一", "一种基于图计算与大语言模型协同的股权穿透分析方法及系统",
         "发明专利", "双层决策架构 + 6维壳评分 + 子图序列化推理桥 + 路径置信度评估"),
        ("第二", "一种面向金融知识图谱的多源事件与股权变动对齐方法及装置",
         "发明专利", "事件聚类 + LLM实体抽取 + 时间轴对齐 + 交叉分析溯源"),
    ]
    for i, (pri, name, ptype, scope) in enumerate(patent_rows, 1):
        table.rows[i].cells[0].text = pri
        table.rows[i].cells[1].text = name
        table.rows[i].cells[2].text = ptype
        table.rows[i].cells[3].text = scope

    doc.add_paragraph()

    add_heading_styled(doc, "5.2 申请流程建议", 2)
    steps_apply = [
        "第一步 — 新颖性检索（1-2周）：在中国专利数据库（CNIPA）、SOOPAT、Google Patents中检索类似技术方案，确认'图计算+LLM双层决策'、'6维壳评分'、'子图序列化推理桥'等关键特征的首次公开时间",
        "第二步 — 专利文书撰写（2-4周）：委托专利代理机构撰写申请文件。重点在权利要求书中完整覆盖混合决策边界的技术路线，避免被绕过。附上graph_llm_bridge.py的核心代码作为实施例",
        "第三步 — 提交申请（1周）：通过CPC电子申请系统向国家知识产权局提交",
        "第四步 — 审查答复（6-18个月）：应对审查意见，可能需要限定权利要求范围",
        "⚠️ 重要提醒：如果计划发表学术论文，务必先提交专利申请再投稿，否则论文公开后会破坏专利的新颖性",
    ]
    for step in steps_apply:
        doc.add_paragraph(step)

    add_heading_styled(doc, "5.3 专利保护的核心技术特征", 2)
    features = [
        "混合决策边界机制：规则处理确定区间 + LLM处理模糊区间的双层架构",
        "6维壳评分模型：实体类型、名称关键词、持股角色、持股分散度、注册资本、名称特征的加权评分",
        "子图→文本序列化方法：将NetworkX局部有向图自动转为LLM可理解的结构化文本",
        "LLM路径置信度评估：对多条穿透链进行综合排序和自然语言解释",
        "公告聚类与股权时间轴对齐：HDBSCAN聚类 + LLM标签 + 与股权变更节点的交叉分析",
    ]
    for f in features:
        doc.add_paragraph(f, style='List Bullet')

    doc.add_paragraph()

    # ── 免责声明 ──
    doc.add_paragraph()
    p = doc.add_paragraph(
        "免责声明：本报告仅为技术分析建议，不构成法律意见。专利申请应委托具有资质的专利代理机构进行，"
        "并由专业代理人撰写申请文件。本报告的专利性分析基于对现有公开技术的了解，"
        "最终是否能够获得授权取决于国家知识产权局的审查结果。"
    )
    p.runs[0].font.size = Pt(8)
    p.runs[0].font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    p.runs[0].font.italic = True

    # ── 保存 ──
    doc.save(OUT)
    print(f"✅ 已保存: {OUT}")
    return OUT


if __name__ == "__main__":
    path = build_doc()
    print(f"文件路径: {path}")
