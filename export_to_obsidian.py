"""
================================================================================
 知识图谱 → Obsidian 导出器 (增强版)
 将 GraphStore 中的实体和持股关系导出为 Obsidian Markdown 文件
 利用 [[wikilink]] 和 YAML frontmatter 在 Obsidian 图谱视图中呈现
 ================================================================================
 增强功能:
 - 多级穿透链路展示（5层 BFS 穿透）
 - Mermaid 流程图可视化关键控制链
 - 壳公司检测标注
 - 强制覆盖模式 (--force)
 - 按股票代码精准导出 (--stock)
 - 金融中介噪音过滤
 - 增强 frontmatter (degree, penetration_depth, shell_score)
 ================================================================================
"""
import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Optional

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

from graph_core import GraphStore, Entity, HoldingEdge
from penetration_engine import PenetrationEngine, ShellDetector, PenetrationPath


# ============================================================================
# 配置
# ============================================================================

# 实体类型 → Obsidian 文件夹
TYPE_FOLDERS = {
    "自然人": "01-自然人",
    "民营企业": "02-民营企业",
    "国有企业": "03-国有企业",
    "境外实体": "04-境外实体",
    "基金/资管": "05-基金资管",
    "合伙企业": "06-合伙企业",
    "政府机构": "07-政府机构",
    "未知": "08-其他实体",
}

# 实体类型 → Obsidian 标签
TYPE_TAGS = {
    "自然人": "#自然人",
    "民营企业": "#民营企业",
    "国有企业": "#国有企业",
    "境外实体": "#境外实体",
    "基金/资管": "#基金/资管",
    "合伙企业": "#合伙企业",
    "政府机构": "#政府机构",
    "未知": "#其他",
}

# 实体类型 → 图标
TYPE_ICONS = {
    "自然人": "👤",
    "民营企业": "🏢",
    "国有企业": "🏛️",
    "境外实体": "🌐",
    "基金/资管": "💰",
    "合伙企业": "🤝",
    "政府机构": "🏫",
    "未知": "❓",
}

# 金融中介关键词（减少图谱噪音：这些实体只导出持股>阈值的）
NOISE_TYPES = {"基金/资管"}
NOISE_MIN_PCT = 2.0  # 持股低于此比例不导出单个链接


# ============================================================================
# Markdown 生成器
# ============================================================================

def _sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    if not name:
        return "unknown"
    forbidden = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']
    for ch in forbidden:
        name = name.replace(ch, '-')
    if len(name) > 80:
        name = name[:77] + "..."
    return name.strip()


def _entity_frontmatter(entity: Entity, extra: dict = None) -> str:
    """生成增强的 YAML frontmatter"""
    lines = ["---"]
    lines.append(f'entity_id: {entity.entity_id}')
    lines.append(f'name: "{entity.canonical_name}"')
    lines.append(f'type: {entity.entity_type}')
    lines.append(f'icon: "{TYPE_ICONS.get(entity.entity_type, "❓")}"')

    if entity.is_listed:
        lines.append(f"listed: true")
        lines.append(f'stock_code: {entity.stock_code}')
    if entity.is_transparent:
        lines.append(f"transparent: true")
    if entity.shell_score > 0.2:
        lines.append(f"shell_score: {entity.shell_score:.2f}")

    # 标签：实体类型标签 + 可能壳标签
    tags = [TYPE_TAGS.get(entity.entity_type, '#其他'), '#实体']
    if entity.shell_score >= 0.4:
        tags.append('#壳公司')
    lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")

    # 别名
    aliases_list = [a for a in entity.aliases if a != entity.canonical_name]
    if aliases_list:
        aliases_str = ", ".join(f'"{a}"' for a in list(aliases_list)[:5])
        lines.append(f"aliases: [{aliases_str}]")

    if extra:
        for k, v in extra.items():
            if isinstance(v, str):
                lines.append(f'{k}: "{v}"')
            elif isinstance(v, list):
                lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                lines.append(f"{k}: {v}")

    lines.append("---")
    return "\n".join(lines)


def _build_mermaid_diagram(
    entity_name: str,
    penetration_paths: list,
    max_chains: int = 5,
    min_pct: float = 3.0,
) -> str:
    """
    为穿透链路构建 Mermaid flowchart 图
    只展示有效持股 >= min_pct 的链路
    """
    if not penetration_paths:
        return ""

    # 收集节点和边
    nodes = set()
    edges = []
    chain_count = 0

    for path_obj in penetration_paths[:max_chains]:
        if isinstance(path_obj, dict):
            chain_nodes = path_obj.get("path", [])
            chain_pcts = path_obj.get("holding_pcts", [])
            effective = path_obj.get("effective_pct", 0)
        else:
            # PenetrationPath 对象
            chain_nodes = path_obj.node_names
            chain_pcts = path_obj.holding_pcts
            effective = path_obj.effective_pct

        if effective < min_pct and chain_count > 0:
            continue

        chain_count += 1
        # 反转路径：从控制人 → 目标公司
        for i in range(len(chain_nodes) - 1):
            src = chain_nodes[i][:15]
            tgt = chain_nodes[i + 1][:15]
            pct = chain_pcts[i] if i < len(chain_pcts) else 0

            src_id = _sanitize_filename(src).replace('-', '_').replace('.', '_')
            tgt_id = _sanitize_filename(tgt).replace('-', '_').replace('.', '_')

            nodes.add((src_id, src))
            nodes.add((tgt_id, tgt))
            edges.append((src_id, tgt_id, pct))

    if not edges:
        return ""

    lines = ["```mermaid", "graph TD"]
    # 节点定义 + 样式
    for nid, nlabel in nodes:
        # 判断节点类型给颜色
        if nlabel == entity_name[:15]:
            color = "#4285f4"  # 目标公司蓝色
        elif "有限合伙" in nlabel or "基金" in nlabel or "资管" in nlabel:
            color = "#ababab"  # 通道/壳灰色
        elif len(nlabel) <= 4:
            color = "#ff8c00"  # 自然人橙色
        elif "国" in nlabel or "政府" in nlabel or "委" in nlabel:
            color = "#db3236"  # 国有红色
        else:
            color = "#4285f4"  # 企业蓝色

        safe_label = nlabel.replace('"', "'")
        lines.append(f'    {nid}["{safe_label}"]')
        lines.append(f"    style {nid} fill:{color},stroke:#333,color:#fff")

    # 边 + 标签
    for src_id, tgt_id, pct in edges:
        lines.append(f'    {src_id} -->|"{pct:.1f}%"| {tgt_id}')

    lines.append("```")
    return "\n".join(lines)


def _entity_body(
    entity: Entity,
    store: GraphStore,
    upstream: list[dict],
    downstream: list[dict],
    controllers: list[dict] = None,
    penetration_chains: list = None,
    shell_detector: Optional[ShellDetector] = None,
) -> str:
    """生成增强的实体笔记正文（含多级穿透和 Mermaid 图）"""
    lines = []

    # 标题
    icon = TYPE_ICONS.get(entity.entity_type, "❓")
    lines.append(f"# {icon} {entity.canonical_name}")
    lines.append("")

    # 壳公司警告
    if entity.shell_score >= 0.4:
        lines.append(f"> ⚠️ **壳/通道实体** — 壳概率评分: {entity.shell_score:.2f}")
        lines.append(f"> 该实体可能为持股平台/通道，非最终控制人，建议继续向上穿透")
        lines.append("")

    # 基本信息卡片
    lines.append("## 📋 基本信息")
    lines.append("")
    lines.append("| 属性 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 实体类型 | {TYPE_ICONS.get(entity.entity_type, '')} {entity.entity_type} |")

    if entity.is_listed:
        lines.append(f"| 上市状态 | 🏢 上市公司 ({entity.stock_code}) |")
    if entity.is_transparent:
        lines.append(f"| 穿透标记 | ⚠️ 需穿透（{entity.entity_type}通常为通道/壳） |")

    if shell_detector:
        score = shell_detector.score(entity.entity_id)
        if score > 0.2:
            gauge = "█" * min(int(score * 10), 10) + "░" * max(0, 10 - int(score * 10))
            lines.append(f"| 壳概率评分 | `{gauge}` {score:.2f} |")

    # 别名
    extra_aliases = [a for a in entity.aliases if a != entity.canonical_name]
    if extra_aliases:
        lines.append(f"| 别名 | {', '.join(list(extra_aliases)[:5])} |")

    # 图谱度数
    in_deg = len(upstream)
    out_deg = len(downstream)
    lines.append(f"| 图谱度数 | ⬆️ {in_deg} 股东 / ⬇️ {out_deg} 投资 |")

    lines.append("")

    # ── 上游：谁持有我 ──
    if upstream:
        # 对于金融中介类型的上游，过滤低持股
        filtered_upstream = []
        for sh in upstream:
            stype = sh.get("source_type", "未知")
            pct = sh.get("pct", 0)
            if stype in NOISE_TYPES and pct < NOISE_MIN_PCT:
                continue
            filtered_upstream.append(sh)

        if filtered_upstream:
            lines.append("## ⬆️ 股东（谁持有我）")
            lines.append("")
            lines.append("| # | 股东 | 类型 | 持股比例 |")
            lines.append("|---|------|------|----------|")
            for i, sh in enumerate(filtered_upstream[:30], 1):
                name = sh.get("source_name", "未知")
                stype = sh.get("source_type", "未知")
                pct = sh.get("pct", 0)
                bar = "█" * max(1, int(pct / 3.3))
                link = f"[[{_sanitize_filename(name)}|{name}]]"
                icon = TYPE_ICONS.get(stype, "")
                lines.append(f"| {i} | {link} | {icon} {stype} | {pct:.2f}% {bar} |")
            lines.append("")

    # ── 下游：我持有谁 ──
    if downstream:
        filtered_down = []
        for h in downstream:
            ttype = h.get("target_type", "未知")
            pct = h.get("pct", 0)
            if ttype in NOISE_TYPES and pct < NOISE_MIN_PCT:
                continue
            filtered_down.append(h)

        if filtered_down:
            lines.append("## ⬇️ 对外投资（我持有谁）")
            lines.append("")
            lines.append("| # | 被投资方 | 类型 | 持股比例 |")
            lines.append("|---|------|------|----------|")
            for i, h in enumerate(filtered_down[:30], 1):
                name = h.get("target_name", "未知")
                ttype = h.get("target_type", "未知")
                pct = h.get("pct", 0)
                bar = "█" * max(1, int(pct / 3.3))
                link = f"[[{_sanitize_filename(name)}|{name}]]"
                icon = TYPE_ICONS.get(ttype, "")
                lines.append(f"| {i} | {link} | {icon} {ttype} | {pct:.2f}% {bar} |")
            lines.append("")

    # ── 穿透分析 ──
    if penetration_chains:
        # 按深度分组
        chains_by_depth = defaultdict(list)
        for chain in penetration_chains:
            if isinstance(chain, dict):
                depth = chain.get("depth", 0)
            else:
                depth = chain.depth
            chains_by_depth[depth].append(chain)

        lines.append("## 🔍 多级穿透链路")
        lines.append("")
        lines.append(f"> 共发现 **{len(penetration_chains)}** 条穿透路径")
        lines.append("")

        # 完整链路表格
        lines.append("| # | 深度 | 有效持股 | 控制链 | 终止原因 |")
        lines.append("|---|------|----------|--------|----------|")
        for i, chain in enumerate(penetration_chains[:15], 1):
            if isinstance(chain, dict):
                path = " → ".join(chain.get("path", []))
                depth = chain.get("depth", 0)
                eff = chain.get("effective_pct", 0)
                terminal = chain.get("terminals_at", "")
            else:
                path = " → ".join(chain.node_names)
                depth = chain.depth
                eff = chain.effective_pct
                terminal = chain.terminals_at

            # 标注路径中每个节点的类型
            lines.append(f"| {i} | {depth}层 | **{eff:.2f}%** | {path[:80]} | {terminal[:20]} |")
        lines.append("")

        # ── Mermaid 可视化 ──
        mermaid = _build_mermaid_diagram(
            entity.canonical_name,
            penetration_chains,
            max_chains=5,
            min_pct=2.0,
        )
        if mermaid:
            lines.append("## 📊 控制链可视化")
            lines.append("")
            lines.append(mermaid)
            lines.append("")

    # ── 最终控制人 ──
    if controllers:
        lines.append("## 🎯 最终控制人")
        lines.append("")

        # 分类：自然人 vs 国有 vs 其他
        person_ctrls = [c for c in controllers if c.get("controller_type") == "自然人"]
        state_ctrls = [c for c in controllers if c.get("controller_type") in ("国有企业", "政府机构")]
        other_ctrls = [c for c in controllers if c not in person_ctrls and c not in state_ctrls]

        if person_ctrls:
            lines.append("### 👤 自然人控制")
            lines.append("")
            for ctrl in person_ctrls[:5]:
                ctrl_name = ctrl.get("controller_name", "?")
                chain = ctrl.get("chain", [])
                eff_pct = ctrl.get("effective_pct", 0)
                depth = ctrl.get("depth", 0)
                pcts = ctrl.get("chain_pcts", [])
                link = f"[[{_sanitize_filename(ctrl_name)}|{ctrl_name}]]"
                lines.append(f"- **{link}** 有效持股 **{eff_pct:.2f}%** (穿透{depth}层)")
                # 逐跳展示
                chain_parts = []
                for j in range(len(chain) - 1):
                    p = pcts[j] if j < len(pcts) else 0
                    chain_parts.append(f"{chain[j][:20]} ──{p:.1f}%──▶ ")
                chain_parts.append(chain[-1][:20])
                lines.append(f"  `{''.join(chain_parts)}`")
            lines.append("")

        if state_ctrls:
            lines.append("### 🏛️ 国有控制")
            lines.append("")
            for ctrl in state_ctrls[:3]:
                ctrl_name = ctrl.get("controller_name", "?")
                eff_pct = ctrl.get("effective_pct", 0)
                depth = ctrl.get("depth", 0)
                link = f"[[{_sanitize_filename(ctrl_name)}|{ctrl_name}]]"
                lines.append(f"- **{link}** 有效持股 **{eff_pct:.2f}%** (穿透{depth}层)")
            lines.append("")

        if other_ctrls:
            lines.append("### 其他控制方")
            lines.append("")
            for ctrl in other_ctrls[:3]:
                ctrl_name = ctrl.get("controller_name", "?")
                ctrl_type = ctrl.get("controller_type", "?")
                eff_pct = ctrl.get("effective_pct", 0)
                link = f"[[{_sanitize_filename(ctrl_name)}|{ctrl_name}]]"
                lines.append(f"- **{link}** ({ctrl_type}) 有效持股 **{eff_pct:.2f}%**")
            lines.append("")

    # ── 壳检测信息 ──
    if shell_detector and entity.shell_score < 0.4:
        score = shell_detector.score(entity.entity_id)
        if score >= 0.4:
            should = shell_detector.should_penetrate_through(entity.entity_id)
            lines.append("## ⚠️ 壳/通道检测")
            lines.append("")
            lines.append(f"- 壳概率评分: **{score:.2f}** (阈值 0.4)")
            lines.append(f"- 建议穿透: {'✅ 是' if should else '❌ 否'}")
            lines.append(f"- 判断依据: 实体类型({entity.entity_type})")
            if entity.entity_type in ("基金/资管", "合伙企业"):
                lines.append(f"  - {entity.entity_type} 为强制穿透类型")
            lines.append("")

    # ── 图谱关联 Dataview ──
    lines.append("## 🔗 图谱关联")
    lines.append("")
    lines.append("```dataview")
    lines.append("TABLE type, stock_code, shell_score")
    lines.append(f"FROM [[{_sanitize_filename(entity.canonical_name)}]]")
    lines.append("WHERE file.name != this.file.name")
    lines.append("SORT type ASC")
    lines.append("```")
    lines.append("")

    # 局部图谱按钮提示
    lines.append("> 💡 **提示**: 在 Obsidian 中点击右上角「更多选项」→「打开局部图谱」")
    lines.append("> 可查看以本实体为中心的关系网络。使用标签面板筛选实体类型。")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# 导出器
# ============================================================================

class ObsidianExporter:
    """
    将 GraphStore 导出为 Obsidian 可识别的 Markdown 知识库 (增强版)
    """

    def __init__(self, store: GraphStore, output_dir: str):
        self.store = store
        self.output_dir = output_dir
        self.engine = PenetrationEngine(store)
        self.shell_detector = ShellDetector(store)

    def export_all(self, max_entities: int = 5000, force: bool = False):
        """
        导出所有实体到 Obsidian

        Args:
            max_entities: 最多导出实体数
            force: True=强制覆盖已存在的文件
        """
        print(f"[ObsidianExporter] 开始导出到 {self.output_dir}...")
        if force:
            print(f"[ObsidianExporter] 模式: 强制覆盖")

        # 创建文件夹结构
        for folder in TYPE_FOLDERS.values():
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        # 按优先级排序实体
        entities = list(self.store.entities.values())
        entities.sort(key=self._entity_priority, reverse=True)

        exported = 0
        skipped = 0
        for entity in entities[:max_entities]:
            try:
                result = self._export_entity(entity, force=force)
                if result:
                    exported += 1
                else:
                    skipped += 1
                if (exported + skipped) % 1000 == 0:
                    print(f"  已处理 {exported + skipped} 个实体 (导出 {exported}, 跳过 {skipped})...")
            except Exception as e:
                pass

        # 创建索引页
        self._export_index()

        # 创建图谱统计页
        self._export_stats()

        print(f"[ObsidianExporter] 完成! 导出 {exported}, 跳过 {skipped}")
        return exported

    def export_by_stock(self, stock_code: str, max_depth: int = 5, force: bool = True):
        """
        按股票代码精准导出：只导出该股票及其穿透链上的所有实体

        Args:
            stock_code: 股票代码（如 603439.SH）
            max_depth: 穿透深度
            force: 是否覆盖已存在的笔记
        """
        # 解析股票代码
        entity_id = self.store.find_by_stock_code(stock_code)
        if not entity_id:
            print(f"[ObsidianExporter] 未找到股票: {stock_code}")
            return 0

        entity = self.store.get_entity(entity_id)
        print(f"[ObsidianExporter] 精准导出: {entity.canonical_name} ({stock_code})")

        # 穿透获取所有关联实体
        paths = self.engine.penetrate_upward(entity_id, max_depth=max_depth)
        related_ids = {entity_id}

        for p in paths:
            for nid in p.node_ids:
                related_ids.add(nid)

        # 也获取下游
        down_paths = self.engine.penetrate_downward(entity_id, max_depth=2)
        for p in down_paths:
            for nid in p.node_ids:
                related_ids.add(nid)

        print(f"[ObsidianExporter] 穿透链涉及 {len(related_ids)} 个实体 (深度{max_depth})")

        # 确保文件夹存在
        for folder in TYPE_FOLDERS.values():
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        # 导出所有相关实体
        exported = 0
        for eid in related_ids:
            ent = self.store.get_entity(eid)
            if ent:
                if self._export_entity(ent, force=force):
                    exported += 1

        # 创建索引
        self._export_index()
        self._export_stats()

        print(f"[ObsidianExporter] 精准导出完成! 导出 {exported} 个实体")
        return exported

    def export_deep_cases(self, top_n: int = 20, min_depth: int = 3, force: bool = True):
        """
        导出穿透最深的前 N 只股票及其完整链路

        Args:
            top_n: 选取前 N 只
            min_depth: 最小穿透深度
            force: 是否强制覆盖
        """
        print(f"[ObsidianExporter] 扫描深度穿透案例...")

        # 找出所有上市公司
        listed_entities = [e for e in self.store.entities.values() if e.is_listed]
        print(f"  共 {len(listed_entities)} 家上市公司，正在评估穿透深度...")

        # 计算每只股票的穿透深度
        scored = []
        for i, ent in enumerate(listed_entities):
            paths = self.engine.penetrate_upward(ent.entity_id, max_depth=5)
            if paths:
                max_d = max(p.depth for p in paths)
                has_natural = any(
                    p.node_types[-1] == "自然人" if p.node_types else False
                    for p in paths
                )
                total_chains = len(paths)
                scored.append((ent, max_d, has_natural, total_chains))

            if (i + 1) % 500 == 0:
                print(f"    已评估 {i + 1}/{len(listed_entities)}...")

        # 排序：深度优先，其次有自然人控制
        scored.sort(key=lambda x: (-x[1], -x[3], -x[2]))
        deep_cases = [s for s in scored if s[1] >= min_depth][:top_n]

        if not deep_cases:
            print(f"  未找到深度>={min_depth}的案例，扩大搜索...")
            deep_cases = scored[:top_n]

        print(f"  选取 Top {len(deep_cases)} 深度穿透案例:")
        for ent, d, has_nat, chains in deep_cases:
            icon = "👤" if has_nat else "🏢"
            print(f"    {icon} {ent.canonical_name} — 最深{d}层, {chains}条链路")

        # 确保文件夹存在
        for folder in TYPE_FOLDERS.values():
            os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

        # 导出每个案例及其关联实体
        all_related = set()
        stock_list = []

        for ent, max_d, has_nat, chains in deep_cases:
            stock_list.append({
                "entity": ent,
                "max_depth": max_d,
                "has_natural": has_nat,
                "total_chains": chains,
            })

            paths = self.engine.penetrate_upward(ent.entity_id, max_depth=5)
            all_related.add(ent.entity_id)
            for p in paths:
                for nid in p.node_ids:
                    all_related.add(nid)

        # 导出所有关联实体
        exported = 0
        for eid in all_related:
            ent = self.store.get_entity(eid)
            if ent:
                if self._export_entity(ent, force=force):
                    exported += 1

        # 创建深度案例索引
        self._export_deep_case_index(stock_list)

        # 创建其他索引
        self._export_index()
        self._export_stats()

        print(f"[ObsidianExporter] 深度案例导出完成! {len(stock_list)} 只股票, {exported} 个实体")
        return exported, stock_list

    def _entity_priority(self, entity: Entity) -> int:
        """实体导出优先级"""
        score = 0
        if entity.is_listed:
            score += 1000
        if entity.entity_type == "自然人":
            score += 500
        if entity.entity_type == "国有企业":
            score += 300
        degree = self.store.G.degree(entity.entity_id)
        score += min(degree, 200)
        return score

    def _export_entity(self, entity: Entity, force: bool = False) -> bool:
        """导出单个实体。返回 True=已导出, False=跳过"""
        folder = TYPE_FOLDERS.get(entity.entity_type, "08-其他实体")
        filename = _sanitize_filename(entity.canonical_name) + ".md"
        filepath = os.path.join(self.output_dir, folder, filename)

        # 已存在且不强制覆盖 → 跳过
        if os.path.exists(filepath) and not force:
            return False

        # 获取关系数据
        upstream = self.store.get_neighbors(entity.entity_id, "in")
        downstream = self.store.get_neighbors(entity.entity_id, "out")

        # ── 穿透分析（所有实体都尝试穿透） ──
        controllers = None
        penetration_chains = None
        if entity.is_listed or entity.entity_type not in ("自然人", "政府机构"):
            try:
                paths = self.engine.penetrate_upward(entity.entity_id, max_depth=5)
                if paths:
                    penetration_chains = paths
                    controllers = self.engine.get_ultimate_controllers(
                        entity.entity_id, max_depth=5
                    )
            except Exception:
                pass

        # 计算 extra frontmatter
        extra = {}
        in_deg = len(upstream)
        out_deg = len(downstream)
        extra["in_degree"] = in_deg
        extra["out_degree"] = out_deg
        if penetration_chains:
            max_pen_depth = max(
                (p.depth if hasattr(p, 'depth') else p.get("depth", 0))
                for p in penetration_chains
            )
            extra["max_penetration_depth"] = max_pen_depth

        # 生成内容
        frontmatter = _entity_frontmatter(entity, extra)
        body = _entity_body(
            entity, self.store,
            upstream, downstream,
            controllers=controllers,
            penetration_chains=penetration_chains,
            shell_detector=self.shell_detector,
        )
        content = frontmatter + "\n\n" + body

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return True

    def _export_deep_case_index(self, stock_list: list):
        """生成深度穿透案例索引页"""
        lines = []
        lines.append("---")
        lines.append("tags: [MOC, 穿透案例, 深度分析]")
        lines.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("---")
        lines.append("")
        lines.append("# 🔍 深度穿透案例索引")
        lines.append("")
        lines.append(f"> 自动筛选穿透深度 ≥ 3 层的典型案例，共 {len(stock_list)} 只")
        lines.append("")
        lines.append("## 案例列表")
        lines.append("")
        lines.append("| # | 上市公司 | 最深穿透 | 有自然人 | 链路数 |")
        lines.append("|---|----------|----------|----------|--------|")

        for i, case in enumerate(stock_list, 1):
            ent = case["entity"]
            d = case["max_depth"]
            has_nat = "👤 是" if case["has_natural"] else "🏢 否"
            chains = case["total_chains"]
            folder = TYPE_FOLDERS.get(ent.entity_type, "08-其他实体")
            link = f"[[{folder}/{_sanitize_filename(ent.canonical_name)}|{ent.canonical_name}]]"
            lines.append(f"| {i} | {link} | {d}层 | {has_nat} | {chains}条 |")

        lines.append("")
        lines.append("## 使用说明")
        lines.append("")
        lines.append("1. 点击上方链接进入具体上市公司的穿透分析笔记")
        lines.append("2. 查看 **Mermaid 流程图** 直观理解控制链结构")
        lines.append("3. 在 **图谱视图** 中观察节点间的关系网络")
        lines.append("4. 使用 **标签面板** 筛选不同类型的实体")
        lines.append("")

        filepath = os.path.join(self.output_dir, "穿透案例深度分析.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_index(self):
        """生成总索引页 (MOC)"""
        type_counts = defaultdict(int)
        listed = []
        top_natural = []
        top_state = []

        for entity in self.store.entities.values():
            type_counts[entity.entity_type] += 1
            if entity.is_listed:
                listed.append(entity)
            if entity.entity_type == "自然人":
                degree = self.store.G.degree(entity.entity_id)
                if degree >= 3:
                    top_natural.append((entity, degree))
            if entity.entity_type == "国有企业":
                degree = self.store.G.degree(entity.entity_id)
                if degree >= 5:
                    top_state.append((entity, degree))

        top_natural.sort(key=lambda x: -x[1])
        top_state.sort(key=lambda x: -x[1])

        lines = []
        lines.append("---")
        lines.append("tags: [MOC, 索引, 股权穿透]")
        lines.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("---")
        lines.append("")
        lines.append("# 🏦 股权穿透知识图谱 — 总索引")
        lines.append("")
        lines.append(f"> 📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"> 📊 实体总数: **{self.store.entity_count():,}** | 关系总数: **{self.store.edge_count():,}**")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 🧭 快速导航")
        lines.append("")
        lines.append("| 目标 | 入口 |")
        lines.append("|------|------|")
        lines.append("| 📖 图谱使用说明 | [[图谱使用指南]] |")
        lines.append("| 🔍 深度穿透案例 | [[穿透案例深度分析]] |")
        lines.append("| 📈 统计数据 | [[99-图谱统计]] |")
        lines.append("| 🌐 打开图谱视图 | 点击左侧边栏「图谱视图」图标 |")
        lines.append("")

        # 按类型的实体分布
        lines.append("## 📊 实体分布")
        lines.append("")
        lines.append("| 类型 | 数量 | 文件夹 | 图谱颜色 |")
        lines.append("|------|------|--------|----------|")
        colors = {
            "自然人": "🟠 橙色", "民营企业": "🔵 蓝色",
            "国有企业": "🔴 红色", "境外实体": "🟢 绿色",
            "基金/资管": "⚪ 灰色", "合伙企业": "🟣 紫色",
            "政府机构": "🟡 黄色", "未知": "⚫ 黑色",
        }
        for etype, folder in TYPE_FOLDERS.items():
            count = type_counts.get(etype, 0)
            color = colors.get(etype, "")
            lines.append(f"| {TYPE_ICONS.get(etype, '')} {etype} | {count:,} | [[{folder}/\\|{folder}]] | {color} |")
        lines.append("")

        # 上市公司列表
        lines.append(f"## 🏢 上市公司 ({len(listed)})")
        lines.append("")
        # 按代码分组显示（每个市场一个子列表）
        for market, market_name in [(".SH", "上海主板"), (".SZ", "深圳"), (".BJ", "北交所")]:
            market_stocks = [e for e in listed if e.stock_code and market in e.stock_code]
            if market_stocks:
                lines.append(f"### {market_name} ({len(market_stocks)}只)")
                lines.append("")
                for ent in sorted(market_stocks, key=lambda e: e.stock_code or "")[:30]:
                    folder = TYPE_FOLDERS.get(ent.entity_type, "08-其他实体")
                    link = f"[[{folder}/{_sanitize_filename(ent.canonical_name)}|{ent.canonical_name}]]"
                    lines.append(f"- {link}")
                if len(market_stocks) > 30:
                    lines.append(f"- ... 还有 {len(market_stocks) - 30} 只")
                lines.append("")

        # 重要自然人和国有企业
        if top_natural:
            lines.append(f"## 👤 重要自然人股东 (Top {min(30, len(top_natural))})")
            lines.append("")
            for ent, deg in top_natural[:30]:
                folder = TYPE_FOLDERS.get(ent.entity_type, "08-其他实体")
                link = f"[[{folder}/{_sanitize_filename(ent.canonical_name)}|{ent.canonical_name}]]"
                lines.append(f"- {link} (关联 {deg} 条)")
            lines.append("")

        # 图谱视图使用提示
        lines.append("## 💡 图谱视图技巧")
        lines.append("")
        lines.append("1. **颜色区分**: 不同颜色代表不同实体类型（见上方颜色表）")
        lines.append("2. **节点大小**: 关联越多节点越大 → 快速定位核心实体")
        lines.append("3. **局部图谱**: 打开任意笔记 → 右上角「更多」→「打开局部图谱」查看邻域")
        lines.append("4. **标签筛选**: 在图谱视图中使用标签面板过滤特定类型实体")
        lines.append("5. **搜索聚焦**: 在图谱视图中搜索实体名 → 高亮显示")
        lines.append("6. **拖拽探索**: 拖拽节点重新布局，发现隐藏的关系结构")
        lines.append("")

        filepath = os.path.join(self.output_dir, "00-股权穿透知识图谱.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _export_stats(self):
        """生成图谱统计页"""
        stats = self.store.stats()

        # 计算壳实体数量
        shell_count = sum(
            1 for eid in self.store.entities
            if self.shell_detector.is_shell(eid)
        )

        lines = []
        lines.append("---")
        lines.append("tags: [统计, 仪表盘]")
        lines.append(f"created: {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("---")
        lines.append("")
        lines.append("# 📈 图谱统计仪表盘")
        lines.append("")
        lines.append(f"> 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("## 核心指标")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 实体总数 | {stats['total_entities']:,} |")
        lines.append(f"| 持股关系总数 | {stats['total_edges']:,} |")
        lines.append(f"| 上市公司数 | {len(self.store.stock_index):,} |")
        lines.append(f"| 疑似壳/通道实体 | {shell_count:,} ({shell_count/max(1, stats['total_entities'])*100:.1f}%) |")
        lines.append("")

        lines.append("## 实体类型分布 (Dataview 动态)")
        lines.append("")
        lines.append("```dataview")
        lines.append("TABLE length(rows) as 数量")
        lines.append("FROM #实体")
        lines.append("GROUP BY type")
        lines.append("SORT rows.length DESC")
        lines.append("```")
        lines.append("")

        lines.append("## 上市公司 (Dataview 动态)")
        lines.append("")
        lines.append("```dataview")
        lines.append("TABLE stock_code, type, max_penetration_depth as 穿透深度")
        lines.append("FROM #实体")
        lines.append("WHERE listed = true")
        lines.append("SORT stock_code ASC")
        lines.append("```")
        lines.append("")

        lines.append("## 壳公司列表 (Dataview 动态)")
        lines.append("")
        lines.append("```dataview")
        lines.append("TABLE type, shell_score as 壳评分")
        lines.append("FROM #壳公司")
        lines.append("SORT shell_score DESC")
        lines.append("LIMIT 50")
        lines.append("```")
        lines.append("")

        filepath = os.path.join(self.output_dir, "99-图谱统计.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============================================================================
# 快速入口
# ============================================================================

def export_to_obsidian(
    store: GraphStore,
    output_dir: str = None,
    max_entities: int = 3000,
    force: bool = False,
):
    """
    一键导出到 Obsidian

    Args:
        store: 已构建的 GraphStore
        output_dir: 输出目录
        max_entities: 最大导出实体数
        force: 是否强制覆盖
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "obsidian-kb",
        )

    exporter = ObsidianExporter(store, output_dir)
    return exporter.export_all(max_entities=max_entities, force=force)


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    from entity_resolver import build_graph_from_shareholders

    print("=" * 60)
    print("  知识图谱 → Obsidian 导出工具 (增强版)")
    print("=" * 60)

    # 解析参数
    force = "--force" in sys.argv
    stock_code = None
    batch_stocks = []
    top_n = None
    max_entities = 3000

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--force":
            force = True
        elif args[i] == "--stock" and i + 1 < len(args):
            stock_code = args[i + 1]
            i += 1
        elif args[i] == "--batch" and i + 1 < len(args):
            batch_stocks = args[i + 1].split(",")
            i += 1
        elif args[i] == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1])
            i += 1
        elif args[i] == "--max" and i + 1 < len(args):
            max_entities = int(args[i + 1])
            i += 1
        i += 1

    # 1. 加载数据
    t0 = time.time()
    print("\n[1/3] 加载股东数据...")
    try:
        df = pd.read_excel("data_raw/2.股东持股-股权穿透/clean.xlsx")
        print(f"  加载了 {len(df):,} 行")
    except FileNotFoundError:
        print("  数据文件未找到! 请确认 data_raw/2.股东持股-股权穿透/clean.xlsx 存在")
        sys.exit(1)

    # 2. 构建图谱
    print("\n[2/3] 构建股权知识图谱...")
    store, aliases = build_graph_from_shareholders(df)
    t1 = time.time()
    print(f"  耗时: {t1 - t0:.1f}s")
    print(f"  实体: {store.entity_count():,} | 边: {store.edge_count():,}")

    # 3. 导出
    print("\n[3/3] 导出到 Obsidian...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obsidian-kb")
    exporter = ObsidianExporter(store, output_dir)

    if stock_code:
        # 单只股票精准导出
        num = exporter.export_by_stock(stock_code, max_depth=5, force=force)
        print(f"\n✅ 精准导出完成! {num} 个实体 → {output_dir}")
    elif batch_stocks:
        # 批量导出
        total = 0
        for sc in batch_stocks:
            n = exporter.export_by_stock(sc.strip(), max_depth=5, force=force)
            total += n
        print(f"\n✅ 批量导出完成! 总计 {total} 个实体")
    elif top_n:
        # 深度案例导出
        num, cases = exporter.export_deep_cases(top_n=top_n, force=force)
        print(f"\n✅ 深度案例导出完成! {num} 个实体")
    else:
        # 全量导出
        num = export_to_obsidian(store, output_dir, max_entities=max_entities, force=force)
        print(f"\n✅ 全量导出完成! {num} 个实体 → {output_dir}")

    t2 = time.time()
    print(f"总耗时: {t2 - t0:.1f}s")
    print()
    print("📂 在 Obsidian 中打开 obsidian-kb/ 文件夹作为 Vault")
    print("🗺️  点击左侧「图谱视图」查看股权关系网络")
    print("🎨 使用标签面板按实体类型筛选")
