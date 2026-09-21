"""
=============================================================================
 任务2 完整实现: 股权穿透 + 事件脉络融合
 - LLM实体关系抽取
 - 股权变更 + 舆情事件对齐
 - 事件溯源报告生成
=============================================================================
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equity_penetration import EquityGraph
from event_clustering import EventClusterer
from datetime import datetime

class EquityEventFusion:
    """股权穿透 + 事件脉络融合引擎"""

    def __init__(self):
        print("[融合] 加载股权图谱...")
        self.equity = EquityGraph()
        print("[融合] 加载事件聚类...")
        self.events = EventClusterer()
        print("[融合] 就绪")

    def llm_extract_entities(self, text, llm_client=None):
        """
        用LLM从非结构化文本中抽取实体和关系
        入: 公告标题/研报摘要
        出: [(实体1, 关系, 实体2), ...]
        """
        if llm_client is None:
            return self._rule_extract(text)

        prompt = f"""从以下金融文本中提取实体关系, 只输出JSON数组:

文本: {text}

关系类型: 控股、参股、减持、增持、质押、法人代表、关联方、处罚、问询、立案、重组

输出格式: [["实体1", "关系", "实体2"], ...]
只输出JSON, 不要解释:"""

        result = llm_client.chat([
            {"role":"system","content":"你是金融实体关系抽取器。只输出JSON数组。"},
            {"role":"user","content":prompt}
        ], temp=0, max_tokens=500, timeout=15)

        if result:
            try:
                import json
                if "```" in result:
                    result = result.split("```")[1].split("```")[0]
                return json.loads(result)
            except:
                pass
        return self._rule_extract(text)

    def _rule_extract(self, text):
        """关键词规则抽取实体关系"""
        relations = []
        text_str = str(text)

        # 持股/控股
        for pattern, rel in [
            (r'(.{2,10})减持(.{2,10})', '减持'),
            (r'(.{2,10})增持(.{2,10})', '增持'),
            (r'(.{2,10})控股(.{2,10})', '控股'),
            (r'(.{2,10})持有(.{2,10})', '持股'),
            (r'(.{2,10})质押(.{2,10})', '质押'),
        ]:
            for m in re.finditer(pattern, text_str):
                relations.append([m.group(1).strip(), rel, m.group(2).strip()])

        return relations

    def generate_traceability_report(self, stock_code, llm_client=None):
        """生成完整事件溯源报告: 股权变更 + 舆情事件对齐"""
        code = stock_code.strip()
        if '.' not in code:
            for s in ['.SH','.SZ','.BJ']:
                if (code+s) in self.equity.reverse:
                    code = code + s; break

        # 1. 股权穿透
        controllers = self.equity.find_controller(code, max_depth=4)
        flags = self.equity.check_flags(code)

        # 2. 事件时间线
        timeline = self.events.stock_timeline(code)

        # 3. 时间轴对齐
        lines = [
            f"## {code} 事件溯源报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
            "### 一、股权控制结构",
            ""
        ]

        # 分层展示
        for depth in range(1, 5):
            depth_chain = [c for c in controllers if c['depth'] == depth]
            if not depth_chain: break
            lines.append(f"#### 第{depth}层控制链 ({len(depth_chain)}条)")
            lines.append("")
            for c in depth_chain[:5]:
                tag = "个人" if c['controller_type']=='个人' else "企业"
                lines.append(f"- [{tag}] **{c['chain']}**")
                lines.append(f"  有效持股: {c['effective_pct']:.2f}%")
            lines.append("")

        if flags.get('flags'):
            lines.append("#### 股权风险信号")
            for fl in flags['flags']:
                lines.append(f"- ! **{fl['flag']}**: {fl['detail']}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append("### 二、舆情事件时间线")
        lines.append("")

        if timeline['total_events'] > 0:
            lines.append(f"总事件: {timeline['total_events']}条 | 风险占比: {timeline['risk_ratio']}")
            lines.append(f"时间跨度: {timeline['first_event']} ~ {timeline['last_event']}")
            lines.append(f"事件类型: {timeline['event_types']}")
            lines.append("")

            # 按年份分组
            events_by_year = {}
            for e in timeline['timeline']:
                year = e['date'][:4]
                events_by_year.setdefault(year, []).append(e)

            for year in sorted(events_by_year.keys(), reverse=True):
                lines.append(f"#### {year}年 ({len(events_by_year[year])}条)")
                lines.append("")
                for e in events_by_year[year][:10]:
                    risk = "!" if e['is_risk'] else " "
                    tags = ", ".join(e['types'])
                    lines.append(f"- [{risk}] **{e['date']}** [{tags}]")
                    lines.append(f"  {e['title'][:90]}")
                lines.append("")
        else:
            lines.append("无事件记录")
            lines.append("")

        # 4. 交叉分析: 股权变更节点与事件发酵时间对齐
        lines.append("---")
        lines.append("")
        lines.append("### 三、股权-事件交叉分析")
        lines.append("")

        # 抽取所有事件日期
        event_dates = set()
        for e in timeline['timeline']:
            event_dates.add(e['date'][:7])  # YYYY-MM

        # 股权相关事件
        equity_events = [e for e in timeline['timeline']
                        if any(t in ['股权风险','高管违规'] for t in e['types'])]
        other_events = [e for e in timeline['timeline']
                       if not any(t in ['股权风险','高管违规'] for t in e['types'])]

        if equity_events:
            lines.append(f"#### 股权直接相关事件 ({len(equity_events)}条)")
            lines.append("")
            for e in equity_events[:5]:
                lines.append(f"- **{e['date']}** {e['title'][:80]}")
            lines.append("")

        if other_events:
            lines.append(f"#### 经营/监管事件 ({len(other_events)}条)")
            lines.append("")
            for e in other_events[:5]:
                lines.append(f"- **{e['date']}** {e['title'][:80]}")
            lines.append("")

        # 5. LLM总结(如果有)
        if llm_client and controllers and timeline['total_events'] > 0:
            summary_prompt = f"""基于以下信息, 用3-5句话总结这家公司的风险画像:

股权结构: {controllers[0]['chain']} (有效{controllers[0]['effective_pct']:.2f}%)
事件数: {timeline['total_events']}条, 风险占比{timeline['risk_ratio']}
主要事件类型: {timeline['event_types']}

请简要总结:"""

            summary = llm_client.chat([
                {"role":"system","content":"你是金融风险分析师。简洁总结。"},
                {"role":"user","content":summary_prompt}
            ], temp=0.3, max_tokens=300, timeout=15)

            if summary:
                lines.append("---")
                lines.append("")
                lines.append("### 四、风险画像总结")
                lines.append("")
                lines.append(summary)

        return "\n".join(lines)

    def extract_and_enrich(self, stock_code, llm_client=None):
        """
        完整流程: 抽取实体 → 补全图谱 → 生成溯源报告
        """
        code = stock_code.strip()
        if '.' not in code:
            for s in ['.SH','.SZ','.BJ']:
                if (code+s) in self.equity.reverse:
                    code = code + s; break

        # 获取该股票的所有公告
        timeline = self.events.stock_timeline(code)
        relations = []

        # 从公告中抽取实体关系
        for e in timeline['timeline'][:20]:
            extracted = self.llm_extract_entities(e['title'], llm_client)
            relations.extend(extracted)

        # 去重
        unique_relations = []
        seen = set()
        for r in relations:
            key = str(r)
            if key not in seen:
                seen.add(key)
                unique_relations.append(r)

        # 生成溯源报告
        report = self.generate_traceability_report(code, llm_client)

        return {
            "stock": code,
            "extracted_relations": unique_relations,
            "relation_count": len(unique_relations),
            "report": report
        }


if __name__ == "__main__":
    fusion = EquityEventFusion()

    # 测试: 生成 603377 的溯源报告 (事件最多的股票)
    print("\n" + "=" * 60)
    print("  事件溯源报告示例: 603377.SH")
    print("=" * 60)
    report = fusion.generate_traceability_report("603377.SH")
    print(report)

    print("\n" + "=" * 60)
    print("  实体关系抽取示例")
    print("=" * 60)
    test_texts = [
        "ST东时:关于公司实际控制人收到中国证券监督管理委员会立案告知书的公告",
        "九华旅游:关于副总经理被立案审查调查并留置的公告",
        "玲珑轮胎:关于公司及相关人员收到山东证监局警示函的公告",
    ]
    for t in test_texts:
        rels = fusion.llm_extract_entities(t)
        print(f"\n文本: {t[:60]}...")
        print(f"抽取: {rels}")
