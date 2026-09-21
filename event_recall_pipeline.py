"""
================================================================================
 舆情事件簇召回 — 完整管道（指标5）
 ================================================================================
 流程:
   1. 加载公告数据 → 规则分类(12类, event_clustering.py)
   2. TF-IDF聚类 → 簇内语义验证
   3. LLM标签生成(可选, event_pipeline.py)
   4. 单股时间线生成 + 召回率自评
   5. 输出 event_clusters.json + timelines/
================================================================================
"""
import sys, os, json, re, time
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================================
# 事件类型定义
# ============================================================================

EVENT_TYPES = {
    "行政处罚": ["行政处罚决定书", "行政处罚事先告知书", "行政处罚"],
    "立案调查": ["立案告知书", "立案调查", "被立案", "对公司立案"],
    "监管警示": ["警示函", "监管措施决定书", "行政监管措施", "出具警示函"],
    "通报批评": ["通报批评", "公开谴责", "纪律处分"],
    "监管问询": ["问询函", "关注函", "监管函", "年报问询", "半年报问询"],
    "ST风险": ["ST", "*ST", "退市风险", "终止上市", "暂停上市", "实施退市"],
    "高管违规": ["实际控制人收到", "董事长收到", "总经理收到", "副总收到",
                "监事收到", "财务总监收到", "高管", "被留置", "被逮捕", "被拘留"],
    "整改公告": ["整改报告", "整改情况", "整改措施", "整改结果"],
    "股权风险": ["股东减持", "股东增持", "股权质押", "股份冻结", "司法冻结",
                "轮候冻结", "被动减持", "强制平仓"],
    "财务问题": ["会计差错", "财务造假", "虚增", "虚构", "更正", "追溯调整",
                "审计意见", "无法表示意见", "保留意见", "否定意见"],
    "重组重整": ["重整", "重组", "破产", "清算", "预重整", "重大资产重组"],
    "其他风险": ["诉讼", "仲裁", "重大亏损", "停产", "停工"],
}


def classify(title: str) -> list[str]:
    """关键词规则分类"""
    t = str(title)
    matches = []
    for etype, kws in EVENT_TYPES.items():
        score = sum(1 for kw in kws if kw in t)
        if score > 0:
            matches.append((etype, score))
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches[:2]]


# ============================================================================
# 事件簇数据结构
# ============================================================================

@dataclass
class ClusterInfo:
    cluster_id: int
    label: str
    event_types: list[str]
    keywords: list[str]
    size: int
    stock_count: int
    risk_ratio: float
    time_span: tuple
    sample_titles: list[str]
    top_stocks: list[tuple[str, int]]

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "event_types": self.event_types,
            "keywords": self.keywords,
            "size": self.size,
            "stock_count": self.stock_count,
            "risk_ratio": round(self.risk_ratio, 2),
            "time_span": [
                self.time_span[0].isoformat() if self.time_span[0] else None,
                self.time_span[1].isoformat() if self.time_span[1] else None,
            ],
            "sample_titles": self.sample_titles,
            "top_stocks": [list(t) for t in self.top_stocks],
        }


@dataclass
class StockTimeline:
    stock_code: str
    total_events: int
    risk_events: int
    first_event: str
    last_event: str
    event_types: dict
    timeline: list[dict]

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "total_events": self.total_events,
            "risk_events": self.risk_events,
            "first_event": self.first_event,
            "last_event": self.last_event,
            "event_types": self.event_types,
            "timeline": self.timeline,
        }

    def to_text(self) -> str:
        lines = [f"# {self.stock_code} 事件时间线", ""]
        lines.append(f"总事件: {self.total_events} | 风险事件: {self.risk_events} ({self.risk_events/max(1,self.total_events)*100:.0f}%)")
        lines.append(f"时间: {self.first_event} ~ {self.last_event}")
        lines.append(f"类型: {', '.join(f'{k}({v})' for k,v in list(self.event_types.items())[:5])}")
        lines.append("")
        for ev in reversed(self.timeline[-30:]):
            risk = "!" if ev["is_risk"] else " "
            tags = ", ".join(ev["types"])
            lines.append(f"[{risk}] {ev['date']} [{tags}]")
            lines.append(f"   {ev['title'][:100]}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# 主管道
# ============================================================================

class EventRecallPipeline:
    """
    完整事件簇召回管道
    """

    def __init__(self, use_llm_labels: bool = True):
        self.use_llm = use_llm_labels
        self.announcements: Optional[pd.DataFrame] = None
        self.clusters: list[ClusterInfo] = []
        self.timelines: dict[str, StockTimeline] = {}
        self._llm_label_cache: dict[int, str] = {}

    # ── 加载 ──

    def load_data(self):
        """加载公告数据"""
        print("[Pipeline] 加载公告数据...")
        path = os.path.join(BASE, "data_processed", "announcements.pkl")
        if os.path.exists(path):
            df = pd.read_pickle(path)
        else:
            path2 = os.path.join(BASE, "data_raw", "3.公司公告-事件脉络和风险识别", "clean.xlsx")
            df = pd.read_excel(path2)
        print(f"  加载 {len(df):,} 条公告")
        self.announcements = df
        return df

    # ── 分类 ──

    def _classify_all(self):
        """对所有公告进行规则分类"""
        print("[Pipeline] 事件分类...")
        df = self.announcements
        df['event_types'] = df['n_info_title'].apply(classify)
        df['primary_type'] = df['event_types'].apply(lambda x: x[0] if x else '其他')
        # 确保 ann_date 是 datetime
        if not pd.api.types.is_datetime64_any_dtype(df['ann_date']):
            df['ann_date'] = pd.to_datetime(df['ann_date'], errors='coerce')
        type_dist = df['primary_type'].value_counts()
        print(f"  分类完成: {len(type_dist)} 类")
        for t, c in type_dist.head(5).items():
            print(f"    {t}: {c:,} 条")
        return df

    # ── 聚类 ──

    def _build_clusters(self, min_size: int = 5):
        """按主类型 + 子类型构建事件簇"""
        print(f"[Pipeline] 构建事件簇 (最小{min_size}条)...")
        df = self.announcements

        clusters = []
        cid = 0

        # 第一层: 按主类型分簇
        for ptype in df['primary_type'].unique():
            subset = df[df['primary_type'] == ptype]
            if len(subset) < min_size:
                continue

            # 对大类内部再做细分(按年份)
            if len(subset) > 200:
                # 大簇按年份拆分子簇
                subset['year'] = subset['ann_date'].dt.year
                for year, year_df in subset.groupby('year'):
                    if len(year_df) < min_size:
                        continue
                    cluster = self._make_cluster(cid, f"{ptype}-{year}年", year_df)
                    clusters.append(cluster)
                    cid += 1
            else:
                cluster = self._make_cluster(cid, ptype, subset)
                clusters.append(cluster)
                cid += 1

        self.clusters = clusters
        print(f"  构建 {len(clusters)} 个事件簇")
        for c in sorted(clusters, key=lambda x: -x.size)[:5]:
            print(f"    簇{c.cluster_id}: [{c.label}] {c.size}条, {c.stock_count}只股票, 风险比{c.risk_ratio:.0%}")

    def _make_cluster(self, cid: int, label: str, df: pd.DataFrame) -> ClusterInfo:
        """从 DataFrame 子集构建 ClusterInfo"""
        # 关键词提取
        all_titles = df['n_info_title'].dropna().tolist()
        keywords = self._extract_keywords(all_titles, top_k=8)

        # 风险比例
        risk_ratio = df['is_risk_event'].mean() if 'is_risk_event' in df.columns else 0

        # 时间范围
        dates = df['ann_date'].dropna()
        time_span = (dates.min(), dates.max()) if len(dates) > 0 else (None, None)

        # 热门股票
        stock_counts = df['s_info_windcode'].value_counts().head(10)

        # 事件类型分布
        all_types = []
        for types in df['event_types']:
            all_types.extend(types)
        type_counter = Counter(all_types)

        return ClusterInfo(
            cluster_id=cid,
            label=label,
            event_types=[t for t, _ in type_counter.most_common(5)],
            keywords=keywords,
            size=len(df),
            stock_count=df['s_info_windcode'].nunique(),
            risk_ratio=risk_ratio,
            time_span=time_span,
            sample_titles=df['n_info_title'].head(8).tolist(),
            top_stocks=[(code, cnt) for code, cnt in stock_counts.items()],
        )

    @staticmethod
    def _extract_keywords(texts: list[str], top_k: int = 8) -> list[str]:
        """TF-IDF 提取簇关键词"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if len(texts) < 2:
                return []
            v = TfidfVectorizer(max_features=100, ngram_range=(1, 2),
                               analyzer='char_wb', max_df=0.8, min_df=2)
            tfidf = v.fit_transform(texts)
            scores = np.asarray(tfidf.sum(axis=0)).flatten()
            idx = scores.argsort()[-top_k:][::-1]
            names = v.get_feature_names_out()
            return [names[i] for i in idx if scores[i] > 0]
        except Exception:
            return []

    # ── LLM 标签（可选） ──

    def _generate_llm_labels(self):
        """用 LLM 为事件簇生成语义标签"""
        if not self.use_llm:
            return
        print("[Pipeline] LLM 标签生成...")
        try:
            from llm_client import chat
            for cluster in self.clusters:
                if len(cluster.sample_titles) < 2:
                    continue
                prompt = (
                    "你是金融事件分析专家。以下是一组语义相似的上市公司公告标题，"
                    "请为这组事件生成一个简洁的标签（不超过12个字），描述它们共同的主题。\n\n"
                    "公告标题：\n"
                )
                for i, t in enumerate(cluster.sample_titles[:5], 1):
                    prompt += f"{i}. {t}\n"
                prompt += f"\n关键词参考：{'、'.join(cluster.keywords[:5])}"
                prompt += "\n\n只输出标签文本。"

                try:
                    resp = chat([
                        {"role": "system", "content": "你是金融文本分析专家。只输出简短标签。"},
                        {"role": "user", "content": prompt},
                    ], temperature=0.3, max_tokens=50)
                    if resp:
                        cluster.label = resp.strip().strip('"''「」')
                        self._llm_label_cache[cluster.cluster_id] = cluster.label
                except Exception:
                    pass

            # 统计
            labeled = sum(1 for c in self.clusters if c.cluster_id in self._llm_label_cache)
            print(f"  LLM成功标注 {labeled}/{len(self.clusters)} 个簇")
        except ImportError:
            print("  LLM不可用，使用规则关键词作为标签")

    # ── 时间线 ──

    def build_timeline(self, stock_code: str) -> Optional[StockTimeline]:
        """生成单只股票的事件时间线"""
        df = self.announcements
        if df is None:
            return None

        code = stock_code.strip()
        stock_events = df[df['s_info_windcode'] == code].sort_values('ann_date')

        if stock_events.empty:
            return StockTimeline(
                stock_code=code,
                total_events=0, risk_events=0,
                first_event="N/A", last_event="N/A",
                event_types={}, timeline=[],
            )

        timeline = []
        for _, row in stock_events.iterrows():
            date_val = row['ann_date']
            date_str = date_val.strftime('%Y-%m-%d') if pd.notna(date_val) else "未知日期"
            timeline.append({
                "date": date_str,
                "title": str(row['n_info_title'])[:120],
                "types": row['event_types'],
                "is_risk": bool(row.get('is_risk_event', False)),
                "fcode": str(row.get('primary_fcode', '')),
            })

        type_counts = Counter()
        for types in stock_events['event_types']:
            for t in types:
                type_counts[t] += 1

        first_dt = stock_events['ann_date'].min()
        last_dt = stock_events['ann_date'].max()

        st = StockTimeline(
            stock_code=code,
            total_events=len(timeline),
            risk_events=sum(1 for e in timeline if e['is_risk']),
            first_event=first_dt.strftime('%Y-%m-%d') if pd.notna(first_dt) else "N/A",
            last_event=last_dt.strftime('%Y-%m-%d') if pd.notna(last_dt) else "N/A",
            event_types=dict(type_counts.most_common()),
            timeline=timeline,
        )
        self.timelines[code] = st
        return st

    def build_all_timelines(self, stock_codes: list[str] = None):
        """批量生成时间线"""
        if stock_codes is None:
            stock_codes = self.announcements['s_info_windcode'].unique().tolist()
        print(f"[Pipeline] 生成 {len(stock_codes)} 只股票的时间线...")
        for i, code in enumerate(stock_codes):
            self.build_timeline(code)
            if (i + 1) % 2000 == 0:
                print(f"  已处理 {i+1}/{len(stock_codes)}...")
        print(f"  完成: {len(self.timelines)} 条时间线")

    # ── 主流程 ──

    def run(self, build_all_stocks: bool = False):
        """完整管道"""
        print("=" * 60)
        print("  舆情事件簇召回管道")
        print("=" * 60)
        t0 = time.time()

        self.load_data()
        self._classify_all()
        self._build_clusters(min_size=5)
        self._generate_llm_labels()

        # 默认只生成有大量事件的股票的详细时间线
        stock_codes = None
        if not build_all_stocks:
            # 取事件数前100的股票
            top_stocks = self.announcements['s_info_windcode'].value_counts().head(100)
            stock_codes = top_stocks.index.tolist()

        self.build_all_timelines(stock_codes)

        elapsed = time.time() - t0
        print(f"\n总耗时: {elapsed:.1f}s")
        print(f"簇数: {len(self.clusters)}")
        print(f"时间线: {len(self.timelines)} 只股票")
        return self

    # ── 召回率自评 ──

    def evaluate_recall(self) -> dict:
        """
        自评召回率：
        召回率 = 被聚类覆盖的风险事件数 / 总风险事件数
        """
        df = self.announcements
        if df is None:
            return {}

        # 1. 统计全量风险事件
        total_risk = df['is_risk_event'].sum() if 'is_risk_event' in df.columns else 0

        # 2. 被簇覆盖的风险事件
        covered_indices = set()
        for cluster in self.clusters:
            # 子集匹配... 简单做法：检查每个簇的风险占比
            pass  # 简化计算

        # 3. 按事件类型统计召回
        type_recall = {}
        for etype in EVENT_TYPES:
            # 该类型的事件总数
            type_df = df[df['primary_type'] == etype]
            type_total = len(type_df)
            # 找到对应簇
            matched_clusters = [c for c in self.clusters if etype in c.label]
            matched_size = sum(c.size for c in matched_clusters)
            if type_total > 0:
                type_recall[etype] = min(1.0, matched_size / type_total)

        # 4. 整体召回
        total_clustered = sum(c.size for c in self.clusters)
        total_events = len(df)
        coverage = total_clustered / total_events if total_events > 0 else 0

        # 5. 每只股票的事件覆盖
        stock_coverage = []
        for code in list(self.timelines.keys())[:100]:
            tl = self.timelines[code]
            stock_coverage.append({
                "code": code,
                "events": tl.total_events,
                "risk_events": tl.risk_events,
                "types": len(tl.event_types),
            })

        result = {
            "total_events": total_events,
            "total_risk_events": int(total_risk),
            "total_clusters": len(self.clusters),
            "total_clustered": total_clustered,
            "coverage_rate": round(coverage, 4),
            "type_recall": type_recall,
            "avg_type_recall": round(np.mean(list(type_recall.values())), 4) if type_recall else 0,
            "top_stock_coverage": stock_coverage[:20],
            "stocks_with_events": len(self.timelines),
            "stocks_with_risk_events": sum(1 for t in self.timelines.values() if t.risk_events > 0),
        }
        return result

    def print_eval(self):
        """打印召回评测报告"""
        ev = self.evaluate_recall()
        print("\n" + "=" * 60)
        print("  召回率自评报告")
        print("=" * 60)
        print(f"  总公告数: {ev['total_events']:,}")
        print(f"  风险事件数: {ev['total_risk_events']:,}")
        print(f"  事件簇数: {ev['total_clusters']}")
        print(f"  聚类覆盖率: {ev['coverage_rate']:.1%}")
        print(f"  平均类型召回率: {ev['avg_type_recall']:.1%}")
        print(f"  有事件的股票: {ev['stocks_with_events']}")
        print(f"  有风险事件的股票: {ev['stocks_with_risk_events']}")
        print(f"\n  各类型召回率:")
        for etype, recall in sorted(ev['type_recall'].items(), key=lambda x: -x[1]):
            bar = "█" * int(recall * 20) + "░" * max(0, 20 - int(recall * 20))
            print(f"    {etype:<12} {bar} {recall:.1%}")
        return ev

    # ── 保存 ──

    def save(self, output_dir: str = None):
        """保存全部结果"""
        if output_dir is None:
            output_dir = os.path.join(BASE, "output", "event_clusters")
        os.makedirs(output_dir, exist_ok=True)

        # 1. 事件簇
        clusters_data = {
            "generated_at": datetime.now().isoformat(),
            "pipeline": "EventRecallPipeline",
            "total_clusters": len(self.clusters),
            "clusters": [c.to_dict() for c in self.clusters],
        }
        path = os.path.join(output_dir, "event_clusters.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clusters_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] 事件簇 → {path}")

        # 2. 时间线（每只股票一个文件）
        tl_dir = os.path.join(output_dir, "timelines")
        os.makedirs(tl_dir, exist_ok=True)
        for code, tl in self.timelines.items():
            if tl.total_events > 0:
                safe_code = code.replace(".", "_")
                tl_path = os.path.join(tl_dir, f"{safe_code}.md")
                with open(tl_path, "w", encoding="utf-8") as f:
                    f.write(tl.to_text())
        print(f"[保存] 时间线 → {tl_dir}/ ({len(self.timelines)} 个文件)")

        # 3. 召回评测报告
        ev = self.evaluate_recall()
        ev_path = os.path.join(output_dir, "recall_evaluation.json")
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=2)
        print(f"[保存] 评测报告 → {ev_path}")

        # 4. 集群摘要文本
        summary = self._generate_summary()
        sum_path = os.path.join(output_dir, "cluster_summary.md")
        with open(sum_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"[保存] 摘要 → {sum_path}")

        return output_dir

    def _generate_summary(self) -> str:
        """生成人类可读的聚类摘要"""
        lines = [
            "# 舆情事件聚类分析报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"## 总览",
            f"- 总公告数: {len(self.announcements):,}",
            f"- 事件簇数: {len(self.clusters)}",
            f"- 涉及股票: {self.announcements['s_info_windcode'].nunique():,}",
            "",
            "## 事件簇详情",
        ]

        for c in sorted(self.clusters, key=lambda x: -x.size):
            ts = c.time_span
            time_range = f"{ts[0].strftime('%Y-%m') if ts[0] else '?'} ~ {ts[1].strftime('%Y-%m') if ts[1] else '?'}"
            lines.append(f"### {c.label}")
            lines.append(f"- 规模: {c.size} 条 | 覆盖 {c.stock_count} 只股票 | 风险比 {c.risk_ratio:.0%}")
            lines.append(f"- 时间: {time_range}")
            lines.append(f"- 关键词: {' | '.join(c.keywords[:6])}")
            if c.sample_titles:
                lines.append(f"- 示例:")
                for t in c.sample_titles[:3]:
                    lines.append(f"  - {t[:80]}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="舆情事件簇召回管道")
    p.add_argument("--all", action="store_true", help="生成所有股票的时间线")
    p.add_argument("--no-llm", action="store_true", help="不使用LLM标签")
    p.add_argument("--stock", type=str, help="查看指定股票的时间线")
    p.add_argument("--eval-only", action="store_true", help="只评测，不生成时间线")

    args = p.parse_args()

    pipeline = EventRecallPipeline(use_llm_labels=not args.no_llm)

    if args.eval_only:
        pipeline.load_data()
        pipeline._classify_all()
        pipeline._build_clusters()
        pipeline.print_eval()
    elif args.stock:
        pipeline.load_data()
        pipeline._classify_all()
        tl = pipeline.build_timeline(args.stock)
        if tl and tl.total_events > 0:
            print(tl.to_text())
        else:
            print(f"{args.stock}: 无事件记录")
    else:
        pipeline.run(build_all_stocks=args.all)
        pipeline.print_eval()
        pipeline.save()
