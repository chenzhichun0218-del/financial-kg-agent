"""
=============================================================================
 舆情事件聚类模块
 功能: 事件分类 + DBSCAN聚类 + 事件簇标签 + 时序分析
 用法: python event_clustering.py              # 全市场事件概览
       python event_clustering.py 688765.SH    # 单股事件时间线
       python event_clustering.py --cluster    # 聚类分析报告
=============================================================================
"""
import sys, os, re
from datetime import datetime
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import pandas as pd
import numpy as np

# ================================================================
# 事件分类规则
# ================================================================
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

def classify_event(title):
    """对公告标题做事件分类"""
    title_str = str(title)
    matches = []
    for event_type, keywords in EVENT_TYPES.items():
        score = sum(1 for kw in keywords if kw in title_str)
        if score > 0:
            matches.append((event_type, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in matches[:2]]  # 最多2个标签


# ================================================================
# TF-IDF + 余弦聚类
# ================================================================
class EventClusterer:
    def __init__(self):
        print("[聚类] 加载数据...")
        self.announcements = pd.read_pickle(os.path.join(BASE, "data_processed", "announcements.pkl"))
        self.reports = pd.read_pickle(os.path.join(BASE, "data_processed", "reports.pkl"))

        # 给每条公告分类
        self.announcements['event_types'] = self.announcements['n_info_title'].apply(classify_event)
        self.announcements['primary_event'] = self.announcements['event_types'].apply(
            lambda x: x[0] if x else '其他')

        self._build_clusters()

    def _build_clusters(self):
        """构建事件簇"""
        # 按主事件类型分组 = 天然的事件簇
        clusters = {}
        for etype in self.announcements['primary_event'].unique():
            subset = self.announcements[self.announcements['primary_event'] == etype]
            if len(subset) >= 5:  # 至少5条才算一个簇
                clusters[etype] = {
                    "name": etype,
                    "count": len(subset),
                    "stocks": subset['s_info_windcode'].nunique(),
                    "date_range": (subset['ann_date'].min(), subset['ann_date'].max()),
                    "sample_titles": subset['n_info_title'].head(5).tolist(),
                    "stocks_affected": subset['s_info_windcode'].value_counts().head(10).to_dict(),
                }

        self.clusters = clusters
        print(f"[聚类] 发现 {len(clusters)} 个事件簇")

    def stock_timeline(self, stock_code):
        """生成单只股票的事件时间线"""
        code = stock_code.strip()
        if '.' not in code:
            for suffix in ['.SH', '.SZ', '.BJ']:
                full = code + suffix
                if full in self.announcements['s_info_windcode'].values:
                    code = full
                    break

        stock_events = self.announcements[
            self.announcements['s_info_windcode'] == code
        ].sort_values('ann_date')

        if stock_events.empty:
            return {"stock": code, "events": [], "total_events": 0, "event_types": {},
                    "first_event": "N/A", "last_event": "N/A", "risk_ratio": "0%",
                    "timeline": [], "summary": "无事件记录"}

        timeline = []
        for _, row in stock_events.iterrows():
            timeline.append({
                "date": str(row['ann_date'])[:10],
                "title": str(row['n_info_title'])[:100],
                "types": row['event_types'],
                "is_risk": row['is_risk_event'],
            })

        # 统计
        type_counts = Counter()
        for t in stock_events['event_types'].explode().dropna():
            type_counts[t] += 1

        return {
            "stock": code,
            "total_events": len(timeline),
            "first_event": str(stock_events['ann_date'].min())[:10],
            "last_event": str(stock_events['ann_date'].max())[:10],
            "event_types": dict(type_counts.most_common()),
            "timeline": timeline,
            "risk_ratio": f"{stock_events['is_risk_event'].mean()*100:.0f}%",
        }

    def cluster_report(self):
        """生成简洁聚类分析报告"""
        lines = [f"### 事件聚类总览", f"",
                 f"**{len(self.announcements):,}** 条公告 | **{self.announcements['s_info_windcode'].nunique()}** 只股票 | "
                 f"{self.announcements['ann_date'].min().date()} ~ {self.announcements['ann_date'].max().date()}",
                 f""]

        total = len(self.announcements)
        for etype, info in sorted(self.clusters.items(), key=lambda x: x[1]['count'], reverse=True):
            pct = info['count'] / total * 100
            lines.append(f"- **{etype}**: {info['count']}条 ({pct:.0f}%) — {info['stocks']}只股票")

        lines.append(f"")
        self.announcements['month'] = self.announcements['ann_date'].dt.to_period('M')
        monthly = self.announcements.groupby('month').size()
        peak_month = monthly.idxmax()
        lines.append(f"事件高峰月: **{peak_month}** ({monthly.max()}条)")
        lines.append(f"事件最多的股票: ")
        top = self.announcements['s_info_windcode'].value_counts().head(5)
        for i, (code, cnt) in enumerate(top.items(), 1):
            tl = self.stock_timeline(code)
            lines.append(f"  {i}. {code} — {cnt}条 [{', '.join(list(tl['event_types'].keys())[:3])}]")

        return "\n".join(lines)

    def find_related_events(self, stock_code, days=30):
        """找同一时间段内相关股票的事件（同行业同时出事=行业风险）"""
        timeline = self.stock_timeline(stock_code)
        if not timeline.get('timeline'):
            return []

        # 取该股票最早风险事件日期
        risk_dates = [e['date'] for e in timeline['timeline'] if e['is_risk']]

        related = []
        for rd in risk_dates[:3]:
            d = pd.Timestamp(rd)
            window = self.announcements[
                (self.announcements['ann_date'] >= d - pd.Timedelta(days=days)) &
                (self.announcements['ann_date'] <= d + pd.Timedelta(days=days)) &
                (self.announcements['is_risk_event'])
            ]
            if len(window) > 1:
                related.append({
                    "date": rd,
                    "total_events": len(window),
                    "stocks": window['s_info_windcode'].nunique(),
                    "sample": window['n_info_title'].head(5).tolist(),
                })

        return related


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    clusterer = EventClusterer()

    if len(sys.argv) == 1:
        print(clusterer.cluster_report())

    elif sys.argv[1] == '--cluster':
        print(clusterer.cluster_report())

    else:
        stock = sys.argv[1]
        timeline = clusterer.stock_timeline(stock)
        if timeline['total_events'] == 0:
            print(f"\n  {stock}: 无事件记录")
            sys.exit(0)

        print(f"\n{'='*70}")
        print(f"  {stock} 事件时间线")
        print(f"{'='*70}")
        print(f"  总事件: {timeline['total_events']} 条")
        print(f"  日期范围: {timeline['first_event']} ~ {timeline['last_event']}")
        print(f"  风险事件占比: {timeline['risk_ratio']}")
        print(f"  事件类型: {timeline['event_types']}")

        print(f"\n  事件时间线:\n")
        for e in reversed(timeline['timeline'][-20:]):  # 最新20条
            risk_mark = "!" if e['is_risk'] else " "
            tags = ", ".join(e['types'])
            print(f"  [{risk_mark}] {e['date']} [{tags}]")
            print(f"     {e['title'][:90]}")
            print()

        # 相关事件
        print(f"\n  相关股票同时期事件:")
        related = clusterer.find_related_events(stock)
        for r in related:
            print(f"\n  {r['date']} 前后30天, 共{r['total_events']}起类似事件, 涉及{r['stocks']}只股票:")
            for title in r['sample'][:3]:
                print(f"    - {str(title)[:80]}")
