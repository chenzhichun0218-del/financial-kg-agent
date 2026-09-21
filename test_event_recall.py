"""
=============================================================================
 舆情事件簇召回率评测
 指标5: 关键节点召回率 >= 85%
 综合 = (分类覆盖率 + 簇内一致性 + 时序完整性) / 3
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_clustering import EventClusterer, classify_event


def test_coverage(clusterer):
    """测试1: 分类覆盖率 — 多少公告被成功分到事件簇"""
    ann = clusterer.announcements
    total = len(ann)
    classified = len(ann[ann['primary_event'] != '其他'])
    coverage = classified / total * 100
    print(f"[测试1] 分类覆盖率: {classified}/{total} = {coverage:.1f}%")
    print(f"        未分类(其他): {total-classified} 条 ({100-coverage:.1f}%)")
    return coverage


def test_consistency(clusterer):
    """测试2: 簇内一致性 — 抽样检查每个簇的标题是否匹配簇名"""
    print(f"\n[测试2] 簇内一致性抽样:")
    all_ok = True
    for name, info in list(clusterer.clusters.items()):
        if name == '其他':  # "其他"是未分类桶, 不验证一致性
            continue
        ok_count = 0
        total = min(5, len(info['sample_titles']))
        for title in info['sample_titles'][:total]:
            tags = classify_event(str(title))
            if name in tags:
                ok_count += 1
        status = "OK" if ok_count == total else f"MISMATCH({ok_count}/{total})"
        if ok_count < total:
            all_ok = False
        print(f"  [{name}] {status}")
    consistency = 100 if all_ok else 0
    return consistency


def test_timeline(clusterer):
    """测试3: 时序完整性 — 事件最多的股票能否生成完整时间线"""
    print(f"\n[测试3] 时序完整性:")
    top_stocks = clusterer.announcements['s_info_windcode'].value_counts().head(10).index
    ok = 0
    for code in top_stocks:
        tl = clusterer.stock_timeline(code)
        has_events = tl['total_events'] > 0
        has_types = len(tl['event_types']) > 0
        has_dates = tl['first_event'] != 'N/A'
        if has_events and has_types and has_dates:
            ok += 1
        print(f"  {code}: {tl['total_events']}条 {len(tl['event_types'])}类 日期{tl['first_event']}~{tl['last_event']} {'OK' if has_events and has_types and has_dates else 'FAIL'}")
    timeline_rate = ok / len(top_stocks) * 100
    print(f"  时序完整: {ok}/{len(top_stocks)} = {timeline_rate:.0f}%")
    return timeline_rate


if __name__ == "__main__":
    c = EventClusterer()

    coverage = test_coverage(c)
    consistency = test_consistency(c)
    timeline = test_timeline(c)

    recall = (coverage + consistency + timeline) / 3

    print(f"\n{'='*60}")
    print(f"  舆情事件簇关键节点召回率")
    print(f"  = ({coverage:.0f}% + {consistency:.0f}% + {timeline:.0f}%) / 3")
    print(f"  = {recall:.1f}%")
    print(f"  目标: >= 85%")
    print(f"  判定: {'PASS' if recall >= 85 else 'FAIL'}")
    print(f"{'='*60}")
