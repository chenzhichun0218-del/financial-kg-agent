"""
================================================================================
 财报反欺诈评测 — 基于监管事件簇的半监督标签
 ================================================================================
 思路: 不用 fra_detector 自己的 risk_score 做标签(循环论证),
       改用外部监管事件(立案/处罚/ST)作为近似正例。
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 标签构建
# ============================================================================

def build_labels():
    """
    用监管事件簇构建半监督标签:

    POSITIVE (真实有问题):
      同时满足:
        1. 有 立案调查 或 行政处罚 事件 (监管已确认)
        2. 有 ST风险 或 财务问题 事件 (财务已暴露)

    NEGATIVE (大概率干净):
      满足:
        1. 在公告数据库中, 但 is_risk_event == 0 (无任何风险事件)
        2. 所有公告都是中性事件 (年报发布、分红公告等)

    EXCLUDED (不参与评测):
      - 只有监管警示/问询 (可能是误报)
      - 只有通报批评 (不确定)
      - 啥事件都没有的冷门股票
    """

    # 加载
    ann = pd.read_pickle(os.path.join(BASE, "data_processed", "announcements.pkl"))
    fin = pd.read_pickle(os.path.join(BASE, "data_processed", "financials_merged.pkl"))

    # 确保日期格式
    if not pd.api.types.is_datetime64_any_dtype(ann['ann_date']):
        ann['ann_date'] = pd.to_datetime(ann['ann_date'], errors='coerce')

    # 分类每条公告
    from event_recall_pipeline import classify as event_classify
    ann['event_types'] = ann['n_info_title'].apply(event_classify)
    ann['primary_type'] = ann['event_types'].apply(lambda x: x[0] if x else '其他')

    # 每只股票的事件类型汇总
    stock_events = defaultdict(set)
    stock_all_types = defaultdict(list)
    for _, row in ann.iterrows():
        code = row['s_info_windcode']
        stock_events[code].add(row['primary_type'])
        stock_all_types[code].extend(row['event_types'])

    # 构建标签
    POSITIVE_TYPES = {'立案调查', '行政处罚'}      # 监管确认有问题
    NEGATIVE_TYPES = {'监管问询', '监管警示', '通报批评'}  # 可能是误报
    STRONG_RISK = {'ST风险', '财务问题'}            # 财务已暴露

    positives = []  # 有问题的
    negatives = []  # 干净的
    excluded = []   # 不确定

    for code, etypes in stock_events.items():
        has_positive = bool(etypes & POSITIVE_TYPES)
        has_strong = bool(etypes & STRONG_RISK)
        has_negative_only = bool(etypes & NEGATIVE_TYPES) and not has_positive and not has_strong
        is_all_clean = (etypes == {'其他'}) or (len(etypes) == 0)

        if has_positive and has_strong:
            # 立案/处罚 + ST/财务问题 = 高置信度有问题
            positives.append(code)
        elif has_positive:
            # 只有立案/处罚，没有ST → 可能还没到那一步，但有监管确认
            positives.append(code)
        elif is_all_clean or has_negative_only:
            # 完全没风险 或 仅有问询/警示 = 大概率干净
            negatives.append(code)
        else:
            excluded.append(code)

    # 过滤：只保留在财务数据中也存在的股票
    fin_codes = set(fin['stock_code'].unique())
    positives = [c for c in positives if c in fin_codes]
    negatives = [c for c in negatives if c in fin_codes]

    print(f"标签构建完成:")
    print(f"  POSITIVE (有问题): {len(positives)} 只")
    print(f"  NEGATIVE (干净):   {len(negatives)} 只")
    print(f"  EXCLUDED (不确定): {len(excluded)} 只")
    print(f"  (其中有财务数据的: P={len(positives)}, N={len(negatives)})")

    return positives, negatives, excluded, stock_events


# ============================================================================
# 评测
# ============================================================================

def evaluate_fraud_detector(positive_codes: list[str], negative_codes: list[str]):
    """
    用外部标签评测 FraudDetector 的 F1 分数。
    不再用 risk_score 定义标签 → 避免循环论证。
    """
    from fraud_detector import FraudDetector

    fd = FraudDetector()
    all_codes = positive_codes + negative_codes

    y_true = []
    y_pred = []

    for code in all_codes:
        r = fd.analyze(code)
        if r.get('error'):
            continue

        # True label: 来自监管事件, 不是来自 risk_score
        is_positive = code in positive_codes
        y_true.append(1 if is_positive else 0)

        # Predicted: risk_score >= 阈值 → 预警
        threshold = 30  # 风险评分 >= 30 即预警
        y_pred.append(1 if r['risk_score'] >= threshold else 0)

    if len(y_true) == 0:
        return {"error": "无可用样本"}

    # 计算指标
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0

    return {
        "total_samples": len(y_true),
        "positive_samples": sum(y_true),
        "negative_samples": len(y_true) - sum(y_true),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "threshold_used": threshold,
    }


def print_report(result: dict):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("  财报反欺诈评测报告 (基于监管事件标签)")
    print("=" * 60)
    print(f"  样本总数: {result['total_samples']}")
    print(f"  正例(有问题): {result['positive_samples']}")
    print(f"  负例(干净):   {result['negative_samples']}")
    print(f"  阈值: risk_score >= {result['threshold_used']}")
    print()
    print(f"  True Positive:  {result['true_positive']}")
    print(f"  False Positive: {result['false_positive']}")
    print(f"  False Negative: {result['false_negative']}")
    print(f"  True Negative:  {result['true_negative']}")
    print()
    print(f"  Precision: {result['precision']:.1%}")
    print(f"  Recall:    {result['recall']:.1%}")
    print(f"  F1-Score:  {result['f1']:.1%}")
    print(f"  Accuracy:  {result['accuracy']:.1%}")

    f1_pct = result['f1'] * 100
    if f1_pct >= 85:
        print(f"\n  判断: PASS (F1={f1_pct:.1f}% >= 85%)")
    else:
        print(f"\n  判断: FAIL (F1={f1_pct:.1f}% < 85%)")
        print(f"  改进建议:")
        if result['recall'] < 0.8:
            print(f"    - 召回率偏低: 漏报多, 降低阈值或增加检测规则")
        if result['precision'] < 0.8:
            print(f"    - 精确率偏低: 误报多, 提高阈值或收紧规则条件")

    return result


# ============================================================================
# 规则级别的诊断
# ============================================================================

def diagnose_rules(positive_codes: list[str], negative_codes: list[str]):
    """诊断每条规则对正/负例的区分能力"""
    from fraud_detector import FraudDetector
    fd = FraudDetector()

    # 收集每条规则在正负例上的触发率
    rule_stats = defaultdict(lambda: {"pos_trigger": 0, "neg_trigger": 0, "pos_count": 0, "neg_count": 0})

    # 用 analyze 的逐条规则得分来诊断
    # FraudDetector.analyze() 返回每个规则的触发情况

    for code in positive_codes[:200]:  # 抽样加速
        try:
            r = fd.analyze(code)
            for alert in r.get('alerts', []):
                rule_name = alert.get('rule', 'unknown')
                rule_stats[rule_name]["pos_trigger"] += 1
            rule_stats["total"]["pos_count"] += 1
        except Exception:
            pass

    for code in negative_codes[:200]:
        try:
            r = fd.analyze(code)
            for alert in r.get('alerts', []):
                rule_name = alert.get('rule', 'unknown')
                rule_stats[rule_name]["neg_trigger"] += 1
            rule_stats["total"]["neg_count"] += 1
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  规则区分度诊断")
    print("=" * 60)
    print(f"  {'规则':<25} {'正例触发':>8} {'负例触发':>8} {'区分度':>8}")
    print("  " + "-" * 50)

    for rule_name, stats in rule_stats.items():
        pos_rate = stats['pos_trigger'] / max(1, len(positive_codes))
        neg_rate = stats['neg_trigger'] / max(1, len(negative_codes))
        diff = pos_rate - neg_rate  # 正越多, 区分度越好
        bar = "+" if diff > 0.1 else ("-" if diff < 0 else "=")
        print(f"  {rule_name:<25} {pos_rate:>7.1%} {neg_rate:>7.1%} {bar} {diff:>+6.1%}")

    return rule_stats


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("构建标签...")
    positives, negatives, excluded, _ = build_labels()

    print("\n评测 FraudDetector...")
    result = evaluate_fraud_detector(positives, negatives)
    print_report(result)

    print("\n诊断规则区分度...")
    diagnose_rules(positives, negatives)

    # 展示几个正例/负例
    print(f"\n正例样本 (有问题):")
    for code in positives[:5]:
        print(f"  {code}")
    print(f"\n负例样本 (干净):")
    for code in negatives[:5]:
        print(f"  {code}")
