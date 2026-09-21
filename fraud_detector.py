"""
=============================================================================
 财报反欺诈模块
 功能：多期趋势分析 + 单期异常检测 + LLM 研判报告生成
 用法：python fraud_detector.py 000008.SZ
       python fraud_detector.py --top 20
       python fraud_detector.py --report 000008.SZ --llm
=============================================================================
"""
import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data_processed")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


class FraudDetector:
    """
    财报欺诈检测器
    包含 12 条规则：6 条单期异常 + 6 条多期趋势异常
    """

    def __init__(self):
        self.fin = pd.read_pickle(os.path.join(DATA_DIR, "financials_merged.pkl"))
        self.ann = pd.read_pickle(os.path.join(DATA_DIR, "announcements.pkl"))
        self._precompute()

    def _precompute(self):
        """预计算所有衍生指标和增长率"""
        df = self.fin.sort_values(['stock_code', 'report_date'])

        # 同比（和去年同期比）
        for col in ['operating_revenue', 'net_profit', 'inventory',
                     'accounts_receivable', 'operating_cashflow']:
            if col in df.columns:
                df[f'{col}_yoy'] = df.groupby('stock_code')[col].pct_change(4, fill_method=None)

        # 环比（和上季度比）
        for col in ['inventory', 'accounts_receivable']:
            if col in df.columns:
                df[f'{col}_qoq'] = df.groupby('stock_code')[col].pct_change(1, fill_method=None)

        self.data = df

    # ================================================================
    # 单期绝对值规则
    # ================================================================

    def _rule_inventory_buildup(self, row):
        """存货/营收 > 1.0 — 货卖不出去"""
        v = row.get('inventory_to_revenue')
        if pd.isna(v) or v <= 1.0:
            return None
        score = min((v - 1.0) * 15, 25)
        return {"rule": "存货积压", "score": score,
                "detail": f"存货是年营收的 {v:.1f} 倍，严重积压，存在跌价风险",
                "data": f"存货={row['inventory']/1e8:.2f}亿, 营收={row['operating_revenue']/1e8:.2f}亿"}

    def _rule_cashflow_gap(self, row):
        """经营现金流/净利润 < 0.3 且净利润>0 — 纸面富贵"""
        v = row.get('cashflow_to_profit')
        profit = row.get('net_profit', 0) or 0
        if pd.isna(v) or profit <= 0:
            return None
        if v < 0.3:
            score = min((0.3 - v) * 50, 25)
            tag = "现金流为负，利润全是纸面" if v < 0 else "现金流仅为利润的" + f"{v*100:.0f}%"
            return {"rule": "现金流悖离", "score": score,
                    "detail": f"{tag}——赚的钱没有真正到账",
                    "data": f"经营CF={row['operating_cashflow']/1e8:.2f}亿, 净利润={profit/1e8:.2f}亿"}
        return None

    def _rule_receivable_surge(self, row):
        """应收/营收 > 0.8 — 靠赊销做收入"""
        v = row.get('receivable_to_revenue')
        if pd.isna(v) or v <= 0.8:
            return None
        score = min((v - 0.8) * 12, 20)
        return {"rule": "应收款畸高", "score": score,
                "detail": f"应收款是营收的 {v:.1f} 倍，收入质量极差——客户都没付款",
                "data": f"应收={row['accounts_receivable']/1e8:.2f}亿, 营收={row['operating_revenue']/1e8:.2f}亿"}

    def _rule_debt_crisis(self, row):
        """负债率 > 80% — 资不抵债边缘"""
        v = row.get('debt_ratio')
        if pd.isna(v) or v <= 0.8:
            return None
        score = min((v - 0.8) * 50, 15)
        if v > 1.0:
            return {"rule": "资不抵债", "score": 25,
                    "detail": f"负债率 {v*100:.1f}%，欠的钱比资产还多，技术上已破产",
                    "data": f"负债={row['total_liabilities']/1e8:.2f}亿, 资产={row['total_assets']/1e8:.2f}亿"}
        return {"rule": "负债过高", "score": score,
                "detail": f"负债率 {v*100:.1f}%，远超80%警戒线",
                "data": f"负债={row['total_liabilities']/1e8:.2f}亿, 资产={row['total_assets']/1e8:.2f}亿"}

    def _rule_goodwill_bomb(self, row):
        """商誉/净资产 > 30% — 商誉减值炸弹"""
        v = row.get('goodwill_to_equity')
        if pd.isna(v) or v <= 0.3:
            return None
        score = min(v * 30, 15)
        return {"rule": "商誉减值风险", "score": score,
                "detail": f"商誉占净资产 {v*100:.1f}%，一旦减值将直接冲击利润",
                "data": f"商誉={row['goodwill']/1e8:.2f}亿, 净资产={row['total_equity']/1e8:.2f}亿"}

    def _rule_profit_quality(self, row):
        """毛利率 < 5% 且营收>0 — 卖一单亏一单"""
        v = row.get('gross_margin')
        rev = row.get('operating_revenue', 0) or 0
        if pd.isna(v) or rev <= 0 or v > 0.05:
            return None
        return {"rule": "毛利率极低", "score": 10,
                "detail": f"毛利率仅 {v*100:.1f}%，几乎没有盈利空间",
                "data": f"营收={rev/1e8:.2f}亿, 毛利={v*100:.1f}%"}

    # ================================================================
    # 多期趋势规则
    # ================================================================

    def _rule_inventory_vs_revenue_trend(self, row):
        """存货增速远超营收增速（同比）"""
        inv_yoy = row.get('inventory_yoy')
        rev_yoy = row.get('operating_revenue_yoy')
        if pd.isna(inv_yoy) or pd.isna(rev_yoy):
            return None
        if inv_yoy > 0.5 and inv_yoy > rev_yoy * 3:
            score = min((inv_yoy - rev_yoy) * 10, 20)
            return {"rule": "存货增速异常", "score": score,
                    "detail": f"存货同比增 {inv_yoy*100:.0f}%，但营收仅增 {rev_yoy*100:.0f}%——产了卖不掉",
                    "data": f"存货增速={inv_yoy*100:.0f}%, 营收增速={rev_yoy*100:.0f}%"}
        return None

    def _rule_receivable_vs_revenue_trend(self, row):
        """应收增速远超营收增速"""
        rcv_yoy = row.get('accounts_receivable_yoy')
        rev_yoy = row.get('operating_revenue_yoy')
        if pd.isna(rcv_yoy) or pd.isna(rev_yoy):
            return None
        if rcv_yoy > 0.5 and rcv_yoy > rev_yoy * 2:
            score = min((rcv_yoy - rev_yoy) * 8, 15)
            return {"rule": "应收增速异常", "score": score,
                    "detail": f"应收款同比增 {rcv_yoy*100:.0f}%，营收仅增 {rev_yoy*100:.0f}%——靠赊销冲业绩",
                    "data": f"应收增速={rcv_yoy*100:.0f}%, 营收增速={rev_yoy*100:.0f}%"}
        return None

    def _rule_profit_vs_cashflow_trend(self, row):
        """利润增长但现金流恶化"""
        profit_yoy = row.get('net_profit_yoy')
        cf_yoy = row.get('operating_cashflow_yoy')
        if pd.isna(profit_yoy) or pd.isna(cf_yoy):
            return None
        if profit_yoy > 0.2 and cf_yoy < -0.3:
            return {"rule": "利润现金背离", "score": 20,
                    "detail": f"净利润同比增 {profit_yoy*100:.0f}%，但经营现金流反而恶化 {cf_yoy*100:.0f}%——利润质量差",
                    "data": f"利润增速={profit_yoy*100:.0f}%, 现金流增速={cf_yoy*100:.0f}%"}
        return None

    def _rule_consecutive_cashflow_negative(self, history):
        """连续多期经营现金流为负"""
        if len(history) < 3:
            return None
        recent = history.tail(3)
        if (recent['operating_cashflow'] < 0).sum() >= 3:
            return {"rule": "持续现金流失血", "score": 20,
                    "detail": "连续3期经营现金流为负，主营业务无法造血",
                    "data": f"近3期经营CF: {recent['operating_cashflow'].iloc[-3:].tolist()}"}
        return None

    def _rule_sudden_profit_jump(self, row):
        """利润突然暴增 200%+（可能是变卖资产或会计变更）"""
        v = row.get('net_profit_yoy')
        if pd.isna(v) or v <= 2.0:
            return None
        profit = row.get('net_profit', 0) or 0
        return {"rule": "利润异常暴增", "score": 10,
                "detail": f"净利润同比暴增 {v*100:.0f}%，需核实是否来自主营业务",
                "data": f"净利润={profit/1e8:.2f}亿, 增速={v*100:.0f}%"}

    def _rule_gross_margin_collapse(self, row):
        """毛利率连续大幅下滑"""
        v = row.get('gross_margin')
        prev_gm = row.get('_prev_gross_margin')
        if pd.isna(v) or pd.isna(prev_gm) or prev_gm <= 0:
            return None
        drop = prev_gm - v
        if drop > 0.1:
            return {"rule": "毛利率骤降", "score": 15,
                    "detail": f"毛利率从 {prev_gm*100:.1f}% 降至 {v*100:.1f}%，竞争恶化或成本失控",
                    "data": f"毛利率变化={prev_gm*100:.1f}% -> {v*100:.1f}%"}
        return None

    # ================================================================
    # 综合分析
    # ================================================================

    def analyze(self, stock_code):
        """对单只股票执行完整分析"""
        code = stock_code.strip()
        if '.' not in code:
            matches = [c for c in self.data['stock_code'].unique() if code in c]
            if matches:
                code = sorted(matches)[0]

        stock_data = self.data[self.data['stock_code'] == code].sort_values('report_date')
        if stock_data.empty:
            return {"stock_code": code, "error": "未找到数据", "alerts": [], "risk_score": 0}

        latest = stock_data.iloc[-1]

        # 添加上一期毛利率用于趋势对比
        if len(stock_data) >= 2:
            latest['_prev_gross_margin'] = stock_data.iloc[-2]['gross_margin']
        else:
            latest['_prev_gross_margin'] = None

        # 执行全部 12 条规则
        rules = [
            self._rule_inventory_buildup,
            self._rule_cashflow_gap,
            self._rule_receivable_surge,
            self._rule_debt_crisis,
            self._rule_goodwill_bomb,
            self._rule_profit_quality,
            self._rule_inventory_vs_revenue_trend,
            self._rule_receivable_vs_revenue_trend,
            self._rule_profit_vs_cashflow_trend,
            self._rule_sudden_profit_jump,
            self._rule_gross_margin_collapse,
        ]

        alerts = []
        for rule_fn in rules:
            try:
                result = rule_fn(latest)
                if result:
                    alerts.append(result)
            except Exception:
                pass

        # 连续现金流规则（需要历史数据）
        try:
            cf_result = self._rule_consecutive_cashflow_negative(stock_data)
            if cf_result:
                alerts.append(cf_result)
        except Exception:
            pass

        total_score = round(min(sum(a['score'] for a in alerts), 100), 1)

        # 等级
        if total_score >= 60:
            level = "高风险"
        elif total_score >= 30:
            level = "中等风险"
        else:
            level = "低风险"

        # 交叉验证：有公告风险事件额外加分
        risk_events = self.ann[(self.ann['s_info_windcode'] == code) & (self.ann['is_risk_event'])]
        has_risk_event = len(risk_events) > 0
        if has_risk_event and total_score >= 30:
            total_score = min(total_score + 10, 100)

        return {
            "stock_code": code,
            "report_period": str(latest['report_period']),
            "risk_score": round(total_score, 1),
            "risk_level": level,
            "alerts": sorted(alerts, key=lambda x: x['score'], reverse=True),
            "alert_count": len(alerts),
            "has_risk_announcement": has_risk_event,
            "risk_event_count": len(risk_events),
            "key_metrics": {
                "debt_ratio": latest.get('debt_ratio'),
                "inventory_to_revenue": latest.get('inventory_to_revenue'),
                "cashflow_to_profit": latest.get('cashflow_to_profit'),
                "receivable_to_revenue": latest.get('receivable_to_revenue'),
                "gross_margin": latest.get('gross_margin'),
                "net_profit": latest.get('net_profit'),
                "operating_cashflow": latest.get('operating_cashflow'),
            }
        }

    def scan_all(self, top_n=50):
        """扫描全市场，返回风险最高的股票"""
        codes = self.data['stock_code'].unique()
        results = []
        for i, code in enumerate(codes):
            try:
                r = self.analyze(code)
                if 'error' not in r and r['alert_count'] >= 1:
                    results.append(r)
            except Exception:
                pass
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results[:top_n]

    def generate_llm_report(self, analysis):
        """调用 LLM 生成研判报告"""
        if analysis.get('error'):
            return f"数据错误: {analysis['error']}"

        alerts_text = ""
        for a in analysis['alerts']:
            alerts_text += f"- [{a['rule']}] {a['detail']}\n  数据: {a['data']}\n\n"

        prompt = f"""你是一位资深财务审计师。以下是 {analysis['stock_code']} 的财务异常检测结果，请用通俗易懂的中文给普通投资者写一份分析报告。

风险评分: {analysis['risk_score']}/100（{analysis['risk_level']}）
报告期: {analysis['report_period']}
预警数量: {analysis['alert_count']} 条

预警详情:
{alerts_text}

请按以下结构输出：
1. 总体判断（一句话总结风险程度）
2. 逐条解读（每条预警用大白话解释是什么意思）
3. 综合评估（这些异常放在一起看说明什么）
4. 投资者提示（普通投资者应该注意什么，不构成投资建议）"""

        try:
            client = OpenAI(
                api_key=os.getenv("LLMOPS_API_KEY"),
                base_url=os.getenv("LLMOPS_BASE_URL", "https://llmops.transwarp.io/vibecoding/v1")
            )
            response = client.chat.completions.create(
                model=os.getenv("LLMOPS_MODEL", "openai/deepseek-v4-flash"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM调用失败: {e}\n\n原始数据:\n{alerts_text}"


def print_analysis(result):
    """打印分析结果"""
    print(f"\n{'='*60}")
    print(f"  {result['stock_code']}  财务欺诈风险分析")
    print(f"{'='*60}")
    print(f"  报告期: {result['report_period']}")
    print(f"  风险评分: {result['risk_score']}/100  [{result['risk_level']}]")
    print(f"  预警信号: {result['alert_count']} 条")
    if result.get('has_risk_announcement'):
        print(f"  风险公告: {result['risk_event_count']} 条 (证监局/交易所处罚或问询)")
    print()

    if result['alerts']:
        print("  预警详情:")
        for i, a in enumerate(result['alerts'], 1):
            print(f"  {i}. [{a['rule']}] (+{a['score']}分)")
            print(f"     {a['detail']}")
            print(f"     数据: {a['data']}")
            print()
    else:
        print("  未检测到明显异常。\n")

    print("  关键指标:")
    km = result['key_metrics']
    for k, v in km.items():
        if pd.notna(v):
            if 'ratio' in k or 'margin' in k:
                print(f"    {k}: {v*100:.1f}%" if v is not None else f"    {k}: N/A")
            elif abs(v) > 1e8:
                print(f"    {k}: {v/1e8:.2f}亿" if v is not None else f"    {k}: N/A")
            else:
                print(f"    {k}: {v:.2f}" if v is not None else f"    {k}: N/A")


if __name__ == "__main__":
    detector = FraudDetector()

    if len(sys.argv) == 1:
        # 默认：显示前 20 名
        print("扫描全市场，寻找高风险股票...")
        top = detector.scan_all(20)
        print(f"\n{'='*60}")
        print(f"  财务欺诈风险排行 Top 20")
        print(f"{'='*60}")
        for i, r in enumerate(top, 1):
            alerts_summary = ", ".join(a['rule'][:8] for a in r['alerts'][:3])
            print(f"  {i:2d}. {r['stock_code']:12s}  {r['risk_score']:5.0f}分  [{r['risk_level']}]  "
                  f"{r['alert_count']}项: {alerts_summary}")

    elif sys.argv[1] == '--top':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        top = detector.scan_all(n)
        for i, r in enumerate(top, 1):
            alerts_summary = ", ".join(a['rule'][:8] for a in r['alerts'][:3])
            print(f"  {i:2d}. {r['stock_code']:12s}  {r['risk_score']:5.0f}分  [{r['risk_level']}]  "
                  f"{r['alert_count']}项: {alerts_summary}")

    elif sys.argv[1] == '--report':
        code = sys.argv[2]
        result = detector.analyze(code)
        print_analysis(result)
        if '--llm' in sys.argv:
            print("\n" + "="*60)
            print("  LLM 研判报告")
            print("="*60)
            report = detector.generate_llm_report(result)
            print(report)
            # 保存
            path = os.path.join(REPORT_DIR, f"fraud_report_{code.replace('.','_')}.txt")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存: {path}")

    else:
        code = sys.argv[1]
        result = detector.analyze(code)
        print_analysis(result)
