"""
================================================================================
 金融AI智能助手
 功能：财报反欺诈 + 股权穿透 + 智能对话
================================================================================
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

# ── 知识图谱模块 ──
from graph_core import GraphStore, Entity, HoldingEdge, make_entity_id
from entity_resolver import (
    EntityResolver, AliasDict, FuzzyMatcher, clean_entity_name,
    build_graph_from_shareholders, normalize_name,
)
from penetration_engine import (
    PenetrationEngine, EquityQueryService, ShellDetector, PenetrationPath,
)
from graph_llm_bridge import (
    GraphLLMBridge, subgraph_to_context, path_to_narrative,
    llm_shell_judge, llm_evaluate_paths,
)
from graph_tools import GraphToolExecutor, get_tools_for_llm, TOOL_DEFINITIONS
from event_pipeline import EventPipeline, EventTimeline

load_dotenv()

# ============================================================================
# 第一部分：配置
# ============================================================================

class Config:
    """全局配置，所有路径和参数都在这里"""
    # 数据目录
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE, "data_raw")

    # LLM 配置
    LLM_CLIENT = OpenAI(
        api_key=os.getenv("LLMOPS_API_KEY"),
        base_url=os.getenv("LLMOPS_BASE_URL", "https://llmops.transwarp.io/vibecoding/v1")
    )
    LLM_MODEL = os.getenv("LLMOPS_MODEL", "openai/deepseek-v4-flash")

    # 输出目录
    OUTPUT_DIR = os.path.join(BASE, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)


cfg = Config()

# ============================================================================
# 第二部分：数据加载
# ============================================================================

class DataLoader:
    """加载所有比赛数据"""

    @staticmethod
    def load_qa_test():
        """加载测试问答集"""
        path = os.path.join(cfg.DATA_DIR, "1.测试问答集/clean.xlsx")
        df = pd.read_excel(path)
        print(f"[数据] 测试问答集: {len(df)} 条, {df['session_id'].nunique()} 个会话")
        return df

    @staticmethod
    def load_shareholders():
        """加载股东持股数据"""
        path = os.path.join(cfg.DATA_DIR, "2.股东持股-股权穿透/clean.xlsx")
        df = pd.read_excel(path)
        print(f"[数据] 股东持股: {len(df)} 条, {df['s_info_windcode'].nunique()} 只股票")
        return df

    @staticmethod
    def load_announcements():
        """加载公司公告数据"""
        path = os.path.join(cfg.DATA_DIR, "3.公司公告-事件脉络和风险识别/clean.xlsx")
        df = pd.read_excel(path)
        print(f"[数据] 公司公告: {len(df)} 条")
        return df

    @staticmethod
    def load_financials():
        """加载三大财务报表"""
        balance = pd.read_csv(os.path.join(cfg.DATA_DIR, "4.三大财务报表-财务反欺诈/asharebalancesheet_202605261517.csv"))
        income = pd.read_csv(os.path.join(cfg.DATA_DIR, "4.三大财务报表-财务反欺诈/ashareincome_202605261519.csv"))
        cashflow = pd.read_csv(os.path.join(cfg.DATA_DIR, "4.三大财务报表-财务反欺诈/asharecashflow_202605261518.csv"))
        print(f"[数据] 资产负债表: {len(balance)} 行, 利润表: {len(income)} 行, 现金流量表: {len(cashflow)} 行")
        return balance, income, cashflow

    @staticmethod
    def load_reports():
        """加载券商研报数据"""
        path = os.path.join(cfg.DATA_DIR, "5.券商研报-专家观点/rr_main_202605281537.csv")
        df = pd.read_csv(path)
        print(f"[数据] 券商研报: {len(df)} 条")
        return df


# ============================================================================
# 第三部分：财报反欺诈引擎
# ============================================================================

class FraudDetector:
    """
    财报造假检测器
    核心思路：对比利润表、资产负债表、现金流量表之间的勾稽关系，
    找出"数据打架"的地方
    """

    def __init__(self, balance_df, income_df, cashflow_df):
        self.balance = balance_df
        self.income = income_df
        self.cashflow = cashflow_df
        self.results = []  # 存储所有分析结果

    def get_latest_financials(self, stock_code):
        """
        获取某只股票最新的三张表数据
        股票代码格式: 603439.SH 或 603439
        """
        # 统一代码格式
        if '.' not in stock_code:
            # 尝试匹配
            for df in [self.balance, self.income, self.cashflow]:
                codes = df['s_info_windcode'].unique()
                matches = [c for c in codes if stock_code in c]
                if matches:
                    stock_code = matches[0]
                    break

        # 取最新一期
        b = self.balance[self.balance['s_info_windcode'] == stock_code].sort_values('report_period')
        i = self.income[self.income['s_info_windcode'] == stock_code].sort_values('report_period')
        c = self.cashflow[self.cashflow['s_info_windcode'] == stock_code].sort_values('report_period')

        if b.empty:
            return None, None, None

        return b.iloc[-1], i.iloc[-1] if not i.empty else None, c.iloc[-1] if not c.empty else None

    def check_rule_1_inventory_vs_revenue(self, stock_code):
        """
        规则1：存货增速 vs 营收增速
        如果存货增长速度远超营收增长速度 → 可能是货卖不出去，却还在生产
        """
        b, i, _ = self.get_latest_financials(stock_code)
        if b is None or i is None:
            return None

        # 注意：数据只有最新一期，无法算增速。这里用绝对值做简化判断
        # 完整实现需要拿多期数据对比
        inventory = b.get('inventories', 0) or 0
        revenue = i.get('oper_rev', 0) or i.get('tot_oper_rev', 0) or 0

        if revenue == 0:
            return None

        # 存货占总资产的比例
        total_assets = b.get('tot_assets', 1) or 1
        inv_ratio = inventory / total_assets if total_assets > 0 else 0

        # 存货/营收比
        inv_rev_ratio = inventory / revenue if revenue > 0 else 0

        # 判断标准
        risk = 0
        details = []
        if inv_ratio > 0.3:  # 存货超过总资产30%
            risk += 30
            details.append(f"存货占总资产比例高达 {inv_ratio*100:.1f}%，可能存在库存积压")
        if inv_rev_ratio > 0.5:  # 存货超过营收一半
            risk += 20
            details.append(f"存货/营收比 = {inv_rev_ratio:.2f}，存货规模相对营收过大")

        return {
            "rule": "存货/营收比异常",
            "risk_score": min(risk, 100),
            "detail": "；".join(details) if details else "正常",
            "data": f"存货={inventory/1e8:.2f}亿, 营收={revenue/1e8:.2f}亿, 存货/营收={inv_rev_ratio:.2f}"
        }

    def check_rule_2_cashflow_vs_profit(self, stock_code):
        """
        规则2：经营现金流 vs 净利润
        利润涨了但现金流没涨 → 利润可能是"纸面富贵"
        """
        _, i, c = self.get_latest_financials(stock_code)
        if i is None or c is None:
            return None

        net_profit = i.get('net_profit_is', 0) or i.get('net_profit', 0) or 0
        # 经营现金流字段查找
        op_cf = c.get('net_cash_flows_oper_act', 0) or c.get('cash_recp_sg_and_rs', 0) or 0

        if net_profit <= 0:
            return {"rule": "现金流/利润倒挂", "risk_score": 0,
                    "detail": "净利润为负，本规则不适用", "data": ""}

        ratio = op_cf / net_profit if net_profit != 0 else 0

        if ratio < 0.5:
            risk = int((0.5 - ratio) * 200)  # 差距越大风险越高
            risk = min(risk, 80)
            return {
                "rule": "经营现金流/净利润倒挂",
                "risk_score": risk,
                "detail": f"经营现金流仅为净利润的 {ratio*100:.1f}%，利润质量存疑——赚的钱可能没真正到账",
                "data": f"经营现金流={op_cf/1e8:.2f}亿, 净利润={net_profit/1e8:.2f}亿, 比值={ratio:.2f}"
            }
        return {"rule": "经营现金流/净利润倒挂", "risk_score": 0,
                "detail": "现金流与利润匹配良好", "data": f"比值={ratio:.2f}"}

    def check_rule_3_receivables_vs_revenue(self, stock_code):
        """
        规则3：应收账款增速 vs 营收增速
        应收账款增长远超营收 → 可能靠"赊销"做高收入（客户其实没付款）
        """
        b, i, _ = self.get_latest_financials(stock_code)
        if b is None or i is None:
            return None

        receivables = (b.get('acct_rcv', 0) or 0) + (b.get('notes_rcv', 0) or 0)
        revenue = i.get('oper_rev', 0) or i.get('tot_oper_rev', 0) or 0

        if revenue == 0:
            return None

        ratio = receivables / revenue

        if ratio > 0.5:
            risk = min(int((ratio - 0.5) * 100), 60)
            return {
                "rule": "应收账款占比过高",
                "risk_score": risk,
                "detail": f"应收账款占营收 {ratio*100:.1f}%，比例偏高——收入可能靠赊销撑起来",
                "data": f"应收账款={receivables/1e8:.2f}亿, 营收={revenue/1e8:.2f}亿, 占比={ratio*100:.1f}%"
            }
        return {"rule": "应收账款占比过高", "risk_score": 0,
                "detail": "应收账款占比正常", "data": f"占比={ratio*100:.1f}%"}

    def check_rule_4_goodwill_ratio(self, stock_code):
        """规则4：商誉占比过高 → 并购溢价过高，未来可能减值"""
        b, _, _ = self.get_latest_financials(stock_code)
        if b is None:
            return None

        goodwill = b.get('goodwill', 0) or 0
        equity = b.get('tot_shrhldr_eqy_incl_min_int', 0) or b.get('tot_assets', 1) or 1

        if goodwill == 0:
            return None

        ratio = goodwill / equity if equity > 0 else 0

        if ratio > 0.3:
            risk = min(int(ratio * 100), 50)
            return {
                "rule": "商誉占比过高",
                "risk_score": risk,
                "detail": f"商誉占净资产 {ratio*100:.1f}%，远超30%警戒线——未来存在大额减值风险",
                "data": f"商誉={goodwill/1e8:.2f}亿, 净资产={equity/1e8:.2f}亿"
            }
        return None

    def check_rule_5_debt_ratio(self, stock_code):
        """规则5：资产负债率异常高"""
        b, _, _ = self.get_latest_financials(stock_code)
        if b is None:
            return None

        total_liab = b.get('tot_liab', 0) or 0
        total_assets = b.get('tot_assets', 1) or 1
        ratio = total_liab / total_assets

        if ratio > 0.8:
            return {
                "rule": "资产负债率过高",
                "risk_score": int((ratio - 0.8) * 200),
                "detail": f"资产负债率高达 {ratio*100:.1f}%，偿债压力大",
                "data": f"负债={total_liab/1e8:.2f}亿, 总资产={total_assets/1e8:.2f}亿"
            }
        return None

    def analyze_stock(self, stock_code):
        """对一只股票执行全部规则检查，返回综合评分"""
        checks = [
            self.check_rule_1_inventory_vs_revenue,
            self.check_rule_2_cashflow_vs_profit,
            self.check_rule_3_receivables_vs_revenue,
            self.check_rule_4_goodwill_ratio,
            self.check_rule_5_debt_ratio,
        ]

        alerts = []
        total_risk = 0
        for check in checks:
            try:
                result = check(stock_code)
                if result and result.get('risk_score', 0) > 0:
                    alerts.append(result)
                    total_risk += result['risk_score']
            except Exception as e:
                pass  # 某些字段可能不存在

        total_risk = min(total_risk, 100)

        # 风险等级
        if total_risk >= 60:
            level = "[高风险]"
        elif total_risk >= 30:
            level = "[中等风险]"
        else:
            level = "[低风险]"

        return {
            "stock_code": stock_code,
            "risk_score": total_risk,
            "risk_level": level,
            "alerts": alerts,
            "alert_count": len(alerts)
        }


# ============================================================================
# 第四部分：股权穿透分析
# ============================================================================

class EquityGraph:
    """
    股权穿透分析器
    基于 GraphStore + PenetrationEngine 的新一代图系统
    支持多跳穿透、壳公司检测、LLM辅助判断
    """

    def __init__(self, shareholder_df):
        print("[图谱] 构建股权关系图（新一代引擎）...")
        # 构建底层图存储
        self.store, self.alias_dict = build_graph_from_shareholders(shareholder_df)
        # 初始化穿透引擎
        self.engine = PenetrationEngine(self.store)
        self.query_service = EquityQueryService(self.store)
        # 实体解析器
        self.resolver = EntityResolver()
        self.resolver.build_index(
            list(self.store.name_index.keys()),
            graph_store=self.store,
        )
        self.resolver.alias_dict = self.alias_dict
        self.query_service.resolver = self.resolver
        # LLM-图桥
        self.llm_bridge = GraphLLMBridge(self.store, self.engine)

        print(f"[图谱] 构建完成: {self.store.entity_count():,} 个实体, "
              f"{self.store.edge_count():,} 条持股关系")

    def query_penetration(self, entity_name, max_depth=5):
        """
        查询某个实体的股权穿透链路
        返回格式兼容旧接口
        """
        result = self.query_service.query(entity_name, max_depth=max_depth)

        if "error" in result:
            return []

        chains = []
        for chain in result.get("penetration_chains", []):
            chains.append({
                "path": " → ".join(chain["path"]),
                "depth": chain["depth"],
                "effective_pct": chain["effective_pct"],
                "nodes": chain["path"],
            })

        # 按有效持股比排序
        chains.sort(key=lambda x: (-x["effective_pct"], x["depth"]))
        return chains[:20]

    def get_top_shareholders(self, stock_code, top_n=10):
        """获取某只股票的前N大股东"""
        entity_id = self.store.find_by_stock_code(stock_code)
        if not entity_id:
            entity_id = self.store.find_by_name(stock_code)

        if not entity_id:
            return []

        shareholders = self.store.get_neighbors(entity_id, "in")
        shareholders.sort(key=lambda x: x.get("pct", 0), reverse=True)

        results = []
        for sh in shareholders[:top_n]:
            results.append({
                "holder": sh.get("source_name", "未知"),
                "pct": sh.get("pct", 0),
                "type": sh.get("source_type", "未知"),
            })
        return results

    def get_ultimate_controllers(self, entity_name, max_depth=5):
        """获取最终控制人"""
        result = self.query_service.query(entity_name, max_depth=max_depth)
        return result.get("ultimate_controllers", [])

    def get_entity_info(self, entity_name):
        """获取实体详细信息"""
        entity_id = self.store.find_by_name(entity_name)
        if not entity_id:
            entity_id = self.store.find_by_stock_code(entity_name)
        if not entity_id:
            return None
        entity = self.store.get_entity(entity_id)
        if entity:
            return entity.to_dict()
        return None

    def stats(self):
        """返回图谱统计"""
        return self.store.stats()


# ============================================================================
# 第五部分：智能 Agent
# ============================================================================

class FinanceAgent:
    """
    智能金融助手 Agent
    整合财报分析 + 股权穿透 + LLM 对话
    """

    def __init__(self):
        print("[Agent] 初始化...")
        # 加载数据
        loader = DataLoader()
        self.qa_df = loader.load_qa_test()
        self.shareholder_df = loader.load_shareholders()
        self.announcement_df = loader.load_announcements()
        self.balance_df, self.income_df, self.cashflow_df = loader.load_financials()
        self.report_df = loader.load_reports()

        # 初始化子模块
        self.fraud_detector = FraudDetector(self.balance_df, self.income_df, self.cashflow_df)
        self.equity_graph = EquityGraph(self.shareholder_df)

        # ── 初始化事件管道 ──
        try:
            self.event_pipeline = EventPipeline()
            self.event_pipeline.fit(self.announcement_df, use_llm_labels=True)
        except Exception as e:
            print(f"[Agent] 事件管道初始化跳过: {e}")
            self.event_pipeline = None

        # ── 初始化工具执行器 ──
        self.tool_executor = GraphToolExecutor(
            graph_store=self.equity_graph.store,
            event_pipeline=self.event_pipeline,
            chat_fn=self._call_llm_raw,
        )

        # 对话历史
        self.conversation_history = []
        # 用户关注的股票（长期记忆）
        self.tracked_stocks = set()

        print("[Agent] 初始化完成！")

    def _call_llm_raw(self, messages, temperature=0.3, max_tokens=2000):
        """内部 LLM 调用方法（供 GraphToolExecutor 等处使用）"""
        try:
            response = cfg.LLM_CLIENT.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return None

    def chat(self, user_message):
        """
        处理用户消息的完整流程：
        1. 识别意图
        2. 调用对应工具
        3. LLM 生成回复
        """
        # 添加到历史
        self.conversation_history.append({"role": "user", "content": user_message})

        # 提取股票代码
        stock_code = self._extract_stock_code(user_message)
        if stock_code:
            self.tracked_stocks.add(stock_code)

        # 意图识别
        intent = self._classify_intent(user_message)

        # 执行工具
        tool_result = ""
        if intent == "financial_analysis" and stock_code:
            tool_result = self._run_fraud_analysis(stock_code)
        elif intent == "equity_penetration":
            entity = self._extract_entity_name(user_message) or stock_code or ""
            tool_result = self._run_equity_analysis(entity)
        elif intent == "stock_query" and stock_code:
            tool_result = self._run_stock_overview(stock_code)
        elif intent == "greeting":
            tool_result = ""

        # 构建 LLM prompt
        system_prompt = self._build_system_prompt(tool_result)
        messages = [{"role": "system", "content": system_prompt}]
        # 加入最近的对话历史（不超过10轮）
        recent = self.conversation_history[-20:]
        messages.extend(recent)

        # 调用 LLM
        try:
            response = cfg.LLM_CLIENT.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"抱歉，AI服务暂时不可用：{e}\n\n不过我可以直接给你分析结果：\n{tool_result}"

        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _extract_stock_code(self, text):
        """从文本中提取股票代码（6位数字）"""
        import re
        patterns = [r'\b(\d{6})\b', r'(\d{6}\.[A-Z]{2,3})']
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1)
        return None

    def _extract_entity_name(self, text):
        """提取实体名称（中文人名或公司名）"""
        import re
        # 简单策略：找"XXX的股权"这种模式
        match = re.search(r'[「「]?(.{2,8})[」」]?的?(股权|穿透|控股|持股)', text)
        if match:
            return match.group(1)
        return None

    def _classify_intent(self, text):
        """简单意图识别：根据关键词判断用户想干什么"""
        if any(w in text for w in ['你好', 'hello', '嗨', 'hi']):
            return "greeting"
        if any(w in text for w in ['财报', '财务', '利润', '造假', '风险', '排雷', '健康', '现金流']):
            return "financial_analysis"
        if any(w in text for w in ['股权', '穿透', '控股', '股东', '持股', '控制']):
            return "equity_penetration"
        if any(w in text for w in ['股票', '代码', '查询', '分析一下']):
            return "stock_query"
        return "general"

    def _run_fraud_analysis(self, stock_code):
        """执行财报反欺诈分析"""
        result = self.fraud_detector.analyze_stock(stock_code)
        lines = [
            f"## {stock_code} 财务健康度分析",
            f"风险评分: {result['risk_score']}/100",
            f"风险等级: {result['risk_level']}",
            f"发现 {result['alert_count']} 个预警信号",
            ""
        ]
        for alert in result['alerts']:
            lines.append(f"### ⚠️ {alert['rule']}")
            lines.append(f"- {alert['detail']}")
            lines.append(f"- 数据: {alert.get('data', '')}")
            lines.append("")
        return "\n".join(lines)

    def _run_equity_analysis(self, entity):
        """执行股权穿透分析（新一代引擎）"""
        if not entity:
            return "请提供要查询的实体名称或股票代码"

        # 使用新的 query_service
        result = self.equity_graph.query_service.query(entity, max_depth=5)

        if "error" in result:
            return f"未找到 {entity} 的股权信息: {result['error']}"

        resolved = result.get("resolved_entity", entity)
        lines = [f"## 🔗 {resolved} 股权穿透分析", ""]

        # 实体基本信息
        lines.append(f"- 实体类型: {result.get('entity_type', '未知')}")
        lines.append(f"- 是否上市公司: {'是' if result.get('is_listed') else '否'}")
        if result.get("stock_code"):
            lines.append(f"- 股票代码: {result['stock_code']}")
        lines.append("")

        # 前十大股东
        top = result.get("top_shareholders", [])
        if top:
            lines.append("### 📊 前十大股东")
            for i, s in enumerate(top[:10], 1):
                lines.append(f"{i}. {s['name']} ({s['type']}) — 持股 {s['pct']}%")
            lines.append("")

        # 穿透链路
        chains = result.get("penetration_chains", [])
        if chains:
            lines.append(f"### 🔍 穿透链路（共{len(chains)}条，展示前5条）")
            for i, chain in enumerate(chains[:5], 1):
                path_str = " → ".join(chain["path"])
                lines.append(f"**路径{i}** (深度{chain['depth']}, 有效持股{chain['effective_pct']}%):")
                lines.append(f"  {path_str}")
                lines.append(f"  终止原因: {chain['terminals_at']}")
                lines.append("")

        # 最终控制人
        controllers = result.get("ultimate_controllers", [])
        if controllers:
            lines.append("### 🎯 最终控制人")
            for c in controllers[:5]:
                lines.append(f"- **{c['controller_name']}** ({c['controller_type']})")
                lines.append(f"  控制链: {' → '.join(c['chain'])}")
                lines.append(f"  有效持股: {c['effective_pct']}%, 深度: {c['depth']}层")
            lines.append("")

        return "\n".join(lines) if (top or chains) else f"未找到 {entity} 的股权信息"

    def _run_stock_overview(self, stock_code):
        """股票综合概览"""
        fraud = self.fraud_detector.analyze_stock(stock_code)
        top = self.equity_graph.get_top_shareholders(stock_code, top_n=5)

        lines = [
            f"## 📈 {stock_code} 综合概览",
            f"风险评分: {fraud['risk_score']}/100 — {fraud['risk_level']}",
            ""
        ]
        if top:
            lines.append("### 大股东")
            for s in top[:3]:
                lines.append(f"- {s['holder']}: {s['pct']}%")

        # 相关研报
        reports = self.report_df[self.report_df['sec_code'].astype(str).str.contains(
            stock_code.replace('.SH', '').replace('.SZ', ''), na=False)]
        if len(reports) > 0:
            lines.append(f"\n### 相关研报 ({len(reports)} 篇)")
            for _, r in reports.head(3).iterrows():
                title = str(r.get('title', ''))[:50]
                lines.append(f"- {title}...")

        return "\n".join(lines)

    def _build_system_prompt(self, tool_result):
        """构建系统提示词"""
        prompt = """你是一个专业的金融AI助手，为中国A股投资者提供智能分析服务。

你的能力包括：
1. 财报反欺诈分析 — 检查上市公司的财务健康度，发现造假预警信号
2. 股权穿透查询 — 多层级挖掘公司的实际控制关系，穿透壳公司/基金/合伙企业
3. 事件时间线 — 对齐股权变更与舆情事件（监管处罚、资产重组等）
4. 研报信息检索 — 查找券商研究观点

回复要求：
- 用通俗易懂的中文，让普通投资者也能理解
- 如果数据正常，如实告知；如果有风险，明确指出
- 给出具体的数字和理由，不要说空话
- 对于股权穿透结果，重点关注"最终控制人是谁"和"通过什么路径控制"
"""
        if tool_result:
            prompt += f"\n\n以下是系统分析到的数据，请基于这些数据回答用户：\n\n{tool_result}"

        if self.tracked_stocks:
            prompt += f"\n\n用户之前关注过这些股票：{', '.join(self.tracked_stocks)}"

        return prompt


# ============================================================================
# 第六部分：Gradio Web 界面
# ============================================================================

def create_web_app():
    """创建 Gradio Web 演示界面"""
    try:
        import gradio as gr
    except ImportError:
        print("请先安装 gradio: pip install gradio")
        return None

    print("[Web] 初始化 Agent（首次加载需要一些时间）...")
    agent = FinanceAgent()

    def respond(message, history):
        """Gradio 回调函数"""
        reply = agent.chat(message)
        return reply

    def analyze_stock_direct(stock_code):
        """直接分析某只股票（不用对话）"""
        if not stock_code or len(stock_code.strip()) < 3:
            return "请输入有效的股票代码（如 603439）"

        fraud = agent.fraud_detector.analyze_stock(stock_code.strip())
        top = agent.equity_graph.get_top_shareholders(stock_code.strip(), 5)

        lines = [
            f"# {stock_code} 分析报告",
            "",
            f"## 财务健康度",
            f"- 风险评分: **{fraud['risk_score']}/100**",
            f"- 风险等级: **{fraud['risk_level']}**",
            ""
        ]

        if fraud['alerts']:
            lines.append("## 预警信号")
            for alert in fraud['alerts']:
                lines.append(f"### ⚠️ {alert['rule']} [风险+{alert['risk_score']}]")
                lines.append(f"{alert['detail']}")
                lines.append(f"数据: {alert.get('data', '')}")
                lines.append("")

        if top:
            lines.append("## 大股东")
            for i, s in enumerate(top, 1):
                lines.append(f"{i}. **{s['holder']}** ({s['type']}) — {s['pct']}%")

        return "\n".join(lines)

    # 构建界面
    with gr.Blocks(title="金融AI智能助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏦 金融AI智能助手
        **功能：财报反欺诈 · 股权穿透 · 智能问答**
        —— 基于 Agentic AI 的金融长上下文推理系统
        """)

        with gr.Tabs():
            # Tab 1: 对话模式
            with gr.Tab("💬 智能对话"):
                gr.ChatInterface(
                    fn=respond,
                    chatbot=gr.Chatbot(height=500),
                    textbox=gr.Textbox(placeholder="输入你的问题，例如：分析一下603439的财务健康度", container=False),
                    title="",
                    description="试试问：看看XX股份的股权结构 / 分析XX的财报有没有风险",
                )

            # Tab 2: 快速分析
            with gr.Tab("📊 快速分析"):
                gr.Markdown("输入股票代码，一键生成分析报告")
                with gr.Row():
                    code_input = gr.Textbox(label="股票代码", placeholder="例如：603439", scale=3)
                    analyze_btn = gr.Button("开始分析", variant="primary", scale=1)
                report_output = gr.Markdown("")

                analyze_btn.click(fn=analyze_stock_direct, inputs=code_input, outputs=report_output)

                gr.Examples(
                    examples=["603439", "600238", "601033", "002202"],
                    inputs=code_input,
                    label="快速尝试这些股票"
                )

            # Tab 3: 系统说明
            with gr.Tab("📖 关于"):
                gr.Markdown("""
                ## 系统架构

                本系统由三个核心模块组成：

                1. **财报反欺诈引擎** — 对资产负债表、利润表、现金流量表进行跨科目勾稽检查
                   - 存货/营收比检测
                   - 经营现金流/净利润倒挂检测
                   - 应收账款异常检测
                   - 商誉减值风险检测
                   - 资产负债率预警

                2. **股权穿透分析器** — 基于64万条股东数据构建股权关系图
                   - 多跳穿透查询（最多3层）
                   - 有效持股比例累乘计算
                   - 十大股东排名

                3. **智能对话 Agent** — 集成 LLM 的自然语言交互
                   - 自动意图识别
                   - 对话记忆管理
                   - 多工具协作调度

                ## 技术栈
                - Python + Pandas（数据处理）
                - OpenAI SDK（LLM调用）
                - Gradio（Web界面）
                - 星环科技 LLMOps 平台
                """)

    return demo


# ============================================================================
# 第七部分：命令行模式（不需要 Web 界面时使用）
# ============================================================================

def run_cli():
    """命令行交互模式"""
    print("=" * 60)
    print("  金融AI智能助手 - 命令行模式")
    print("  输入 'quit' 退出, 'demo' 看演示")
    print("=" * 60)

    agent = FinanceAgent()

    while True:
        try:
            msg = input("\n你: ").strip()
            if not msg:
                continue
            if msg.lower() == 'quit':
                print("再见！")
                break
            if msg.lower() == 'demo':
                msg = "分析一下603439的财务健康度"
                print(f"(演示模式) 你: {msg}")

            print("\nAI: ", end="", flush=True)
            reply = agent.chat(msg)
            print(reply)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n再见！")
            break


# ============================================================================
# 第八部分：入口
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Web 模式
        demo = create_web_app()
        if demo:
            demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # 快速测试模式
        print("=" * 60)
        print("  快速测试")
        print("=" * 60)
        loader = DataLoader()
        balance, income, cashflow = loader.load_financials()
        detector = FraudDetector(balance, income, cashflow)

        # 随便选几只股票测试
        test_stocks = balance['s_info_windcode'].dropna().unique()[:5]
        for code in test_stocks:
            result = detector.analyze_stock(code)
            print(f"\n{code}: 风险分={result['risk_score']}, 等级={result['risk_level']}, 预警={result['alert_count']}个")
            for alert in result['alerts']:
                print(f"  ⚠️ {alert['rule']}: {alert['detail'][:80]}...")
    else:
        # 默认：命令行模式
        run_cli()
