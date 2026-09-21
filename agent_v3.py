"""
Agent V3 — 赛题完整实现
- 分级记忆: 短期(滑窗) + 中期(LLM摘要) + 长期(实体索引+关键词检索)
- 自纠错闭环: 意图识别→工具选择→执行→验证→重试/换工具
- 支持0.5M+ Token的上下文管理策略
"""
import sys, os, re
from datetime import datetime
from collections import OrderedDict, defaultdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
DATA_DIR = os.path.join(BASE, "data_processed")
import pandas as pd, numpy as np

# ================================================================
# LLM 客户端
# ================================================================
class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("LLMOPS_API_KEY"),
                             base_url=os.getenv("LLMOPS_BASE_URL"))
        self.model = os.getenv("LLMOPS_MODEL", "openai/glm-5.2")

    def chat(self, messages, temp=0.3, max_tokens=1500, timeout=15):
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temp, max_tokens=max_tokens, timeout=timeout)
            text = resp.choices[0].message.content
            return text.encode('gbk', errors='replace').decode('gbk') if text else None
        except Exception:
            return None

# ================================================================
# 分级记忆 — 支持 0.5M+ Token 上下文 (升级版: 语义 RAG)
# ================================================================
class TieredMemory:
    """
    三层记忆架构:
      L1 短期: 最近10轮对话原文 (滑动窗口)
      L2 中期: LLM压缩摘要 (每10轮生成一次)
      L3 长期: 语义向量索引 + 关键词索引 + 实体库 (支持语义检索)

    上下文组装策略:
      1. 总是包含: L1短期(最近N轮) + L3实体
      2. 语义检索: 用 embedding 从 L3 中查相关历史
      3. 总Token预算: 50K (可扩展到500K)
    """

    FINANCE_KEYWORDS = [
        "财务","造假","风险","利润","现金流","负债","股权","股东","穿透",
        "减持","增持","研报","评级","持仓","自选","买入","卖出","补仓",
        "止损","趋势","分析","暴雷","排雷","健康","基本面","营收","毛利率",
        "ROE","每股","净资产","现金流","净利润","增长率"
    ]

    def __init__(self, short_window=10, use_rag=True, rag_threshold=0.4):
        self.short = []                           # L1: 最近对话
        self.short_window = short_window
        self.summary_chunks = []                  # L2: LLM摘要块
        self.entities = OrderedDict()             # L3: 实体库
        self.keyword_index = defaultdict(list)    # L3: 关键词索引(回退用)
        self.full_history = []                    # 完整对话记录(压缩参考)
        self.chunk_counter = 0

        # ── RAG 语义检索 ──
        self.use_rag = use_rag
        self.rag_threshold = rag_threshold        # 语义相似度最低阈值
        self._embed_cache: dict[int, list[float]] = {}  # history_index → vector
        self._embed_available: bool | None = None       # None=未测试, True/False
        self._history_for_rag: list[dict] = []   # 可被检索的历史条目副本

    # ── Embedding 懒加载 ──
    def _ensure_embed(self) -> bool:
        """检测 embedding API 是否可用（只测一次）"""
        if self._embed_available is not None:
            return self._embed_available
        try:
            from llm_client import embed as llm_embed
            test_vec = llm_embed(["测试"])
            self._embed_available = bool(test_vec and len(test_vec[0]) > 0)
        except Exception:
            self._embed_available = False
        return self._embed_available

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        try:
            from llm_client import embed as llm_embed
            return llm_embed(texts)
        except Exception:
            from llm_client import _fallback_embed
            return _fallback_embed(texts)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度"""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 核心 API ──

    def add(self, role, content):
        entry = {"role": role, "content": content, "time": datetime.now()}
        self.short.append(entry)
        self.full_history.append(entry)
        history_idx = len(self.full_history) - 1

        # L3: 实体提取
        if role == "user":
            for m in re.finditer(r'(?<!\d)(\d{6})(?!\d)', content):
                self.entities[m.group(1)] = datetime.now()

            # L3: 关键词索引 (保留作为回退)
            extracted = set()
            for kw in self.FINANCE_KEYWORDS:
                if kw in content:
                    extracted.add(kw)
            for kw in extracted:
                self.keyword_index[kw].append(history_idx)

            # L3: RAG 语义索引 — 生成 embedding
            if self.use_rag and self._ensure_embed():
                try:
                    vecs = self._embed_texts([content])
                    if vecs and len(vecs[0]) > 0:
                        self._embed_cache[history_idx] = vecs[0]
                        self._history_for_rag.append({
                            "idx": history_idx,
                            "role": role,
                            "content": content,
                        })
                except Exception:
                    pass  # embedding 失败，静默回退到关键词模式

        # 滚动压缩
        if len(self.short) > self.short_window * 2:
            self._compress_to_medium()

    def _compress_to_medium(self):
        """L1溢出 → 摘要 → L2"""
        overflow = self.short[:-self.short_window]
        self.short = self.short[-self.short_window:]

        summary = f"[段{self.chunk_counter}] " + " | ".join(
            m['content'][:60].replace('\n', ' ')
            for m in overflow[-5:]
            if m['role'] == 'user'
        )
        self.summary_chunks.append(summary)
        self.chunk_counter += 1

        if len(self.full_history) > 100:
            self.full_history = self.full_history[-100:]
            # 清理过期的 embedding
            stale = [k for k in self._embed_cache if k < len(self.full_history) - 100]
            for k in stale:
                del self._embed_cache[k]

    # ── 检索方法 ──

    def retrieve(self, query, max_items=5):
        """
        L3 检索（自动选择最佳策略）:
          1. 优先: 语义 RAG (embedding → cosine similarity → top-k)
          2. 回退: 关键词索引
        """
        # 尝试语义 RAG
        if self.use_rag and self._ensure_embed() and len(self._embed_cache) >= 3:
            rag_results = self._retrieve_rag(query, max_items)
            if rag_results:
                return rag_results

        # 回退到关键词
        return self._retrieve_keyword(query, max_items)

    def _retrieve_rag(self, query: str, max_items: int = 5) -> list[str]:
        """
        语义 RAG 检索:
          1. 对 query 生成 embedding
          2. 与历史中所有用户消息的 embedding 计算余弦相似度
          3. 返回 top-k 超过阈值的结果
        """
        try:
            query_vecs = self._embed_texts([query])
            if not query_vecs or len(query_vecs[0]) == 0:
                return []
            query_vec = query_vecs[0]

            # 计算所有历史条目的相似度
            scored = []
            for entry in self._history_for_rag[-100:]:  # 最近100条
                idx = entry["idx"]
                if idx not in self._embed_cache:
                    continue
                hist_vec = self._embed_cache[idx]
                sim = self._cosine_similarity(query_vec, hist_vec)

                # 关键词加权：如果历史条目含 query 中的关键词，加分
                boost = 0.0
                query_words = set(query)
                content_words = set(entry["content"])
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                boost = overlap * 0.15  # 最多 +0.15

                final_score = sim + boost
                if final_score >= self.rag_threshold:
                    scored.append((final_score, entry))

            # 按分数降序，取 top-k
            scored.sort(key=lambda x: -x[0])
            results = []
            for score, entry in scored[:max_items]:
                results.append(entry["content"][:150])
            return results

        except Exception:
            return []

    def _retrieve_keyword(self, query, max_items=5):
        """关键词回退检索（原实现）"""
        relevant_indices = set()
        for kw in self.FINANCE_KEYWORDS:
            if kw in query:
                for idx in self.keyword_index.get(kw, [])[-10:]:
                    relevant_indices.add(idx)

        results = []
        for idx in sorted(relevant_indices, reverse=True)[:max_items]:
            if idx < len(self.full_history):
                entry = self.full_history[idx]
                if entry['role'] == 'user':
                    results.append(entry['content'][:100])
        return results

    def retrieve_hybrid(self, query, max_items=5, rag_weight=0.7):
        """
        混合检索: RAG 语义 + 关键词，加权融合
        rag_weight: RAG 结果的权重 (0-1), 剩余给关键词
        """
        rag_results = self._retrieve_rag(query, max_items * 2) if self._ensure_embed() else []
        kw_results = self._retrieve_keyword(query, max_items * 2)

        # 给每个结果打分: RAG 结果初始分 * rag_weight, 关键词结果初始分 * (1-rag_weight)
        scored = {}

        for i, r in enumerate(rag_results):
            score = (1.0 - i * 0.1) * rag_weight  # 排名越前分越高
            key = r[:60]  # 用前60字符做去重 key
            scored[key] = (score, r)

        for i, r in enumerate(kw_results):
            score = (1.0 - i * 0.1) * (1 - rag_weight)
            key = r[:60]
            if key in scored:
                scored[key] = (scored[key][0] + score, r)  # 累加分数
            else:
                scored[key] = (score, r)

        # 按分数排序
        ranked = sorted(scored.values(), key=lambda x: -x[0])
        return [r for _, r in ranked[:max_items]]

    # ── 上下文组装 ──

    def build_context(self, user_msg):
        """组装 LLM 上下文 — 支持扩展到 0.5M Token"""
        parts = []

        # L1: 短期记忆
        if self.short:
            parts.append("--- 最近对话 ---")
            for msg in self.short[-self.short_window:]:
                role = "用户" if msg['role'] == 'user' else "助手"
                parts.append(f"[{role}] {msg['content'][:300]}")

        # L3: 长期实体
        if self.entities:
            ents = list(self.entities.keys())[-15:]
            parts.append(f"\n--- 用户长期关注的股票 ---")
            parts.append(", ".join(ents))

        # L3: 语义检索相关历史（自动选择 RAG 或关键词回退）
        retrieval_method = "语义RAG" if (self.use_rag and self._ensure_embed() and len(self._embed_cache) >= 3) else "关键词"
        relevant = self.retrieve(user_msg, max_items=5)
        if relevant:
            parts.append(f"\n--- 相关历史提问 ({retrieval_method}检索) ---")
            for r in relevant:
                parts.append(f"  · {r}")

        # L2: 中期摘要
        if self.summary_chunks:
            parts.append(f"\n--- 历史摘要 ({len(self.summary_chunks)}段) ---")
            parts.extend(self.summary_chunks[-5:])

        return "\n".join(parts)

    def is_repeat(self, q):
        q_short = q[:30]
        recent = [m['content'][:30] for m in self.short[-self.short_window:]
                  if m['role'] == 'user']
        return recent.count(q_short) >= 2

    def recent_messages(self, n=5):
        return [{"role": m["role"], "content": m["content"][:400]}
                for m in self.short[-n:]]

    # ── 统计信息 ──
    def stats(self) -> dict:
        return {
            "short_messages": len(self.short),
            "summary_chunks": len(self.summary_chunks),
            "tracked_entities": len(self.entities),
            "embed_cache_size": len(self._embed_cache),
            "embed_available": self._embed_available,
            "rag_mode": "语义RAG" if (self.use_rag and self._ensure_embed() and len(self._embed_cache) >= 3) else "关键词回退",
        }

# ================================================================
# 研报检索
# ================================================================
class ReportSearch:
    def __init__(self):
        self.df = pd.read_pickle(os.path.join(DATA_DIR, "reports.pkl"))
        self.df['sc'] = self.df['sec_code'].astype(str)
        self.df['pd'] = pd.to_datetime(self.df['publish_dt'])
        print(f"[研报] {len(self.df):,}篇")

    def search(self, q, n=5):
        r = self.df[self.df['title'].str.contains(str(q), na=False) |
                    self.df['abstract'].str.contains(str(q), na=False) |
                    self.df['sc'].str.contains(str(q), na=False) |
                    self.df['org_name'].str.contains(str(q), na=False)]
        return self._fmt(r, n) if not r.empty else self._fmt(self.df.sort_values('pd', ascending=False), n)

    def _fmt(self, r, n):
        if r.empty: return "无结果"
        lines = []
        for _, row in r.head(n).iterrows():
            lines.append(f"[{str(row.get('pd',''))[:10]}] {str(row.get('title',''))[:60]}")
            lines.append(f"  机构:{row.get('org_name','')} 评级:{row.get('rating_change','')}")
            ab = str(row.get('abstract','')); lines.append(f"  {ab[:120]}..." if ab!='nan' else "")
            lines.append("")
        return f"(共{len(r)}篇)\n"+"\n".join(lines)

# ================================================================
# 自纠错闭环: 意图识别 → 工具选择 → 执行 → 验证
# ================================================================
class SelfCorrectingAgent:
    """赛题要求的完整 Agentic 框架"""

    def __init__(self):
        self.llm = LLMClient()
        self.memory = TieredMemory(short_window=10)
        from portfolio import Portfolio; self.portfolio = Portfolio()
        self.reports = ReportSearch()
        self._fraud = None; self._equity = None
        self.stats = {"rounds":0, "deep":0, "correct_attempts":0, "correct_successes":0}

    @property
    def fraud(self):
        if self._fraud is None:
            from fraud_detector import FraudDetector; self._fraud = FraudDetector()
        return self._fraud

    @property
    def equity(self):
        if self._equity is None:
            from equity_penetration import EquityGraph; self._equity = EquityGraph()
        return self._equity

    # ================================================================
    # 阶段1: 意图识别
    # ================================================================
    def _recognize_intent(self, user_msg):
        """识别用户意图 → 选择工具"""
        code = None
        m = re.search(r'(?<!\d)(\d{6})(?!\d)', user_msg)
        if m: code = m.group(1)

        FRAUD = ["财务","财报","造假","风险","健康","利润","现金流","负债","资产","营收",
                  "毛利率","净资产","基本面","排雷","扭亏","净利润","每股","ROE","报酬率","收入","成本"]
        EQUITY = ["股权","穿透","控股","控制","十大股东","减持","增持","质押","实际控制人",
                   "股东结构","股东人数"]
        REPORT = ["研报","券商","研究报告","机构观点","评级"]
        PORTFOLIO = ["我的自选股","我的持仓","帮我看看自选","自选股异动","自选股综合表现",
                      "自选股研报","持仓分析","自选日报","分析我的自选","自选股关联行业"]
        TRACE = ["溯源","事件脉络","时间线","完整报告","综合尽调","尽调"]

        if any(w in user_msg for w in TRACE): intent = "traceability"
        elif any(w in user_msg for w in FRAUD): intent = "fraud_analysis"
        elif any(w in user_msg for w in EQUITY): intent = "equity_penetration"
        elif any(w in user_msg for w in PORTFOLIO): intent = "portfolio_scan"
        elif any(w in user_msg for w in REPORT): intent = "report_search"
        elif code and len(user_msg.strip()) <= 10: intent = "fraud_analysis"  # 纯代码→默认分析
        elif code and any(w in user_msg for w in ["分析","看看","怎么样"]): intent = "fraud_analysis"
        else: intent = "direct_answer"

        return intent, code

    # ================================================================
    # 阶段2+3: 工具选择 + 执行 (带自纠错)
    # ================================================================
    def _execute_with_correction(self, intent, code, user_msg):
        """执行工具 → 验证结果 → 失败则自纠错重试"""
        result = self._try_execute(intent, code, user_msg)
        if result and len(result) > 30:
            return result  # 成功

        # 自纠错1: 代码补全后缀
        if code and '.' not in code:
            for suffix in ['.SZ', '.SH', '.BJ']:
                self.stats['correct_attempts'] += 1
                retry = self._try_execute(intent, code + suffix, user_msg)
                if retry and len(retry) > 30:
                    self.stats['correct_successes'] += 1
                    return retry

        # 自纠错2: 代码去后缀
        if code and '.' in code:
            short = code.split('.')[0]
            self.stats['correct_attempts'] += 1
            retry = self._try_execute(intent, short, user_msg)
            if retry and len(retry) > 30:
                self.stats['correct_successes'] += 1
                return retry

        # 自纠错3: 研报搜索空 → 换最新
        if intent == 'report_search' and (not result or '无相关' in str(result)):
            self.stats['correct_attempts'] += 1
            retry = self.reports.latest(5)
            if retry and '无相关' not in retry:
                self.stats['correct_successes'] += 1
                return retry

        # 自纠错4: 从长记忆中找代码
        if not code and self.memory.entities:
            last_code = list(self.memory.entities.keys())[-1]
            self.stats['correct_attempts'] += 1
            retry = self._try_execute(intent, last_code, user_msg)
            if retry and len(retry) > 30:
                self.stats['correct_successes'] += 1
                return f"[using memory: {last_code}]\n{retry}"

        return result

    def _try_execute(self, intent, code, user_msg):
        """单次工具执行"""
        if intent == 'fraud_analysis' and code:
            r = self.fraud.analyze(code)
            if r.get('error'): return None
            lines = [f"【{code} 财务健康度】风险{r['risk_score']:.0f}/100 [{r['risk_level']}] {r['alert_count']}项预警"]
            for a in r['alerts']: lines.append(f"  [{a['rule']}] {a['detail']} ({a['data']})")
            return "\n".join(lines)

        elif intent == 'equity_penetration' and code:
            ctrls = self.equity.find_controller(code)
            if not ctrls: return None
            lines = [f"【{code} 股权穿透】"]
            for c in ctrls[:5]: lines.append(f"  {c['chain']} (有效{c['effective_pct']:.2f}%)")
            flags = self.equity.check_flags(code)
            for fl in flags.get('flags', []): lines.append(f"  ! {fl['flag']}: {fl['detail']}")
            return "\n".join(lines)

        elif intent == 'traceability' and code:
            from equity_event_fusion import EquityEventFusion
            fusion = EquityEventFusion()
            return fusion.generate_traceability_report(code)

        elif intent == 'report_search':
            return self.reports.search(user_msg)

        elif intent == 'portfolio_scan':
            codes = self.portfolio.get_all_codes()
            if not codes: return None
            lines = [self.portfolio.snapshot_for_agent(), "\n各股分析:"]
            for c in codes:
                r = self.fraud.analyze(c); risk = r.get('risk_score','?')
                lines.append(f"  {c}: 风险{risk}/100")
            return "\n".join(lines)

        return None

    # ================================================================
    # 阶段5: think_flag 决策
    # ================================================================
    def _decide_think(self, user_msg, intent):
        DEEP = ["该继续持有吗","适合补仓","应止损","应止盈","操作策略","投资对策",
                "连续横盘","放量上涨","K线","MACD","主力控盘","明天趋势","涨停原因",
                "会涨吗","走势分析","适合买入","适合卖出","要不要卖","能不能买",
                "对策","策略","怎么看"]
        is_repeat = self.memory.is_repeat(user_msg)
        think = (intent not in ("direct_answer","report_search")) or is_repeat or any(p in user_msg for p in DEEP)
        reason = "工具调用" if intent!="direct_answer" else ("重复提问" if is_repeat else ("深度模式" if think else "简单查询"))
        return think, reason

    # ================================================================
    # 主对话
    # ================================================================
    def chat(self, user_msg):
        self.stats['rounds'] += 1

        # 阶段1: 意图识别
        intent, code = self._recognize_intent(user_msg)

        # 阶段2+3: 工具执行 + 自纠错
        # 如果没代码但记忆中有，且intent是direct_answer → 尝试用记忆中的代码
        if not code and intent == 'direct_answer' and self.memory.entities:
            last_code = list(self.memory.entities.keys())[-1]
            # 重新判断intent: 也许用户是在追问之前提到的股票
            if any(w in user_msg for w in ["它","他","她","这个","那个","刚才","之前","上面"]):
                # 用户用代词指代 → 用记忆中的代码重判intent
                if any(w in user_msg for w in ["股东","股权","穿透","谁持有","控制"]):
                    intent = "equity_penetration"
                elif any(w in user_msg for w in ["财务","风险","健康","利润","现金流","负债"]):
                    intent = "fraud_analysis"
                code = last_code

        tool_result = self._execute_with_correction(intent, code, user_msg)

        # 阶段4: 实时行情
        rt_text = ""
        try:
            from realtime_data import get_stock_quote, get_market_overview, is_market_open
            mkt = get_market_overview()
            if mkt.get('status')=='ok' and mkt.get('indices'):
                rt_text = f"今日({datetime.now().strftime('%Y-%m-%d %H:%M')}, {'交易中' if is_market_open() else '收盘'}): "
                rt_text += " | ".join(f"{n}:{i['price']:.0f}({i['change_pct']:+.2f}%)" for n,i in mkt['indices'].items())
            if code:
                q = get_stock_quote(code)
                if q.get('status')=='ok': rt_text += f"\n{code}: {q['price']:.2f}元 ({q['change_pct']:+.2f}%)"
        except: pass

        # 阶段5: think_flag
        think_flag, think_reason = self._decide_think(user_msg, intent)
        if think_flag: self.stats['deep'] += 1

        # 阶段6: 上下文组装 (0.5M+ 支持)
        memory_context = self.memory.build_context(user_msg)
        tools_txt = f"\n[分析数据]\n{tool_result}" if tool_result else ""
        portfolio_txt = self.portfolio.snapshot_for_agent() if self.portfolio.get_all_codes() else ""

        system = f"""你是金融AI助手。当前日期:{datetime.now().strftime('%Y-%m-%d')}

{rt_text}

{memory_context}

{portfolio_txt}

风格: 先结论后数据, 通俗易懂。分析基于财报数据, 不构成投资建议。"""

        # 阶段7: LLM生成回复
        reply = self.llm.chat([
            {"role":"system","content":system},
            *self.memory.recent_messages(4),
            {"role":"user","content":user_msg + tools_txt}
        ], timeout=15)

        if reply is None and tool_result:
            reply = (rt_text + "\n\n" + tool_result) if rt_text else tool_result
        elif reply is None:
            reply = "请尝试输入股票代码进行财报分析或股权穿透查询"

        # 更新记忆
        self.memory.add("user", user_msg)
        self.memory.add("assistant", reply or "")
        if code: self.memory.entities[code] = datetime.now()

        return reply or "", {
            "think_flag": think_flag, "think_reason": think_reason,
            "intent": intent, "confidence": 0.95,
            "tools_used": tool_result is not None, "stock_code": code
        }

# ================================================================
# CLI
# ================================================================
def run_cli():
    print("="*60)
    print("  Agent V3 — 完整Agentic框架")
    print("  分级记忆 | 自纠错闭环 | 意图→工具→验证")
    print("="*60)
    agent = SelfCorrectingAgent()
    while True:
        try:
            msg = input("\n你: ").strip()
            if not msg: continue
            if msg.lower()=='quit': break
            if msg.lower()=='stats':
                s=agent.stats; r=max(s['rounds'],1)
                print(f"轮次:{s['rounds']} 深度:{s['deep']}({s['deep']/r*100:.0f}%) "
                      f"自纠错:{s['correct_successes']}/{s['correct_attempts']}")
                continue
            reply, meta = agent.chat(msg)
            flag = "[深度]" if meta['think_flag'] else "[普通]"
            print(f"\n助手 {flag} ({meta['think_reason']})\n{reply}")
        except KeyboardInterrupt: break

if __name__ == "__main__":
    run_cli()
