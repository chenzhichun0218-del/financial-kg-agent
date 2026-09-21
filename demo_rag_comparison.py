"""
================================================================================
 RAG vs 关键词检索 — 对比演示
 展示升级后的语义 RAG 相比原关键词匹配的优势
================================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from agent_v3 import TieredMemory

# 模拟10轮金融对话
CONVERSATIONS = [
    "帮我分析一下603439的财务健康度",
    "贵州三力的毛利率最近有变化吗，成本控制怎么样",
    "它的应收账款占比高不高，回款能力如何",
    "603439的大股东是谁，股权结构稳定吗",
    "张海实际控制了多少家公司，资本版图有多大",
    "最近这家公司有什么风险事件或者监管关注吗",
    "它的现金流情况怎么样，经营质量好不好",
    "和同行业比，三力的盈利水平处于什么位置",
    "这个价位适合长期持有吗，有没有爆雷风险",
    "帮我看看自选股里哪些最近有减持公告",
]

def test_keyword():
    """原版：纯关键词匹配"""
    m = TieredMemory(use_rag=False)
    for c in CONVERSATIONS:
        m.add("user", c)
        m.add("assistant", f"分析结果: {c[:20]}...")

    # 测试1: 同义词 — "赚钱能力" vs "盈利水平"
    q1 = "这家公司赚钱能力怎么样"
    r1 = [r[:50] for r in m.retrieve(q1)]
    print(f"查询: '{q1}'")
    print(f"  关键词匹配结果: {r1 if r1 else '(空 — 找不到)'}")
    print(f"  原因: '赚钱'不在55个预定义关键词中")
    print()

    # 测试2: 同义词 — "经营状况" vs "经营质量"
    q2 = "它的经营状况好不好"
    r2 = [r[:50] for r in m.retrieve(q2)]
    print(f"查询: '{q2}'")
    print(f"  关键词匹配结果: {r2 if r2 else '(空 — 找不到)'}")
    print(f"  原因: '经营状况'不在关键词列表中")
    print()

    # 测试3: 能匹配的关键词
    q3 = "这个股票有没有财务风险"
    r3 = [r[:50] for r in m.retrieve(q3)]
    print(f"查询: '{q3}'")
    print(f"  关键词匹配结果: {len(r3)} 条 (命中'财务'+'风险')")
    print()
    print("=" * 60)


def test_rag():
    """新版：语义 RAG"""
    m = TieredMemory(use_rag=True)
    for c in CONVERSATIONS:
        m.add("user", c)
        m.add("assistant", f"分析结果: {c[:20]}...")

    stats = m.stats()
    print(f"RAG 模式: {stats['rag_mode']}")
    print(f"Embedding 缓存: {stats['embed_cache_size']} 条")
    print(f"Embedding API: {'可用' if stats['embed_available'] else '回退(bigram哈希)'}")
    print()

    # 测试1: "赚钱能力" — 语义上最接近"盈利水平"和"毛利率"
    q1 = "这家公司赚钱能力怎么样"
    r1 = m.retrieve(q1, max_items=3)
    print(f"查询: '{q1}'")
    for i, r in enumerate(r1, 1):
        print(f"  {i}. {r[:80]}")
    if not r1:
        print("  (无结果)")
    print()

    # 测试2: "经营状况" — 语义上接近"经营质量"和"现金流情况"
    q2 = "它的经营状况好不好"
    r2 = m.retrieve(q2, max_items=3)
    print(f"查询: '{q2}'")
    for i, r in enumerate(r2, 1):
        print(f"  {i}. {r[:80]}")
    if not r2:
        print("  (无结果)")
    print()

    # 测试3: "老板是谁" — 语义上接近"大股东"和"实际控制"
    q3 = "这家公司的老板是谁"
    r3 = m.retrieve(q3, max_items=3)
    print(f"查询: '{q3}'")
    for i, r in enumerate(r3, 1):
        print(f"  {i}. {r[:80]}")
    if not r3:
        print("  (无结果)")
    print()
    print("=" * 60)


def test_hybrid():
    """混合检索：RAG + 关键词加权融合"""
    m = TieredMemory(use_rag=True)
    for c in CONVERSATIONS:
        m.add("user", c)
        m.add("assistant", f"分析结果: {c[:20]}...")

    # 测试: "财务和股权方面有没有风险"
    q = "财务和股权方面有没有风险"
    print(f"查询: '{q}'")
    print()
    print("--- 纯 RAG 结果 ---")
    for i, r in enumerate(m._retrieve_rag(q, 3), 1):
        print(f"  {i}. {r[:80]}")
    if not m._retrieve_rag(q, 3):
        print("  (无结果)")
    print()
    print("--- 纯关键词结果 ---")
    for i, r in enumerate(m._retrieve_keyword(q, 3), 1):
        print(f"  {i}. {r[:80]}")
    if not m._retrieve_keyword(q, 3):
        print("  (无结果)")
    print()
    print("--- 混合检索结果 (RAG权重0.7) ---")
    for i, r in enumerate(m.retrieve_hybrid(q, 3, rag_weight=0.7), 1):
        print(f"  {i}. {r[:80]}")
    if not m.retrieve_hybrid(q, 3):
        print("  (无结果)")


if __name__ == "__main__":
    print("=" * 60)
    print("  原版关键词匹配 vs 语义RAG 对比演示")
    print("=" * 60)
    print()

    print("【原版: 纯关键词匹配】")
    print("-" * 40)
    test_keyword()

    print()
    print("【升级版: 语义 RAG】")
    print("-" * 40)
    test_rag()

    print()
    print("【混合检索: RAG + 关键词】")
    print("-" * 40)
    test_hybrid()
