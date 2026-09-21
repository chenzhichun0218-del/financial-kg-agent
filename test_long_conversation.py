"""
=============================================================================
 长文本对话功能测试
 模拟 10+ 轮真实对话，测试：记忆保持、意图切换、重复检测、事实召回
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_v3 import AutonomousAgent
from datetime import datetime

agent = AutonomousAgent()

# 模拟对话场景：一个投资者在研究几只股票
conversation = [
    # 第1轮：查行情
    ("今天大盘怎么样", {
        "expected_intent": "direct_answer",
        "expected_think": False,
        "check": "should mention market indices"
    }),
    # 第2轮：财务分析
    ("分析一下 688765 的财务风险", {
        "expected_intent": "fraud_analysis",
        "expected_think": True,
        "check": "should mention risk score and alerts"
    }),
    # 第3轮：股权追问
    ("它的股东结构怎么样", {
        "expected_intent": "equity_penetration",
        "expected_think": True,
        "check": "should remember 688765 from previous round"
    }),
    # 第4轮：切换股票
    ("那 600519 茅台呢，财务健康吗", {
        "expected_intent": "fraud_analysis",
        "expected_think": True,
        "check": "should switch to 600519"
    }),
    # 第5轮：研报查询
    ("东吴证券对这两只股票有什么研报", {
        "expected_intent": "report_search",
        "expected_think": True,
        "check": "should search for both stocks or at least mention 东吴"
    }),
    # 第6轮：简单闲聊
    ("谢谢，先休息一下", {
        "expected_intent": "direct_answer",
        "expected_think": False,
        "check": "should respond casually"
    }),
    # 第7轮：回来继续
    ("刚才分析的688765，再帮我看看股权穿透", {
        "expected_intent": "equity_penetration",
        "expected_think": True,
        "check": "should remember 688765 (not 600519)"
    }),
    # 第8轮：重复提问测试
    ("它的股东结构怎么样", {
        "expected_intent": "equity_penetration",
        "expected_think": True,
        "check": "should detect as repeat and go deeper"
    }),
    # 第9轮：投资建议
    ("688765现在适合买入吗", {
        "expected_intent": "direct_answer",
        "expected_think": True,
        "check": "should give cautious analysis, mention risks"
    }),
    # 第10轮：综合回顾
    ("帮我总结一下今天聊的这几只股票", {
        "expected_intent": "direct_answer",
        "expected_think": True,
        "check": "should mention both 688765 and 600519"
    }),
]

print("=" * 70)
print("  长文本对话测试 — 10 轮连续对话")
print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 70)

results = []
pass_count = 0
fail_count = 0

for i, (question, expected) in enumerate(conversation, 1):
    print(f"\n{'─' * 50}")
    print(f"第 {i} 轮: {question}")
    print(f"期望: intent={expected['expected_intent']}, think={expected['expected_think']}")

    try:
        reply, meta = agent.chat(question)
    except Exception as e:
        print(f"  [X] 调用失败: {e}")
        results.append({"round": i, "status": "ERROR", "error": str(e)})
        fail_count += 1
        continue

    # 评估
    checks = []
    # 1. 意图匹配
    intent_ok = meta['intent'] == expected['expected_intent']
    checks.append(("意图匹配", intent_ok))

    # 2. think_flag
    think_ok = meta['think_flag'] == expected['expected_think']
    checks.append(("think_flag", think_ok))

    # 3. 回复非空
    reply_ok = len(reply) > 20
    checks.append(("回复有效", reply_ok))

    # 4. 事实检查
    fact_ok = True
    if "688765" in expected.get('check', ''):
        fact_ok = "688765" in reply
        checks.append(("提及688765", fact_ok))
    if "600519" in expected.get('check', '') and "688765" not in expected.get('check', ''):
        fact_ok = "600519" in reply
        checks.append(("提及600519", fact_ok))
    if "both" in expected.get('check', ''):
        both_ok = "688765" in reply and "600519" in reply
        checks.append(("提及两只股票", both_ok))

    # 5. 重复检测
    if "repeat" in expected.get('check', ''):
        repeat_ok = meta.get('think_reason', '').count('重复') > 0 or meta['think_flag'] == True
        checks.append(("重复检测", repeat_ok))

    all_pass = all(passed for _, passed in checks)

    status = "[OK]" if all_pass else "[!]"
    print(f"  {status} intent={meta['intent']} think={meta['think_flag']}")
    for check_name, ok in checks:
        print(f"    {'[OK]' if ok else '[X]'} {check_name}")
    print(f"  回复预览: {reply[:120]}...")

    # 记忆状态
    ents = list(agent.memory.entities.keys())[-5:]
    print(f"  记忆中的股票: {ents}")

    results.append({"round": i, "all_pass": all_pass, "checks": checks})
    if all_pass:
        pass_count += 1
    else:
        fail_count += 1

# ================================================================
# 总结
# ================================================================
print(f"\n{'=' * 70}")
print(f"  测试总结")
print(f"{'=' * 70}")
print(f"  通过: {pass_count}/{len(conversation)} 轮")
print(f"  未通过: {fail_count}/{len(conversation)} 轮")

# 记忆测试
print(f"\n  长期记忆测试:")
print(f"  对话结束后记住的股票: {list(agent.memory.entities.keys())}")
expected_in_memory = ["688765", "600519"]
for code in expected_in_memory:
    found = code in agent.memory.entities
    print(f"    {'[OK]' if found else '[X]'} {code} {'在记忆中' if found else '丢失'}")

# 统计
print(f"\n  性能统计:")
s = agent.stats
print(f"  总轮次: {s['rounds']}")
print(f"  深度推理: {s['deep_think']} ({s['deep_think']/s['rounds']*100:.0f}%)")
correct = s.get('self_correct_successes', 0)
attempts = s.get('self_correct_attempts', 0)
print(f"  自纠错: {correct}/{attempts}" if attempts > 0 else f"  自纠错: 无需触发")

# 事实召回率
fact_checks = sum(1 for r in results for c in r['checks'] if '提及' in c[0] or '重复' in c[0])
fact_passed = sum(1 for r in results for c in r['checks'] if ('提及' in c[0] or '重复' in c[0]) and c[1])
if fact_checks > 0:
    print(f"\n  关键事实召回率: {fact_passed}/{fact_checks} = {fact_passed/fact_checks*100:.0f}%")
    print(f"  目标: ≥90%")

print(f"\n  完成时间: {datetime.now().strftime('%H:%M:%S')}")
