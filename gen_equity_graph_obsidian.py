"""
=============================================================================
 股权关系 → Obsidian 可视化图谱
 把每个股东/公司生成一篇笔记，用 [[链接]] 表示持股关系
 Obsidian 图谱视图 = 股权穿透可视化
=============================================================================
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equity_penetration import EquityGraph
from datetime import datetime

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obsidian-kb-equity")

def safe_name(name):
    """清理名称中的特殊字符，确保 Obsidian 链接可用"""
    name = str(name).strip()
    # 去掉文件名不允许的字符
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, '')
    return name[:80]

def build_graph_notes(stock_code, max_depth=3):
    """为一只股票构建完整的股权关系图笔记"""
    g = EquityGraph()
    code = stock_code.strip()

    if '.' not in code:
        for suffix in ['.SH', '.SZ', '.BJ']:
            if (code + suffix) in g.reverse:
                code = code + suffix
                break

    if code not in g.reverse:
        print(f"  {stock_code}: 无数据")
        return 0

    ctrls = g.find_controller(code, max_depth=max_depth)
    created = set()

    # Step 1: 为目标公司创建笔记
    target_name = safe_name(code)
    target_note = f"""---
tags: [股权图谱, 上市公司]
stock_code: {code}
created: {datetime.now().strftime('%Y-%m-%d')}
---

# {code}

类型: 上市公司(穿透目标)

## 股东列表

"""
    shareholders = sorted(g.reverse[code].items(), key=lambda x: x[1]['pct'], reverse=True)
    for holder, info in shareholders[:20]:
        holder_safe = safe_name(holder)
        pct = info['pct']
        htype = info.get('type', '')
        bar = '█' * max(1, int(pct / 3))
        target_note += f"- [[{holder_safe}]] — **{pct:.1f}%** {bar} ({htype})\n"

    target_note += f"""
## 穿透链路

"""
    for c in ctrls[:10]:
        target_note += f"- {c['chain']} (有效{c['effective_pct']:.2f}%)\n"

    target_note += f"""
## 相关
- [[00-股权穿透知识图谱]]
"""
    with open(os.path.join(OUT, f"{target_name}.md"), 'w', encoding='utf-8') as f:
        f.write(target_note)
    created.add(target_name)

    # Step 2: 为每个股东创建笔记
    all_entities = {}  # {name: [(target, pct, type)]}

    # 直接从 reverse 图获取
    for holder, info in shareholders[:20]:
        holder_safe = safe_name(holder)
        if holder_safe not in all_entities:
            all_entities[holder_safe] = []
        all_entities[holder_safe].append({
            'target': code,
            'pct': info['pct'],
            'type': info.get('type', ''),
            'depth': 1,
        })

    # 从控制链路获取更深层的实体
    for c in ctrls:
        nodes = c['chain'].split(" -> ")
        for i, node in enumerate(nodes[:-1]):
            node_safe = safe_name(node)
            target_node = safe_name(nodes[i+1])
            if node_safe not in all_entities:
                all_entities[node_safe] = []
            # Check if this edge is already recorded
            if not any(e['target'] == target_node for e in all_entities[node_safe]):
                all_entities[node_safe].append({
                    'target': target_node,
                    'pct': c['effective_pct'],  # 使用有效持股比例
                    'type': c['controller_type'] if i == 0 else '企业',
                    'depth': i + 1,
                })

    # 为每个实体生成笔记
    for entity_name, holdings in all_entities.items():
        if entity_name in created:
            continue

        # 判断类型
        is_person = len(entity_name) <= 4  # 简短名称通常是个人
        is_stock = bool(re.match(r'\d{6}', entity_name))  # 6位数字=股票代码

        note = f"""---
tags: [{'个人' if is_person else '企业' if not is_stock else '上市公司'}, 股权图谱]
entity: {entity_name}
created: {datetime.now().strftime('%Y-%m-%d')}
---

# {entity_name}

类型: {'自然人' if is_person else '上市公司' if is_stock else '企业'}

## 持股情况

"""
        for h in holdings:
            target_safe = safe_name(h['target'])
            note += f"- 持有 [[{target_safe}]] — **{h['pct']:.1f}%**\n"

        # 如果这个实体本身也是上市公司，列出它的股东
        if entity_name in g.reverse or any(entity_name in k for k in g.reverse):
            matching_code = entity_name if entity_name in g.reverse else None
            if not matching_code:
                for k in g.reverse:
                    if entity_name in k or k in entity_name:
                        matching_code = k
                        break
            if matching_code and matching_code != code:
                sub_holders = sorted(g.reverse[matching_code].items(), key=lambda x: x[1]['pct'], reverse=True)[:5]
                if sub_holders:
                    note += f"\n## 自身股东 (作为 {matching_code})\n\n"
                    for sh, si in sub_holders:
                        sh_safe = safe_name(sh)
                        note += f"- [[{sh_safe}]] — {si['pct']:.1f}%\n"

        note += f"""
## 相关
- [[{target_name}]]
- [[00-股权穿透知识图谱]]
"""
        with open(os.path.join(OUT, f"{entity_name}.md"), 'w', encoding='utf-8') as f:
            f.write(note)
        created.add(entity_name)

    return len(created)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python gen_equity_graph_obsidian.py 688765.SH")
        print("      python gen_equity_graph_obsidian.py --deep 000498.SZ")
        sys.exit(0)

    stocks = []
    max_depth = 3
    for arg in sys.argv[1:]:
        if arg == '--deep':
            max_depth = 5
        else:
            stocks.append(arg)

    if not stocks:
        stocks = ['688765.SH']

    total = 0
    for s in stocks:
        print(f"构建 {s} 的图谱...")
        n = build_graph_notes(s, max_depth=max_depth)
        print(f"  生成了 {n} 篇笔记")
        total += n

    print(f"\n共 {total} 篇笔记 → {OUT}")
    print("在 Obsidian 中打开 obsidian-kb-equity 文件夹")
    print("点击左侧「图谱视图」即可看到股权关系网络")
