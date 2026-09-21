"""
=============================================================================
 股权穿透 Obsidian 笔记生成器
 输入股票代码 → 自动穿透 → 生成可读笔记
 用法: python gen_equity_note.py 688765.SH
       python gen_equity_note.py --batch 688765.SH 603439.SH 300838.SZ
=============================================================================
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equity_penetration import EquityGraph
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obsidian-kb-equity")

def gen_note(stock_code):
    """为一只股票生成股权穿透笔记"""
    g = EquityGraph()
    code = stock_code.strip()

    # 模糊补全
    if '.' not in code:
        for suffix in ['.SH', '.SZ', '.BJ']:
            full = code + suffix
            if full in g.reverse:
                code = full
                break

    if code not in g.reverse:
        return f"# {stock_code}\n\n未在股东数据库中找到 {stock_code}"

    # 穿透分析
    ctrls = g.find_controller(code, max_depth=4)
    flags = g.check_flags(code)
    top = sorted(g.reverse[code].items(), key=lambda x: x[1]['pct'], reverse=True)[:10]

    # 统计
    max_depth = max((c['depth'] for c in ctrls), default=0)
    has_person = any(c['controller_type'] == '个人' for c in ctrls)
    multi_hop = any(c['depth'] >= 2 for c in ctrls)

    # 生成 Markdown
    name_md = code.replace('.', '_')
    lines = []
    lines.append(f"---")
    lines.append(f"tags: [股权穿透, 案例]")
    lines.append(f"stock: {code}")
    lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"max_depth: {max_depth}")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"# {code} 股权穿透分析")
    lines.append(f"")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 最大穿透深度: **{max_depth} 层**")
    lines.append(f"> 控制类型: {'自然人实控' if has_person else '企业/其他控制'}")
    lines.append(f"> {'存在多层穿透链' if multi_hop else '全部为直接持股'}")
    lines.append(f"")

    # 前十大股东
    lines.append(f"## 前十大股东")
    lines.append(f"")
    lines.append(f"| # | 股东 | 持股比例 | 类型 |")
    lines.append(f"|---|------|----------|------|")
    for i, (holder, info) in enumerate(top, 1):
        htype = info.get('type', '未知')
        pct = info['pct']
        bar = '█' * max(1, int(pct / 5))
        lines.append(f"| {i} | {holder[:25]} | {pct:.2f}% {bar} | {htype} |")
    lines.append(f"")

    # 控制链路
    lines.append(f"## 控制链路 (共 {len(ctrls)} 条)")
    lines.append(f"")

    # 按深度分层
    for depth in range(1, max_depth + 1):
        depth_ctrls = [c for c in ctrls if c['depth'] == depth]
        if not depth_ctrls:
            continue
        lines.append(f"### 第 {depth} 层 ({len(depth_ctrls)} 条)")
        lines.append(f"")
        for i, c in enumerate(depth_ctrls[:10], 1):
            tag = "👤" if c['controller_type'] == '个人' else "🏢"
            lines.append(f"{i}. {tag} **{c['chain']}**")
            lines.append(f"   - 有效持股: **{c['effective_pct']:.2f}%**")
            if depth >= 2:
                # 标出每一跳
                nodes = c['chain'].split(" -> ")
                lines.append(f"   - 链路详解:")
                for j, (fr, to) in enumerate(zip(nodes[:-1], nodes[1:])):
                    pct_val = c.get('pcts', [0]*(depth))[j] if j < len(c.get('pcts',[])) else 0
                    lines.append(f"     {j+1}. {fr[:20]} →({pct_val}%)→ {to[:20]}")
            lines.append(f"")

    # 股权风险
    if flags.get('flags'):
        lines.append(f"## 股权风险信号")
        lines.append(f"")
        for f in flags['flags']:
            lines.append(f"- ⚠️ **{f['flag']}**: {f['detail']}")
        lines.append(f"")

    # 相关链接
    lines.append(f"## 相关笔记")
    lines.append(f"")
    lines.append(f"- [[00-股权穿透知识图谱]]")
    lines.append(f"- [[什么是股权穿透]]")
    lines.append(f"- [[BFS多跳算法]]")
    lines.append(f"- [[有效持股计算]]")
    if flags.get('flags'):
        lines.append(f"- [[股权风险信号]]")
    lines.append(f"")

    content = "\n".join(lines)

    # 保存
    filename = f"穿透-{name_md}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    return f"已生成: {filename} (深度{max_depth}层, {len(ctrls)}条链路, {len(flags.get('flags',[]))}个风险信号)"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("用法: python gen_equity_note.py 688765.SH")
        print("      python gen_equity_note.py --batch 688765.SH 603439.SH")
        sys.exit(0)

    if sys.argv[1] == '--batch':
        stocks = sys.argv[2:]
    else:
        stocks = [sys.argv[1]]

    for s in stocks:
        result = gen_note(s)
        print(result)

    print(f"\n笔记保存在: {OUTPUT_DIR}")
    print("用 Obsidian 打开 obsidian-kb-equity 文件夹即可查看")
