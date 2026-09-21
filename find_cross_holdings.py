"""找上市公司持有上市公司的案例 —— 这是多跳穿透的数据基础"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
from entity_resolver import build_graph_from_shareholders

df = pd.read_excel('data_raw/2.股东持股-股权穿透/clean.xlsx')
store, _ = build_graph_from_shareholders(df)

# 构建股票代码集合
stock_codes = set(store.stock_index.keys())

print(f"stock_index 中股票代码数: {len(stock_codes)}")

# 在所有股东列表中找也是上市公司的股东
cross_listed = []
for sc in list(stock_codes)[:5000]:
    eid = store.find_by_stock_code(sc)
    if not eid:
        continue
    entity = store.get_entity(eid)
    if not entity:
        continue
    shareholders = store.get_neighbors(eid, 'in')
    for sh in shareholders:
        sh_name = sh.get('source_name', '')
        # 尝试各种方式找到这个股东作为上市公司
        sh_code = None
        # 方法1: 直接查 stock_index
        for k, v in store.stock_index.items():
            if v == sh['source_id']:
                sh_code = k
                break
        # 方法2: 查 name_index
        if not sh_code:
            found_eid = store.find_by_name(sh_name)
            if found_eid:
                for k, v in store.stock_index.items():
                    if v == found_eid:
                        sh_code = k
                        break

        if sh_code and sh_code != sc:
            cross_listed.append({
                'target': sc,
                'target_name': entity.canonical_name,
                'shareholder_name': sh_name,
                'shareholder_code': sh_code,
                'pct': sh.get('pct', 0),
            })

print(f"\n找到 {len(cross_listed)} 条上市公司→上市公司的持股关系")
if cross_listed:
    print("\n示例:")
    for case in cross_listed[:10]:
        print(f"  {case['shareholder_name'][:30]} ({case['shareholder_code']})")
        print(f"    持有 {case['target']} — {case['pct']}%")

# 如果找到了，测试一下多跳
if cross_listed:
    print(f"\n=== 验证多跳穿透 ===")
    from penetration_engine import PenetrationEngine
    engine = PenetrationEngine(store)

    tested = 0
    for case in cross_listed:
        if tested >= 3:
            break
        eid = store.find_by_stock_code(case['target'])
        paths = engine.penetrate_upward(eid, max_depth=5)
        max_d = max((p.depth for p in paths), default=0)
        if max_d >= 2:
            tested += 1
            print(f"\n目标: {case['target']}")
            print(f"  最大深度: {max_d}")
            for p in paths:
                if p.depth >= 2:
                    print(f"  {p.node_names}")
                    print(f"  有效持股: {p.effective_pct}%, 终止: {p.terminals_at}")
        else:
            print(f"\n{case['target']}: 深度仍为{max_d}，穿透未生效")
            if paths:
                for p in paths[:3]:
                    if p.node_names[-1] == case['shareholder_name'][:50]:
                        print(f"  该股东在链路末尾: {p.node_names}")
                        print(f"  终止原因: {p.terminals_at}")
