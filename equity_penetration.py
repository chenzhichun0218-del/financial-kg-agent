"""
=============================================================================
 股权穿透模块
 功能：构建股权关系图 + 多跳路径查询 + 有效持股计算 + 控制链识别
 用法：python equity_penetration.py 300838.SZ
       python equity_penetration.py --top 300838.SZ
       python equity_penetration.py --chain 王旭宁
=============================================================================
"""
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict, deque

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data_processed")


# 金融中介机构 — 它们持股再多也不是实际控制人
FINANCIAL_INTERMEDIARIES = [
    '香港中央結算', '香港中央结算', '中国证券金融', '中央汇金',
    '社保基金', '基本养老保险', '养老金', '企业年金',
    '中证500', '中证1000', '沪深300', '创业板', '科创50',
    '上证50', '深证100', '交易型开放式指数', 'ETF',
    '摩根士丹利', '摩根大通', '瑞士联合银行', '巴克莱银行',
    '花旗银行', '高盛', '美林', '德意志银行',
    '中信证券', '华泰证券', '国泰君安', '海通证券',
    '南方基金', '华夏基金', '易方达', '嘉实', '博时',
    '招商基金', '广发基金', '富国', '鹏华', '汇添富',
    '中国人寿', '中国平安', '太平洋保险', '新华人寿',
    '合格境外机构投资者', 'QFII', 'RQFII',
    '陆股通', '深股通', '沪股通',
    '转融通', '融资融券',
]


class EquityGraph:
    """股权关系图 — 支持多跳穿透查询"""

    def __init__(self):
        print("[股权穿透] 构建股权关系图...")

        sh = pd.read_pickle(os.path.join(DATA_DIR, "shareholders.pkl"))

        # 图结构：{实体名: {目标: {pct, type}}}
        # 正向：股东 -> 持股的目标公司
        self.forward = defaultdict(dict)
        # 反向：公司 -> 所有股东
        self.reverse = defaultdict(dict)

        # 实体别名映射（处理名称不统一的问题）
        self.aliases = {}

        # 过滤金融中介
        mask_fin = sh['s_holder_name'].apply(self._is_financial)
        sh_filtered = sh[~mask_fin]
        skipped = mask_fin.sum()

        for _, row in sh_filtered.iterrows():
            holder = str(row['s_holder_name']).strip()
            target = str(row['s_info_windcode']).strip()
            pct = float(row['s_holder_pct']) if pd.notna(row['s_holder_pct']) else 0
            htype = row.get('holder_type', '未知')

            if pct <= 0:
                continue

            # 正反向图
            self.forward[holder][target] = {'pct': pct, 'type': htype}
            self.reverse[target][holder] = {'pct': pct, 'type': htype}

        print(f"[股权穿透] 节点: {len(self.forward):,} (过滤掉{skipped:,}个金融中介)")
        print(f"[股权穿透] 边: {sum(len(v) for v in self.forward.values()):,} 条")

        # 构建公司名 → 股票代码映射（从研报数据）
        self._build_name_mapping()

    def _build_name_mapping(self):
        """构建公司名 → 股票代码的映射，用于多跳穿透"""
        self.name_to_code = {}
        self.code_to_name = {}
        try:
            reports = pd.read_pickle(os.path.join(DATA_DIR, "reports.pkl"))
            for _, r in reports[['sec_code', 'sec_name']].drop_duplicates().iterrows():
                name = str(r['sec_name']).strip()
                code = str(r['sec_code']).strip()
                if name and name != 'nan':
                    self.name_to_code[name] = code
                    self.code_to_name[code] = name
                    # 去掉后缀的简称也加入映射
                    core = name.replace('股份有限公司','').replace('有限责任公司','').replace('有限公司','')
                    if core != name:
                        self.name_to_code[core] = code
        except Exception:
            pass
        print(f"[股权穿透] 公司名映射: {len(self.name_to_code)} 条")

    def _resolve_name_to_code(self, name):
        """把公司名称转为股票代码（带交易所后缀），用于穿透查询"""
        name = str(name).strip()
        short = None
        if name in self.name_to_code:
            short = self.name_to_code[name]
        else:
            # 模糊匹配
            for known, code in self.name_to_code.items():
                if len(name) >= 4 and len(known) >= 4:
                    if name[:4] == known[:4] or known[:4] in name or name[:4] in known:
                        short = code
                        break
        if not short:
            return None
        # 找完整代码（带 .SH/.SZ/.BJ）
        for full_code in self.reverse:
            if short in full_code:
                return full_code
        return None

    def _is_financial(self, name):
        """判断是否为金融中介（不参与实际控制）"""
        name_str = str(name)
        return any(kw in name_str for kw in FINANCIAL_INTERMEDIARIES)

    def _normalize(self, name):
        """名称标准化"""
        name = str(name).strip()
        # 去除常见后缀差异
        for suffix in ['股份有限公司', '有限责任公司', '有限公司', '（上市）', '(上市)']:
            if name.endswith(suffix):
                base = name[:-len(suffix)]
                self.aliases[base] = name
        return name

    # ================================================================
    # 多跳穿透查询
    # ================================================================

    def penetrate(self, entity, max_depth=4, min_pct=1.0):
        """
        从实体出发，BFS 查找所有穿透链路
        返回: [{path, depths, effective_pct, edges}]
        """
        entity = str(entity).strip()
        results = []

        # BFS: 队列存 (当前实体, 路径, 累计持股比例链, 边详情链)
        queue = deque()
        queue.append((entity, [entity], [], []))
        visited = set()

        while queue:
            current, path, pcts, edges = queue.popleft()
            depth = len(path) - 1

            if depth >= max_depth:
                continue

            if current in visited:
                continue
            visited.add(current)

            # 查找当前实体持有的所有目标
            holdings = self.forward.get(current, {})

            # 也尝试模糊匹配
            if not holdings:
                for key in self.forward:
                    if current in key or key in current:
                        holdings = self.forward[key]
                        current = key  # 标准化为全名
                        break

            for target, info in holdings.items():
                edge_pct = info['pct']
                if edge_pct < min_pct:
                    continue

                new_path = path + [target]
                new_pcts = pcts + [edge_pct]
                new_edges = edges + [{
                    'from': current,
                    'to': target,
                    'pct': edge_pct,
                    'type': info.get('type', '')
                }]

                effective = self._calc_effective(new_pcts)

                results.append({
                    'chain': ' -> '.join(new_path),
                    'depth': len(new_path) - 1,
                    'effective_pct': effective,
                    'edges': new_edges,
                    'pcts': new_pcts,
                })

                queue.append((target, new_path, new_pcts, new_edges))

        # 按深度和有效持股排序
        results.sort(key=lambda x: (x['depth'], -x['effective_pct']))
        return results

    def _calc_effective(self, pcts):
        """多层有效持股 = 每层持股比例累乘"""
        result = 1.0
        for p in pcts:
            result *= p / 100.0
        return round(result * 100, 4)

    # ================================================================
    # 反向追溯：谁在控制这家公司
    # ================================================================

    def find_controller(self, stock_code, max_depth=4):
        """
        反向追溯：从一家上市公司往上找，看最终是谁在控制
        返回控制链路
        """
        code = self._resolve_code(stock_code)
        if not code:
            return []

        chains = []
        queue = deque()
        queue.append((code, [code], [], []))

        while queue:
            current, path, pcts, edges = queue.popleft()
            depth = len(path) - 1

            if depth >= max_depth:
                continue

            shareholders = self.reverse.get(current, {})

            # 按持股比排序，只取前 5 大
            top_holders = sorted(shareholders.items(), key=lambda x: x[1]['pct'], reverse=True)[:5]

            for holder, info in top_holders:
                edge_pct = info['pct']
                htype = info.get('type', '')

                new_path = [holder] + path
                new_pcts = [edge_pct] + pcts
                new_edges = [{
                    'from': holder,
                    'to': current,
                    'pct': edge_pct,
                    'type': htype,
                }] + edges

                effective = self._calc_effective(new_pcts)

                chains.append({
                    'chain': ' -> '.join(new_path),
                    'depth': len(new_path) - 1,
                    'effective_pct': effective,
                    'controller': holder,
                    'controller_type': htype,
                    'target': code,
                })

                # 如果是企业，尝试找到它的股票代码，继续往上追溯
                if htype == '企业':
                    deeper_code = self._resolve_name_to_code(holder)
                    if deeper_code and deeper_code in self.reverse:
                        queue.append((deeper_code, new_path, new_pcts, new_edges))
                # 如果是个人，到达终点

        # 去重（同一控制人只保留最深最高的一条）
        chains.sort(key=lambda x: (x['depth'], -x['effective_pct']), reverse=True)

        seen_controllers = set()
        deduped = []
        for c in chains:
            if c['controller'] not in seen_controllers:
                deduped.append(c)
                seen_controllers.add(c['controller'])

        return deduped[:20]

    def _resolve_code(self, stock_code):
        """模糊匹配股票代码"""
        code = str(stock_code).strip()
        if '.' not in code:
            for key in self.reverse:
                if code in key:
                    return key
        return code if code in self.reverse else None

    # ================================================================
    # 正向挖掘：一个实体控制了多少公司
    # ================================================================

    def find_empire(self, entity_name):
        """
        正向挖掘：输入一个人名或公司名，找出他/它控制的所有上市公司
        """
        entity = str(entity_name).strip()
        controlled = []

        # 直接持股
        direct = self.forward.get(entity, {})

        # 模糊匹配
        if not direct:
            for key in self.forward:
                if entity in key or key in entity:
                    direct = self.forward[key]
                    entity = key
                    break

        for target, info in direct.items():
            controlled.append({
                'stock': target,
                'direct_pct': info['pct'],
                'level': 1,
                'holder_type': info.get('type', ''),
            })

        # 间接持股（BFS 一层）
        for target, info in direct.items():
            if info.get('type') == '企业':
                sub = self.forward.get(target, {})
                for sub_target, sub_info in sub.items():
                    effective = info['pct'] * sub_info['pct'] / 100
                    controlled.append({
                        'stock': sub_target,
                        'direct_pct': sub_info['pct'],
                        'effective_pct': effective,
                        'level': 2,
                        'via': target,
                        'holder_type': sub_info.get('type', ''),
                    })

        controlled.sort(key=lambda x: x.get('effective_pct', x['direct_pct']), reverse=True)
        return controlled

    # ================================================================
    # 审计检查
    # ================================================================

    def check_flags(self, stock_code):
        """检查股权方面的红旗信号"""
        code = self._resolve_code(stock_code)
        if not code:
            return {"error": "未找到该股票"}

        flags = []
        shareholders = self.reverse.get(code, {})

        # 1. 大股东质押率高（数据中没有质押字段，用持股集中度替代）
        total_pct = sum(v['pct'] for v in shareholders.values())
        if total_pct < 30:
            flags.append({"flag": "股权分散", "detail": f"前十大股东合计持股仅 {total_pct:.1f}%，容易被举牌或恶意收购"})

        # 2. 实控人不明 — 追溯不到自然人
        controllers = self.find_controller(code, max_depth=3)
        has_person = any(c['controller_type'] == '个人' for c in controllers)
        if not has_person and controllers:
            flags.append({"flag": "无自然人实控", "detail": "追溯3层仍未找到自然人，可能是国资、外资或无实际控制人"})

        # 3. 一层直接持股过高（可能是壳公司嫌疑）
        top1 = max(shareholders.items(), key=lambda x: x[1]['pct']) if shareholders else (None, None)
        if top1[1] and top1[1]['pct'] > 70:
            flags.append({"flag": "一股独大", "detail": f"{top1[0]} 持股 {top1[1]['pct']:.1f}%，缺乏制衡机制"})

        # 4. 控制链太复杂
        deep_controllers = self.find_controller(code, max_depth=5)
        if any(c['depth'] >= 3 for c in deep_controllers):
            flags.append({"flag": "控制链过长", "detail": "控制链超过3层，存在隐性关联交易风险"})

        return {
            "stock_code": code,
            "total_known_holdings": round(total_pct, 1),
            "flags": flags,
            "top_controllers": controllers[:5],
            "shareholder_count": len(shareholders),
        }


# ================================================================
# CLI
# ================================================================

def print_controller_analysis(graph, stock_code):
    """打印控制链分析"""
    print(f"\n{'='*70}")
    print(f"  {stock_code} 股权穿透分析")
    print(f"{'='*70}")

    # 反向追溯
    controllers = graph.find_controller(stock_code)
    print(f"\n  [控制链路] 发现 {len(controllers)} 条潜在控制链:\n")
    for i, c in enumerate(controllers[:10], 1):
        indent = "    " if i > 1 else ""
        tag = "[个人实控]" if c['controller_type'] == '个人' else "[企业控制]"
        print(f"  {i}. {tag} {c['chain']}")
        print(f"     有效持股: {c['effective_pct']:.2f}%, 深度: {c['depth']} 层")
        print()

    # 审计标记
    flags = graph.check_flags(stock_code)
    if flags.get('flags'):
        print("  [股权风险]\n")
        for f in flags['flags']:
            print(f"  ! {f['flag']}: {f['detail']}")
        print()


def print_empire_analysis(graph, entity_name):
    """打印资本版图分析"""
    print(f"\n{'='*70}")
    print(f"  {entity_name} 资本版图分析")
    print(f"{'='*70}")

    empire = graph.find_empire(entity_name)

    if not empire:
        print(f"\n  未找到 '{entity_name}' 的持股记录")
        return

    direct = [e for e in empire if e['level'] == 1]
    indirect = [e for e in empire if e['level'] == 2]

    print(f"\n  直接控制: {len(direct)} 家")
    for e in direct[:10]:
        print(f"    {e['stock']} — {e['direct_pct']:.1f}%")

    if indirect:
        print(f"\n  间接控制: {len(indirect)} 家")
        for e in indirect[:10]:
            print(f"    {e['stock']} — {e['effective_pct']:.2f}% (通过 {e.get('via','')})")

    print(f"\n  总计: {len(empire)} 家上市公司")


if __name__ == "__main__":
    graph = EquityGraph()

    if len(sys.argv) == 1:
        # 默认演示几只股票
        demo_stocks = ['300838.SZ', '688765.SH', '603439.SH']
        for s in demo_stocks:
            print_controller_analysis(graph, s)

    elif sys.argv[1] == '--chain':
        name = sys.argv[2]
        print_empire_analysis(graph, name)

    elif sys.argv[1] == '--top':
        stock = sys.argv[2]
        print_controller_analysis(graph, stock)

    elif sys.argv[1] == '--flags':
        stock = sys.argv[2]
        flags = graph.check_flags(stock)
        for k, v in flags.items():
            print(f"{k}: {v}")

    else:
        stock = sys.argv[1]
        print_controller_analysis(graph, stock)
