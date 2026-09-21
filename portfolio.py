"""
个人投资组合管理 + 股票代码/名称自动补全
"""
import json, os, pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE, "my_portfolio.json")

# 股票代码 ↔ 名称映射表（从研报和财务数据构建）
_name_to_code = {}
_code_to_name = {}
_mapping_loaded = False

def _load_mapping():
    global _name_to_code, _code_to_name, _mapping_loaded
    if _mapping_loaded:
        return
    try:
        reports = pd.read_pickle(os.path.join(BASE, "data_processed", "reports.pkl"))
        for _, r in reports[['sec_code', 'sec_name']].drop_duplicates().iterrows():
            code = str(r['sec_code']).strip()
            name = str(r['sec_name']).strip()
            if code and name and name != 'nan':
                _code_to_name[code] = name
                _name_to_code[name] = code
                # 去后缀简称
                core = name.replace('股份有限公司','').replace('有限责任公司','').replace('有限公司','')
                if core != name:
                    _name_to_code[core] = code
    except Exception:
        pass
    _mapping_loaded = True

def auto_fill(code="", name=""):
    """输入代码或名称之一，自动补全另一个"""
    _load_mapping()
    if code and not name:
        code = code.strip()
        # 先精确匹配
        if code in _code_to_name:
            return code, _code_to_name[code]
        # 去掉后缀再试 (000498.SZ → 000498)
        short = code.replace('.SH','').replace('.SZ','').replace('.BJ','')
        if short in _code_to_name:
            return code, _code_to_name[short]
        if short != code and code in _code_to_name:
            return code, _code_to_name[code]
    if name and not code:
        name = name.strip()
        if name in _name_to_code:
            return _name_to_code[name], name
        for known, c in _name_to_code.items():
            if name[:4] in known or known[:4] in name:
                return c, known
    if code and name:
        return code.strip(), name.strip()
    return code.strip() if code else "", name.strip() if name else ""


class Portfolio:
    def __init__(self):
        self.data = self._load()

    def _load(self):
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "watchlist": [],
            "holdings": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _save(self):
        self.data['updated_at'] = datetime.now().isoformat()
        with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def add_watchlist(self, code, name=""):
        code, name = auto_fill(code, name)
        if not code:
            return "请填写股票代码或名称"
        for item in self.data['watchlist']:
            if item['code'] == code:
                return f"{code} {name} 已在自选股中"
        self.data['watchlist'].append({
            "code": code, "name": name or code, "added_at": datetime.now().isoformat()
        })
        self._save()
        return f"已添加 {code} {name} 到自选股"

    def add_holding(self, code, shares, cost_price, name=""):
        code, name = auto_fill(code, name)
        if not code:
            return "请填写股票代码或名称"
        shares = int(shares)
        cost_price = float(cost_price)
        for item in self.data['holdings']:
            if item['code'] == code:
                old_total = item['shares'] * item['cost_price']
                new_total = shares * cost_price
                item['shares'] += shares
                item['cost_price'] = round((old_total + new_total) / item['shares'], 3)
                item['bought_at'] = datetime.now().isoformat()
                self._save()
                return f"已追加 {code} {name}: 共 {item['shares']} 股, 成本 {item['cost_price']:.2f}"

        self.data['holdings'].append({
            "code": code, "name": name or code,
            "shares": shares, "cost_price": cost_price,
            "bought_at": datetime.now().isoformat()
        })
        self._save()
        return f"已添加 {code} {name}: {shares} 股 @ {cost_price}"

    def remove(self, code):
        before_w = len(self.data['watchlist'])
        before_h = len(self.data['holdings'])
        self.data['watchlist'] = [i for i in self.data['watchlist'] if i['code'] != code]
        self.data['holdings'] = [i for i in self.data['holdings'] if i['code'] != code]
        after_w = len(self.data['watchlist'])
        after_h = len(self.data['holdings'])
        self._save()
        msgs = []
        if before_w > after_w: msgs.append(f"已从自选股移除 {code}")
        if before_h > after_h: msgs.append(f"已从持仓移除 {code}")
        return "; ".join(msgs) if msgs else f"未找到 {code}"

    def get_watchlist_codes(self):
        return [i['code'] for i in self.data['watchlist']]

    def get_holding_codes(self):
        return [i['code'] for i in self.data['holdings']]

    def get_all_codes(self):
        return list(set(self.get_watchlist_codes() + self.get_holding_codes()))

    def summary(self):
        """简洁的持仓摘要，每只股票独立成行，空行分隔"""
        lines = ["### 我的投资组合", ""]

        if self.data['watchlist']:
            lines.append(f"**自选股** ({len(self.data['watchlist'])} 只)")
            lines.append("")
            for i, item in enumerate(self.data['watchlist']):
                name_display = f" — {item.get('name', '')}" if item.get('name') and item.get('name') != item['code'] else ''
                lines.append(f"{i+1}. `{item['code']}`{name_display}")
                lines.append("")
            lines.append("")

        if self.data['holdings']:
            lines.append(f"**持仓** ({len(self.data['holdings'])} 只)")
            lines.append("")
            total_cost = 0
            for i, item in enumerate(self.data['holdings']):
                cost = item['shares'] * item['cost_price']
                total_cost += cost
                name_display = f" — {item.get('name', '')}" if item.get('name') and item.get('name') != item['code'] else ''
                lines.append(f"{i+1}. `{item['code']}`{name_display}")
                lines.append(f"   {item['shares']}股 × {item['cost_price']:.2f}元 = **{cost/10000:.2f}万**")
                lines.append("")
            lines.append(f"*总成本: **{total_cost/10000:.2f}万***")

        if not self.data['watchlist'] and not self.data['holdings']:
            lines.append("*暂无自选股或持仓*")

        return "\n".join(lines)

    def snapshot_for_agent(self):
        lines = ["[用户投资组合]"]
        if self.data['watchlist']:
            codes = ", ".join(i['code'] for i in self.data['watchlist'])
            lines.append(f"自选股: {codes}")
        if self.data['holdings']:
            for item in self.data['holdings']:
                lines.append(f"持仓: {item['code']} {item['shares']}股 成本{item['cost_price']:.2f}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    pf = Portfolio()
    if len(sys.argv) == 1:
        print(pf.summary())
    elif sys.argv[1] == "add-watch":
        code, name = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
        print(pf.add_watchlist(code, name))
    elif sys.argv[1] == "add-hold":
        code, shares, price = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
        name = sys.argv[5] if len(sys.argv) > 5 else ""
        print(pf.add_holding(code, shares, price, name))
    elif sys.argv[1] == "remove":
        print(pf.remove(sys.argv[2]))
