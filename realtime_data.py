"""
=============================================================================
 实时行情 — 腾讯财经API (免费, 无需安装额外包)
=============================================================================
"""
import json, urllib.request, re
from datetime import datetime


def _fetch(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode('gbk')
    except Exception:
        return None


def is_market_open():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 930 <= t <= 1130 or 1300 <= t <= 1500


def _code_to_qt(stock_code):
    """600519 → sh600519, 000001 → sz000001"""
    code = stock_code.strip().replace('.SH','').replace('.SZ','').replace('.BJ','')
    if code.startswith(('6','9')):
        return f"sh{code}"
    else:
        return f"sz{code}"


def _parse_qt(raw):
    """解析腾讯行情字符串"""
    if not raw or '="' not in raw:
        return None
    data = raw.split('="')[1].split('~')
    if len(data) < 50:
        return None
    return {
        "name": data[1],
        "code": data[2],
        "price": float(data[3]) if data[3] else 0,
        "prev_close": float(data[4]) if data[4] else 0,
        "open": float(data[5]) if data[5] else 0,
        "volume": int(data[6]) if data[6] else 0,
        "change_pct": float(data[32]) if data[32] else 0,
        "high": float(data[33]) if data[33] else 0,
        "low": float(data[34]) if data[34] else 0,
        "time": data[30],
    }


def get_stock_quote(stock_code):
    qt_code = _code_to_qt(stock_code)
    raw = _fetch(f"https://qt.gtimg.cn/q={qt_code}")
    d = _parse_qt(raw)
    if not d:
        return {"status": "not_found"}
    return {
        "status": "ok",
        "code": d['code'],
        "name": d['name'],
        "price": d['price'],
        "change_pct": d['change_pct'],
        "open": d['open'],
        "high": d['high'],
        "low": d['low'],
        "volume": d['volume'],
        "market_open": is_market_open(),
    }


def get_market_overview():
    raw = _fetch("https://qt.gtimg.cn/q=sh000001,sz399001,sz399006")
    if not raw:
        return {"status": "not_found"}
    indices = {}
    names = {'000001': '上证指数', '399001': '深证成指', '399006': '创业板指'}
    for line in raw.strip().split('\n'):
        d = _parse_qt(line)
        if d and d['code'] in names:
            indices[names[d['code']]] = {
                "price": d['price'],
                "change_pct": d['change_pct'],
            }
    return {"status": "ok", "indices": indices, "market_open": is_market_open()}


def get_portfolio_prices(codes):
    if not codes:
        return {}
    qt_codes = ",".join(_code_to_qt(c) for c in codes)
    raw = _fetch(f"https://qt.gtimg.cn/q={qt_codes}")
    if not raw:
        return {}
    result = {}
    for line in raw.strip().split('\n'):
        d = _parse_qt(line)
        if d:
            result[d['code']] = {"price": d['price'], "change_pct": d['change_pct'], "name": d['name']}
    return result


if __name__ == "__main__":
    print(f"交易时段: {is_market_open()}")
    q = get_stock_quote('600519')
    for k, v in q.items():
        print(f"  {k}: {v}")
    m = get_market_overview()
    print(f"\n市场: {m}")
