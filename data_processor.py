"""
=============================================================================
 数据处理模块 — 加载、清洗、标准化全部比赛数据
 产出：processed/ 目录下的清洗后数据，供后续模块直接使用
=============================================================================
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE, "data_raw")
DATA_OUT = os.path.join(BASE, "data_processed")
os.makedirs(DATA_OUT, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================
def int_to_date(series):
    """将 int 格式日期 (20240102) 转为 datetime"""
    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')


def report_int_to_date(series):
    """将报告期 int (20231231) 转为 datetime"""
    return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')


def summary(df, name):
    """打印数据集摘要"""
    print(f"\n{'='*60}")
    print(f"[{name}] {df.shape[0]:,} 行 x {df.shape[1]} 列")
    print(f"{'='*60}")
    print(f"内存: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_miss = missing[missing > 50]
    if len(high_miss) > 0:
        print(f"高缺失列 (>50%): {dict(high_miss)}")


# ============================================================
# 1. 测试问答集
# ============================================================
def process_qa_test():
    print("\n>>> 处理测试问答集...")
    df = pd.read_excel(os.path.join(DATA_RAW, "1.测试问答集/clean.xlsx"))

    # 添加问题长度、是否需要深度思考
    df['question_len'] = df['question'].str.len()
    df['question_words'] = df['question'].apply(lambda x: len(str(x)))  # 中文字数
    df['need_deep_think'] = df['think_flag'].astype(int)

    # Session 统计
    session_stats = df.groupby('session_id').agg(
        question_count=('question', 'count'),
        deep_think_ratio=('need_deep_think', 'mean'),
        avg_question_len=('question_len', 'mean')
    ).reset_index()

    df.to_pickle(os.path.join(DATA_OUT, "qa_test.pkl"))
    session_stats.to_pickle(os.path.join(DATA_OUT, "qa_session_stats.pkl"))
    summary(df, "测试问答集")
    print(f"  35个session, 平均{len(df)//35}条/会话, 需深度思考{df['need_deep_think'].sum()}条")
    return df


# ============================================================
# 2. 股东持股数据
# ============================================================
def process_shareholders():
    print("\n>>> 处理股东持股数据...")
    df = pd.read_excel(os.path.join(DATA_RAW, "2.股东持股-股权穿透/clean.xlsx"))

    # 日期转换
    df['ann_date'] = int_to_date(df['ann_dt'])
    df['holder_enddate'] = int_to_date(df['s_holder_enddate'])
    if 'report_period' in df.columns:
        df['report_date'] = report_int_to_date(df['report_period'])

    # 持有人类别标记
    df['holder_type'] = df['s_holder_holdercategory'].map({1: '个人', 2: '企业'})

    # 去重：同一股东+同一股票+同一天告，保留最新的
    df = df.sort_values('ann_date', ascending=False)
    df = df.drop_duplicates(subset=['s_holder_name', 's_info_windcode', 'ann_date'], keep='first')

    # 只保留关键列
    cols_keep = ['s_info_windcode', 's_holder_name', 's_holder_pct',
                 's_holder_holdercategory', 'holder_type', 's_holder_quantity',
                 'ann_date', 'holder_enddate', 'report_date',
                 's_holder_aname', 's_holder_nat']
    df_out = df[[c for c in cols_keep if c in df.columns]].copy()

    # 持股比例异常值检查（>100%为错误）
    abnormal = df_out[df_out['s_holder_pct'] > 100]
    if len(abnormal) > 0:
        print(f"  WARNING: {len(abnormal)} rows with pct > 100%, capped at 100")
        df_out.loc[df_out['s_holder_pct'] > 100, 's_holder_pct'] = 100

    # ── 实体类型标准化 ──
    # 从 graph_core 导入映射函数
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from graph_core import map_entity_type, TRANSPARENT_TYPES
        df_out['entity_type_canonical'] = df_out['s_holder_nat'].apply(map_entity_type)
        df_out['is_transparent'] = df_out['entity_type_canonical'].apply(
            lambda x: x in TRANSPARENT_TYPES
        )
        type_dist = df_out['entity_type_canonical'].value_counts().to_dict()
        print(f"  实体类型分布: {type_dist}")
    except ImportError:
        print("  (graph_core 尚未就绪，跳过实体类型标准化)")

    df_out.to_pickle(os.path.join(DATA_OUT, "shareholders.pkl"))
    summary(df_out, "股东持股")
    print(f"  股票: {df_out['s_info_windcode'].nunique():,}")
    print(f"  股东: {df_out['s_holder_name'].nunique():,}")
    print(f"  个人/企业: {df_out['holder_type'].value_counts().to_dict()}")
    print(f"  日期: {df_out['ann_date'].min().date()} ~ {df_out['ann_date'].max().date()}")
    return df_out


# ============================================================
# 3. 公司公告数据
# ============================================================
def process_announcements():
    print("\n>>> 处理公司公告数据...")
    df = pd.read_excel(os.path.join(DATA_RAW, "3.公司公告-事件脉络和风险识别/clean.xlsx"))

    # 日期
    if 'ann_dt' in df.columns:
        df['ann_date'] = pd.to_datetime(df['ann_dt'])

    # 公告类型码拆分（多个码用 | 分隔）
    df['fcode_list'] = df['n_info_fcode'].apply(
        lambda x: str(x).split('|') if pd.notna(x) else []
    )
    df['fcode_count'] = df['fcode_list'].apply(len)
    df['primary_fcode'] = df['fcode_list'].apply(lambda x: x[0] if x else '')

    # 提取关键信息：是否风险相关
    risk_keywords = ['处罚', '监管', '警示', '立案', '调查', '违规', '整改', '处分', '留置', '问询']
    df['is_risk_event'] = df['n_info_title'].apply(
        lambda x: any(kw in str(x) for kw in risk_keywords)
    )

    cols_keep = ['s_info_windcode', 'ann_date', 'n_info_title',
                 'primary_fcode', 'fcode_count', 'is_risk_event', 'n_info_fcode']
    df_out = df[[c for c in cols_keep if c in df.columns]].copy()

    df_out.to_pickle(os.path.join(DATA_OUT, "announcements.pkl"))
    summary(df_out, "公司公告")
    print(f"  股票: {df_out['s_info_windcode'].nunique():,}")
    print(f"  风险事件: {df_out['is_risk_event'].sum()} 条 ({df_out['is_risk_event'].mean()*100:.1f}%)")
    print(f"  日期: {df_out['ann_date'].min().date()} ~ {df_out['ann_date'].max().date()}")
    return df_out


# ============================================================
# 4. 三大财务报表 — 合并与标准化
# ============================================================
def process_financials():
    print("\n>>> 处理三大财务报表...")

    # 加载原始CSV
    balance = pd.read_csv(os.path.join(DATA_RAW, "4.三大财务报表-财务反欺诈/asharebalancesheet_202605261517.csv"))
    income = pd.read_csv(os.path.join(DATA_RAW, "4.三大财务报表-财务反欺诈/ashareincome_202605261519.csv"))
    cashflow = pd.read_csv(os.path.join(DATA_RAW, "4.三大财务报表-财务反欺诈/asharecashflow_202605261518.csv"))

    # --- 统一日期 ---
    for df in [balance, income, cashflow]:
        df['report_date'] = report_int_to_date(df['report_period'])
        df['ann_date'] = pd.to_datetime(df['ann_dt'], format='%Y%m%d', errors='coerce')

    # --- 资产负债表：核心字段 ---
    bs_cols = {
        's_info_windcode': 'stock_code',
        'report_date': 'report_date',
        'report_period': 'report_period',
        'tot_assets': 'total_assets',
        'tot_liab': 'total_liabilities',
        'tot_shrhldr_eqy_incl_min_int': 'total_equity',
        'tot_cur_assets': 'current_assets',
        'tot_cur_liab': 'current_liabilities',
        'monetary_cap': 'cash',
        'inventories': 'inventory',
        'acct_rcv': 'accounts_receivable',
        'notes_rcv': 'notes_receivable',
        'goodwill': 'goodwill',
        'fix_assets': 'fixed_assets',
        'int_rcv': 'interest_receivable',
        'prepay': 'prepayments',
        'lt_borrow': 'long_term_borrow',
        'st_borrow': 'short_term_borrow',
        'accounts_payable': 'accounts_payable',
    }
    bs_clean = balance[[k for k in bs_cols if k in balance.columns]].rename(
        columns={k: v for k, v in bs_cols.items() if k in balance.columns})

    # --- 利润表：核心字段 ---
    is_cols = {
        's_info_windcode': 'stock_code',
        'report_date': 'report_date',
        'report_period': 'report_period',
        'oper_rev': 'operating_revenue',
        'tot_oper_rev': 'total_operating_revenue',
        'tot_oper_cost': 'total_operating_cost',
        'oper_profit': 'operating_profit',
        'net_profit_incl_min_int_inc': 'net_profit',
        'net_profit_excl_min_int_inc': 'net_profit_attributable',
        'less_selling_dist_exp': 'selling_expense',
        'less_gerl_admin_exp': 'admin_expense',
        'less_fin_exp': 'finance_expense',
        'tot_profit': 'total_profit',
    }
    is_clean = income[[k for k in is_cols if k in income.columns]].rename(
        columns={k: v for k, v in is_cols.items() if k in income.columns})

    # --- 现金流量表：核心字段 ---
    cf_cols = {
        's_info_windcode': 'stock_code',
        'report_date': 'report_date',
        'report_period': 'report_period',
        'net_cash_flows_oper_act': 'operating_cashflow',
        'cash_recp_sg_and_rs': 'cash_from_sales',
        'net_cash_flows_inv_act': 'investing_cashflow',
        'net_cash_flows_fnc_act': 'financing_cashflow',
    }
    cf_clean = cashflow[[k for k in cf_cols if k in cashflow.columns]].rename(
        columns={k: v for k, v in cf_cols.items() if k in cashflow.columns})

    # --- 合并为一张大宽表 ---
    merged = bs_clean.merge(is_clean, on=['stock_code', 'report_date', 'report_period'],
                            how='outer', suffixes=('', '_inc'))
    merged = merged.merge(cf_clean, on=['stock_code', 'report_date', 'report_period'],
                          how='outer', suffixes=('', '_cf'))

    # --- 过滤：只保留同时有三张表数据的记录 ---
    merged = merged.dropna(subset=['total_assets', 'operating_revenue', 'operating_cashflow'], how='all')

    # --- 衍生指标 ---
    merged['debt_ratio'] = merged['total_liabilities'] / merged['total_assets']
    merged['inventory_to_revenue'] = merged['inventory'] / merged['operating_revenue'].replace(0, np.nan)
    merged['receivable_to_revenue'] = (merged['accounts_receivable'].fillna(0) +
                                        merged['notes_receivable'].fillna(0)) / merged['operating_revenue'].replace(0, np.nan)
    merged['cashflow_to_profit'] = merged['operating_cashflow'] / merged['net_profit'].replace(0, np.nan)
    merged['goodwill_to_equity'] = merged['goodwill'] / merged['total_equity'].replace(0, np.nan)
    merged['gross_margin'] = (merged['total_operating_revenue'].fillna(merged['operating_revenue']) -
                               merged['total_operating_cost']) / merged['total_operating_revenue'].fillna(merged['operating_revenue']).replace(0, np.nan)

    # 保存
    merged.to_pickle(os.path.join(DATA_OUT, "financials_merged.pkl"))
    bs_clean.to_pickle(os.path.join(DATA_OUT, "balance_sheet.pkl"))
    is_clean.to_pickle(os.path.join(DATA_OUT, "income_statement.pkl"))
    cf_clean.to_pickle(os.path.join(DATA_OUT, "cashflow_statement.pkl"))

    summary(merged, "合并财务数据")
    print(f"  股票: {merged['stock_code'].nunique():,}")
    print(f"  报告期: {merged['report_period'].nunique()} 期")
    print(f"  报告期范围: {merged['report_date'].min().date()} ~ {merged['report_date'].max().date()}")
    print(f"  含资产负债表数据: {merged['total_assets'].notna().sum():,}")
    print(f"  含利润表数据: {merged['operating_revenue'].notna().sum():,}")
    print(f"  含现金流量数据: {merged['operating_cashflow'].notna().sum():,}")

    # 打印衍生指标摘要
    for col in ['debt_ratio', 'inventory_to_revenue', 'cashflow_to_profit', 'goodwill_to_equity']:
        if col in merged.columns:
            valid = merged[col].dropna()
            if len(valid) > 0:
                print(f"  {col}: median={valid.median():.3f}, std={valid.std():.3f}")

    return merged


# ============================================================
# 5. 券商研报数据
# ============================================================
def process_reports():
    print("\n>>> 处理券商研报数据...")
    df = pd.read_csv(os.path.join(DATA_RAW, "5.券商研报-专家观点/rr_main_202605281537.csv"))

    # 日期
    df['publish_dt'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['write_dt'] = pd.to_datetime(df['write_date'], errors='coerce')

    # 评级变化标记
    df['is_upgrade'] = df['rating_change'].isin(['上调', '调高', '调升'])
    df['is_downgrade'] = df['rating_change'].isin(['下调', '调低'])
    df['is_maintain'] = df['rating_change'] == '维持'

    # 摘要长度
    df['abstract_len'] = df['abstract'].fillna('').str.len()

    # 保留关键列
    cols_keep = ['sec_code', 'sec_name', 'title', 'abstract', 'abstract_len',
                 'org_name', 'author', 'publish_dt', 'report_type',
                 'rating_change', 'is_upgrade', 'is_downgrade', 'is_maintain',
                 'industry_l1', 'industry_l2', 'industry_l3']
    df_out = df[[c for c in cols_keep if c in df.columns]].copy()

    df_out.to_pickle(os.path.join(DATA_OUT, "reports.pkl"))
    summary(df_out, "券商研报")
    print(f"  股票: {df_out['sec_code'].nunique():,}")
    print(f"  券商: {df_out['org_name'].nunique():,}")
    print(f"  上调/下调/维持: {df_out['is_upgrade'].sum()}/{df_out['is_downgrade'].sum()}/{df_out['is_maintain'].sum()}")
    print(f"  日期: {df_out['publish_dt'].min().date()} ~ {df_out['publish_dt'].max().date()}")
    print(f"  行业数: {df_out['industry_l1'].nunique()}")
    return df_out


# ============================================================
# 6. 数据画像报告
# ============================================================
def generate_profile():
    """生成数据画像报告文本"""
    print("\n>>> 生成数据画像...")
    lines = []
    lines.append("=" * 60)
    lines.append(f"数据画像报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    # QA test
    df = pd.read_pickle(os.path.join(DATA_OUT, "qa_test.pkl"))
    lines.append(f"\n[测试问答集] {len(df)}条, {df['session_id'].nunique()}会话, "
                 f"需深度思考{df['need_deep_think'].sum()}条({df['need_deep_think'].mean()*100:.1f}%)")

    # Shareholders
    df = pd.read_pickle(os.path.join(DATA_OUT, "shareholders.pkl"))
    lines.append(f"\n[股东持股] {len(df):,}条, {df['s_info_windcode'].nunique()}只股票, "
                 f"{df['s_holder_name'].nunique()}个股东")
    lines.append(f"  个人/企业: {df['holder_type'].value_counts().to_dict()}")
    top_holder = df['s_holder_name'].value_counts().head(5)
    lines.append(f"  出现最多的股东: {dict(top_holder)}")

    # Announcements
    df = pd.read_pickle(os.path.join(DATA_OUT, "announcements.pkl"))
    lines.append(f"\n[公司公告] {len(df):,}条, {df['s_info_windcode'].nunique()}只股票")
    lines.append(f"  风险事件: {df['is_risk_event'].sum()}条 ({df['is_risk_event'].mean()*100:.1f}%)")

    # Financials
    df = pd.read_pickle(os.path.join(DATA_OUT, "financials_merged.pkl"))
    lines.append(f"\n[财务报表] {len(df):,}条记录, {df['stock_code'].nunique()}只股票")
    lines.append(f"  报告期: {df['report_date'].min().date()} ~ {df['report_date'].max().date()}")
    lines.append(f"  负债率中位数: {df['debt_ratio'].median():.1%}")
    lines.append(f"  现金流/利润中位数: {df['cashflow_to_profit'].median():.2f}")

    # Reports
    df = pd.read_pickle(os.path.join(DATA_OUT, "reports.pkl"))
    lines.append(f"\n[券商研报] {len(df):,}条, {df['sec_code'].nunique()}只股票, {df['org_name'].nunique()}家券商")
    lines.append(f"  维持/上调/下调: {df['is_maintain'].sum()}/{df['is_upgrade'].sum()}/{df['is_downgrade'].sum()}")
    top_industry = df['industry_l1'].value_counts().head(5)
    lines.append(f"  行业Top5: {dict(top_industry)}")

    report = "\n".join(lines)

    with open(os.path.join(DATA_OUT, "data_profile.txt"), 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    return report


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  数据预处理开始")
    print("=" * 60)

    qa = process_qa_test()
    shareholders = process_shareholders()
    announcements = process_announcements()
    financials = process_financials()
    reports = process_reports()

    generate_profile()

    print(f"\n{'='*60}")
    print(f"  全部处理完成！清洗后数据在: {DATA_OUT}")
    print(f"{'='*60}")
