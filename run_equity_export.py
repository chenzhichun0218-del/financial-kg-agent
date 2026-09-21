"""
================================================================================
 股权穿透知识图谱 → Obsidian 一键导出
 ================================================================================
 用法:
   python run_equity_export.py                           # 全量导出(3000实体)
   python run_equity_export.py --force                   # 强制覆盖全部
   python run_equity_export.py --stock 603439.SH         # 单只股票深度穿透导出
   python run_equity_export.py --stock 603439.SH --depth 6  # 自定义穿透深度
   python run_equity_export.py --top 20                  # 导出穿透最深的20只
   python run_equity_export.py --batch 603439.SH,600238.SH,601033.SH  # 批量
   python run_equity_export.py --all --force             # 全量强制覆盖(5000实体)
================================================================================
"""
import sys
import os
import time
import argparse

# 修复 Windows GBK 编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entity_resolver import build_graph_from_shareholders
from export_to_obsidian import ObsidianExporter, export_to_obsidian


def main():
    parser = argparse.ArgumentParser(
        description="股权穿透知识图谱 → Obsidian 导出工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                   全量导出3000实体
  %(prog)s --force                           强制覆盖模式
  %(prog)s --stock 603439.SH                 单只股票深度穿透
  %(prog)s --stock 603439.SH --depth 6       自定义穿透深度(最大6层)
  %(prog)s --top 20                          穿透最深的20只股票
  %(prog)s --batch 603439.SH,600238.SH       批量导出指定股票
  %(prog)s --all --force                     全量强制覆盖5000实体
        """,
    )

    # 导出模式（互斥组）
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stock", type=str, metavar="CODE",
                      help="单只股票深度穿透导出 (如 603439.SH)")
    mode.add_argument("--top", type=int, metavar="N",
                      help="导出穿透最深的前N只股票及其完整链路")
    mode.add_argument("--batch", type=str, metavar="CODES",
                      help="批量导出指定股票 (逗号分隔)")
    mode.add_argument("--all", action="store_true",
                      help="全量导出模式")

    # 通用参数
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已存在的笔记文件")
    parser.add_argument("--max", type=int, default=3000, metavar="N",
                        help="全量模式最大导出实体数 (默认3000)")
    parser.add_argument("--depth", type=int, default=5, metavar="N",
                        help="穿透深度 (默认5层, 最大6层)")
    parser.add_argument("--output", type=str, default=None, metavar="DIR",
                        help="输出目录 (默认 obsidian-kb/)")
    parser.add_argument("--data", type=str,
                        default="data_raw/2.股东持股-股权穿透/clean.xlsx",
                        metavar="PATH", help="股东数据路径")

    args = parser.parse_args()

    print("=" * 60)
    print("  股权穿透知识图谱 → Obsidian 导出")
    print("=" * 60)
    print(f"  模式: ", end="")
    if args.stock:
        print(f"单只股票 ({args.stock})")
    elif args.top:
        print(f"深度案例 Top {args.top}")
    elif args.batch:
        print(f"批量导出 ({args.batch})")
    elif args.all:
        print(f"全量 (最多 {args.max} 实体)")
    else:
        print(f"默认全量 (最多 {args.max} 实体)")
    print(f"  穿透深度: {args.depth} 层")
    print(f"  覆盖模式: {'是' if args.force else '否（跳过已有文件）'}")
    print()

    # ── Step 1: 加载数据 ──
    t0 = time.time()
    print("[1/3] 加载股东数据...")
    try:
        df = pd.read_excel(args.data)
        print(f"  ✅ 加载了 {len(df):,} 行股东数据")
    except FileNotFoundError:
        print(f"  ❌ 数据文件未找到: {args.data}")
        print(f"  请确认路径正确")
        sys.exit(1)

    # ── Step 2: 构建图谱 ──
    print("\n[2/3] 构建股权知识图谱...")
    store, aliases = build_graph_from_shareholders(df)
    t1 = time.time()
    print(f"  ✅ 构建完成 ({t1 - t0:.1f}s)")
    print(f"     实体: {store.entity_count():,} 个")
    print(f"     持股关系: {store.edge_count():,} 条")
    print(f"     上市公司: {len(store.stock_index):,} 家")

    # ── Step 3: 导出 ──
    print("\n[3/3] 导出 Obsidian 笔记...")
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "obsidian-kb"
    )
    exporter = ObsidianExporter(store, output_dir)

    if args.stock:
        num = exporter.export_by_stock(
            args.stock, max_depth=args.depth, force=args.force
        )
        print(f"\n  ✅ 导出 {num} 个实体")

    elif args.batch:
        stocks = [s.strip() for s in args.batch.split(",") if s.strip()]
        total = 0
        for sc in stocks:
            n = exporter.export_by_stock(sc, max_depth=args.depth, force=args.force)
            total += n
            print(f"      {sc}: {n} 个实体")
        print(f"\n  ✅ 批量导出完成! 总计 {total} 个实体")

    elif args.top:
        num, cases = exporter.export_deep_cases(
            top_n=args.top, min_depth=3, force=args.force
        )
        print(f"\n  ✅ 导出 {num} 个实体 ({len(cases)} 只深度穿透股票)")

    else:
        # 默认全量
        num = export_to_obsidian(store, output_dir,
                                 max_entities=args.max, force=args.force)
        print(f"\n  ✅ 导出 {num} 个实体")

    t2 = time.time()
    print(f"\n{'=' * 60}")
    print(f"  总耗时: {t2 - t0:.1f}s")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")
    print()
    print("📂 下一步:")
    print("   1. 打开 Obsidian")
    print("   2. 点击「打开其他 Vault」→ 选择 obsidian-kb/ 文件夹")
    print("   3. 点击左侧边栏「📊 图谱视图」查看股权关系网络")
    print("   4. 按 Ctrl+P → 输入 'graph' → 选择「打开局部图谱」查看单个实体")
    print()
    print("🎨 图谱颜色含义:")
    print("   🟠 橙色 = 自然人  🔵 蓝色 = 民营企业  🔴 红色 = 国有企业")
    print("   🟢 绿色 = 境外实体  ⚪ 灰色 = 基金/资管  🟣 紫色 = 合伙企业")
    print()


if __name__ == "__main__":
    main()
