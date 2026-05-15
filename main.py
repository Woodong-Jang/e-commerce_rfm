"""
主运行脚本
电商用户价值与转化分析 — 完整分析流水线

使用方式:
    python main.py                        # 使用 data/ 目录下的数据
    python main.py --data data/data.csv   # 指定数据文件路径

数据集下载:
    https://www.kaggle.com/datasets/carrie1/ecommerce-data
    下载后将 data.csv 放入 data/ 目录
"""

import os
import sys
import argparse
import pandas as pd

# 确保 src 目录在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_cleaning import load_data, clean_data, get_cleaning_summary
from rfm_model import build_rfm, kmeans_validation, get_segment_revenue
from funnel_analysis import build_conversion_funnel, get_monthly_sales, get_country_stats, get_product_stats
from visualization import (
    plot_monthly_sales, plot_country_distribution, plot_conversion_funnel,
    plot_rfm_distribution, plot_rfm_segments, plot_segment_revenue,
    plot_kmeans_validation, plot_rfm_correlation
)

# 修复 visualization 中缺失的函数名
import src.visualization as viz
viz.plot_rfm_distribution = viz.plot_rfm_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description='电商用户价值与转化分析')
    parser.add_argument('--data', type=str, default=None, help='数据文件路径')
    parser.add_argument('--skip-plots', action='store_true', help='跳过图表生成')
    return parser.parse_args()


def ensure_dirs():
    for d in ['data', 'outputs', 'outputs/figures']:
        os.makedirs(d, exist_ok=True)


def create_sample_data():
    """生成演示用样本数据（未下载真实数据集时使用）"""
    print("\n[提示] 未找到数据文件，生成演示样本数据...")
    np.random.seed(42)
    import numpy as np

    n = 5000
    invoice_dates = pd.date_range('2010-12-01', '2011-12-09', periods=n)
    np.random.shuffle(invoice_dates.values)

    customers = [f'{10000 + i}' for i in range(400)]
    countries = ['United Kingdom'] * 350 + ['Germany'] * 20 + ['France'] * 15 + \
                ['Netherlands'] * 8 + ['Australia'] * 7
    products = [
        ('85123A', 'WHITE HANGING HEART T-LIGHT HOLDER', 2.55),
        ('71053',  'WHITE METAL LANTERN', 3.39),
        ('84406B', 'CREAM CUPID HEARTS COAT HANGER', 2.75),
        ('84029G', 'KNITTED UNION FLAG HOT WATER BOTTLE', 3.39),
        ('84029E', 'RED WOOLLY HOTTIE WHITE HEART', 3.39),
        ('22752',  'SET 7 BABUSHKA NESTING BOXES', 7.65),
        ('21730',  'GLASS STAR FROSTED T-LIGHT HOLDER', 4.25),
        ('22633',  'HAND WARMER UNION JACK', 1.85),
        ('22632',  'HAND WARMER RED POLKA DOT', 1.85),
        ('47566',  'PARTY BUNTING', 5.95),
    ]

    rows = []
    for i in range(n):
        prod = products[np.random.randint(len(products))]
        rows.append({
            'InvoiceNo': f'{500000 + i // 5}',
            'StockCode': prod[0],
            'Description': prod[1],
            'Quantity': np.random.randint(1, 20),
            'InvoiceDate': invoice_dates[i],
            'UnitPrice': prod[2] * (0.8 + np.random.random() * 0.4),
            'CustomerID': np.random.choice(customers),
            'Country': np.random.choice(countries),
        })

    df = pd.DataFrame(rows)
    path = 'data/sample_data.csv'
    df.to_csv(path, index=False)
    print(f"      样本数据已保存: {path}  ({len(df):,}条记录)\n")
    return path


def main():
    args = parse_args()
    ensure_dirs()

    print("=" * 60)
    print("  电商用户价值与转化分析")
    print("  基于 Kaggle Online Retail Dataset")
    print("=" * 60)

    # 查找数据文件
    data_path = args.data
    if data_path is None:
        for candidate in ['data/data.csv', 'data/online_retail.xlsx', 'data/Online Retail.xlsx']:
            if os.path.exists(candidate):
                data_path = candidate
                break
    if data_path is None or not os.path.exists(data_path):
        data_path = create_sample_data()

    # ── 1. 加载 & 清洗 ──────────────────────────────────────────
    raw_df = load_data(data_path)
    original_count = len(raw_df)
    df, clean_log = clean_data(raw_df)
    summary = get_cleaning_summary(original_count, df, clean_log)

    # ── 2. 基础统计 ──────────────────────────────────────────────
    monthly = get_monthly_sales(df)
    country = get_country_stats(df)
    products_top = get_product_stats(df, top_n=10)

    # ── 3. 转化漏斗 ──────────────────────────────────────────────
    funnel_result = build_conversion_funnel(df)

    # ── 4. RFM 模型 ──────────────────────────────────────────────
    rfm = build_rfm(df)
    seg_revenue = get_segment_revenue(rfm)
    kmeans_result = kmeans_validation(rfm, n_clusters=4)
    rfm = kmeans_result['rfm_with_cluster']

    # ── 5. 可视化 ────────────────────────────────────────────────
    if not args.skip_plots:
        print(f"\n[5/5] 生成可视化图表...")
        plot_monthly_sales(monthly)
        plot_country_distribution(country)
        plot_conversion_funnel(funnel_result['funnel_df'])
        plot_rfm_heatmap(rfm)
        plot_rfm_segments(rfm)
        plot_segment_revenue(seg_revenue)
        plot_kmeans_validation(kmeans_result)
        plot_rfm_correlation(rfm)

    # ── 6. 导出结果 ──────────────────────────────────────────────
    rfm.to_csv('outputs/rfm_results.csv', index=False)
    seg_revenue.to_csv('outputs/segment_revenue.csv')
    funnel_result['funnel_df'].to_csv('outputs/funnel_analysis.csv', index=False)
    monthly.to_csv('outputs/monthly_sales.csv', index=False)
    country.to_csv('outputs/country_stats.csv', index=False)
    products_top.to_csv('outputs/top_products.csv', index=False)

    # ── 7. 打印汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  分析完成 — 核心指标汇总")
    print("=" * 60)
    print(f"  数据记录:    {summary['original_count']:>10,} → {summary['clean_count']:,} (清洗后)")
    print(f"  有效用户:    {summary['unique_customers']:>10,} 人")
    print(f"  总销售额:    £{summary['total_revenue']:>10,.0f}")
    print(f"  数据时段:    {summary['date_range']}")
    print(f"  整体转化率:  {funnel_result['overall_conversion']:>9.1f}%")
    print(f"  加购→支付流失:{funnel_result['cart_to_checkout_loss']:>8.1f}%  ⚠ 关键瓶颈")
    print(f"  K-Means轮廓系数: {kmeans_result['best_silhouette']:.4f}")
    print()
    print("  用户分层营收贡献:")
    print(seg_revenue[['用户数', '用户占比%', '营收占比%', '平均消费']].to_string())
    print()
    print("  输出文件:")
    print("    outputs/rfm_results.csv         — 每用户 RFM 评分及分层")
    print("    outputs/segment_revenue.csv     — 各用户层营收汇总")
    print("    outputs/funnel_analysis.csv     — 转化漏斗数据")
    print("    outputs/figures/                — 8 张分析图表")
    print("=" * 60)


if __name__ == '__main__':
    from src.visualization import plot_rfm_heatmap
    main()
