"""
可视化模块 — 生成所有分析图表（Seaborn + Matplotlib）
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

# ---- 全局样式 ----
PALETTE = {
    '冠军用户': '#639922',
    '忠实用户': '#378ADD',
    '潜力用户': '#EF9F27',
    '流失风险': '#D85A30',
    '沉默用户': '#888780',
    '新客用户': '#7F77DD',
}
sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
FIGDIR = 'outputs/figures'


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"      图表已保存: {path}")
    return path


# =========================================================
# 1. 数据概览：月度销售趋势
# =========================================================
def plot_monthly_sales(monthly_df) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 销售额趋势
    ax = axes[0]
    x = range(len(monthly_df))
    ax.fill_between(x, monthly_df['销售额'] / 1e4, alpha=0.15, color='#378ADD')
    ax.plot(x, monthly_df['销售额'] / 1e4, color='#378ADD', linewidth=2.5, marker='o', markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly_df['YearMonth'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('销售额（万英镑）')
    ax.set_title('月度销售额趋势', fontweight='bold', pad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:.0f}W'))

    # 月度用户数
    ax2 = axes[1]
    colors = ['#D85A30' if v == monthly_df['用户数'].max() else '#5DCAA5' for v in monthly_df['用户数']]
    ax2.bar(x, monthly_df['用户数'], color=colors, width=0.7, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(monthly_df['YearMonth'], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('活跃用户数')
    ax2.set_title('月度活跃用户数', fontweight='bold', pad=10)

    plt.tight_layout(pad=2)
    return save(fig, '01_monthly_sales.png')


# =========================================================
# 2. 国家销售分布
# =========================================================
def plot_country_distribution(country_df) -> str:
    top = country_df.head(8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 销售额条形图
    ax = axes[0]
    bars = ax.barh(top['Country'][::-1], top['销售额'][::-1] / 1e6, color='#378ADD', edgecolor='white')
    bars[len(bars)-1].set_color('#D85A30')  # 最高的用红色
    ax.set_xlabel('销售额（百万英镑）')
    ax.set_title('各国销售额 Top 8', fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'£{v:.1f}M'))

    # 饼图（按销售额）
    ax2 = axes[1]
    labels = top['Country'].tolist()
    sizes = top['销售额'].tolist()
    colors_pie = ['#378ADD','#5DCAA5','#EF9F27','#D85A30','#7F77DD','#888780','#639922','#F09595']
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=None, autopct='%1.1f%%', colors=colors_pie,
        startangle=140, pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax2.legend(wedges, labels, loc='lower right', fontsize=8, framealpha=0.8)
    ax2.set_title('销售额国家占比', fontweight='bold')

    plt.tight_layout()
    return save(fig, '02_country_distribution.png')


# =========================================================
# 3. 转化漏斗
# =========================================================
def plot_conversion_funnel(funnel_df) -> str:
    fig, ax = plt.subplots(figsize=(9, 6))

    stages = funnel_df['阶段'].tolist()
    values = funnel_df['用户数'].tolist()
    pcts = funnel_df['整体转化率%'].tolist()
    max_val = values[0]

    colors = ['#378ADD', '#5DCAA5', '#EF9F27', '#D85A30', '#639922']
    bar_height = 0.55

    for i, (stage, val, pct) in enumerate(zip(stages, values, pcts)):
        width = val / max_val
        left = (1 - width) / 2
        y = len(stages) - 1 - i

        rect = mpatches.FancyBboxPatch(
            (left, y - bar_height/2), width, bar_height,
            boxstyle='round,pad=0.01', linewidth=0,
            facecolor=colors[i], alpha=0.85
        )
        ax.add_patch(rect)

        ax.text(0.5, y, f"{stage}  {val:,}人  ({pct}%)",
                ha='center', va='center', fontsize=11, color='white', fontweight='bold')

        # 流失率标注
        if i > 0:
            loss = funnel_df.iloc[i]['环节流失率%']
            ax.annotate(f'↓ 流失 {loss}%',
                        xy=(0.5, y + 0.5), ha='center', fontsize=9,
                        color='#A32D2D' if loss > 50 else '#888780',
                        fontweight='bold' if loss > 50 else 'normal')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(stages))
    ax.axis('off')
    ax.set_title('用户转化漏斗分析', fontsize=14, fontweight='bold', pad=15)

    # 关键发现标注框
    ax.text(0.02, 0.02,
            '⚠ 关键瓶颈：加购→支付 流失率 62%\n  建议：购物车提醒 + 简化结算流程',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCEBEB', edgecolor='#D85A30', alpha=0.9),
            color='#A32D2D')

    plt.tight_layout()
    return save(fig, '03_conversion_funnel.png')


# =========================================================
# 4. RFM 分布热力图
# =========================================================
def plot_rfm_heatmap(rfm_df) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ['Recency（最近购买天数）', 'Frequency（购买频次）', 'Monetary（消费金额£）']
    cols   = ['Recency', 'Frequency', 'Monetary']
    colors = ['#D85A30', '#378ADD', '#639922']

    for ax, col, title, color in zip(axes, cols, titles, colors):
        data = rfm_df[col]
        if col == 'Monetary':
            data = np.log1p(data)
            xlabel = 'log(消费金额)'
        else:
            xlabel = col
        sns.histplot(data, bins=40, ax=ax, color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
        ax.axvline(data.median(), color='#2C2C2A', linestyle='--', linewidth=1.2, label=f'中位数={data.median():.0f}')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('用户数')
        ax.legend(fontsize=8)

    plt.suptitle('RFM 三维度分布', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    return save(fig, '04_rfm_distribution.png')


# =========================================================
# 5. 用户分层气泡图
# =========================================================
def plot_rfm_segments(rfm_df) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))

    seg_stats = rfm_df.groupby('Segment').agg(
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean'),
        Count=('CustomerID', 'count')
    ).reset_index()

    for _, row in seg_stats.iterrows():
        seg = row['Segment']
        color = PALETTE.get(seg, '#888780')
        size = (row['Count'] / seg_stats['Count'].max()) * 3000 + 200
        ax.scatter(row['Frequency'], row['Monetary'],
                   s=size, color=color, alpha=0.75, edgecolors='white', linewidth=1.5)
        ax.annotate(f"{seg}\n({row['Count']}人)",
                    (row['Frequency'], row['Monetary']),
                    textcoords='offset points', xytext=(8, 4),
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xlabel('平均购买频次', fontsize=11)
    ax.set_ylabel('平均消费金额（£）', fontsize=11)
    ax.set_title('RFM 用户分层气泡图（气泡大小=用户数）', fontsize=13, fontweight='bold')
    handles = [mpatches.Patch(color=c, label=s) for s, c in PALETTE.items()]
    ax.legend(handles=handles, fontsize=8, loc='upper left')

    plt.tight_layout()
    return save(fig, '05_rfm_segments_bubble.png')


# =========================================================
# 6. 用户分层营收贡献
# =========================================================
def plot_segment_revenue(seg_revenue_df) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    seg = seg_revenue_df.reset_index()
    colors = [PALETTE.get(s, '#888780') for s in seg['Segment']]

    # 营收占比饼图
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        seg['营收占比%'], labels=None, autopct='%1.1f%%',
        colors=colors, startangle=90,
        pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts: at.set_fontsize(9)
    ax1.legend(wedges, seg['Segment'], loc='lower left', fontsize=8)
    ax1.set_title('各用户层营收贡献占比', fontweight='bold')

    # 用户数 vs 营收对比
    ax2 = axes[1]
    x = np.arange(len(seg))
    width = 0.35
    b1 = ax2.bar(x - width/2, seg['用户占比%'], width, label='用户占比%', color='#378ADD', alpha=0.8)
    b2 = ax2.bar(x + width/2, seg['营收占比%'], width, label='营收占比%', color='#639922', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seg['Segment'], rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('%')
    ax2.set_title('用户占比 vs 营收占比', fontweight='bold')
    ax2.legend()
    ax2.bar_label(b1, fmt='%.0f%%', fontsize=8, padding=2)
    ax2.bar_label(b2, fmt='%.0f%%', fontsize=8, padding=2)

    plt.tight_layout()
    return save(fig, '06_segment_revenue.png')


# =========================================================
# 7. K-Means 验证图
# =========================================================
def plot_kmeans_validation(kmeans_result: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 肘部法则
    ax1 = axes[0]
    ax1.plot(kmeans_result['k_range'], kmeans_result['inertia'],
             'o-', color='#378ADD', linewidth=2.5, markersize=6)
    ax1.set_xlabel('聚类数 k')
    ax1.set_ylabel('Inertia（簇内平方和）')
    ax1.set_title('肘部法则 — 最优 k 值选择', fontweight='bold')
    ax1.axvline(x=4, color='#D85A30', linestyle='--', alpha=0.7, label='推荐 k=4')
    ax1.legend()

    # 轮廓系数
    ax2 = axes[1]
    bars = ax2.bar(kmeans_result['k_range'], kmeans_result['silhouette'],
                   color='#5DCAA5', alpha=0.8, edgecolor='white')
    best_k_idx = kmeans_result['silhouette'].index(max(kmeans_result['silhouette']))
    bars[best_k_idx].set_color('#D85A30')
    ax2.set_xlabel('聚类数 k')
    ax2.set_ylabel('轮廓系数 Silhouette Score')
    ax2.set_title('轮廓系数验证（越高越好）', fontweight='bold')
    ax2.axhline(y=0.3, color='gray', linestyle=':', alpha=0.6, label='合理阈值 0.3')
    ax2.legend()

    plt.tight_layout()
    return save(fig, '07_kmeans_validation.png')


# =========================================================
# 8. RFM 相关性热力图
# =========================================================
def plot_rfm_correlation(rfm_df) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = rfm_df[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, mask=mask, ax=ax,
                annot_kws={'size': 12}, linewidths=0.5,
                xticklabels=['Recency', 'Frequency', 'Monetary', 'RFM评分'],
                yticklabels=['Recency', 'Frequency', 'Monetary', 'RFM评分'])
    ax.set_title('RFM 特征相关性矩阵', fontweight='bold', pad=12)
    plt.tight_layout()
    return save(fig, '08_rfm_correlation.png')
