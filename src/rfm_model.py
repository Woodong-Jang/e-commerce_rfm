"""
RFM 模型模块
- 五分位评分法构建 RFM 分数
- K-Means 聚类验证分层合理性
- 用户分层标签
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    构建 RFM 特征表
    
    Parameters
    ----------
    df : 清洗后的订单数据
    snapshot_date : 分析基准日期（默认取最大日期+1天）
    
    Returns
    -------
    rfm : 含 R/F/M 原始值及评分的 DataFrame
    """
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    print(f"[4/5] 构建 RFM 模型 (基准日期: {snapshot_date.date()})")

    rfm = df.groupby('CustomerID').agg(
        Recency   = ('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
        Frequency = ('InvoiceNo',    'nunique'),
        Monetary  = ('TotalAmount',  'sum')
    ).reset_index()

    # ---- 五分位评分 ----
    # Recency: 天数越少 → 分数越高（反向）
    rfm['R_Score'] = pd.qcut(rfm['Recency'],   5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    # Frequency: 频次越高 → 分数越高（rank 处理并列）
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: 金额越高 → 分数越高
    rfm['M_Score'] = pd.qcut(rfm['Monetary'],  5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # ---- 用户分层 ----
    rfm['Segment'] = rfm.apply(_assign_segment, axis=1)

    print(f"      用户总数: {len(rfm):,}")
    print(f"      RFM评分范围: {rfm['RFM_Score'].min()} ~ {rfm['RFM_Score'].max()}")
    print(f"\n      用户分层结果:")
    seg_summary = rfm.groupby('Segment').agg(
        用户数=('CustomerID', 'count'),
        平均Recency=('Recency', 'mean'),
        平均Frequency=('Frequency', 'mean'),
        平均Monetary=('Monetary', 'mean')
    ).round(1)
    print(seg_summary.to_string())

    return rfm


def _assign_segment(row) -> str:
    """根据 RFM 评分分配用户层"""
    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
    score = row['RFM_Score']

    if score >= 12 and r >= 4:
        return '冠军用户'
    elif score >= 10 or (r >= 4 and f >= 3):
        return '忠实用户'
    elif score >= 8 or (r >= 3 and f >= 2 and m >= 2):
        return '潜力用户'
    elif r <= 2 and f >= 2:
        return '流失风险'
    elif r <= 2 and f == 1:
        return '沉默用户'
    else:
        return '新客用户'


def kmeans_validation(rfm: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    用 K-Means 聚类验证 RFM 分层合理性
    
    Returns: 含聚类标签、轮廓系数、各簇统计的字典
    """
    print(f"\n[K-Means] 开始 K-Means 聚类验证 (k={n_clusters})")

    # 对数变换（处理偏态分布）+ 标准化
    features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    features['Recency']   = np.log1p(features['Recency'])
    features['Frequency'] = np.log1p(features['Frequency'])
    features['Monetary']  = np.log1p(features['Monetary'])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # 肘部法则：计算 k=2~8 的 inertia
    inertia_list = []
    sil_list = []
    k_range = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertia_list.append(km.inertia_)
        sil_list.append(silhouette_score(X, labels, sample_size=min(1000, len(X))))

    # 最优 k 的聚类结果
    km_best = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['KMeans_Cluster'] = km_best.fit_predict(X)

    sil = silhouette_score(X, rfm['KMeans_Cluster'], sample_size=min(1000, len(X)))
    print(f"      轮廓系数(Silhouette Score): {sil:.4f}  (越接近1越好，>0.3即合理)")

    # 各簇统计
    cluster_stats = rfm.groupby('KMeans_Cluster').agg(
        用户数=('CustomerID', 'count'),
        平均Recency=('Recency', 'mean'),
        平均Frequency=('Frequency', 'mean'),
        平均Monetary=('Monetary', 'mean'),
        平均RFM评分=('RFM_Score', 'mean')
    ).round(2)
    print(f"\n      K-Means 各簇统计:")
    print(cluster_stats.to_string())

    return {
        'rfm_with_cluster': rfm,
        'k_range': list(k_range),
        'inertia': inertia_list,
        'silhouette': sil_list,
        'best_silhouette': sil,
        'cluster_stats': cluster_stats
    }


def get_segment_revenue(rfm: pd.DataFrame) -> pd.DataFrame:
    """计算各用户层营收贡献"""
    seg = rfm.groupby('Segment').agg(
        用户数=('CustomerID', 'count'),
        总消费=('Monetary', 'sum'),
        平均消费=('Monetary', 'mean'),
        平均购买次数=('Frequency', 'mean'),
        平均距今天数=('Recency', 'mean')
    ).round(2)
    seg['用户占比%'] = (seg['用户数'] / seg['用户数'].sum() * 100).round(1)
    seg['营收占比%'] = (seg['总消费'] / seg['总消费'].sum() * 100).round(1)
    return seg.sort_values('总消费', ascending=False)
