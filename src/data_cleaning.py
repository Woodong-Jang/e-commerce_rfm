"""
数据清洗模块
Kaggle Online Retail Dataset: https://www.kaggle.com/datasets/carrie1/ecommerce-data
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath: str) -> pd.DataFrame:
    """加载原始数据"""
    print(f"[1/5] 加载数据: {filepath}")
    try:
        df = pd.read_csv(filepath, encoding='latin1')
    except Exception:
        df = pd.read_excel(filepath, dtype={'CustomerID': str})
    print(f"      原始记录数: {len(df):,}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """完整数据清洗流程"""
    original_count = len(df)
    log = {}

    # 1. 标准化列名
    df.columns = df.columns.str.strip()

    # 2. 删除缺失CustomerID
    df = df.dropna(subset=['CustomerID'])
    log['缺失CustomerID'] = original_count - len(df)
    print(f"[2/5] 删除缺失CustomerID: -{log['缺失CustomerID']:,} 条")

    # 3. 删除取消订单（InvoiceNo以C开头）
    before = len(df)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    log['取消订单'] = before - len(df)
    print(f"      删除取消订单(C前缀): -{log['取消订单']:,} 条")

    # 4. 删除Quantity <= 0
    before = len(df)
    df = df[df['Quantity'] > 0]
    log['负数Quantity'] = before - len(df)
    print(f"      删除负数Quantity: -{log['负数Quantity']:,} 条")

    # 5. 删除UnitPrice <= 0
    before = len(df)
    df = df[df['UnitPrice'] > 0]
    log['负数UnitPrice'] = before - len(df)
    print(f"      删除无效UnitPrice: -{log['负数UnitPrice']:,} 条")

    # 6. 处理异常值（IQR方法，保留合理批发大单）
    before = len(df)
    Q1 = df['UnitPrice'].quantile(0.01)
    Q3 = df['UnitPrice'].quantile(0.99)
    df = df[(df['UnitPrice'] >= Q1) & (df['UnitPrice'] <= Q3)]
    log['价格异常值'] = before - len(df)
    print(f"      删除价格极端异常值(1%-99%): -{log['价格异常值']:,} 条")

    # 7. 类型转换
    df['CustomerID'] = df['CustomerID'].astype(str).str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)

    # 8. 特征工程
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Hour'] = df['InvoiceDate'].dt.hour

    print(f"[3/5] 清洗完成:")
    print(f"      有效记录数: {len(df):,}")
    print(f"      有效用户数: {df['CustomerID'].nunique():,}")
    print(f"      时间范围: {df['InvoiceDate'].min().date()} ~ {df['InvoiceDate'].max().date()}")
    print(f"      总销售额: £{df['TotalAmount'].sum():,.0f}")

    return df, log


def get_cleaning_summary(original_count: int, df: pd.DataFrame, log: dict) -> dict:
    """返回清洗摘要"""
    return {
        'original_count': original_count,
        'clean_count': len(df),
        'removed_count': original_count - len(df),
        'removal_rate': f"{(original_count - len(df)) / original_count * 100:.1f}%",
        'unique_customers': df['CustomerID'].nunique(),
        'unique_products': df['StockCode'].nunique(),
        'total_revenue': df['TotalAmount'].sum(),
        'date_range': f"{df['InvoiceDate'].min().date()} ~ {df['InvoiceDate'].max().date()}",
        'log': log
    }
