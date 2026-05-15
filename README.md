# E-commerce RFM

基于 Kaggle Online Retail Dataset 的电商用户价值分析项目。项目通过数据清洗、销售统计、转化漏斗、RFM 用户分层和 K-Means 聚类验证，帮助识别高价值用户、潜力用户、流失风险用户，并输出可复用的分析结果。

## 项目功能

- 清洗电商订单数据，处理缺失用户、取消订单、无效数量、无效价格和价格异常值
- 计算月度销售额、国家销售分布、畅销商品等基础经营指标
- 构建电商转化漏斗，定位关键流失环节
- 基于 Recency、Frequency、Monetary 构建 RFM 用户价值模型
- 使用 K-Means 聚类和轮廓系数验证用户分层合理性
- 生成 CSV 分析结果和 Matplotlib/Seaborn 可视化图表

## 项目结构

```text
e-commerce_rfm/
├── data/                    # 原始数据目录
│   └── data.csv             # Online Retail 数据集，通常不纳入版本控制
├── fonts/                   # 字体文件
├── notebooks/               # Notebook 演示
├── outputs/                 # 分析结果输出目录
│   ├── rfm_results.csv
│   └── segment_revenue.csv
├── src/
│   ├── data_cleaning.py     # 数据加载与清洗
│   ├── funnel_analysis.py   # 转化漏斗和经营指标统计
│   ├── rfm_model.py         # RFM 建模与 K-Means 验证
│   └── visualization.py     # 图表生成
├── main.py                  # 主运行入口
├── requirements.txt         # Python 依赖
└── README.md
```

## 环境要求

建议使用 Python 3.9 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

主要依赖包括：

- pandas / numpy：数据处理
- matplotlib / seaborn：可视化
- scikit-learn：K-Means 聚类与轮廓系数
- openpyxl：Excel 数据读取
- jupyter / ipykernel：Notebook 演示

## 数据准备

项目默认使用 Kaggle Online Retail Dataset：

[https://www.kaggle.com/datasets/carrie1/ecommerce-data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

下载后，将数据文件放入 `data/` 目录，并命名为以下任意一种：

- `data/data.csv`
- `data/online_retail.xlsx`
- `data/Online Retail.xlsx`

如果没有提供真实数据，`main.py` 会尝试生成演示用样本数据。

数据字段应包含：

| 字段 | 含义 |
| --- | --- |
| InvoiceNo | 订单编号 |
| StockCode | 商品编码 |
| Description | 商品描述 |
| Quantity | 购买数量 |
| InvoiceDate | 订单时间 |
| UnitPrice | 商品单价 |
| CustomerID | 用户 ID |
| Country | 国家或地区 |

## 运行方式

使用默认数据路径：

```bash
python main.py
```

指定数据文件：

```bash
python main.py --data data/data.csv
```

跳过图表生成，仅输出分析结果：

```bash
python main.py --skip-plots
```

## 输出结果

运行完成后，项目会在 `outputs/` 目录下生成分析结果：

| 文件 | 说明 |
| --- | --- |
| `outputs/rfm_results.csv` | 每个用户的 RFM 指标、评分、用户分层和聚类标签 |
| `outputs/segment_revenue.csv` | 各用户层级的用户数、消费额、营收占比等 |
| `outputs/funnel_analysis.csv` | 转化漏斗各阶段用户量和转化率 |
| `outputs/monthly_sales.csv` | 月度销售额、订单数和用户数 |
| `outputs/country_stats.csv` | 各国家销售表现 |
| `outputs/top_products.csv` | 销售额最高的商品 |
| `outputs/figures/` | 可视化图表输出目录 |

## 分析方法

### 数据清洗

清洗流程包括删除缺失 `CustomerID` 的记录、剔除取消订单、过滤非正数数量和价格，并使用分位数方法处理极端价格异常值。清洗后新增 `TotalAmount`、`YearMonth`、`DayOfWeek` 和 `Hour` 等分析字段。

### RFM 用户分层

RFM 模型从三个维度衡量用户价值：

- Recency：最近一次购买距分析日期的天数，越小表示越活跃
- Frequency：购买次数，越高表示购买越频繁
- Monetary：累计消费金额，越高表示贡献越大

项目使用五分位评分方法生成 `R_Score`、`F_Score`、`M_Score`，并根据综合分数划分用户层级。

### 聚类验证

项目使用 K-Means 对 RFM 特征进行聚类，并通过轮廓系数和肘部法则辅助判断用户分群效果。