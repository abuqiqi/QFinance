# 量子核分类模块实施计划

## 面向个人投资顾问助手的投资风险分类Demo

## 一、模块定位

本模块面向个人投资顾问助手中的投资计划风险评估任务，负责将结构化投资计划转化为量子核分类模型可处理的特征输入，并输出多类投资风险的分类结果。

本模块不负责完整Agent系统、不负责真实交易拦截、不负责交易执行，也不直接面向交易系统。其作用是作为后台分类模型，为上层Agent提供定量风险判断依据。

模块输入来自上游Agent或特征构造模块，主要包括用户画像、投资计划和股票特征；模块输出包括三类风险概率、风险标签、综合风险等级和模型置信度，供下游解释Agent生成自然语言风险分析。

---

## 二、Demo阶段任务范围

Demo阶段只考虑三类投资风险：

```text
1. 仓位集中风险 concentration_risk
2. 个股波动风险 volatility_risk
3. 流动性风险 liquidity_risk
```

暂不考虑以下复杂风险：

```text
事件风险
基本面风险
市场风险
行业风险
风格匹配风险
买入时点风险
```

原因是这些风险需要新闻、公告、财报、宏观市场、长期用户画像等额外数据，第一版Demo会增加系统复杂度。后续如果Demo效果较好，可以在现有框架上继续扩展标签和特征。

---

## 三、整体技术流程

模块整体流程如下：

```text
模拟投资计划数据
   ↓
原始字段整理
   ↓
特征工程
   ↓
标签生成
   ↓
特征归一化
   ↓
量子核分类模型训练
   ↓
经典基线模型对比
   ↓
模型评估
   ↓
分类结果输出
   ↓
交付给上层Agent解释
```

在系统中的位置如下：

```text
用户自然语言问题
   ↓
上游Agent解析投资计划
   ↓
结构化投资计划JSON
   ↓
量子核分类模块
   ↓
风险概率 / 风险标签 / 综合风险等级
   ↓
风险解释Agent
```

---

## 四、模拟数据组织方式

Demo阶段建议同时保留两种数据格式：

```text
1. JSON格式：用于模拟上游Agent传入的结构化投资计划；
2. CSV格式：用于批量训练、测试和模型评估。
```

### 4.1 JSON格式

JSON格式用于单条样本输入和Agent接口模拟。

示例：

```json
{
  "sample_id": "S000001",
  "user_profile": {
    "user_id": "U0001",
    "total_asset": 500000,
    "cash_available": 300000,
    "risk_preference": "moderate",
    "current_stock_position_value": 80000
  },
  "trade_plan": {
    "action": "buy",
    "market": "SSE",
    "stock_code": "600006",
    "trade_time": "2026-06-08 10:30:00",
    "volume_lot": 1000,
    "volume_share": 100000,
    "estimated_price": 5.2,
    "estimated_amount": 520000
  },
  "stock_features": {
    "volatility_20d": 0.36,
    "return_5d": 0.12,
    "abs_return_5d": 0.12,
    "avg_volume_20d": 2000000,
    "turnover_rate": 0.018
  },
  "derived_features": {
    "estimated_amount_ratio": 1.04,
    "current_position_ratio": 0.16,
    "holding_ratio_after_trade": 1.20,
    "trade_volume_to_avg_volume": 0.05
  },
  "labels": {
    "concentration_risk": 1,
    "volatility_risk": 1,
    "liquidity_risk": 0,
    "overall_risk_level": "high"
  }
}
```

### 4.2 CSV格式

CSV格式用于训练模型。建议将嵌套JSON展开为二维表。

字段示例：

```text
sample_id,
total_asset,
cash_available,
risk_preference_code,
current_stock_position_value,
current_position_ratio,
action_code,
market_code,
stock_code,
trade_hour,
volume_lot,
volume_share,
estimated_price,
estimated_amount,
estimated_amount_ratio,
holding_ratio_after_trade,
volatility_20d,
return_5d,
abs_return_5d,
avg_volume_20d,
turnover_rate,
trade_volume_to_avg_volume,
concentration_risk,
volatility_risk,
liquidity_risk,
overall_risk_level
```

---

## 五、模拟数据字段设计

模拟数据分为四类字段：用户画像字段、投资计划字段、股票特征字段和派生特征字段。

### 5.1 用户画像字段

| 字段名                            | 含义        | 用途         |
| ------------------------------ | --------- | ---------- |
| `user_id`                      | 用户编号      | 区分不同用户     |
| `total_asset`                  | 用户总资产     | 判断仓位集中风险   |
| `cash_available`               | 可用现金      | 判断交易规模是否过大 |
| `risk_preference`              | 用户风险偏好    | 调整仓位集中风险阈值 |
| `current_stock_position_value` | 当前该股票持仓市值 | 判断买入后单股暴露  |
| `current_position_ratio`       | 当前该股票持仓占比 | 模型输入特征     |

风险偏好建议编码为：

```text
conservative = 0
moderate = 1
aggressive = 2
```

### 5.2 投资计划字段

| 字段名                | 含义            | 用途            |
| ------------------ | ------------- | ------------- |
| `action`           | 买入/卖出         | Demo阶段可先只做buy |
| `market`           | 市场，例如SSE/SZSE | 区分交易所         |
| `stock_code`       | 股票代码          | 标识股票          |
| `trade_time`       | 计划交易时间        | 可提取交易小时       |
| `trade_hour`       | 交易小时          | 可作为辅助特征       |
| `volume_lot`       | 买入手数          | 原始用户输入        |
| `volume_share`     | 买入股数          | 1手=100股       |
| `estimated_price`  | 估计成交价         | 计算计划买入金额      |
| `estimated_amount` | 计划买入金额        | 判断仓位风险        |

其中：

```text
volume_share = volume_lot × 100
estimated_amount = volume_share × estimated_price
```

### 5.3 股票特征字段

| 字段名              | 含义        | 用途       |
| ---------------- | --------- | -------- |
| `volatility_20d` | 近20日波动率   | 判断个股波动风险 |
| `return_5d`      | 近5日收益率    | 判断短期波动   |
| `abs_return_5d`  | 近5日收益率绝对值 | 判断短期剧烈波动 |
| `avg_volume_20d` | 近20日日均成交量 | 判断流动性风险  |
| `turnover_rate`  | 换手率       | 判断股票活跃程度 |

Demo阶段这些字段可以随机模拟生成，不需要接入真实行情数据。

### 5.4 派生特征字段

| 字段名                          | 含义             | 用途        |
| ---------------------------- | -------------- | --------- |
| `estimated_amount_ratio`     | 计划买入金额/总资产     | 仓位风险核心特征  |
| `holding_ratio_after_trade`  | 买入后该股票持仓占总资产比例 | 仓位风险核心特征  |
| `trade_volume_to_avg_volume` | 计划买入股数/日均成交量   | 流动性风险核心特征 |

计算方式：

```text
estimated_amount_ratio = estimated_amount / total_asset

holding_ratio_after_trade =
(current_stock_position_value + estimated_amount) / total_asset

trade_volume_to_avg_volume =
volume_share / avg_volume_20d
```

---

## 六、标签体系设计

Demo阶段采用三类二分类标签。

```text
concentration_risk ∈ {0, 1}
volatility_risk ∈ {0, 1}
liquidity_risk ∈ {0, 1}
```

同时生成一个综合风险等级：

```text
overall_risk_level ∈ {low, medium, medium_high, high}
```

### 6.1 仓位集中风险标签

根据买入后单只股票持仓占比生成。

基础规则：

```text
如果 holding_ratio_after_trade >= threshold，则 concentration_risk = 1；
否则 concentration_risk = 0。
```

阈值根据用户风险偏好调整：

| 用户风险偏好       |   阈值 |
| ------------ | ---: |
| conservative | 0.20 |
| moderate     | 0.30 |
| aggressive   | 0.40 |

也就是说：

```text
保守型用户买入后单股持仓超过20%，标记为仓位集中风险；
稳健型用户超过30%，标记为仓位集中风险；
激进型用户超过40%，标记为仓位集中风险。
```

### 6.2 个股波动风险标签

根据近20日波动率和近5日涨跌幅生成。

规则：

```text
如果 volatility_20d >= 0.35，则 volatility_risk = 1；
或者 abs_return_5d >= 0.15，则 volatility_risk = 1；
否则 volatility_risk = 0。
```

### 6.3 流动性风险标签

根据计划买入量与日均成交量的比例生成。

规则：

```text
如果 trade_volume_to_avg_volume >= 0.10，则 liquidity_risk = 1；
否则 liquidity_risk = 0。
```

### 6.4 综合风险等级

综合风险等级根据三个风险标签和风险概率生成。

Demo阶段可以先用简单规则：

```text
如果三个风险均为0：overall_risk_level = low
如果一个风险为1：overall_risk_level = medium
如果两个风险为1：overall_risk_level = medium_high
如果三个风险均为1：overall_risk_level = high
```

后续可以改为加权评分：

```text
overall_score =
0.4 × concentration_risk_prob
+ 0.35 × volatility_risk_prob
+ 0.25 × liquidity_risk_prob
```

---

## 七、模型输入设计

量子核分类模型不直接处理原始JSON，也不直接处理股票代码、自然语言问题或时间字符串。模型输入为经过特征工程和归一化后的数值向量。

Demo阶段建议使用8维特征：

```text
x = [
  estimated_amount_ratio,
  holding_ratio_after_trade,
  current_position_ratio,
  risk_preference_code,
  volatility_20d,
  abs_return_5d,
  turnover_rate,
  trade_volume_to_avg_volume
]
```

对应含义如下：

| 维度 | 特征名                          | 主要对应风险 |
| -: | ---------------------------- | ------ |
|  1 | `estimated_amount_ratio`     | 仓位集中风险 |
|  2 | `holding_ratio_after_trade`  | 仓位集中风险 |
|  3 | `current_position_ratio`     | 仓位集中风险 |
|  4 | `risk_preference_code`       | 仓位集中风险 |
|  5 | `volatility_20d`             | 个股波动风险 |
|  6 | `abs_return_5d`              | 个股波动风险 |
|  7 | `turnover_rate`              | 流动性风险  |
|  8 | `trade_volume_to_avg_volume` | 流动性风险  |

输入维度为8，因此量子特征映射使用8个量子比特。

特征归一化建议：

```text
将所有输入特征归一化到 [0, π]
```

原因是量子特征映射通常将输入特征作为旋转角度参数，归一化到固定区间有利于模型稳定。

---

## 八、量子核模型构建

### 8.1 模型思路

量子核分类的核心思想是：

```text
将经典特征向量 x 编码为量子态 |φ(x)⟩；
计算两个样本在量子特征空间中的相似度 K(x_i, x_j)；
将量子核矩阵输入SVM完成分类。
```

模型流程：

```text
8维风险特征
   ↓
特征归一化到[0, π]
   ↓
量子特征映射
   ↓
量子核矩阵计算
   ↓
SVM分类器
   ↓
风险标签/风险概率输出
```

### 8.2 量子特征映射

Demo阶段建议使用：

```text
ZZFeatureMap
feature_dimension = 8
reps = 1 或 2
entanglement = linear
```

推荐第一版配置：

```text
feature_dimension = 8
reps = 2
entanglement = linear
```

理由：

```text
8维特征对应8个量子比特；
linear entanglement计算开销较小；
reps=2比reps=1表达能力稍强，但仍适合Demo。
```

### 8.3 多标签分类实现方式

Demo阶段建议采用**三个独立的二分类量子核SVM**。

```text
Model 1: QKSVM_concentration
Model 2: QKSVM_volatility
Model 3: QKSVM_liquidity
```

每个模型独立训练一个标签：

```text
QKSVM_concentration: 输入X，训练y_concentration
QKSVM_volatility: 输入X，训练y_volatility
QKSVM_liquidity: 输入X，训练y_liquidity
```

每个模型输出对应风险概率和二分类标签。

这种方式最简单，后续扩展新风险类型时，只需要新增一个二分类器。

### 8.4 经典基线模型

为了评估量子核分类模块，建议保留经典基线：

```text
Classical SVM with RBF kernel
Random Forest
Logistic Regression
```

最小Demo至少保留：

```text
Classical SVM with RBF kernel
Quantum Kernel SVM
```

对比目的不是证明量子模型必然更优，而是展示：

```text
在相同输入特征和相同标签体系下，量子核模型可以作为一种高维相似度建模方法参与投资风险分类。
```

---

## 九、训练流程

### 9.1 数据生成

生成模拟样本，例如：

```text
总样本数：1000-3000
训练集：70%
验证集：10%
测试集：20%
```

如果量子核计算开销较大，第一版可缩小为：

```text
总样本数：500-1000
训练集：300-700
测试集：100-300
```

### 9.2 数据划分

按样本划分：

```text
train / validation / test
```

需要尽量保持三类风险标签比例相对均衡。

如果某一类风险样本太少，可以在模拟生成时控制比例，例如：

```text
concentration_risk正样本比例：30%-40%
volatility_risk正样本比例：30%-40%
liquidity_risk正样本比例：20%-30%
```

### 9.3 特征归一化

对训练集拟合归一化器：

```text
scaler.fit(X_train)
```

对训练集、验证集、测试集使用同一个归一化器：

```text
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

归一化范围：

```text
[0, π]
```

### 9.4 模型训练

对每个风险标签分别训练：

```text
for risk_type in [concentration, volatility, liquidity]:
    1. 构建量子特征映射
    2. 构建量子核
    3. 训练QSVC
    4. 在验证集/测试集上预测
    5. 输出分类指标
```

### 9.5 概率输出

SVM天然输出的是分类边界，不一定直接给出概率。Demo阶段可以采用两种方式之一：

```text
方案A：使用带概率校准的SVM输出概率；
方案B：使用decision function经过sigmoid转换得到近似概率。
```

为了工程简单，第一版可以输出：

```text
risk_score ∈ [0, 1]
```

并说明它是模型风险得分，不强行称为严格概率。

---

## 十、模型评估设计

每个风险标签分别报告指标。

### 10.1 单标签指标

对每一类风险输出：

```text
Accuracy
Precision
Recall
F1-score
AUC
```

示例表：

| 风险类型   | Accuracy | Precision | Recall |   F1 |  AUC |
| ------ | -------: | --------: | -----: | ---: | ---: |
| 仓位集中风险 |     0.86 |      0.84 |   0.88 | 0.86 | 0.91 |
| 个股波动风险 |     0.82 |      0.80 |   0.83 | 0.81 | 0.88 |
| 流动性风险  |     0.79 |      0.76 |   0.74 | 0.75 | 0.84 |

### 10.2 量子模型与经典模型对比

建议至少对比：

```text
Classical SVM
Quantum Kernel SVM
```

对比表：

| 模型                 | 仓位风险F1 | 波动风险F1 | 流动性风险F1 | 平均F1 |
| ------------------ | -----: | -----: | ------: | ---: |
| Classical SVM      |   0.84 |   0.80 |    0.73 | 0.79 |
| Quantum Kernel SVM |   0.86 |   0.81 |    0.75 | 0.81 |

注意表述要稳妥：

```text
量子核模型在Demo模拟数据上表现出可行性；
其优势主要体现为高维特征空间中的相似度建模能力；
后续需要在更丰富数据和更多风险标签上进一步验证。
```

---

## 十一、分类输出格式

分类模块最终向上层Agent输出JSON。

### 11.1 单条样本输出格式

```json
{
  "sample_id": "S000001",
  "model_version": "qksvm_demo_v1",
  "input_summary": {
    "stock_code": "600006",
    "action": "buy",
    "volume_lot": 1000,
    "volume_share": 100000,
    "estimated_amount": 520000
  },
  "risk_scores": {
    "concentration_risk": 0.82,
    "volatility_risk": 0.69,
    "liquidity_risk": 0.35
  },
  "risk_labels": {
    "concentration_risk": 1,
    "volatility_risk": 1,
    "liquidity_risk": 0
  },
  "overall_risk": {
    "overall_score": 0.66,
    "overall_level": "medium_high"
  },
  "confidence": 0.74,
  "model_outputs": {
    "classifier_type": "quantum_kernel_svm",
    "feature_map": "ZZFeatureMap",
    "num_qubits": 8,
    "feature_dimension": 8
  }
}
```

### 11.2 风险分数含义

建议定义：

```text
0.00 - 0.30：低风险
0.30 - 0.60：中等风险
0.60 - 0.80：较高风险
0.80 - 1.00：高风险
```

对应解释：

```text
risk_score越高，表示该投资计划越接近该类风险样本。
```

不要在Demo阶段过度声称：

```text
risk_score = 真实发生风险的概率
```

更稳妥的说法是：

```text
risk_score表示模型判断该计划具有某类风险特征的强度。
```

---

## 十二、建议代码目录

```text
quantum-risk-classifier/
├── data/
│   ├── raw/
│   │   ├── investment_samples.jsonl
│   │   └── investment_samples.csv
│   ├── processed/
│   │   ├── features.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── outputs/
│       └── predictions.jsonl
│
├── src/
│   ├── data_generation/
│   │   ├── generate_users.py
│   │   ├── generate_stocks.py
│   │   ├── generate_trade_plans.py
│   │   └── generate_labels.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── normalize_features.py
│   │
│   ├── models/
│   │   ├── train_classical_svm.py
│   │   ├── train_quantum_kernel.py
│   │   ├── predict_quantum_kernel.py
│   │   └── evaluate.py
│   │
│   ├── interfaces/
│   │   ├── input_schema.py
│   │   └── output_schema.py
│   │
│   └── utils/
│       ├── config.py
│       └── metrics.py
│
├── configs/
│   └── qksvm_demo_v1.yaml
│
├── notebooks/
│   └── demo_quantum_kernel_classifier.ipynb
│
├── results/
│   ├── metrics_classical_svm.csv
│   ├── metrics_quantum_kernel.csv
│   └── comparison_table.csv
│
└── README.md
```

---

## 十三、阶段安排

### 阶段一：任务定义与数据结构确认

目标：明确模块输入、输出和标签体系。

任务：

```text
1. 确认三类风险标签；
2. 确认JSON输入格式；
3. 确认CSV训练格式；
4. 确认8维模型输入特征；
5. 确认输出JSON格式。
```

产出：

```text
input_schema.json
output_schema.json
feature_list.md
label_definition.md
```

---

### 阶段二：模拟数据生成

目标：生成可用于训练和测试的模拟投资计划数据。

任务：

```text
1. 生成用户画像；
2. 生成股票特征；
3. 生成投资计划；
4. 计算派生特征；
5. 根据规则生成三类风险标签；
6. 导出JSONL和CSV数据。
```

产出：

```text
investment_samples.jsonl
investment_samples.csv
features.csv
```

---

### 阶段三：特征工程与数据预处理

目标：将模拟数据转成模型可用输入。

任务：

```text
1. 选取8维数值特征；
2. 编码risk_preference；
3. 处理异常值；
4. 划分训练集、验证集、测试集；
5. 将特征归一化到[0, π]。
```

产出：

```text
train.csv
val.csv
test.csv
scaler.pkl
```

---

### 阶段四：经典基线模型实现

目标：建立可对比的经典模型基线。

任务：

```text
1. 实现经典SVM分类器；
2. 对三类风险分别训练二分类模型；
3. 输出每类风险预测结果；
4. 计算Accuracy、Precision、Recall、F1、AUC。
```

产出：

```text
classical_svm_models
metrics_classical_svm.csv
```

---

### 阶段五：量子核分类模型实现

目标：完成量子核SVM训练和预测。

任务：

```text
1. 构建8维ZZFeatureMap；
2. 构建FidelityQuantumKernel；
3. 为三类风险分别训练QSVC；
4. 输出每类风险分数和标签；
5. 计算模型指标；
6. 与经典SVM进行对比。
```

推荐配置：

```text
num_qubits = 8
feature_dimension = 8
feature_map = ZZFeatureMap
reps = 2
entanglement = linear
classifier = QSVC
```

产出：

```text
qksvm_concentration_model
qksvm_volatility_model
qksvm_liquidity_model
metrics_quantum_kernel.csv
comparison_table.csv
```

---

### 阶段六：输出接口封装

目标：将模型输出封装为上游Agent可读取的JSON格式。

任务：

```text
1. 设计predict接口；
2. 输入单条结构化投资计划；
3. 自动提取8维特征；
4. 调用三个量子核分类器；
5. 输出risk_scores、risk_labels和overall_risk；
6. 生成predictions.jsonl。
```

产出：

```text
predict_quantum_kernel.py
predictions.jsonl
output_example.json
```

---

### 阶段七：结果整理与Demo交付

目标：形成可展示的量子核分类Demo结果。

任务：

```text
1. 整理模型评估表；
2. 整理经典模型与量子核模型对比结果；
3. 准备单条样本输入输出示例；
4. 撰写模块说明；
5. 准备对接Agent系统的接口文档。
```

产出：

```text
README.md
model_report.md
接口说明文档
样例输入输出JSON
模型对比表
```

---

## 十四、第一版最小可行Demo

第一版最小可行Demo应完成以下内容：

```text
1. 生成至少500条模拟投资计划样本；
2. 每条样本包含用户画像、投资计划、股票特征和三类风险标签；
3. 提取8维模型输入特征；
4. 训练经典SVM基线；
5. 训练三个量子核SVM二分类器；
6. 输出三类风险分数；
7. 输出综合风险等级；
8. 生成一条完整样例：
   输入：“买入600006股票1000手”
   输出：仓位集中风险、个股波动风险、流动性风险及综合风险等级。
```

---

## 十五、后续扩展方向

如果第一版Demo效果较好，可以继续扩展：

### 15.1 增加风险标签

```text
事件风险 event_risk
基本面风险 fundamental_risk
市场风险 market_risk
行业风险 sector_risk
风格匹配风险 style_mismatch_risk
买入时点风险 timing_risk
```

### 15.2 增加数据字段

```text
新闻情绪
公告事件
财务指标
估值分位数
行业景气度
市场指数波动率
用户历史交易行为
用户最大可接受回撤
```

### 15.3 增强量子模型

```text
尝试不同量子特征映射；
比较ZZFeatureMap、PauliFeatureMap等；
调整reps和纠缠结构；
探索量子核对齐方法；
探索量子核与经典核融合。
```

### 15.4 与Agent系统对接

```text
由Agent负责自然语言解析；
由本模块负责结构化输入后的量子核分类；
由Agent负责风险解释和建议生成。
```

---

## 十六、模块边界说明

本模块负责：

```text
模拟数据构造；
风险标签生成；
特征工程；
量子核分类模型训练；
经典基线对比；
分类结果JSON输出。
```

本模块不负责：

```text
自然语言交互；
完整Agent调度；
真实行情数据接入；
真实交易系统拦截；
投资建议合规审查；
自动下单或交易执行。
```

本模块的定位是：

```text
为个人投资顾问助手提供量子增强的风险分类能力。
```
