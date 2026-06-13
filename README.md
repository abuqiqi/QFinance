# Quantum Risk Classifier Demo

基于模拟投资计划数据，对仓位集中、个股波动和流动性三类风险分别训练经典 RBF SVM 与量子核 SVM。该项目只用于技术演示，不构成投资建议。

## 环境

```bash
uv python install 3.11
uv sync --extra dev
```

## 快速运行

```bash
uv run quantum-risk export-schemas
uv run quantum-risk generate-data --count 600
uv run quantum-risk prepare-data
uv run quantum-risk train-classical
uv run quantum-risk train-quantum --subset-size 180
uv run quantum-risk train-quantum --full-data --experiment full_shared_baseline
uv run quantum-risk tune-quantum --budget-minutes 60
uv run quantum-risk predict --input examples/input_valid.json --output results/prediction.json
uv run pytest
```

跳过耗时的量子阶段进行流水线验证：

```bash
uv run quantum-risk run-demo --count 600 --skip-quantum
```

## Agent 对接案例

分类模块不直接理解用户自然语言。负责 Agent 系统的服务需要先解析用户意图，再结合用户档案和行情服务补齐字段，最终生成符合 `schemas/input_schema.json` 的投资计划 JSON。

### 1. 用户自然语言

用户本次输入可以是：

> 我想以每股 5.2 元买入 1000 手 600006，请帮我评估风险。

Agent 从本次对话中提取：

| 字段 | 值 | 来源 |
| --- | --- | --- |
| `action` | `buy` | “买入” |
| `stock_code` | `600006` | 股票代码 |
| `volume_lot` | `1000` | 计划买入手数，A 股 1 手等于 100 股 |
| `estimated_price` | `5.2` | 用户期望价格；没有指定时应查询行情或继续询问 |

### 2. 用户档案信息

以下信息通常由 Agent 从用户档案或账户服务读取，不要求用户每次重复输入：

> 用户总资产 50 万元，可用资金 30 万元，风险偏好为稳健型，目前持有价值 8 万元的 600006。

对应字段：

```json
{
  "user_id": "U0001",
  "total_asset": 500000,
  "cash_available": 300000,
  "risk_preference": "moderate",
  "current_stock_position_value": 80000
}
```

`current_stock_position_value` 必须是本次计划交易股票的当前持仓市值，而不是全部股票持仓市值。

### 3. 当前股票特征

以下信息应由行情服务提供，不应让用户填写，也不应在正式对接时随机生成：

```json
{
  "volatility_20d": 0.36,
  "return_5d": 0.12,
  "avg_volume_20d": 2000000,
  "turnover_rate": 0.018
}
```

- `volatility_20d`：近 20 日波动率。
- `return_5d`：近 5 日收益率，`0.12` 表示上涨 12%。
- `avg_volume_20d`：近 20 日日均成交股数。
- `turnover_rate`：换手率，`0.018` 表示 1.8%。

### 4. Agent 补充的字段

- `sample_id`：由 Agent 生成的请求追踪 ID。
- `market`：可选字段。Agent 可以根据证券主数据补充 `SSE` 或 `SZSE`；不提供时不会影响当前模型预测。该字段不参与特征计算和风险判断。
- `trade_time`：用户指定时间；未指定时使用计划执行时间或当前请求时间。

Agent 不需要计算 `volume_share`、预计交易金额、持仓比例或计划交易量占日均成交量比例。这些派生字段由分类模块根据原始输入重新计算，避免上下游计算不一致。

### 5. 最终模型输入

完整示例位于 [`examples/input_valid.json`](examples/input_valid.json)：

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
    "stock_code": "600006",
    "trade_time": "2026-06-08 10:30:00",
    "volume_lot": 1000,
    "estimated_price": 5.2
  },
  "stock_features": {
    "volatility_20d": 0.36,
    "return_5d": 0.12,
    "avg_volume_20d": 2000000,
    "turnover_rate": 0.018
  }
}
```

当前 Demo 只接受 `buy`。`market` 不是必填字段；如果提供，只接受 `SSE` 或 `SZSE`。输入字段的完整约束见 [`schemas/input_schema.json`](schemas/input_schema.json)。

## 当前 Demo 数据如何生成

当前训练和评估使用的 `user_profile`、`trade_plan`、`stock_features` 均为模拟数据，不来自真实用户、账户或行情系统。生成器会先按目标比例决定样本是否应具有某类风险，再反向生成相关字段，因此这些数据用于验证模型和接口流程，不代表真实市场的数据分布。

所有随机过程默认使用固定种子 `20260613`。相同样本数和随机种子会生成相同数据。

### User Profile 生成规则

- `risk_preference` 按固定概率抽取：
  - `conservative`：30%。
  - `moderate`：50%。
  - `aggressive`：20%。
- `total_asset`：在 200,000 至 2,000,000 元之间随机生成整数，包含两个端点。
- `current_position_ratio`：当前该股票持仓占总资产的比例，从以下范围均匀生成：

  ```text
  [0, min(0.18, concentration_threshold × 0.55))
  ```

  `concentration_threshold` 根据风险偏好取 20%、30% 或 40%。
- `current_stock_position_value`：

  ```text
  total_asset × current_position_ratio
  ```

- `cash_available`：先取计划交易金额的 1.0–1.5 倍，再限制为不超过总资产：

  ```text
  min(total_asset, estimated_amount × uniform(1.0, 1.5))
  ```

- `user_id`：从 `U0001` 至 `U0200` 随机选择。不同投资计划样本可能属于同一个模拟用户，但生成器不会维护该用户跨样本的一致资产状态。

### Trade Plan 生成规则

- `action`：固定为 `buy`。
- 生成器先以 36% 概率设置 `concentration_target=true`，用于控制仓位集中风险正样本比例。
- 根据用户风险偏好确定集中风险阈值：保守型 20%、稳健型 30%、激进型 40%。
- 目标交易后持仓比例：
  - 风险目标样本：从 `[threshold, min(0.85, threshold + 0.25))` 均匀生成。
  - 非风险目标样本：生成低于对应阈值的交易后持仓比例，并确保计划买入比例为正。
- 初始计划交易金额：

  ```text
  total_asset × (holding_ratio_after_trade - current_position_ratio)
  ```

- `estimated_price`：从每股 `[4.0, 80.0)` 元均匀生成，保留四位小数。
- `volume_lot`：根据初始交易金额和价格换算为手数并四舍五入，最少为 1 手：

  ```text
  max(1, round(estimated_amount / estimated_price / 100))
  ```

- `volume_share`：

  ```text
  volume_lot × 100
  ```

- `estimated_amount`：按取整后的手数重新计算：

  ```text
  volume_share × estimated_price
  ```

- `trade_time`：以 `2026-01-05 09:30:00` 为起点，在之后 0–179 分钟内随机选择一分钟。
- `stock_code`：从 1 至 999,998 随机生成整数，再补齐为六位字符串。
- `market`：模拟数据生成器仍会保留该字段，代码以 `6` 开头时设为 `SSE`，否则设为 `SZSE`。这是 Demo 简化规则，不是完整的证券代码识别逻辑。Agent 单条输入中该字段可省略，并且不参与当前模型计算。

### Stock Features 生成规则

生成器先以 35% 概率设置波动风险目标：

- 波动风险目标样本中，65% 使用较高 20 日波动率：

  ```text
  volatility_20d ~ uniform(0.35, 0.65)
  return_5d ~ uniform(-0.22, 0.22)
  ```

- 其余波动风险目标样本使用较低 20 日波动率，但强制产生较大的 5 日涨跌：

  ```text
  volatility_20d ~ uniform(0.12, 0.34)
  abs(return_5d) ~ uniform(0.15, 0.28)
  return_5d 的正负方向随机选择
  ```

- 非波动风险目标样本：

  ```text
  volatility_20d ~ uniform(0.10, 0.345)
  return_5d ~ uniform(-0.145, 0.145)
  ```

- `abs_return_5d` 不是独立随机生成，直接计算为：

  ```text
  abs(return_5d)
  ```

生成器再以 25% 概率设置流动性风险目标：

- 流动性风险目标样本的 `trade_volume_to_avg_volume` 从 `[0.10, 0.30)` 均匀生成。
- 非流动性风险目标样本从 `[0.005, 0.095)` 均匀生成。
- `avg_volume_20d` 不是独立随机数，而是根据计划买入股数反算：

  ```text
  avg_volume_20d = volume_share / trade_volume_to_avg_volume
  ```

- `turnover_rate` 从 `[0.002, 0.12)` 均匀生成，即约 0.2%–12%，保留六位小数。当前它与成交量、流通股本和流动性标签没有关联，只是模拟的辅助输入字段。

对应标签规则为：

```text
volatility_risk = 1
当 volatility_20d >= 0.35 或 abs_return_5d >= 0.15

liquidity_risk = 1
当 trade_volume_to_avg_volume >= 0.10
```

### 代码和生成产物

- 单条样本生成逻辑：[`src/quantum_risk_classifier/data_generation.py`](src/quantum_risk_classifier/data_generation.py) 中的 `_sample_record()`。
- 批量生成入口：同一文件中的 `generate_dataset()`。
- JSONL、CSV 和数据概况写出逻辑：同一文件中的 `write_dataset()`。
- CLI 生成命令：

  ```bash
  uv run quantum-risk generate-data --count 600 --seed 20260613
  ```

- 生成产物：
  - `data/raw/investment_samples.jsonl`：保留嵌套结构的模拟样本。
  - `data/raw/investment_samples.csv`：展开后的训练数据。
  - `data/raw/data_profile.json`：样本数量、缺失值、标签比例和部分数值范围。

`examples/input_valid.json` 是手工编写的接口演示值，不是生成器现场随机抽取的样本。正式对接 Agent 后，`user_profile` 应来自用户档案或账户服务，`trade_plan` 应来自用户自然语言解析，`stock_features` 应来自真实行情服务；模拟生成器仅用于测试、演示和模型实验。

## 单条预测接口

项目已经支持读取一个投资计划 JSON 并写出一个预测 JSON：

```bash
uv run quantum-risk predict \
  --input examples/input_valid.json \
  --output results/prediction.json
```

成功时退出码为 `0`，预测同时写入指定文件并输出到标准输出。完整结果示例位于 [`results/prediction.json`](results/prediction.json)。

输入不符合 Schema 时退出码为 `2`，标准错误输出为结构化错误，并包含具体字段，例如：

```bash
uv run quantum-risk validate-input --input examples/input_invalid.json
```

归一化器或模型文件缺失时同样返回退出码 `2` 和 `FileNotFoundError`。Agent 应将其视为服务不可用或部署不完整，记录错误并停止生成风险结论，不能伪造默认结果。

## 输出解读与用户报告

核心输出示例：

```json
{
  "risk_scores": {
    "concentration_risk": 0.878,
    "volatility_risk": 0.501,
    "liquidity_risk": 0.227
  },
  "risk_labels": {
    "concentration_risk": 1,
    "volatility_risk": 1,
    "liquidity_risk": 0
  },
  "overall_risk": {
    "overall_score": 0.583,
    "overall_level": "medium_high"
  },
  "confidence": 0.435
}
```

- `risk_scores`：模型判断计划具有对应风险特征的强度，范围为 `[0,1]`，不是真实风险发生概率。
- `risk_labels`：`1` 表示触发该类风险，`0` 表示未触发。
- `overall_risk.overall_level`：Agent 面向用户总结时使用的综合等级。
- `overall_risk.overall_score`：三类风险得分的加权结果。
- `confidence`：三个结果距离各自分类边界的平均程度；越低表示越接近边界，措辞应更保守。
- `model_outputs`：模型类型、量子特征和参数，只用于调试、审计和版本追踪，不建议展示给普通用户。

Agent 还应使用 `input_summary` 和原始输入解释风险依据，而不是只复述分数。本示例中：

- 1000 手等于 100,000 股，预计交易金额为 520,000 元。
- 当前持仓 80,000 元，交易后该股票持仓约 600,000 元，高于用户总资产 500,000 元，仓位集中风险明显。
- 20 日波动率为 36%，达到 Demo 的波动风险规则区间。
- 计划交易量占日均成交量 `100000 / 2000000 = 5%`，未达到 10% 的流动性风险规则。

Agent 可以整理为以下用户报告：

> **综合结论：中高风险。** 该计划主要存在仓位集中风险和个股波动风险。计划买入金额约 52 万元，加上现有约 8 万元持仓后，对 600006 的总持仓约为 60 万元，已经超过当前 50 万元总资产，建议先核对资金和买入数量。该股票近 20 日波动率约为 36%，短期价格波动较大。计划买入 10 万股，约占近 20 日日均成交量的 5%，本次未识别出明显流动性风险。模型综合得分约为 0.583，但波动风险判断接近分类边界，结论应谨慎解释。
>
> 本结果来自模拟数据训练的技术 Demo，只用于风险提示，不构成投资建议或交易指令。

Agent 生成报告时必须保留最后的 Demo 免责声明，不应将 `risk_scores` 表述为真实亏损概率，也不应根据模型结果自动下单。

## 结果

- `data/raw/data_profile.json`：样本量、标签分布和数值范围。
- `data/processed/feature_metadata.json`：特征顺序、划分样本和分布。
- `data/processed/features.csv`：固定顺序的 8 维特征与标签。
- `results/metrics_classical.csv`：完整测试集经典模型指标。
- `results/comparison.csv`：相同子集上的经典与量子模型对比。
- `results/comparison_full.csv`：全量共享量子核与经典模型对比。
- `results/quantum_search_results.csv`：专用量子核搜索记录。
- `results/metrics_quantum_optimized.csv`：全量专用量子核最终指标。
- `results/prediction.json`：单条计划预测结果。
- `results/run_manifest.json`：模型版本、随机种子、依赖版本和产物状态。

风险分数是模型判断计划具有某类风险特征的强度，不是真实风险发生概率。模拟数据上的量子模型结果不能证明实际投资场景中的量子优势。

当前全量实验中，共享 8 维量子核的宏平均 F1 约为 `0.045`；风险专用 2–4 维量子核调优后的宏平均 F1 约为 `0.978`。详细配置与限制见 `model_report.md`。
