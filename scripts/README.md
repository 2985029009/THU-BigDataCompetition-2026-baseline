# 脚本导航

## 数据脚本

- `data/fetch_hs300_data.py`：下载沪深 300 成分股历史数据，输出到项目根目录的 `data/`。

## 评测脚本

- `evaluation/evaluate_initial16_reproduction.py`：评测历史 197→16 复现实验。

## 自动化实验与诊断

- `../codex_generated/scripts/experiments/`：滚动验证、消融和基线实验。
- `../codex_generated/scripts/diagnostics/`：市场状态及特征诊断。

## 根目录兼容入口

以下脚本仍被 README、Shell 命令或自动化脚本直接引用，因此保留在项目根目录：

- `evaluate_model.py`
- `get_stock_data.py`

训练、预测等核心实现位于 `../code/src/`，赛事评分与测试辅助脚本位于 `../test/`。
