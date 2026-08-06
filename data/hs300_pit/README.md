# 沪深300 Point-in-Time 数据目录

本目录用于存放时点真实（PIT）的沪深300研究数据。目录中的大体积下载产物均被
Git忽略，只保留本文档。

## 构建

Token只能通过环境变量提供，禁止写入代码或命令行参数：

```powershell
$env:TUSHARE_TOKEN = '<在本地设置，不要提交>'
.\.venv\Scripts\python.exe data\build_hs300_pit.py --resume
```

Linux/macOS：

```bash
export TUSHARE_TOKEN='<在本地设置，不要提交>'
.venv/bin/python data/build_hs300_pit.py --resume
```

脚本会先检查 `trade_cal`、`index_weight`、`daily`、`adj_factor`、
`daily_basic` 和 `suspend_d` 权限。任何必要接口不可用都会立即停止。

## 正式产物

- `membership_intervals.csv`：历史成分有效区间；
- `daily_membership.csv`：逐交易日官方股票池；
- `model_data.csv`：含预热历史、PIT掩码和统一交易日标签的模型输入；
- `membership_reconciliation.json`：公告回放与月度快照核对结果；
- `quality_report.json`：数据质量检查；
- `manifest.json`：版本、来源、日期和文件SHA-256。

只有 `manifest.json` 的 `status` 为 `validated` 时，训练程序才允许使用该数据。
旧的固定期末股票池数据不得与本目录数据拼接。
