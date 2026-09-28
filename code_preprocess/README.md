# 第零步：数据预处理

## 输出文件

运行后只产生后续实质建模会直接使用的 CSV：

### A / 问题一
- `A/q1_quality.csv`：A1+A2+A3 全量质量信号，8 个列表指标已按指定规则压成标量；用于 CRITIC-TOPSIS、冲突分析与域级 Q。
- `A/q1_mixture_loss.csv`：A4-A15 的 17 维配比 + 13 维 Loss；用于 Scheffe/ElasticNet 配比模型。
- `A/q1_domain_mapping.csv`：A16+A17；包含 17 个配方域到质量域的映射，以及每个配方域的抽样规模、平均文本长度等领域统计，用于质量域连接和配比结果解释。
- `A/q1_text_validation.csv`：仅在 `--include-a18` 时生成；用于第一问原始文本/领域合理性辅助验证。

### B / 问题二
- `B/q2_scaling.csv`：B1-B8、B10 的统一 N-D-Q-Loss 面板，并接入 B11 的模型族元数据（族覆盖模型数、权重文件规模等）；用于经典 Scaling Law、SA-NSL 及跨族验证。
- `B/q2_large_models.csv`：B9 百亿参数以上真实模型元数据；用于大模型外推范围与压力测试。

### C / 问题三、四
- `C/q3_context.csv`：C7 架构与上下文长度；用于问题三上下文敏感性分析。
- `C/q4_leaderboard.csv`：C2 排行榜增强表；用于问题四综合能力、类型和开源筛选。
- `C/q4_timeseries.csv`：C3；用于技术演进时间趋势和前沿预测。
- `C/q4_epoch_models.csv`：C4 的规模、算力、数据量、开源等核心字段；用于规模/技术贡献分解。
- `C/q4_loss_benchmark_bridge.csv`：C6；用于 Loss→Benchmark 映射。
- `C/q4_task_metrics.csv`：C8 最新可解析 JSON 展平后的逐任务长表；用于题目要求的逐任务聚合分析。

坏 JSON 数量将打印到控制台。

## 运行

```powershell
python run_all.py --root ../real_attachments --out ../processed --include-a18
```

每次运行时会重建 `processed/A`、`processed/B`、`processed/C`，因此旧版本的 Parquet 和多余 CSV 不会残留。
