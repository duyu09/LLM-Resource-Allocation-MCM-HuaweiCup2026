# 输出清单与论文用途

## Q1 质量评价目录：`q1_quality/`

| 输出 | 含义 / 论文用途 |
|---|---|
| `q1_sample_quality_scores.csv` | 每条 A1/A2/A3 记录的 Equal、Global CRITIC-TOPSIS、Adaptive TOPSIS、Huber、Conflict-aware Q，另含 MAD conflict、可靠度、高冲突标记。后续 Q0、领域 Q、冲突案例均从这里取。 |
| `q1_domain_quality_main.csv` | 最终 7 质量域 Q。arxiv/github 优先使用 A2/A3 全量，其他域使用 A1。 |
| `q1_domain_quality_comparison.csv` | 各评分方案×来源×领域的 Q、冲突率、Bootstrap CI/CV。E1 主表来源。 |
| `q1_quality_method_scorecard.csv` | E1/C0-C2 的压缩比较：A1→扩展集偏移、Bootstrap CV、扰动排名稳定性、是否显式处理冲突。 |
| `q1_expanded_vs_sample.csv` | arxiv/github 的 A1 抽样 vs A2/A3 扩展比较，按不同 Q 方法分别输出。 |
| `q1_quality_weights.csv` | 22 指标 Global CRITIC、领域 CRITIC、shrink 后权重。 |
| `q1_indicator_spearman.csv` | 22×22 Spearman 相关矩阵。 |
| `q1_indicator_extreme_conflict.csv` | 22×22 极端意见相反率矩阵。 |
| `q1_quality_record_overlap.csv` | A1 与 A2/A3 record-ID 重叠检查。用于说明扩展集并非独立抽样。 |
| `q1_quality_perturbation_stability.csv` | 1%/5% 指标扰动后的排名稳定性。 |
| `q1_quality_summary.json` | 最关键：`Q0_global` 与 A1 conflict 95% 阈值。Q3 直接读取 Q0。 |
| `q1_quality_transform.json` | Winsor/Min-Max/方向/中位数参数，保证结果可复现。 |
| `q1_mixture_domain_quality.csv` | A16/A17 映射接上最终质量域 Q 后的 17 域解释表。 |
| `q1_text_validation_domain_stats.csv` | 若存在 A18 预处理结果，给出域级文本长度/Token 统计，用作合理性辅助。 |
| `figures/q1_01...q1_08...png` | 指标相关热图、CRITIC 权重、冲突热图、Q-Conflict 散点、领域提琴、领域 Q、样本/扩展对照、方法热图。 |

## Q1 配比目录：`q1_mixture/`

| 输出 | 含义 |
|---|---|
| `q1_mixture_model_comparison.csv` | Linear / Scheffé / Scheffé-EN / ILR-GP 在 train/test/extrapolation 各尺度的 RMSE/MAE/R²/Spearman。E2 主表。 |
| `q1_mixture_coefficients.csv` | 13 Loss + macro 的一阶/二阶/ElasticNet 系数。 |
- `q1_elasticnet_diagnostics.csv`: selected ElasticNet alpha/l1-ratio, iterations, dual gap, and whether the 10,000-iteration limit was reached for each loss target.
| `q1_domain_interaction_matrix.csv` | 17×17 平均 Scheffé-EN 交互系数：负=互补、正=竞争/替代。 |
| `q1_domain_interactions_long.csv` | 交互矩阵长表，便于筛选最强正/负组合。 |
| `q1_reference_mixture.csv` | Q3 fixed-p 使用的经验参考配方：从 A4/A5 已观测配方中选取宏平均 Loss 最低者，不做全单纯形外推寻优。 |
| `q1_optimal_mixture.csv` | 与 `q1_reference_mixture.csv` 内容一致的兼容别名；不再表示离开支撑域的连续全局极值。 |
| `q1_mixture_predictions.csv` | 各配比样本的 macro Loss 观测/预测。 |
| `models/q1_mixture_models.joblib` | Linear/Scheffé/Scheffé-EN 模型对象。 |
| `models/q1_ilr_gp.joblib` | ILR-GP 模型对象。 |
| `figures/q1m_*.png` | 模型 RMSE、跨尺度 Spearman、交互热图、强交互、预测散点、经验参考配比。 |

## Q2 目录：`q2/`

| 输出 | 含义 |
|---|---|
| `artifacts/scaling_params.json` | Classic Scaling Law 参数 `E,A,B,alpha,beta`。 |
| `artifacts/q_branch.pt` | 主 **Scale-Conditioned Monotonic Quality Network**，学习 `ΔQ(Q,D)`；质量效应强度可随 `log D` 变化，对每个固定 D 保证 Q 单调、满足 `ΔQ(1,D)=0` 且输出有界。 |
| `artifacts/p_branch.pt` | 主 CLR + Domain Embedding + 2-layer 4-head MHSA 配比效率修正 `Δp(p)`。 |
| `artifacts/model_metadata.json` | `p_ref`、17 域列名、N/D/Q 数据范围。Q3 模型接口使用。 |
| `q2_scaling_groupcv.csv` | B1 按 run/model 分组交叉验证。 |
| `q2_scaling_external_validation.csv` | B1/B2/B3/B4/B5/B10 外部/跨族/外推验证。 |
| `q2_q_branch_comparison.csv` | SL0、Additive-Q、Parametric-Deff-Q、Parametric-Deff-Q×D、Spline-Q、**Scale-Conditioned Monotonic QD**、no-D-conditioning、unconstrained-QD、旧无界网络的 B6/B7/B8 比较。 |
| `q2_q_correction_observations.csv` | 由经典 Scaling backbone 反推出的逐样本 `Delta_Q_obs` 监督标签及其构造中间量。 |
| `q2_q_correction_validity.csv` | B6/B7/B8 中 `Delta_Q_obs` 的有效样本比例、无效样本数及 `Delta_Q<=0` 的物理一致率。 |
| `q2_q_monotonicity.csv` | 在 D 的低/中/高分位分别检查主 QD 分支的输出范围、`Q=1` 锚点和 `dDelta_Q/dQ` 违反率，并记录消融。 |
| `q2_q_scale_conditioning_surface.csv` | `Q × D -> Delta_Q` 的完整网格，用于解释“同样质量提升在不同数据规模下价值不同”。 |
| `q2_p_branch_comparison.csv` | CLR-linear、Parametric-Deff、CLR-MLP、Attention 与 raw-p/no-embedding 消融；同时合并 Q1 mixture baselines。**不含 no-p，也不含 p-no-anchor 消融。** |
| `q2_model_comparison_scorecard.csv` | E3/E4 component-wise 总表。**不要当作一个虚构联合测试集**。 |
| `q2_attention_matrix.csv` | 17×17 最后一层平均 Self-Attention；解释“关注关系”，非因果。 |
| `q2_attention_pooling.csv` | 各域的平均 attention pooling 权重。 |
| `q2_domain_transfer_marginal_matrix.csv` | 单纯形方向导数：从域 j 转移少量配比到域 i 对 Loss 的边际影响。 |
| `q2_scheffe_attention_concordance.csv/json` | Q1 Scheffé 交互绝对值与 Q2 Attention 的一致性分析。 |
| `q2_elasticity_quality_parameter_tradeoff.csv` | `εN, εD, εQ`、`∂L/∂Q`，以及 `ΔQ=0.1` 等价参数增量；`∂L/∂D` 由 autograd 计算总导数，包含 `ΔQ(Q,D)` 的 D 条件化效应。 |
| `q2_large_model_extrapolation_predictions.csv` | B9 大模型在参考条件 `Q=1,p=p_ref` 下的 Scaling/CE-ADNSL 外推。 |
| `figures/q2_*.png` | 外部 RMSE、Q/p 分支比较、`Delta_Q_obs` 与 QD 条件化拟合、`Delta_Q(Q,D)` 热图、Attention 热图、弹性、Q↔N 替代曲线。 |

### Q2 Bootstrap：`q2/bootstrap/`

| 输出 | 含义 |
|---|---|
| `q2_bootstrap_scaling_params.csv` | Cluster bootstrap 的 Scaling 参数样本。Q3 鲁棒目标读取。 |
| `ensemble/boot_xxxx/` | 使用 `--refit-neural` 时保存每次 bootstrap 的 Scaling+Q branch+p branch 完整模型。Q3 优先使用。 |

## Q3 目录：`q3/`

| 输出 | 含义 |
|---|---|
| `q3_core_budget_strategies.csv` | 三档预算下 fixed-p / joint-p / robust joint-p 的 `N*,D*,Q*,p*`、Loss、成本分解、CVaR、KKT 指标。Q3 主答案。 |
| `q3_value_of_adaptive_mixture.csv` | 动态调整 p 相对固定 p 的价值。 |
| `q3_optimizer_repeated_runs.csv` | E5 每个优化器在多个随机种子下的逐次结果：目标值、可行性、运行时间、KKT 残差等。 |
| `q3_optimizer_comparison.csv` | E5 汇总：可行率、最佳/平均/标准差目标值、平均运行时间、平均/最大 KKT spread。 |
| `q3_risk_objective_comparison.csv` | Mean / Mean+SD / Mean+CVaR；同时输出 Mean/P95/CVaR 与 scenario-wise mean/P95/worst-5% Regret。E6。 |
| `q3_cost_context_sensitivity.csv` | 三种 g(Q) × C7 context 档敏感性。 |
| `q3_budget_context_phase.csv` | 连续 budget×context 相图完整结果。 |
| `q3_budget_scan.csv` | 主 context 下的连续预算序列，供 PELT。 |
| `q3_pelt_changepoints.csv` | PELT 识别的结构转移候选预算。 |
| `q3_segmented_regression_confirmation.csv` | 单线性 vs breakpoint 分段线性的 BIC 对比；`BIC improvement >10` 作为强支持。 |
| `figures/q3_01_budget_cost_shares.png` | 三档预算的 100% 堆叠成本比例柱图；每根柱总高为 1，分段为 Training / Quality / Attention。可用 `--replot-q3-01` 从已有 `q3_core_budget_strategies.csv` 单独重画。 |
| `figures/q3_08_phase_cost_shares.png` | 连续 budget×context 成本结构图：三个并排热图分别给出 `r_train`、`r_quality`、`r_attn`，统一 0–1 色标并标注 `L_ctx^crit=30000`。取代旧 dominant-cost 离散相图。 |
| `figures/q3_02...q3_07, q3_09...q3_12.png` | 最优 N/D/Q、优化器、CVaR、context 临界点、最优质量相图、预算-Loss 与变化点等。 |

## Q4 能力目录：`q4_capability/`

| 输出 | 含义 |
|---|---|
| `q4_capability_scores.csv` | Raw mean、Z-mean、PCA PC1/PC2 的模型级能力。 |
| `q4_capability_method_comparison.csv` | 三种能力指标的排名一致性、Bootstrap 稳定性、年度趋势粗糙度。 |
| `q4_pca_loadings.csv` | 六 Benchmark 的 PC1 loading。 |
| `q4_pca_explained_variance.csv` | Scree / EVR。 |
| `q4_capability_bootstrap_stability.csv` | PCA/Z-mean bootstrap 排名稳定性。 |
| `q4_capability_leave_one_benchmark.csv` | 删去单一 Benchmark 后 PC1 排名稳定性。 |
| `q4_c8_task_scores.csv` | C8 模型×具体子任务的规范化代表分数。 |
| `q4_c8_group_scores.csv` | C8 六 benchmark group 的子任务宏平均。 |
| `q4_c8_specialization.csv` | `c8_specialization_sd`：模型偏科/均衡度。 |
| `models/capability_pca.joblib` | Bridge 将六 Benchmark 转综合能力时复用。 |
| `figures/q4_01...q4_08...png` | Benchmark 热图、Scree、loading、PCA map、时间、type violin、C8 heatmap、偏科。 |

## Q4 技术分解目录：`q4_decomposition/`

| 输出 | 含义 |
|---|---|
| `q4_model_panel.csv` | Leaderboard capability 与 Epoch N/D/C/open-weight 的建模连接面板；含 L1-L5 五层匹配方法、score、matched model。 |
| `q4_epoch_match_summary.csv` | Epoch 匹配方法数量，论文披露数据连接质量。 |
| `q4_scale_tech_model_comparison.csv` | Linear / full GAM / simplified GAM 的时间块验证。
| `q4_tech_trend_robustness.csv` | full GAM 与简化 GAM 的时间技术趋势 Pearson/Spearman、一致方向等共线性稳健性指标。 |
| `q4_tech_trend_curves.csv` | 两种 GAM 参数化下的技术时间趋势曲线。 |
| `q4_yearly_shapley_decomposition.csv` | 每年能力增量的规模贡献 `phi_scale` 与技术贡献 `phi_tech`。 |
| `q4_gam_partial_dependence.csv` | N/D/C/time 的 GAM 偏效应曲线数据。 |
| `models/scale_tech_gam*.joblib` | 全模型/简化共线性验证模型。 |

## Q4 Loss→Benchmark：`q4_bridge/`

| 输出 | 含义 |
|---|---|
| `q4_bridge_model_comparison.csv` | Linear / Spline / Isotonic / I-Spline 的 weighted error + 单调违反率。E7。 |
| `q4_bridge_curves.csv` | 六 Benchmark × 四模型的拟合曲线。 |
| `q4_bridge_hyperparameters.json` | I-Spline 的 comparability 权重 `omega` 与惩罚 `lambda`。 |
| `models/bridge_models.joblib` | 六 Benchmark 的完整 bridge 模型。 |
| `q4_q3_loss_to_benchmark_projection.csv` | 若提供 Q3 结果：把最优 Loss 转成六 Benchmark + PCA 能力。 |
| `figures/q4b_*.png` | 总体误差比较及六张 Loss→Benchmark 曲线。 |

## Q4 Pareto：`q4_pareto/`

| 输出 | 含义 |
|---|---|
| `q4_pareto_global.csv` | 全时期六维 Pareto 标记。 |
| `q4_pareto_yearly.csv` | 分年份 Pareto 标记。 |
| `figures/q4p_*.png` | PCA 空间中的 Pareto、年度前沿占比。 |

## Q4 前沿：`q4_frontier/`

| 输出 | 含义 |
|---|---|
| `q4_open_model_screening.csv` | 前沿预测的开源权重筛选证据：模型、许可证类别、open-weight 标记、模型类型和 Epoch 匹配方式。
| `q4_frontier_backtest.csv` | Rolling-origin 每个 cutoff/horizon 的实际 Q95，以及 F-A~F-F 六种方法的预测、误差、pinball loss 与前沿覆盖标记。 |
| `q4_frontier_model_comparison.csv` | F-A~F-F 六种前沿方法按 6/12 月分别比较 MAE/RMSE/Pinball/FrontierCoverage。E8。 |
| `q4_compute_frontier_quarterly.csv` | 开源权重模型训练算力的季度 90% 前沿。 |
| `q4_frontier_model_ranking.csv` | F-A~F-F 跨 6/12 月总体回测排序，优先按 Pinball loss。 |
| `q4_frontier_deep_monotonicity.csv` | F-E/F-F 在时间轴与算力轴上的单调性数值检查；`violations=0` 为满足约束。 |
| `q4_frontier_future_scenarios.csv` | 正常、0.75、0.5、0.25 算力增速下，F-A~F-F 六种方法的未来 12/24 月能力前沿预测。 |
| `q4_frontier_bootstrap.csv` | 主 0.5 放缓情景下 F-C Quantile GAM 的季度 block bootstrap（作为原有参考不确定性分析保留）。 |
| `q4_frontier_bootstrap_ci.csv` | 12/24月 2.5%/50%/97.5% 区间。 |
| `figures/q4f_01_backtest_mae.png` | F-A~F-F 在 6/12 月 rolling-origin 回测上的 MAE 比较。 |
| `figures/q4f_02_future_scenarios.png` | 六种模型分面展示四种算力增速情景下的 12/24 月能力前沿。 |
| `figures/q4f_03_bootstrap_fan.png` | F-C Quantile GAM 主放缓情景的 bootstrap 参考区间。 |
| `figures/q4f_04_compute_frontier.png` | 开源权重模型季度训练算力 90% 前沿。 |
| `figures/q4f_05_model_ranking_pinball.png` | F-A~F-F 跨 horizon 总体 Pinball loss 排名。 |
| `models/quantile_gam.joblib` | 0.95 Quantile GAM。 |
| `figures/q4f_*.png` | 历史回测、未来场景、置信扇形、算力前沿。 |
