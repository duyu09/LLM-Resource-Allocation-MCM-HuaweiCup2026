# F题完整建模代码包：预处理之后 → 四问建模 → 图表与结果

本工程直接读取先前 `llm_f_preprocess_csv_with_A17_B11` 预处理程序生成的 `processed/` 目录；**不再读取原始 xz / JSON / Parquet**。主流程严格按四问递进：

`Q1 质量与配比 → Q2 CE-ADNSL → Q3 算力约束优化 → Q4 Benchmark/技术演进/前沿预测`。

## 1. 环境

建议 Python 3.10–3.12。安装：

```bash
pip install -r requirements.txt
```

正式论文实验建议安装 `cvxpy` 与 `ruptures`：

- `cvxpy`：严格求解单调 I-Spline 约束回归、Quantile GAM 的 pinball + 平滑惩罚；
- `ruptures`：PELT 变化点检测。

代码为二者都留了保守 fallback，但正式结果建议使用 requirements 中完整依赖。

PyTorch 自动使用 CPU；Q2 可通过 `--device cuda` 使用 GPU。

## 2. 与预处理输出的兼容性检查

假设：

```text
F/
├─ processed/
│  ├─ A/
│  ├─ B/
│  └─ C/
└─ llm_f_modeling/
```

先运行：

```bash
cd llm_f_modeling
python3 check_inputs.py --processed ../processed
```

必须全部显示 `[OK]`。`A/q1_text_validation.csv` 是可选文件，仅在预处理时使用 `--include-a18` 才存在。

## 3. 质量指标方向配置——运行前必须核验一次

文件：

```text
config/quality_directions.json
```

# 4. 正式流水线命令

下面命令从 `processed/` 开始，到最终全部结果结束。要按顺序执行。

## Step Q1-A：综合质量 Q + 冲突消解 + E1/C0-C2 比较

```bash
python3 -m q1.run_quality \
  --processed ../processed \
  --out ../model_results/q1_quality \
  --bootstrap 1000
```

完成：

- Winsorize + 方向统一；
- Global Spearman-CRITIC；
- Domain-adaptive CRITIC shrinkage；
- TOPSIS；
- MAD conflict index；
- Huber M-estimator；
- Conflict-aware `Q=(1-C)T+C H`；
- Equal / Global-TOPSIS / Adaptive-TOPSIS / Huber / Conflict-aware 比较；
- A1 vs A2/A3 扩展稳定性；
- Bootstrap CI；
- 指标相关热图、冲突热图、提琴图等。

> 程序会检测 A1 与 A2/A3 的 record-ID 重叠。所有行都获得 Q；默认 `--critic-reference unique` 仅在“拟合权重/缩放参数”时让重复 ID 只计一次，避免相同文本双重加权。如论文决定严格逐文件计权，可改 `--critic-reference all`。

## Step Q1-B：17域配比模型 + E2

```bash
python3 -m q1.run_mixture \
  --processed ../processed \
  --out ../model_results/q1_mixture
```

完成四种模型：

1. 一阶 mixture linear；
2. 二阶 Scheffé；
3. Scheffé + ElasticNet（主解释模型）；
4. ILR + Gaussian Process（非线性对照）。

并输出 17×17 协同/替代矩阵、A6–A11 测试和 A12–A15 外推稳定性。Q3 的 fixed-p 基线不再采用二阶响应面在整个单纯形上的连续外推极值，而是从 A4/A5 已观测配方中选择宏平均 Loss 最低的经验参考配方，输出 `q1_reference_mixture.csv`；同时保留同内容的 `q1_optimal_mixture.csv` 兼容旧调用。该修改避免旧版在训练支撑域之外产生负交叉熵的非物理解。

---

## Step Q2：经典 Scaling + Q/p 多方案 + CE-ADNSL + 消融

```bash
python3 -m q2.run_scaling \
  --processed ../processed \
  --q1 ../model_results/q1_mixture \
  --out ../model_results/q2
```

若有 NVIDIA GPU：

```bash
python3 -m q2.run_scaling --processed ../processed --q1 ../model_results/q1_mixture --out ../model_results/q2 --device cuda
```

实现：

- SL0 Classic Scaling Law：`E + A N^-alpha + B D^-beta`；
- Q 分支：Classic / Additive-Q / Parametric-Deff-Q / **Parametric-Deff-Q×D** / Spline-Q / **Scale-Conditioned Monotonic Quality Network**；
- 主 Q 网络使用 B6/B7/B8 中真实存在的联合 `(N,D,Q,Loss)` 结构，学习 `Delta_Q(Q,D)` 而不是旧版 `Delta_Q(Q)`。令 `x=1-Q`、`z_D=(log D-mean)/std`，网络采用 `D` 条件化正斜率 + `D` 条件化单调基函数残差，因此对每个固定 `D` 都保证 `d Delta_Q / dQ >= 0`，并严格满足 `Delta_Q(1,D)=0`；
- 主 Q 网络先以 `SL2b_parametric_Deff_QxD` 作为 epoch-0 初始化，再用 `Delta_Q_obs` 残差监督和真实 Loss 二阶段微调；B7 RMSE 从 epoch 0 起参与 checkpoint 选择。`s_max=8`、soft-cap 阈值 `4`、梯度裁剪 `1.0`，B8 仅作压力/外推测试；
- Q 消融包括：去掉 `D` 条件化、保留 `Q,D` 但取消单调约束，以及旧无界 Q-only 网络诊断。新增 `q2_q_scale_conditioning_surface.csv` 与 `q2_02b_q_scale_conditioning_heatmap.png`，直接展示质量收益如何随数据规模变化；
- p 分支：CLR linear / parametric effective-data / CLR-MLP / CLR+MHSA；**不设置“去除 p”或 p-no-anchor 消融，因为 p 是题目明确要求纳入的变量。**
- 主模型 CE-ADNSL：**Scale-Conditioned Monotonic Q-D branch** + CLR + Domain Embedding + 2-layer 4-head Self-Attention；
- 因为 `Delta_Q` 依赖 `D`，`q2_elasticity_quality_parameter_tradeoff.csv` 中 `dL/dD` 使用 PyTorch autograd 计算总导数，自动包含 `Q×D` 交互对数据弹性的贡献；
- Scheffé interaction 与 Attention 的一致性解释；
- simplex 上的领域转移边际 `M_{i<-j}`；
- N/D/Q 弹性及 `ΔQ=0.1` 的参数等价量；
- B9 大模型参考与 B10 外推验证。

## Step Q2-B：Bootstrap 不确定性

快速的“仅 Scaling backbone 参数 bootstrap”：

```bash
python3 -m q2.bootstrap \
  --processed ../processed \
  --q2 ../model_results/q2 \
  --n-boot 1000
```

严格对应理论方案的“完整网络重拟合 bootstrap”（耗时显著增加）：

```bash
python3 -m q2.bootstrap \
  --processed ../processed \
  --q2 ../model_results/q2 \
  --n-boot 1000 \
  --refit-neural
```

Q3 会优先使用完整神经 bootstrap ensemble；若不存在，则使用 scaling-parameter bootstrap + 主神经分支。

---

## Step Q3-A：三档预算、资源策略、求解器与风险目标比较（E5/E6）

```bash
python3 -m q3.run_optimization \
  --processed ../processed \
  --q1-quality ../model_results/q1_quality \
  --q1-mixture ../model_results/q1_mixture \
  --q2 ../model_results/q2 \
  --out ../model_results/q3
```

默认：

- 预算 `1e19, 1e22, 1e24`；
- `L_ctx` 取 C7 实际值的中位档；
- 质量成本主场景为指数型；
- 比较 fixed-p / joint-p / robust joint-p；其中 fixed-p 读取 Q1 的**已观测经验参考配方**，不读取任何离开配方支撑域的响应面极值；
- 比较 multi-start SLSQP / DE / DE+SLSQP；
- E5 默认 5 个随机种子重复（`--optimizer-repeats` 可改），multi-start SLSQP 正式默认 100 个随机初值（`--slsqp-starts` 可改）；
- 比较 Mean / Mean+SD / Mean+CVaR，并计算 bootstrap 场景下 Regret；
- 输出可行率、目标值标准差、运行时间和 KKT 单位算力边际收益残差。

可指定某个 C7 context：

```bash
python3 -m q3.run_optimization ... --context 32768
```

如果 Q3-A 已经跑完，只想利用现有 `q3_core_budget_strategies.csv` 重画 `q3_01_budget_cost_shares.png`，无需重新优化：

```bash
python3 -m q3.run_optimization \
  --out ../model_results/q3 \
  --replot-q3-01
```

新版 `q3_01` 是 100% 堆叠比例柱图：每个预算只有一根高度为 1 的柱，内部按 Training / Quality / Attention 三类成本比例分段。

## Step Q3-B：三种质量成本函数 × C7 上下文敏感性

```bash
python3 -m q3.run_sensitivity \
  --processed ../processed \
  --q1-quality ../model_results/q1_quality \
  --q1-mixture ../model_results/q1_mixture \
  --q2 ../model_results/q2 \
  --out ../model_results/q3
```

并显式画出：

`C_attn / C_train = eta L_ctx / 6`，临界点 `L_ctx^crit = 30000`。

## Step Q3-C：预算连续扫描 + 资源配置相图

```bash
python3 -m q3.run_phase \
  --processed ../processed \
  --q1-quality ../model_results/q1_quality \
  --q2 ../model_results/q2 \
  --out ../model_results/q3 \
  --logC-min 19 --logC-max 24 --step 0.1
```

该步骤计算量较大。程序在整数 log-budget 处重新使用 DE+SLSQP，其余预算用前一预算解 warm-start SLSQP，失败才重新触发 DE。

若优化已经完成并已生成 `q3_budget_context_phase.csv`，但只需要重新生成相图和预算扫描图，可直接使用：

```bash
python3 -m q3.run_phase \
  --processed ../processed \
  --q1-quality ../model_results/q1_quality \
  --q2 ../model_results/q2 \
  --out ../model_results/q3 \
  --logC-min 19 --logC-max 24 --step 0.1 \
  --plot-only
```

`--plot-only` 不会重新执行任何 DE/SLSQP 优化，只读取现有 `q3_budget_context_phase.csv` 重建 `q3_budget_scan.csv` 与 `q3_08`–`q3_10` 三张图。

新版 `q3_08_phase_cost_shares.png` 不再使用“最大成本项”离散分类。它由三个共享 0–1 色标的连续热图组成，分别展示 `r_train`、`r_quality`、`r_attn` 在 `(log10 budget, L_ctx)` 网格上的比例，并叠加 `L_ctx^crit=30000`。旧的 `q3_08_phase_dominant_cost.png` 会在重画时自动删除。

## Step Q3-D：PELT + 分段回归确认结构性转移

```bash
python3 -m q3.run_changepoint \
  --q3 ../model_results/q3
```

---

## Step Q4-A：三种综合能力指标 + PCA + C8 强制逐任务分析

```bash
python3 -m q4.run_capability \
  --processed ../processed \
  --out ../model_results/q4_capability \
  --bootstrap 1000
```

比较：

- Raw six-benchmark mean；
- Z-score mean；
- PCA PC1（主指标）。

同时对 C8 做模型×子任务层级聚合、六任务热图及“偏科度”分析。

## Step Q4-B：规模扩张 vs 非规模技术进步

```bash
python3 -m q4.run_decomposition \
  --processed ../processed \
  --capability ../model_results/q4_capability \
  --out ../model_results/q4_decomposition
```

实现：

- Linear baseline；
- Additive spline GAM；
- 简化 GAM 共线性稳健性检查；
- full/simplified GAM 的技术时间趋势相关性、方向一致性量化；
- Scale group vs Tech group 两组 Shapley 分解。

Leaderboard 与 Epoch 使用五层分级匹配，并把每条匹配的 `epoch_match_method / epoch_match_score / epoch_matched_model` 写入 `q4_model_panel.csv`：

1. L1：严格 `model_key` exact；
2. L2：去掉 HuggingFace 组织前缀后的 repository basename exact；
3. L3：basename 规范化（大小写/标点统一）后 exact；
4. L4：去掉 `chat/instruct/it/sft/rlhf/...` 等末尾后训练变体后做 base-model exact，并要求参数规模比例 <=1.25；
5. L5：只对 basename 做高置信 fuzzy（默认 >=88），同时要求参数规模比例 <=1.25。

多候选时优先参数规模更接近、发布日期更接近且 N/D/C 字段更完整的 Epoch 记录。这样避免旧版中 `组织/repo` 与普通模型名格式差异导致 0 条有效匹配。

## Step Q4-C：Loss → 六 Benchmark（E7）

```bash
python3 -m q4.run_bridge \
  --processed ../processed \
  --capability ../model_results/q4_capability \
  --out ../model_results/q4_bridge \
  --q3-results ../model_results/q3/q3_core_budget_strategies.csv
```

比较：

- Linear；
- ordinary spline；
- Weighted Isotonic；
- Weighted Monotone I-Spline（主模型）。

正式环境有 `cvxpy` 时，I-Spline 使用非负系数凸优化；输出 weighted RMSE/MAE、LOOCV 与 monotonic violation rate。

如果给 `--q3-results`，还会把 Q3 的最优 Loss 映射成六 Benchmark 和 PCA 综合能力，完成 Q3→Q4 的闭环。

## Step Q4-D：六维能力 Pareto 前沿

```bash
python3 -m q4.run_pareto \
  --processed ../processed \
  --capability ../model_results/q4_capability \
  --out ../model_results/q4_pareto
```

## Step Q4-E：能力前沿比较、历史回测、算力放缓 12/24 月预测（E8）

```bash
python3 -m q4.run_frontier \
  --decomposition ../model_results/q4_decomposition \
  --processed ../processed \
  --out ../model_results/q4_frontier \
  --bootstrap 1000
```

比较六种方案（F-A ~ F-F）：

- F-A：Rolling empirical Q95；
- F-B：Linear Quantile Regression；
- F-C：0.95 Quantile GAM；
- F-D：Deep Quantile MLP（时间/算力/模型类型 -> 0.95 分位能力）；
- F-E：Monotonic Deep Quantile Network，对时间和训练算力施加单调非减约束；
- F-F：Monotonic Deep Quantile Network + 轻量正向 `time × compute` 交互项，在保持单调性的同时允许算力收益随时代变化。

六种方案统一使用 Rolling-Origin Backtesting，在 6/12 个月历史回测上输出 MAE、RMSE、Pinball loss 与前沿覆盖率，并输出综合排序 `q4_frontier_model_ranking.csv`。未来 12/24 个月的正常、0.75、0.5（主放缓场景）、0.25 四种训练算力前沿增速情景会同时给出 F-A ~ F-F 六种方法的预测，而不是预先指定深度学习一定胜出。

原有 Quantile GAM 的季度 block bootstrap 不删除，继续作为 F-C 的参考不确定性区间；F-D/F-E/F-F 不做 1000 次神经重拟合 bootstrap，避免把模型比较变成不可承受的重复训练。F-E/F-F 另输出单调性数值诊断。
同时输出 `q4_open_model_screening.csv`，逐模型记录 open-weight 入选、许可证分类与模型类型，作为论文筛选口径证据。

---

# 5. `--fast` 的用途

Q1/Q2/Q3/Q4 若脚本提供 `--fast`，它只用于开发/字段联调。正式论文结果不要使用 `--fast`。

例如：

```bash
python3 -m q2.run_scaling --processed ../processed --q1 ../model_results/q1_mixture --out ../tmp_q2 --fast
```

---

# 6. 重要方法边界

1. **A 与 B 是独立实验**：CE-ADNSL 分支独立学习后进行结构融合，不建立不存在的联合测试集。
2. **B2、B6–B8、B10 的性质必须在论文中明确**：半合成/估算不是直接实验观测。
3. **Attention 权重不是因果系数**：仅作为“网络关注关系”，通过 Q1 Scheffé interaction 交叉验证其解释一致性。
4. **Q4 GAM-Shapley 是统计分解，不宣称严格因果识别**。
5. **PCA 与 Pareto 不竞争**：PC1 给一维综合能力；Pareto 保留六维不可支配结构。
6. **C7 的 `L_ctx` 是外生敏感性变量**，不作为 Q3 内点决策变量。
7. **方向配置必须核验**：见 `config/quality_directions_NOTES.md`。

# 7. 测试

工程附带与预处理输出同字段的合成数据生成器：

```bash
python3 -m tests.make_synthetic_processed
```

它不是竞赛数据，也不用于论文，只用于检查接口与代码修改后是否仍能运行。

开发时已完成：

- 全项目 `py_compile`；
- Q1 quality、Q1 mixture（含 ILR-GP）快速实跑；
- Q2 主流程（含 PyTorch Attention/消融）快速实跑；
- Q2 bootstrap（参数版与神经重拟合版）小规模实跑；
- Q3 单预算混合优化数值接口实跑；
- Q3 PELT/分段回归模块实跑；
- Q4 capability/C8、decomposition、bridge、Pareto、frontier 快速实跑。

当前会话未提供 `processed/` 的真实全量结果文件，因此无法在这里替你们跑出真实竞赛数值；代码的真实数据字段契约来自你提供的预处理压缩包并已按其列名编写。

详细输出解释见：`OUTPUT_MANIFEST.md`。
