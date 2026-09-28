# Problem F of the 2026 “Huawei Cup” 23rd China Postgraduate Mathematical Contest in Modeling _Resource Allocation Modeling for Improving Large Language Model Capabilities under Compute Constraints_

> This repository organizes and open-sources our competition paper, data preprocessing code, main modeling code, and selected modeling results for **Problem F, “Resource Allocation Modeling for Improving Large Language Model Capabilities under Compute Constraints.”**

## Language

[**繁體中文**](./README.md) | [**English**](./README.en.md) | [**Tiếng Việt**](./README.vi.md)

## I. Project Overview

This problem focuses on how to jointly allocate the following resources for large language models under a limited total compute budget:

- Model parameter scale $N$
- Training data volume $D$
- Data quality $Q$
- Domain-mixture proportions $\mathbf{p}$
- Context length $L_{\mathrm{ctx}}$

The overall modeling framework addresses four questions in sequence: **How should data be evaluated → How should capability be characterized → How should compute be allocated → How does real-world capability evolve** Loss is used as the primary bridging scale across the four parts.

## II. Overall Modeling Workflow

<img src="./images/Overall-Modeling-Flowchart.svg" style="width:60%" />

## III. Core Methods

### 1. Data Quality Evaluation and Domain Mixture Modeling

We first perform winsorization, normalization, and direction unification on multidimensional data-quality indicators, and then combine **Spearman-CRITIC, TOPSIS, MAD-based conflict measurement, and Huber robust aggregation** to construct a comprehensive quality score.

For the mixture of 17 training domains, we further compare methods including the **Scheffé mixture model, ElasticNet, and ILR-GP** to characterize the nonlinear relationship between domain composition and validation Loss.

### 2. Generalized Scaling Law

Starting from the classical $N$ - $D$ Scaling Law:

$$L_0(N,D)=E+A N^{-\alpha}+B D^{-\beta}.$$

We incorporate the effects of data quality $Q$ and domain mixture $\mathbf{p}$ on data-use efficiency through the concept of “effective data volume”:

$$D_{\mathrm{eff}}=D\exp\left(\Delta_Q(Q,D)+\Delta_p(\mathbf{p})\right).$$

The generalized Loss can therefore be written as:

$$L(N,D,Q,\mathbf{p})=E+A N^{-\alpha}+B D_{\mathrm{eff}}^{-\beta}.$$

This formulation allows model scale, data scale, data quality, and domain composition to be analyzed on a unified Loss scale.

### 3. Joint Resource Allocation under Compute Constraints

Given a total compute budget $C_{\max}$, we account for three types of cost: base training, data-quality improvement, and long-context attention:

$$C_{\mathrm{tot}}=6ND+D[g(Q)-g(Q_0)]_++\eta NDL_{\mathrm{ctx}}.$$

We then solve:

$$\min_{N,D,Q,\mathbf{p}} L(N,D,Q,\mathbf{p}),\qquad \text{s.t. } C_{\mathrm{tot}}\le C_{\max}.$$

The numerical solution uses **Differential Evolution + SQP constrained refinement**, while a mean-CVaR formulation is used to examine robust allocation under parameter perturbations.

### 4. Capability Evolution and Frontier Forecasting

Finally, multiple Benchmarks are standardized and **PCA** is used to construct a composite capability index. **GAM, Shapley decomposition, and Loss-Benchmark bridge models** are then employed to analyze the evolution of model capabilities.

For future capability frontiers, conditional quantile methods are used to provide 12-/24-month conditional projections and uncertainty analysis under a scenario of slowing compute growth.

## IV. Selected Results

- The baseline quality score over the full dataset is $Q_0=0.548126$.
- On the 1M same-scale test set, ILR-GP achieves a macro-average Loss prediction RMSE of **0.158609** for domain-mixture modeling.
- Under compute budgets of $10^{19}$, $10^{22}$, and $10^{24}$ FLOPs, the optimal Loss values for the main robust joint-allocation scenario are approximately **3.153866, 2.153194, and 1.915272**, respectively.
- Compared with the fixed empirical mixture, allowing domain proportions to participate in joint optimization reduces predicted Loss by approximately **31.75%, 17.40%, and 9.49%** under the three compute budgets.
- The first principal component of six Benchmarks explains approximately **73.55%** of the total variance.

## V. Repository Structure

```text
.
├── code_f_v9/                  # Main modeling and optimization code
├── code_preprocess/            # Data preprocessing code
├── images/
├── F26107320168.pdf            # Competition paper
├── README.md
├── README.en.md
├── README.vi.md
└── model_results_v2.7z         # Modeling results and outputs
```

## VI. Paper

The full competition paper is available here: **[F26107320168.pdf](./F26107320168.pdf)**

## VII. Usage Notes

- `code_preprocess/`: used for raw-data organization, cleaning, and preprocessing.
- `code_f_v9/`: contains the main modeling, optimization, and analysis code for the four subproblems.
- `model_results_v2.7z`: contains selected model outputs, intermediate results, and plotting data.
- `images/`: stores images used for presentation in the repository.

This repository is a post-competition release intended primarily for mathematical modeling exchange, result reproduction, and academic learning.

## VIII. Authors

**Copyright &copy; 2026 [HE Feifan (何非凡; HÀ Phi Phàm)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1835.htm), [DU Yu (杜宇; ĐỖ Vũ)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm), and YAN Yuanyuan (閆媛媛; DIÊM Viện Viện). Supervisor: [Prof. CHEN Mei (陳梅教授; TRẦN Mai)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/list/index.htm), School of Electronic and Information Engineering, Lanzhou Jiaotong University**
