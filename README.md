# 2026「華為杯」第二十三屆中國研究生數學建模競賽 F 題 _算力約束下提升大語言模型能力的資源配置建模_

> 本倉庫整理並開源本隊針對 **F 題「算力約束下提升大語言模型能力的資源配置建模」** 的參賽論文、資料預處理程式、主要建模程式與部分建模結果。

## 語言

[**繁體中文**](./README.md) | [**English**](./README.en.md) | [**Tiếng Việt**](./README.vi.md)

## I. 專案概述

本題關注在總算力受限的情況下，如何聯合配置大語言模型的：

- 模型參數規模 $N$
- 訓練資料量 $D$
- 資料品質 $Q$
- 不同領域資料配比 $\mathbf{p}$
- 上下文長度 $L_{\mathrm{ctx}}$

整體建模思路依序處理「**資料如何評價 → 能力如何刻畫 → 算力如何分配 → 現實能力如何演進**」四個問題，並以 Loss 為主要橋接尺度。

## II. 總體建模流程

<img src="./images/Overall-Modeling-Flowchart.svg" style="width:60%" />

## III. 核心方法

### 1. 資料品質評價與領域配比

首先對多維資料品質指標進行縮尾、標準化與方向統一，結合 **Spearman-CRITIC、TOPSIS、MAD 衝突度與 Huber 穩健聚合** 建立綜合品質評分。

對 17 個訓練領域的配比問題，進一步比較 **Scheffé 混料模型、ElasticNet 與 ILR-GP** 等方法，以描述領域組成與驗證 Loss 之間的非線性關係。

### 2. 廣義 Scaling Law

以經典 $N$ - $D$ Scaling Law 為基礎：

$$L_0(N,D)=E+A N^{-\alpha}+B D^{-\beta}.$$

並將資料品質 $Q$ 與領域配比 $\mathbf{p}$ 對資料利用效率的影響納入「有效資料量」：

$$D_{\mathrm{eff}}=D\exp\left(\Delta_Q(Q,D)+\Delta_p(\mathbf{p})\right).$$

因此可將廣義 Loss 表示為：

$$L(N,D,Q,\mathbf{p})=E+A N^{-\alpha}+B D_{\mathrm{eff}}^{-\beta}.$$

這使模型規模、資料規模、資料品質與領域組成能在同一 Loss 尺度下進行分析。

### 3. 算力約束下的聯合資源配置

在給定總算力預算 $C_{\max}$ 下，考慮基礎訓練、資料品質提升與長上下文注意力三類成本：

$$C_{\mathrm{tot}}=6ND+D[g(Q)-g(Q_0)]_++\eta NDL_{\mathrm{ctx}}.$$

並求解：

$$\min_{N,D,Q,\mathbf{p}} L(N,D,Q,\mathbf{p}),\qquad \text{s.t. } C_{\mathrm{tot}}\le C_{\max}.$$

數值求解採用 **差分進化（Differential Evolution）+ SQP 約束精修**，並以均值-CVaR 形式考察參數擾動下的魯棒配置。

### 4. 能力演進與前沿預測

最後將多項 Benchmark 標準化，利用 **PCA** 建立綜合能力指標，並透過 **GAM、Shapley 分解與 Loss-Benchmark 橋接模型** 分析模型能力演進。

對未來能力前沿則採用條件分位數方法，在算力增長放緩情景下給出 12 / 24 個月的條件推演與不確定性分析。

## IV. 部分結果

- 全量資料的基準品質評分為 $Q_0=0.548126$。
- 在 1M 同尺度檢驗中，ILR-GP 對領域配比的宏平均 Loss 預測 RMSE 為 **0.158609**。
- 在 $10^{19}$、 $10^{22}$、 $10^{24}$ FLOPs 三檔預算下，主情景魯棒聯合配置的最優 Loss 約為 **3.153866、2.153194、1.915272**。
- 相較固定經驗配比，允許領域配比參與聯合優化時，三檔預算下預測 Loss 分別降低約 **31.75%、17.40%、9.49%**。
- 六項 Benchmark 的第一主成分可解釋約 **73.55%** 的總變異。

## V. 倉庫結構

```text
.
├── code_f_v9/                  # 主要建模與求解程式
├── code_preprocess/            # 資料預處理程式
├── images/
├── F26107320168.pdf            # 參賽論文
├── README.md
├── README.en.md
├── README.vi.md
└── model_results_v2.7z         # 建模結果與輸出
```

## VI. 論文

完整參賽論文可見： **[F26107320168.pdf](./F26107320168.pdf)**

## VII. 使用說明

- `code_preprocess/`：用於原始資料整理、清洗與預處理。
- `code_f_v9/`：包含本題四個問題的主要建模、求解與分析程式。
- `model_results_v2.7z`：保存部分模型輸出、中間結果與圖表資料。
- `images/`：保存展示所使用的圖片。

本倉庫為競賽結束後的整理版本，主要用於數學建模方法交流、結果復現與學術學習。

## VIII. 作者

**Copyright &copy; 2026 蘭州交通大學電子與信息工程學院 [何非凡 (HE Feifan; HÀ Phi Phàm)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1835.htm)、[杜宇 (DU Yu; ĐỖ Vũ)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm)、閆媛媛 (YAN Yuanyuan; DIÊM Viện Viện)。指導教師：[陳梅教授 (Prof. CHEN Mei; TRẦN Mai)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/list/index.htm)**

