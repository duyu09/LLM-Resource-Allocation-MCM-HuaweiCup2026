# Bài F của Cuộc thi Mô hình hóa Toán học Sau đại học Trung Quốc lần thứ 23 “Huawei Cup” năm 2026 _Mô hình phân bổ tài nguyên nhằm nâng cao năng lực mô hình ngôn ngữ lớn dưới ràng buộc năng lực tính toán_

> Kho lưu trữ này tổng hợp và công khai bài dự thi, mã tiền xử lý dữ liệu, mã mô hình hóa chính và một phần kết quả mô hình của nhóm chúng tôi cho **Bài F “Mô hình phân bổ tài nguyên nhằm nâng cao năng lực mô hình ngôn ngữ lớn dưới ràng buộc năng lực tính toán”.**

## Ngôn ngữ

[**繁體中文**](./README.md) | [**English**](./README.en.md) | [**Tiếng Việt**](./README.vi.md)

## I. Tổng quan dự án

Bài toán tập trung vào việc phân bổ đồng thời các tài nguyên sau cho mô hình ngôn ngữ lớn khi tổng ngân sách tính toán bị giới hạn:

- Quy mô tham số mô hình $N$
- Lượng dữ liệu huấn luyện $D$
- Chất lượng dữ liệu $Q$
- Tỷ lệ phối trộn dữ liệu giữa các miền $\mathbf{p}$
- Độ dài ngữ cảnh $L_{\mathrm{ctx}}$

Khung mô hình tổng thể lần lượt giải quyết bốn câu hỏi: **Đánh giá dữ liệu như thế nào → Đặc trưng hóa năng lực như thế nào → Phân bổ năng lực tính toán như thế nào → Năng lực thực tế tiến triển ra sao** Loss được sử dụng làm thước đo cầu nối chính giữa bốn phần.

## II. Quy trình mô hình hóa tổng thể

<img src="./images/Overall-Modeling-Flowchart.svg" style="width:60%" />

## III. Phương pháp cốt lõi

### 1. Đánh giá chất lượng dữ liệu và mô hình hóa tỷ lệ miền

Trước hết, chúng tôi thực hiện winsorization, chuẩn hóa và thống nhất chiều ý nghĩa cho các chỉ báo chất lượng dữ liệu đa chiều, sau đó kết hợp **Spearman-CRITIC, TOPSIS, độ xung đột dựa trên MAD và phép tổng hợp bền vững Huber** để xây dựng điểm chất lượng tổng hợp.

Đối với bài toán phối trộn 17 miền huấn luyện, chúng tôi tiếp tục so sánh các phương pháp gồm **mô hình hỗn hợp Scheffé, ElasticNet và ILR-GP** nhằm mô tả quan hệ phi tuyến giữa thành phần miền và Loss trên tập xác thực.

### 2. Scaling Law tổng quát

Xuất phát từ Scaling Law $N$ - $D$ cổ điển:

$$L_0(N,D)=E+A N^{-\alpha}+B D^{-\beta}.$$

Ảnh hưởng của chất lượng dữ liệu $Q$ và tỷ lệ miền $\mathbf{p}$ lên hiệu quả sử dụng dữ liệu được đưa vào thông qua khái niệm “lượng dữ liệu hiệu dụng”:

$$D_{\mathrm{eff}}=D\exp\left(\Delta_Q(Q,D)+\Delta_p(\mathbf{p})\right).$$

Do đó, Loss tổng quát có thể được biểu diễn dưới dạng:

$$L(N,D,Q,\mathbf{p})=E+A N^{-\alpha}+B D_{\mathrm{eff}}^{-\beta}.$$

Biểu thức này cho phép phân tích quy mô mô hình, quy mô dữ liệu, chất lượng dữ liệu và thành phần miền trên cùng một thang Loss.

### 3. Phân bổ tài nguyên đồng thời dưới ràng buộc tính toán

Với tổng ngân sách tính toán $C_{\max}$ cho trước, chúng tôi xét ba loại chi phí: huấn luyện cơ sở, cải thiện chất lượng dữ liệu và attention cho ngữ cảnh dài:

$$C_{\mathrm{tot}}=6ND+D[g(Q)-g(Q_0)]_++\eta NDL_{\mathrm{ctx}}.$$

Sau đó giải bài toán:

$$\min_{N,D,Q,\mathbf{p}} L(N,D,Q,\mathbf{p}),\qquad \text{s.t. } C_{\mathrm{tot}}\le C_{\max}.$$

Quá trình giải số sử dụng **Differential Evolution + tinh chỉnh ràng buộc bằng SQP**, đồng thời dùng dạng mục tiêu mean-CVaR để khảo sát phương án phân bổ bền vững khi các tham số bị nhiễu.

### 4. Tiến triển năng lực và dự báo biên năng lực

Cuối cùng, nhiều Benchmark được chuẩn hóa và **PCA** được sử dụng để xây dựng chỉ số năng lực tổng hợp. Sau đó, **GAM, phân rã Shapley và mô hình cầu nối Loss-Benchmark** được dùng để phân tích quá trình tiến triển năng lực của mô hình.

Đối với biên năng lực trong tương lai, phương pháp phân vị có điều kiện được sử dụng để đưa ra các kịch bản suy diễn sau 12 / 24 tháng và phân tích bất định trong bối cảnh tốc độ tăng trưởng năng lực tính toán chậm lại.

## IV. Một số kết quả

- Điểm chất lượng cơ sở trên toàn bộ dữ liệu là $Q_0=0.548126$.
- Trên tập kiểm định cùng quy mô 1M, ILR-GP đạt RMSE dự báo Loss trung bình vĩ mô là **0.158609** đối với mô hình tỷ lệ miền.
- Với ba mức ngân sách $10^{19}$, $10^{22}$ và $10^{24}$ FLOPs, Loss tối ưu trong kịch bản phân bổ đồng thời bền vững chính lần lượt xấp xỉ **3.153866, 2.153194 và 1.915272**.
- So với tỷ lệ kinh nghiệm cố định, khi cho phép tỷ lệ miền tham gia tối ưu hóa đồng thời, Loss dự báo giảm khoảng **31.75%, 17.40% và 9.49%** dưới ba mức ngân sách tương ứng.
- Thành phần chính thứ nhất của sáu Benchmark giải thích khoảng **73.55%** tổng phương sai.

## V. Cấu trúc kho lưu trữ

```text
.
├── code_f_v9/                  # Mã mô hình hóa và tối ưu hóa chính
├── code_preprocess/            # Mã tiền xử lý dữ liệu
├── images/
├── F26107320168.pdf            # Bài dự thi
├── README.md
├── README.en.md
├── README.vi.md
└── model_results_v2.7z         # Kết quả và đầu ra mô hình
```

## VI. Bài dự thi

Toàn văn bài dự thi: **[F26107320168.pdf](./F26107320168.pdf)**

## VII. Hướng dẫn sử dụng

- `code_preprocess/`: dùng để tổ chức, làm sạch và tiền xử lý dữ liệu thô.
- `code_f_v9/`: chứa mã mô hình hóa, tối ưu hóa và phân tích chính cho bốn bài toán con.
- `model_results_v2.7z`: lưu một phần đầu ra mô hình, kết quả trung gian và dữ liệu phục vụ vẽ biểu đồ.
- `images/`: lưu các hình ảnh dùng để trình bày trong kho lưu trữ.

Kho lưu trữ này là phiên bản được chỉnh lý sau khi cuộc thi kết thúc, chủ yếu phục vụ trao đổi phương pháp mô hình hóa toán học, tái lập kết quả và học thuật.

## VIII. Tác giả

**Copyright &copy; 2026 [Hà Phi Phàm (HE Feifan; 何非凡)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1835.htm), [Đỗ Vũ (DU Yu; 杜宇)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm), Diêm Viện Viện (YAN Yuanyuan; 閆媛媛), Giảng viên hướng dẫn: [GS. Trần Mai (Prof. CHEN Mei; 陳梅教授)](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/list/index.htm), Học Viện Điện Tử Và Công Nghệ Thông Tin, Đại Học Giao thông Lan Châu**
