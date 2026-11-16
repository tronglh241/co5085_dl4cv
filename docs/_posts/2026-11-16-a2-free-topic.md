---
layout: post
title: "A2 Chủ đề tự chọn và phân tích bài báo"
authors: [trong, thien]
date: 2026-11-16 10:00
toc: true
math: true
comments: false
---

## Giới thiệu

**Chủ đề:** Bài A2 xây dựng hệ thống nhận dạng khuôn mặt cho cả hai tác vụ xác minh và định danh, sử dụng hàm loss **ArcFace** và đánh giá trên các danh tính không xuất hiện trong quá trình huấn luyện.

Bài A2 tái hiện ở quy mô nhỏ ý tưởng của bài báo ArcFace trên tập CelebA và trả lời hai câu hỏi sau:

- **Angular margin của ArcFace cải thiện kết quả bao nhiêu so với softmax?** Nhóm giữ nguyên backbone và chỉ thay đổi hàm loss.
- **Backbone nhỏ phải đánh đổi những gì so với backbone lớn?** Nhóm giữ nguyên hàm loss ArcFace và so sánh ResNet-50 với MobileNetV3.

Thông tin chính của bài:

- **Bài báo:** *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. [Link](https://arxiv.org/abs/1801.07698v1).
- **Tập dữ liệu:** Large-scale CelebFaces Attributes Dataset [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- **Mô hình:** ResNet-50 và MobileNetV3 Large.
- **Code:** thư mục [`scripts/a2`](https://github.com/tronglh241/co5085_dl4cv/tree/main/scripts/a2), [`models/a2`](https://github.com/tronglh241/co5085_dl4cv/tree/main/models/a2), [`configs/a2`](https://github.com/tronglh241/co5085_dl4cv/tree/main/configs/a2) và [`datasets`](https://github.com/tronglh241/co5085_dl4cv/tree/main/datasets) trên GitHub.
- **Báo cáo PDF:** [assignment_2.pdf](https://github.com/tronglh241/co5085_dl4cv/blob/main/reports/assignment_2.pdf)

Kết quả chính:

- ArcFace tốt hơn softmax ở cả hai backbone.
- ResNet-50 và MobileNetV3 gần như ngang nhau về hiệu năng.

## Dữ liệu

### Tập dữ liệu CelebA

CelebA gồm ảnh những người nổi tiếng chụp trong điều kiện không kiểm soát và chỉ được phát hành cho nghiên cứu phi thương mại. Nhóm sử dụng phiên bản `img_align_celeba` với kích thước $218 \times 178$, kèm năm land mark có sẵn, và giữ nguyên cách chia tập mặc định của CelebA.

| Tập | Số ảnh | Số danh tính |
|---|---:|---:|
| Train | 162.770 | 8.192 |
| Validation | 19.867 | 985 |
| Test | 19.962 | 1.000 |
| Tổng | 202.599 | 10.177 |

*Bảng 1. Ba tập dữ liệu của CelebA.*

### Phân tích khám phá dữ liệu

![Phân bố số ảnh mỗi danh tính ở tập train](/assets/a2/eda/identity_hist.png)

*Hình 1. Số ảnh mỗi danh tính ở tập train.*

| Số ảnh mỗi danh tính | Train | Validation | Test |
|---|---:|---:|---:|
| 1 | 36 (0,4%) | 5 (0,5%) | 3 (0,3%) |
| 2–4 | 648 (7,9%) | 73 (7,4%) | 69 (6,9%) |
| 5–9 | 774 (9,4%) | 96 (9,7%) | 104 (10,4%) |
| 10–19 | 1.631 (19,9%) | 182 (18,5%) | 208 (20,8%) |
| 20–29 | 3.225 (39,4%) | 408 (41,4%) | 355 (35,5%) |
| 30+ | 1.878 (22,9%) | 221 (22,4%) | 261 (26,1%) |
| **Tổng** | **8.192** | **985** | **1.000** |

*Bảng 2. Phân bố số ảnh trên mỗi danh tính ở ba tập; phần trăm tính trên số danh tính của từng tập.*

- **Ít ảnh mỗi lớp:** trung vị 21 ảnh, trung bình 19,9 ảnh, phần lớn danh tích có khoảng 20 hoặc 30 ảnh.
- **Một số danh tính quá ít ảnh:** 36 danh tính (0,4%) chỉ có một ảnh và 684 danh tính (8,3%) có dưới 5 ảnh.
- **Các tập có phân bố giống nhau:** validation và test cũng có trung vị 21 ảnh mỗi danh tính, nên điểm đánh giá không bị lệch do phân bố.

![Ảnh cùng danh tính sau khi căn chỉnh](/assets/a2/eda/samples.png)

*Hình 2. Một vài mẫu từ tập CelebA: mỗi hàng là một danh tính. Ảnh cùng người vẫn khác nhau nhiều về tóc, trang điểm, ánh sáng và tư thế.*

![Ảnh gốc và ảnh sau căn chỉnh](/assets/a2/eda/alignment.png)

*Hình 3. Ảnh gốc $218 \times 178$ (trên) và ảnh sau căn chỉnh $112 \times 112$ (dưới). Sau căn chỉnh, mắt, mũi và khóe miệng ở cùng vị trí.*

### Bài toán và cách đánh giá

Mô hình được **huấn luyện như một bài toán phân loại** trên $8.192$ danh tính của tập train, nhưng được **đánh giá trên các danh tính chưa từng thấy**. Tập validation có $985$ danh tính và tập test có $1.000$ danh tính, và cả hai không có danh tính hay ảnh nào chung với tập train. Vì vậy lớp phân loại cuối bị loại bỏ khi đánh giá. Mỗi ảnh được biến thành một vector embedding $512$ chiều, và hai ảnh được so sánh bằng cosine similarity:

$$s(x,y) = \frac{f(x)^\top f(y)}{\lVert f(x)\rVert \, \lVert f(y)\rVert}.$$

Trong công thức này, $x$ và $y$ là hai ảnh khuôn mặt cần so sánh. Hàm $f(\cdot)$ là backbone dùng để trích xuất vector embedding $512$ chiều từ một ảnh. Tích vô hướng $f(x)^\top f(y)$ đo mức độ cùng hướng của hai embedding. Các chuẩn $\lVert f(x)\rVert$ và $\lVert f(y)\rVert$ là độ dài $L_2$ của từng embedding, và việc chia cho chúng loại bỏ ảnh hưởng của độ dài vector. Kết quả $s(x,y) \in [-1, 1]$ chính là cosine của góc giữa hai embedding, và giá trị càng lớn thì hai ảnh càng có khả năng thuộc cùng một người.

- **Verification:** Bài toán hỏi hai ảnh có thuộc cùng một người hay không. Hai ảnh được xem là cùng người khi $s(x,y) \ge \tau$, với ngưỡng $\tau$ được chọn trên tập validation. Tập test có $10.000$ cặp ảnh, gồm $4.885$ cặp cùng một người và $5.115$ cặp là hai người khác nhau. Các metric gồm accuracy tại ngưỡng đã chọn, TAR@FAR=$10^{-3}$ và TAR@FAR=$10^{-2}$.
- **Identification:** Bài toán hỏi ảnh truy vấn thuộc danh tính nào trong gallery. Gallery có $997$ ảnh, mỗi danh tính một ảnh, và có $18.962$ ảnh truy vấn. Mỗi ảnh truy vấn được so với toàn bộ gallery bằng $s(x,y)$, rồi các danh tính được xếp hạng theo độ tương đồng giảm dần. Các metric gồm Rank-1 và Rank-5.

Các metric được hiểu như sau:

- **Accuracy:** tỉ lệ cặp verification bị phân loại đúng tại ngưỡng $\tau$ đã chọn. Ngưỡng được chọn trên validation để accuracy cao nhất, rồi giữ nguyên khi đánh giá test.
- **TAR@FAR:** True Accept Rate (TAR) tại một mức False Accept Rate (FAR) cho trước. FAR là tỉ lệ cặp hai người khác nhau bị nhận nhầm là cùng một người; TAR là tỉ lệ cặp cùng người được nhận đúng. Ví dụ TAR@FAR=$10^{-2}$ là TAR tại mức 1% cặp âm bị chấp nhận sai.
- **Rank-1:** tỉ lệ probe có ảnh gallery đúng được xếp hạng đầu tiên.
- **Rank-5:** tỉ lệ probe có ảnh gallery đúng nằm trong năm ứng viên đầu.

## Chuẩn bị dữ liệu huấn luyện

### Căn chỉnh và tiền xử lý

Nhóm dùng ảnh `img_align_celeba` ($218 \times 178$) và năm điểm mốc có sẵn của CelebA. Ảnh được căn chỉnh về $112 \times 112$ bằng phép biến đổi đồng dạng (tỉ lệ, xoay, tịnh tiến), khớp năm điểm mốc với template ArcFace bằng bình phương tối thiểu:

$$
\min_{s,\,R,\,t}\sum_{k=1}^{5}\left\|sRp_k+t-q_k\right\|^2.
$$

Ảnh sau căn chỉnh được lưu một lần và dùng chung cho cả bốn run. Mọi ảnh sau đó được chuẩn hóa theo mean/std $0{,}5$ cho cả ba kênh. Tăng cường dữ liệu chỉ áp dụng trên tập train.

### Tăng cường dữ liệu

Chỉ tập train được tăng cường; validation, test và các cặp đánh giá chỉ được chuẩn hóa.

| Bước | Tham số |
|---|---|
| Lật ngang | Xác suất 0,5 |
| ColorJitter nhẹ | Độ sáng, tương phản, độ bão hòa, mỗi đại lượng tối đa 0,2 |
| Chuẩn hóa | Mean và std 0,5 |

*Bảng 3. Pipeline tăng cường dữ liệu của tập train.*

- Lật ngang không đổi danh tính và giữ nguyên vị trí các điểm mốc sau căn chỉnh; ColorJitter mô phỏng sự khác nhau về ánh sáng giữa các ảnh.
- Nhóm không dùng xoay hoặc cắt ảnh để không làm lệch mẫu căn chỉnh năm điểm.

### Tập đánh giá và DataLoader

- Các cặp verification và việc chia gallery, probe cho identification được sinh một lần với seed cố định, lưu thành tệp và dùng chung cho các lần chạy thí nghiệm.
- `CelebA` đọc tệp chia tập và ảnh đã căn chỉnh; `VerificationPairs` đọc danh sách cặp. Tập train được xáo trộn mỗi epoch, batch size 128; validation và test không xáo trộn.

## Phân tích bài báo ArcFace

### Phương pháp

Softmax chuẩn không trực tiếp ép các ảnh cùng người gần nhau hơn và các ảnh khác người xa nhau hơn, nên đặc trưng học được phân tách tốt các lớp đã thấy nhưng chưa chắc tốt cho khuôn mặt mới. ArcFace biến đổi logit của softmax theo ba bước:

1. Đặt bias bằng 0, nên $W_j^\top x_i = \lVert W_j\rVert\,\lVert x_i\rVert\cos\theta_j$ với $\theta_j$ là góc giữa đặc trưng $x_i$ và trọng số $W_j$ của lớp $j$.
2. Chuẩn hóa $W_j$ về độ dài 1.
3. Chuẩn hóa $x_i$ và nhân với hệ số tỉ lệ $s$ (paper chọn $s = 64$), nên logit chỉ còn phụ thuộc vào góc.

Sau đó cộng margin góc $m$ vào góc của lớp đúng:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\ne y_i}e^{s\cos\theta_j}}, \qquad m = 0{,}5.$$

Vì $\cos(\theta + m) < \cos\theta$, mẫu phải nằm gần tâm lớp đúng hơn so với softmax chuẩn hóa mới đạt cùng xác suất. Margin là một góc nên tương ứng với khoảng cách cung trên siêu cầu, đó là lý do gọi là *additive angular margin*.

### Thí nghiệm của paper

- **Dữ liệu và mạng:** VGG2 (8.631 danh tính, 3,14 triệu ảnh) và MS-Celeb-1M đã làm sạch (khoảng 85 nghìn danh tính, 3,8 triệu ảnh), ảnh $112 \times 112$ căn chỉnh năm điểm, backbone ResNet cải tiến (LResNet100E-IR), embedding 512 chiều qua BN-Dropout-FC-BN.
- **Huấn luyện:** batch 512, SGD momentum 0,9, weight decay $5 \times 10^{-4}$, learning rate 0,1 chia 10 tại các vòng lặp 100k, 140k, 160k trong tổng 200k.
- **Benchmark:** độ chính xác verification trên LFW, CFP-FP, AgeDB-30; rank-1 identification và TAR@FAR $10^{-6}$ trên MegaFace.
- **So sánh loss (độ chính xác verification, %):**

| Loss | LFW | CFP-FP | AgeDB-30 |
|---|---:|---:|---:|
| Softmax | 99,70 | 91,40 | 95,56 |
| SphereFace ($m=4$) | 99,76 | 93,70 | 97,56 |
| CosineFace ($m=0{,}35$) | 99,80 | 94,40 | 97,91 |
| ArcFace ($m=0{,}5$) | 99,83 | 94,04 | 98,08 |

*Bảng 4. Kết quả so sánh loss trong paper (LResNet100E-IR, huấn luyện trên MS1M).*

Chênh lệch giữa các loss nhỏ ở LFW (gần bão hòa) và rõ hơn ở CFP-FP và AgeDB-30. Khi quét $m$ từ 0,2 đến 0,8 trên một backbone nhỏ, $m = 0{,}5$ cho kết quả cao nhất.

### Hạn chế

Paper không có mục hạn chế riêng. Có thể rút ra các điểm sau:

- **Lớp phân loại lớn:** khi số danh tính lên cỡ hàng triệu, lớp phân loại tốn rất nhiều bộ nhớ GPU; tác giả dùng triplet loss để tinh chỉnh ở quy mô đó.
- **Nhãn nhiễu:** MS1M và MegaFace phải được làm sạch thủ công.
- **Margin cố định:** $m$ và $s$ dùng chung cho mọi lớp; paper chỉ quét $m$ trên LMobileNetE rồi chọn $m=0{,}5$.
- **Miền không đơn điệu:** với $m=0{,}5$, $\cos(\theta+m)$ tăng trở lại khi $\theta>151{,}35^\circ$; paper không xử lý vùng này vì cho rằng nó hiếm khi xảy ra khi huấn luyện từ đầu.

### Đối chiếu với nhóm

| | Paper | Nhóm |
|---|---|---|
| Dữ liệu | VGG2, MS1M làm sạch (85K danh tính) | CelebA, 8.192 danh tính ở tập train |
| Backbone | LResNet100E-IR, so sánh với MobileNet | ResNet-50 và MobileNetV3 Large, khởi tạo từ ImageNet |
| Embedding | BN-Dropout-FC-BN, 512 chiều | BN-Dropout-FC-BN, 512 chiều |
| Loss | ArcFace, $s=64$, $m=0{,}5$ | Softmax và ArcFace, $s=64$, $m=0{,}5$ |
| Batch, learning rate | 512, 0,1 | 128, 0,001 |
| Lịch learning rate | Chia 10 tại vòng lặp cố định | ReduceLROnPlateau theo ROC-AUC validation |
| Thời gian huấn luyện | 200k vòng lặp | Tối đa 100 epoch, dừng sớm |
| Tăng cường dữ liệu | Không nêu | Lật ngang, ColorJitter nhẹ |
| Xử lý $\theta+m>\pi$ | Không xử lý | Logit tuyến tính $\cos\theta - m\sin m$ |

*Bảng 5. Đối chiếu thiết lập của paper và của nhóm.*

- **Tái hiện được:** xu hướng chính của paper, ArcFace tốt hơn softmax trên cùng backbone. Các con số tuyệt đối không so sánh được vì dữ liệu, backbone và benchmark đều khác.
- **Không kiểm chứng được:** giá trị $m = 0{,}5$, nhóm không quét margin.
- **Điều học được:** không sao chép siêu tham số của paper, cần điều chỉnh cho phù hợp với mô hình và dữ liệu; nên báo cáo cả verification và identification vì khác biệt giữa hai loss lộ rõ ở identification.

## Pipeline, huấn luyện và so sánh

### Mô hình

Ảnh $112 \times 112$ đi qua backbone, rồi khối embedding BN, Dropout (0,4), Linear, BN cho ra vector 512 chiều. Khi huấn luyện, vector này đi vào lớp phân loại 8.192 lớp của hàm loss; khi đánh giá, lớp phân loại bị bỏ và vector được dùng trực tiếp cho cosine similarity.

- **Softmax:** lớp tuyến tính có bias trên vector chưa chuẩn hóa, loss cross-entropy.
- **ArcFace:** $s = 64$, $m = 0{,}5$ cố định, không dùng cơ chế tăng dần margin. Với $\theta + m > \pi$, logit của lớp đúng được thay bằng $\cos\theta - m\sin m$ để loss không tăng ở vùng đó.
- **Backbone:** ResNet-50 (24,56 triệu tham số) và MobileNetV3 Large (3,47 triệu tham số), cả hai khởi tạo từ trọng số ImageNet và được huấn luyện toàn bộ (full fine-tune) với cùng một learning rate.

### Thiết lập huấn luyện

- SGD momentum 0,9, weight decay $5 \times 10^{-4}$, batch 128, FP32, seed 42.
- Learning rate khởi tạo 0,001 cho cả bốn thí nghiệm. Giá trị này được chọn sau thử nghiệm ngắn chỉ với ResNet-50 ArcFace (ở 0,01 loss kẹt quanh 22 đến 23 và ROC-AUC validation chỉ đạt 0,849 sau 9 epoch, còn ở 0,001 đạt 0,976), không tinh chỉnh riêng cho softmax hay MobileNetV3.
- Tối đa 100 epoch, dừng sớm với patience 5 theo ROC-AUC validation; ReduceLROnPlateau (factor 0,5, patience 2). Checkpoint tốt nhất theo ROC-AUC validation.
- Huấn luyện trên một GPU A10G (Modal).

Cách chạy lại thí nghiệm: làm theo [SETUP.md](https://github.com/tronglh241/co5085_dl4cv/blob/main/SETUP.md) để cài môi trường, sau đó mở shell trong container và chạy:

```bash
python scripts/a2/prepare_data.py --raw data/celeba --out data/celeba_112 --download
python scripts/train.py configs/a2/train_resnet50_arcface.yml
python scripts/a2/evaluate.py --config configs/a2/test_resnet50_arcface.yml
python scripts/a2/analyze.py
```

Thay `resnet50_arcface` bằng ba cấu hình còn lại trong `configs/a2`. `analyze.py` tổng hợp kết quả các run đã đánh giá thành bảng và hình.

### Hiệu năng tính toán

| Run | Tham số suy luận | Giây/epoch | Bộ nhớ đỉnh (GB) | Epoch tốt nhất / đã chạy | Phút đến epoch tốt nhất |
|---|---:|---:|---:|---:|---:|
| ResNet-50 softmax | 24,56 tr | 116,9 | 2,91 | 27 / 33 | 52,6 |
| ResNet-50 ArcFace | 24,56 tr | 117,6 | 2,94 | 14 / 20 | 27,4 |
| MobileNetV3 softmax | 3,47 tr | 45,6 | 1,49 | 51 / 57 | 38,7 |
| MobileNetV3 ArcFace | 3,47 tr | 47,2 | 1,54 | 20 / 27 | 15,7 |

*Bảng 6. Chi phí của bốn run (A10G, FP32). Thời gian mỗi epoch không gồm validation.*

Nhóm chưa đo độ trễ suy luận, nên số tham số chỉ là chỉ báo gián tiếp cho chi phí khi suy luận.

## Kết quả

### Đường cong huấn luyện

![ROC-AUC trên validation theo epoch](/assets/a2/curves_a2.png)

*Hình 4. ROC-AUC trên tập validation theo epoch của bốn run.*

Hai run ArcFace hội tụ nhanh hơn hẳn: ResNet-50 ArcFace đạt đỉnh ở epoch 14 (0,9795) và MobileNetV3 ArcFace ở epoch 20 (0,9797), trong khi hai run softmax đạt đỉnh ở epoch 27 (0,9720) và 51 (0,9741). Các run dừng sớm sau 20 đến 57 epoch nên trần 100 epoch không bị chạm tới. Loss huấn luyện của hai hàm loss có thang khác nhau (ArcFace từ khoảng 40 xuống 7 đến 12, softmax từ khoảng 9 xuống khoảng 1) nên không so trực tiếp được.

### Kết quả trên tập test

| Run | ROC-AUC | Accuracy (%) | TAR@FAR=$10^{-2}$ (%) | Rank-1 (%) | Rank-5 (%) |
|---|---:|---:|---:|---:|---:|
| ResNet-50 softmax | 0,9721 | 92,41 | 73,37 | 45,26 | 66,48 |
| ResNet-50 ArcFace | 0,9805 | 94,10 | 86,16 | **68,37** | **82,31** |
| MobileNetV3 softmax | 0,9736 | 92,40 | 77,15 | 52,04 | 71,09 |
| MobileNetV3 ArcFace | **0,9816** | 93,99 | 84,28 | 66,50 | 80,61 |

*Bảng 7. Kết quả test. Accuracy dùng ngưỡng chọn trên validation; Rank-1 và Rank-5 là identification trên gallery 997 ảnh và 18.962 probe.*

![Đường ROC trên tập test](/assets/a2/roc_a2.png)

*Hình 5. Đường ROC của verification trên tập test; hai đường ArcFace nằm trên hai đường softmax ở vùng FPR thấp.*

### Softmax so với ArcFace

Khoảng tin cậy bootstrap 95% (ghép cặp trên 10.000 cặp test, 2.000 lần lấy mẫu) của chênh lệch giữa các run:

| Phép so sánh | Metric | Chênh lệch | KTC 95% |
|---|---|---:|---:|
| ResNet-50: ArcFace − softmax | ROC-AUC | +0,0084 | [+0,0061; +0,0108] |
| MobileNetV3: ArcFace − softmax | ROC-AUC | +0,0079 | [+0,0060; +0,0100] |
| ArcFace: MobileNetV3 − ResNet-50 | ROC-AUC | +0,0011 | [−0,0008; +0,0027] |
| Softmax: MobileNetV3 − ResNet-50 | ROC-AUC | +0,0016 | [−0,0007; +0,0041] |

*Bảng 8. Chênh lệch giữa các cặp run trên tập test.*

- Verification: ArcFace tốt hơn softmax ở cả hai backbone, và khoảng tin cậy không chứa 0.
- Identification: Rank-1 tăng 23,1 điểm (ResNet-50) và 14,5 điểm (MobileNetV3), Rank-5 tăng 15,8 và 9,5 điểm. Identification buộc embedding của một ảnh phải gần ảnh cùng người nhất trong 997 danh tính, nên nhạy với việc các lớp có tách nhau đủ xa hay không, đúng là điều margin nhắm tới. Nhóm chưa có thí nghiệm riêng để kiểm chứng cách giải thích này.
- Lưu ý: learning rate 0,001 được chọn vì ArcFace. Ở learning rate 0,01 (chỉ theo dõi trên validation), softmax ResNet-50 đạt ROC-AUC validation 0,9794, cao hơn 0,9720 ở 0,001. Khoảng cách giữa hai loss ở trên là khoảng cách ở một learning rate chung và có thể bị phóng đại.

### ResNet-50 so với MobileNetV3

Với verification, hai backbone không khác biệt có ý nghĩa thống kê (khoảng tin cậy chứa 0 ở cả hai loss). Với identification, hướng chênh lệch không nhất quán: ResNet-50 cao hơn với ArcFace (Rank-1 68,4% so với 66,5%), còn MobileNetV3 cao hơn với softmax (52,0% so với 45,3%). Chi phí thì rất khác nhau: MobileNetV3 có số tham số suy luận nhỏ hơn 7,1 lần, mỗi epoch nhanh hơn 2,5 lần và dùng bộ nhớ ít hơn 1,9 lần. Trong thiết lập này MobileNetV3 là lựa chọn hợp lý hơn, với điều kiện chỉ có một seed và dữ liệu cỡ CelebA.

### Phân tích lỗi

Nhóm xem lỗi của verification trên 10.000 cặp test ở ngưỡng chọn trên validation:

| Run | False accept | FAR (%) | False reject | FRR (%) | Tổng lỗi |
|---|---:|---:|---:|---:|---:|
| ResNet-50 softmax | 300 | 5,87 | 459 | 9,40 | 759 |
| ResNet-50 ArcFace | 197 | 3,85 | 393 | 8,05 | 590 |
| MobileNetV3 softmax | 401 | 7,84 | 359 | 7,35 | 760 |
| MobileNetV3 ArcFace | 254 | 4,97 | 347 | 7,10 | 601 |

*Bảng 9. False accept là cặp khác người bị nhận là cùng người; false reject là cặp cùng người bị loại.*

- **ArcFace giảm chủ yếu false accept:** giảm 34% (ResNet-50) và 37% (MobileNetV3), trong khi false reject chỉ giảm 14% và 3%. Điều này phù hợp với việc margin đẩy các danh tính khác nhau ra xa nhau. Ngưỡng của mỗi run khác nhau nên cách lỗi chia giữa hai loại phụ thuộc vào ngưỡng.
- **Lỗi chung giữa các run:** 208 cặp bị cả bốn run phân loại sai, 1.392 cặp sai ở ít nhất một run; 56% đến 63% lỗi của ArcFace cũng là lỗi của softmax cùng backbone. Phần lớn lỗi nằm ở các cặp khó, nhưng vẫn có phần lỗi riêng của từng mô hình.
- **Ca sai điển hình** (nhận xét định tính từ 12 cặp mỗi loại): false reject thường là cùng một người nhưng khác rõ về trang điểm, màu và kiểu tóc, độ tuổi, biểu cảm, ánh sáng; false accept thường là hai người khác nhau nhưng cùng giới tính, nhóm tuổi và sắc tộc, có kiểu tóc hoặc tư thế tương tự. Nhóm chưa kiểm tra nhãn của các cặp này nên không loại trừ nhiễu nhãn.

![Các cặp cùng người bị loại sai chắc chắn nhất](/assets/a2/errors_false_reject_a2.png)

*Hình 6. Các cặp cùng người bị loại sai chắc chắn nhất của MobileNetV3 ArcFace; số trên mỗi cặp là cosine.*

![Các cặp khác người bị nhận sai chắc chắn nhất](/assets/a2/errors_false_accept_a2.png)

*Hình 7. Các cặp khác người bị nhận sai chắc chắn nhất của MobileNetV3 ArcFace.*

- **Chất lượng ảnh:** với cặp cùng người, nhóm ảnh mờ nhất có tỉ lệ false reject cao hơn nhóm trung bình ở cả bốn run (ví dụ MobileNetV3 ArcFace 9,1% so với 5,5%), nhưng ở ba trong bốn run nhóm sắc nét nhất lại cao hơn nhóm trung bình, nên quan hệ không đơn điệu. Với cặp khác người, tỉ lệ false accept gần như không phụ thuộc độ sắc nét. Nhóm chưa kiểm định và chỉ có một seed, nên chỉ xem là xu hướng.

![Tỉ lệ lỗi theo độ sắc nét](/assets/a2/errors_sharpness_a2.png)

*Hình 8. Tỉ lệ lỗi theo độ sắc nét của ảnh mờ hơn trong cặp (chia ba nhóm bằng nhau): trái là cặp cùng người, phải là cặp khác người.*

## Thảo luận và hạn chế

- **Điều kết quả cho thấy:** trên CelebA, ArcFace cho embedding tốt hơn softmax ở cả hai backbone; khác biệt về verification nhỏ nhưng ổn định, còn khác biệt về identification lớn. Backbone không phải yếu tố quyết định.
- **Đối chiếu với paper:** xu hướng cùng chiều với paper (ArcFace cao hơn softmax, khoảng cách nhỏ ở benchmark dễ như LFW và lớn hơn ở benchmark khó). Độ lớn tuyệt đối không so sánh được.
- **Một seed:** mỗi cấu hình chỉ chạy một lần. Khoảng tin cậy bootstrap chỉ phản ánh nhiễu do lấy mẫu các cặp test, không phản ánh độ biến động giữa các lần huấn luyện; với identification nhóm không tính được khoảng tin cậy.
- **Learning rate chưa tinh chỉnh:** dùng chung 0,001 cho bốn run, chọn bằng thử nghiệm ngắn chỉ với ArcFace trên ResNet-50; softmax không được thử giá trị khác.
- **Quy mô dữ liệu:** khoảng 20 ảnh mỗi danh tính trên 8.192 lớp, nhỏ hơn nhiều so với dữ liệu của paper; kết luận chỉ áp dụng cho quy mô này và cho CelebA.
- **Tập đánh giá:** gallery có một ảnh mỗi danh tính (ảnh đầu tiên theo tên tệp) và không lặp lại với các lựa chọn khác, nên Rank-1 phụ thuộc vào ảnh được chọn.
- **Thiết kế thí nghiệm:** không thử các giá trị khác của $s$ và $m$, không ablation riêng cho tăng cường dữ liệu và khởi tạo từ ImageNet, không đánh giá chéo tập dữ liệu (ví dụ LFW), phân tích lỗi chỉ định tính và chưa theo thuộc tính của CelebA, chưa đo độ trễ suy luận.
- **Dropout trước BatchNorm cuối:** nhóm đã đo ảnh hưởng của việc dropout đứng trước BN cuối của khối embedding (`scripts/a2/check_bn.py`). Hiệu chỉnh lại thống kê của BN này chỉ làm ROC-AUC test thay đổi tối đa 0,0006 và Rank-1 tối đa 0,0016, nhỏ hơn nhiều so với khoảng cách giữa hai loss.
- **Huấn luyện bị gián đoạn:** bốn run bị thu hồi container và tiếp tục từ bản backup, nên thứ tự lấy mẫu dữ liệu có thể khác một lần chạy liền mạch.

## Demo

Script `scripts/a2/demo.py` nhận hai ảnh khuôn mặt, in cosine similarity và kết luận cùng người hay khác người theo ngưỡng chọn trên validation, đồng thời lưu hình hai ảnh kèm kết quả. Script không có bộ phát hiện khuôn mặt, nên ảnh cần được căn chỉnh sẵn về $112 \times 112$ như dữ liệu huấn luyện; ảnh khác kích thước chỉ được cắt giữa và đổi cỡ.

```bash
python scripts/a2/demo.py a.jpg b.jpg --run mobilenetv3_arcface --truth same --out demo_pair.png
```

Video demo: _đang cập nhật liên kết_.
