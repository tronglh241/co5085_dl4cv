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

**Chủ đề:** nhận dạng khuôn mặt (xác minh và định danh) với hàm loss **ArcFace**, đánh giá trên các danh tính chưa từng thấy khi huấn luyện.

Bài A2 tái hiện ở quy mô nhỏ ý tưởng của bài báo ArcFace trên tập CelebA và trả lời hai câu hỏi:

- **Margin góc của ArcFace giúp được bao nhiêu so với softmax?** Giữ nguyên backbone, đổi hàm loss.
- **Backbone nhỏ đánh đổi gì so với backbone lớn?** Giữ nguyên hàm loss ArcFace, so sánh ResNet-50 và MobileNetV3.

Thông tin chính của bài:

- **Bài báo:** J. Deng, J. Guo, N. Xue, S. Zafeiriou, *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019, tr. 4690–4699. [arXiv:1801.07698](https://arxiv.org/abs/1801.07698). Nhóm đọc và trích số liệu từ bản preprint đầu tiên (v1), và ghi rõ chỗ bản này không nêu chi tiết.
- **Tập dữ liệu:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), 202.599 ảnh của 10.177 danh tính, căn chỉnh về $112 \times 112$ bằng năm điểm mốc.
- **Mô hình:** ResNet-50 và MobileNetV3 Large (torchvision, khởi tạo từ ImageNet) nhân hai hàm loss (softmax, ArcFace) thành 4 run, mỗi run một lần với seed 42.
- **Code:** [thư mục `scripts/a2`, `models/a2`, `configs/a2` và `datasets`](https://github.com/tronglh241/co5085_dl4cv/tree/main/scripts/a2) trên GitHub.
- **Báo cáo PDF:** [assignment_2.pdf](https://github.com/tronglh241/co5085_dl4cv/blob/main/reports/assignment_2.pdf), có đầy đủ phân tích paper, thiết lập, kết quả và các hạn chế.
- **Video demo:** _đang cập nhật liên kết_.

Kết quả chính:

- ArcFace tốt hơn softmax ở cả hai backbone. Với verification, ROC-AUC trên test tăng khoảng 0,008 (từ 0,9721 lên 0,9805 ở ResNet-50 và từ 0,9736 lên 0,9816 ở MobileNetV3), khoảng tin cậy bootstrap 95% không chứa 0. Với identification, chênh lệch lớn hơn nhiều: Rank-1 tăng từ 45,3% lên 68,4% (ResNet-50) và từ 52,0% lên 66,5% (MobileNetV3).
- ResNet-50 và MobileNetV3 gần như ngang nhau về verification (ROC-AUC 0,9805 và 0,9816, khoảng tin cậy của chênh lệch chứa 0), nhưng MobileNetV3 có số tham số suy luận nhỏ hơn 7,1 lần và mỗi epoch nhanh hơn 2,5 lần.
- ArcFace đạt checkpoint tốt nhất sau số epoch ít hơn khoảng 1,9 đến 2,5 lần so với softmax (14 so với 27 epoch ở ResNet-50, 20 so với 51 ở MobileNetV3).
- Hai loss được so sánh ở cùng learning rate 0,001, vốn được chọn vì ArcFace cần giá trị nhỏ, nên khoảng cách này có thể lớn hơn khoảng cách nếu mỗi loss có learning rate riêng.

## Dữ liệu

### Bài toán và cách đánh giá

Mô hình được **huấn luyện như một bài toán phân loại** trên 8.192 danh tính của tập train, nhưng được **đánh giá trên danh tính chưa từng thấy**: tập validation (985 danh tính) và test (1.000 danh tính) không có danh tính hoặc ảnh chung với tập train. Vì vậy lớp phân loại cuối bị bỏ khi đánh giá; mỗi ảnh được biến thành một vector 512 chiều (embedding) và hai ảnh được so bằng cosine similarity:

$$s(x,y) = \frac{f(x)^\top f(y)}{\lVert f(x)\rVert \, \lVert f(y)\rVert}.$$

- **Verification:** hai ảnh có cùng người không? Quyết định bằng ngưỡng $\tau$ chọn trên validation. Test có 10.000 cặp (4.885 cặp cùng người, 5.115 cặp khác người). Metric: ROC-AUC, EER, accuracy ở ngưỡng đã chọn, TAR@FAR.
- **Identification (closed-set):** ảnh truy vấn thuộc danh tính nào trong gallery? Gallery có 997 ảnh (mỗi danh tính một ảnh), 18.962 ảnh truy vấn (probe). Metric: Rank-1, Recall@5, mAP. Vì gallery chỉ có một ảnh mỗi danh tính nên mAP trùng với mean reciprocal rank.

### Tập dữ liệu CelebA

CelebA gồm ảnh người nổi tiếng chụp ngoài điều kiện kiểm soát, phát hành cho nghiên cứu phi thương mại. Nhóm dùng phiên bản `img_align_celeba` ($218 \times 178$) cùng năm điểm mốc có sẵn và chia tập theo mặc định của CelebA.

| Tập | Số ảnh | Số danh tính |
|---|---:|---:|
| Train | 162.770 | 8.192 |
| Validation | 19.867 | 985 |
| Test | 19.962 | 1.000 |
| Tổng | 202.599 | 10.177 |

*Bảng 1. Ba tập dữ liệu của CelebA. Nhóm đã kiểm tra các tệp chia tập: không có danh tính chung và không có ảnh chung giữa hai tập bất kỳ.*

### Phân tích khám phá dữ liệu

![Phân bố số ảnh mỗi danh tính ở tập train](/assets/a2/eda/identity_hist.png)

*Hình 1. Số ảnh mỗi danh tính ở tập train (trục tung theo thang logarit).*

- **Ít ảnh mỗi lớp:** trung vị 21 ảnh, trung bình 19,9, 50% danh tính có từ 13 đến 29 ảnh, lớn nhất 35. Không danh tính nào chiếm ưu thế (năm danh tính nhiều ảnh nhất chỉ chiếm 0,1% số ảnh), nên không cần gán trọng số lớp. Tuy nhiên khoảng 20 ảnh mỗi danh tính trên 8.192 lớp là rất ít so với dữ liệu của paper, nên nguy cơ overfit cao và mỗi lớp có ít mẫu để học tâm lớp.
- **Danh tính quá ít ảnh:** 36 danh tính (0,4%) chỉ có một ảnh và 684 danh tính (8,3%) có dưới 5 ảnh.
- **Các tập có phân bố giống nhau:** validation và test cũng có trung vị 21 ảnh mỗi danh tính, nên điểm đánh giá không bị lệch do phân bố.
- **Cặp ảnh và FAR:** với khoảng 5.100 cặp âm ở test, mức FAR $10^{-3}$ chỉ ứng với khoảng 5 cặp, nên TAR@FAR=$10^{-3}$ rất nhiễu và nhóm không dùng nó để kết luận.

![Ảnh cùng danh tính sau khi căn chỉnh](/assets/a2/eda/samples.png)

*Hình 2. Ảnh mẫu sau khi căn chỉnh: mỗi hàng là một danh tính. Ảnh cùng người vẫn khác nhau nhiều về tóc, trang điểm, ánh sáng và tư thế.*

![Ảnh gốc và ảnh sau căn chỉnh](/assets/a2/eda/alignment.png)

*Hình 3. Ảnh gốc $218 \times 178$ (trái) và ảnh sau căn chỉnh $112 \times 112$ (phải). Sau căn chỉnh, mắt, mũi và khóe miệng ở cùng vị trí; ảnh gốc có độ phân giải thấp thì ảnh sau căn chỉnh bị mờ.*

## Chuẩn bị dữ liệu huấn luyện

### Căn chỉnh và tiền xử lý

Mọi ảnh được căn chỉnh một lần về $112 \times 112$ bằng phép biến đổi đồng dạng (tỉ lệ, xoay, tịnh tiến) khớp năm điểm mốc của ảnh với mẫu chuẩn $112 \times 112$ thường dùng trong các pipeline kiểu ArcFace, rồi lưu lại để các lần huấn luyện sau không lặp lại bước này. Nhóm dùng trực tiếp các điểm mốc có sẵn của CelebA nên không cần bộ phát hiện khuôn mặt.

Ảnh được chuẩn hóa bằng mean và std $0{,}5$ cho cả ba kênh.

### Tăng cường dữ liệu

Chỉ tập train được tăng cường; validation, test và các cặp đánh giá chỉ được chuẩn hóa.

| Bước | Tham số |
|---|---|
| Lật ngang | Xác suất 0,5 |
| ColorJitter nhẹ | Độ sáng, tương phản, độ bão hòa, mỗi đại lượng tối đa 0,2 |
| Chuẩn hóa | Mean và std 0,5 |

*Bảng 2. Pipeline tăng cường dữ liệu của tập train.*

- Lật ngang không đổi danh tính và giữ nguyên vị trí các điểm mốc sau căn chỉnh; ColorJitter mô phỏng sự khác nhau về ánh sáng giữa các ảnh.
- Nhóm không dùng xoay hoặc cắt ảnh để không làm lệch mẫu căn chỉnh năm điểm.
- Bản v1 của paper không nêu phép augmentation nào, nên đây là lựa chọn của nhóm.

### Tập đánh giá và DataLoader

- Các cặp verification và việc chia gallery, probe cho identification được sinh một lần với seed cố định, lưu thành tệp và dùng chung cho cả bốn run.
- Mỗi lớp `CelebA` đọc tệp chia tập và ảnh đã căn chỉnh; `VerificationPairs` đọc danh sách cặp. Tập train được xáo trộn mỗi epoch, batch 128; validation và test không xáo trộn.

## Phân tích bài báo ArcFace

### Phương pháp

Softmax chuẩn không trực tiếp ép các ảnh cùng người gần nhau hơn và các ảnh khác người xa nhau hơn, nên đặc trưng học được phân tách tốt các lớp đã thấy nhưng chưa chắc tốt cho khuôn mặt mới. ArcFace biến đổi logit của softmax theo ba bước:

1. Đặt bias bằng 0, nên $W_j^\top x_i = \lVert W_j\rVert\,\lVert x_i\rVert\cos\theta_j$ với $\theta_j$ là góc giữa đặc trưng $x_i$ và trọng số $W_j$ của lớp $j$.
2. Chuẩn hóa $W_j$ về độ dài 1.
3. Chuẩn hóa $x_i$ và nhân với hệ số tỉ lệ $s$ (paper chọn $s = 64$), nên logit chỉ còn phụ thuộc vào góc.

Sau đó cộng margin góc $m$ vào góc của lớp đúng:

$$L = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\ne y_i}e^{s\cos\theta_j}}, \qquad m = 0{,}5.$$

Vì $\cos(\theta + m) < \cos\theta$, mẫu phải nằm gần tâm lớp đúng hơn so với softmax chuẩn hóa mới đạt cùng xác suất. Margin là một góc nên tương ứng với khoảng cách cung trên siêu cầu, đó là lý do gọi là *additive angular margin*. Paper đặt ArcFace cạnh hai cách đặt margin khác: SphereFace nhân góc với một số nguyên $m$ (logit $\cos(m\theta)$, cần hàm từng khúc và trộn thêm softmax để hội tụ), và CosineFace trừ hằng số khỏi cosine (logit $s(\cos\theta - m)$).

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

*Bảng 3. Kết quả so sánh loss trong paper (LResNet100E-IR, huấn luyện trên MS1M).*

Chênh lệch giữa các loss nhỏ ở LFW (gần bão hòa) và rõ hơn ở CFP-FP và AgeDB-30. Khi quét $m$ từ 0,2 đến 0,8 trên một backbone nhỏ, $m = 0{,}5$ cho kết quả cao nhất.

### Hạn chế

Bài báo không có mục hạn chế riêng; phần dưới đây gồm điều paper tự nêu rải rác và nhận xét của nhóm.

- **Do paper nêu:** lớp phân loại tốn bộ nhớ GPU khi số danh tính lên cỡ hàng triệu (tác giả dùng triplet loss để tinh chỉnh ở quy mô đó); danh sách ảnh nhiễu của MegaFace và FaceScrub làm thủ công và không bảo đảm sạch hoàn toàn; $m$ quá lớn làm huấn luyện phân kỳ; logit của lớp đúng tăng trở lại khi $\theta > 151{,}35^\circ$ và paper không xử lý vùng này.
- **Nhận xét của nhóm:** $s$ và $m$ chọn thủ công và dùng chung mọi lớp; không có thí nghiệm về nhãn nhiễu; huấn luyện theo kiểu closed-set nhưng đánh giá open-set, và paper không phân tích lý thuyết cho bước chuyển này; không có độ lệch chuẩn hay số lần chạy lặp, nên chênh lệch 0,13 điểm trên LFW khó diễn giải; bản v1 thiếu pseudo-code và thông tin augmentation.
- **Giả thiết không đúng trên tập của nhóm:** tập train chỉ có khoảng 20 ảnh mỗi danh tính và mạng được fine-tune từ ImageNet thay vì huấn luyện từ đầu. Hệ quả quan sát được là ở learning rate 0,01, ArcFace trên ResNet-50 kẹt (ROC-AUC validation chỉ khoảng 0,91), còn ở 0,001 thì hội tụ bình thường.

### Đối chiếu với nhóm

| | Paper (bản v1) | Nhóm |
|---|---|---|
| Dữ liệu | VGG2, MS1M làm sạch (85K danh tính) | CelebA, 8.192 danh tính ở tập train |
| Backbone | LResNet100E-IR, so sánh với MobileNet | ResNet-50 và MobileNetV3 Large, khởi tạo từ ImageNet |
| Embedding | BN-Dropout-FC-BN, 512 chiều | Cùng cấu trúc |
| Loss | ArcFace, $s=64$, $m=0{,}5$ | Softmax và ArcFace, $s=64$, $m=0{,}5$ |
| Batch, learning rate | 512, 0,1 | 128, 0,001 |
| Lịch learning rate | Chia 10 tại vòng lặp cố định | ReduceLROnPlateau theo ROC-AUC validation |
| Thời gian huấn luyện | 200k vòng lặp | Tối đa 100 epoch, dừng sớm |
| Augmentation | Không nêu | Lật ngang, ColorJitter nhẹ |
| Xử lý $\theta+m>\pi$ | Không xử lý | Logit tuyến tính $\cos\theta - m\sin m$ |

*Bảng 4. Đối chiếu thiết lập của paper và của nhóm.*

- **Tái hiện được:** xu hướng chính của paper, ArcFace tốt hơn softmax trên cùng backbone. Các con số tuyệt đối không so sánh được vì dữ liệu, backbone và benchmark đều khác.
- **Không kiểm chứng được:** giá trị $m = 0{,}5$ (nhóm không quét margin) và độ nhạy với nhãn nhiễu.
- **Điều học được:** không sao chép siêu tham số của paper khi đổi chế độ huấn luyện (learning rate); nên báo cáo cả verification và identification vì khác biệt giữa hai loss lộ rõ ở identification; ghi lại các chi tiết paper không nêu.

## Pipeline, huấn luyện và so sánh

### Mô hình

Ảnh $112 \times 112$ đi qua backbone, rồi khối embedding BN, Dropout (0,4), Linear, BN cho ra vector 512 chiều. Khi huấn luyện, vector này đi vào lớp phân loại 8.192 lớp của hàm loss; khi đánh giá, lớp phân loại bị bỏ và vector được dùng trực tiếp cho cosine similarity.

- **Softmax:** lớp tuyến tính có bias trên vector chưa chuẩn hóa, loss cross-entropy.
- **ArcFace:** $s = 64$, $m = 0{,}5$ cố định, không dùng cơ chế tăng dần margin. Với $\theta + m > \pi$, logit của lớp đúng được thay bằng $\cos\theta - m\sin m$ (theo bản cài đặt tham chiếu của tác giả) để loss không tăng ở vùng đó.
- **Backbone:** ResNet-50 (24,56 triệu tham số suy luận) và MobileNetV3 Large (3,47 triệu), cả hai khởi tạo từ trọng số ImageNet và được huấn luyện toàn bộ (full fine-tune) với cùng một learning rate.

### Thiết lập huấn luyện

- SGD momentum 0,9, weight decay $5 \times 10^{-4}$, batch 128, FP32, seed 42.
- Learning rate khởi tạo 0,001 cho cả bốn run. Giá trị này được chọn sau thử nghiệm ngắn chỉ với ResNet-50 ArcFace (ở 0,01 loss kẹt quanh 22 đến 23 và ROC-AUC validation chỉ đạt 0,849 sau 9 epoch, còn ở 0,001 đạt 0,976), không tinh chỉnh riêng cho softmax hay MobileNetV3.
- Tối đa 100 epoch, dừng sớm với patience 5 theo ROC-AUC validation; ReduceLROnPlateau (factor 0,5, patience 2). Checkpoint tốt nhất theo ROC-AUC validation.
- Huấn luyện trên một GPU A10G (Modal). Bốn run bị thu hồi container giữa chừng và tiếp tục từ bản backup gần nhất.

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

*Bảng 5. Chi phí của bốn run (A10G, FP32). Thời gian mỗi epoch không gồm validation.*

Nhóm chưa đo độ trễ suy luận, nên số tham số chỉ là chỉ báo gián tiếp cho chi phí khi suy luận.

## Kết quả

### Đường cong huấn luyện

![ROC-AUC trên validation theo epoch](/assets/a2/curves_a2.png)

*Hình 4. ROC-AUC trên tập validation theo epoch của bốn run.*

Hai run ArcFace hội tụ nhanh hơn hẳn: ResNet-50 ArcFace đạt đỉnh ở epoch 14 (0,9795) và MobileNetV3 ArcFace ở epoch 20 (0,9797), trong khi hai run softmax đạt đỉnh ở epoch 27 (0,9720) và 51 (0,9741). Các run dừng sớm sau 20 đến 57 epoch nên trần 100 epoch không bị chạm tới. Loss huấn luyện của hai hàm loss có thang khác nhau (ArcFace từ khoảng 40 xuống 7 đến 12, softmax từ khoảng 9 xuống khoảng 1) nên không so trực tiếp được.

### Kết quả trên tập test

| Run | ROC-AUC | EER (%) | Accuracy (%) | TAR@FAR=$10^{-2}$ (%) | Rank-1 (%) | Recall@5 (%) | mAP (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 softmax | 0,9721 | 7,54 | 92,41 | 73,37 | 45,26 | 66,48 | 55,26 |
| ResNet-50 ArcFace | 0,9805 | 6,16 | 94,10 | 86,16 | **68,37** | **82,31** | **74,81** |
| MobileNetV3 softmax | 0,9736 | 7,51 | 92,40 | 77,15 | 52,04 | 71,09 | 60,92 |
| MobileNetV3 ArcFace | **0,9816** | **6,11** | 93,99 | 84,28 | 66,50 | 80,61 | 73,05 |

*Bảng 6. Kết quả test. Accuracy dùng ngưỡng chọn trên validation; Rank-1, Recall@5 và mAP là identification trên gallery 997 ảnh và 18.962 probe.*

Điểm test sát điểm validation (chênh lệch tối đa 0,002), nên việc chọn checkpoint và ngưỡng trên validation không làm lệch kết quả. TAR@FAR=$10^{-3}$ chỉ dựa trên khoảng 5 cặp âm nên nhóm không dùng để kết luận.

![Đường ROC trên tập test](/assets/a2/roc_a2.png)

*Hình 5. Đường ROC của verification trên tập test; hai đường ArcFace nằm trên hai đường softmax ở vùng FPR thấp.*

### Softmax so với ArcFace

Khoảng tin cậy bootstrap 95% (ghép cặp trên 10.000 cặp test, 2.000 lần lấy mẫu) của chênh lệch giữa các run:

| Phép so sánh | Metric | Chênh lệch | KTC 95% |
|---|---|---:|---:|
| ResNet-50: ArcFace − softmax | ROC-AUC | +0,0084 | [+0,0061; +0,0108] |
| ResNet-50: ArcFace − softmax | EER (điểm %) | −1,38 | [−1,97; −0,90] |
| MobileNetV3: ArcFace − softmax | ROC-AUC | +0,0079 | [+0,0060; +0,0100] |
| MobileNetV3: ArcFace − softmax | EER (điểm %) | −1,40 | [−1,88; −0,93] |
| ArcFace: MobileNetV3 − ResNet-50 | ROC-AUC | +0,0011 | [−0,0008; +0,0027] |
| ArcFace: MobileNetV3 − ResNet-50 | EER (điểm %) | −0,05 | [−0,49; +0,44] |
| Softmax: MobileNetV3 − ResNet-50 | ROC-AUC | +0,0016 | [−0,0007; +0,0041] |
| Softmax: MobileNetV3 − ResNet-50 | EER (điểm %) | −0,03 | [−0,55; +0,53] |

*Bảng 7. Chênh lệch giữa các cặp run trên tập test.*

- Verification: ArcFace tốt hơn softmax ở cả hai backbone, và khoảng tin cậy không chứa 0.
- Identification: Rank-1 tăng 23,1 điểm (ResNet-50) và 14,5 điểm (MobileNetV3), Recall@5 tăng 15,8 và 9,5 điểm, mAP tăng 19,6 và 12,1 điểm. Identification buộc embedding của một ảnh phải gần ảnh cùng người nhất trong 997 danh tính, nên nhạy với việc các lớp có tách nhau đủ xa hay không, đúng là điều margin nhắm tới. Nhóm chưa có thí nghiệm riêng để kiểm chứng cách giải thích này.
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

*Bảng 8. False accept là cặp khác người bị nhận là cùng người; false reject là cặp cùng người bị loại.*

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
- **Thiết kế thí nghiệm:** không thử các giá trị khác của $s$ và $m$, không ablation riêng cho augmentation và khởi tạo từ ImageNet, không đánh giá chéo tập dữ liệu (ví dụ LFW), phân tích lỗi chỉ định tính và chưa theo thuộc tính của CelebA, chưa đo độ trễ suy luận.
- **Dropout trước BatchNorm cuối:** nhóm đã đo ảnh hưởng của việc dropout đứng trước BN cuối của khối embedding (`scripts/a2/check_bn.py`). Hiệu chỉnh lại thống kê của BN này chỉ làm ROC-AUC test thay đổi tối đa 0,0006 và Rank-1 tối đa 0,0016, nhỏ hơn nhiều so với khoảng cách giữa hai loss.
- **Huấn luyện bị gián đoạn:** bốn run bị thu hồi container và tiếp tục từ bản backup, nên thứ tự lấy mẫu dữ liệu có thể khác một lần chạy liền mạch.

## Demo

Script `scripts/a2/demo.py` nhận hai ảnh khuôn mặt, in cosine similarity và kết luận cùng người hay khác người theo ngưỡng chọn trên validation, đồng thời lưu hình hai ảnh kèm kết quả. Script không có bộ phát hiện khuôn mặt, nên ảnh cần được căn chỉnh sẵn về $112 \times 112$ như dữ liệu huấn luyện; ảnh khác kích thước chỉ được cắt giữa và đổi cỡ.

```bash
python scripts/a2/demo.py a.jpg b.jpg --run mobilenetv3_arcface --truth same --out demo_pair.png
```

Video demo: _đang cập nhật liên kết_.
