---
layout: post
title: "A2 Chủ đề tự chọn và phân tích bài báo"
author:
    - trong
    - thien
date: 2026-11-16 10:00
toc: true
math: true
comments: false
---

## Giới thiệu

Bài A2 gồm hai phần: phân tích paper **ArcFace** và tái hiện ý tưởng của paper trên một tập dữ liệu nhỏ hơn. ArcFace thay đổi lớp phân loại cuối của mạng: thay vì chỉ dùng softmax, nó chuẩn hóa đặc trưng và trọng số rồi cộng một margin góc vào logit của lớp đúng. Mục tiêu là làm cho các đặc trưng của cùng một người nằm gần nhau hơn trên siêu cầu, còn các đặc trưng của người khác xa nhau hơn.

Chúng tôi huấn luyện bốn cấu hình trên CelebA:

- ResNet-50 với softmax
- ResNet-50 với ArcFace
- MobileNetV3 Large với softmax
- MobileNetV3 Large với ArcFace

Các câu hỏi chính:

- **ArcFace có tốt hơn softmax không?** So sánh trên cùng backbone, cùng learning rate và cùng dữ liệu.
- **Backbone có phải yếu tố quyết định không?** So sánh ResNet-50 và MobileNetV3 Large ở cùng loss.
- **Chênh lệch nằm ở verification hay identification?** Verification đo trên cặp ảnh, identification đo trên gallery/probe.
- **Chi phí là bao nhiêu?** Số tham số suy luận, thời gian huấn luyện và bộ nhớ GPU.

Thông tin chính:

- **Paper:** [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698v1), CVPR 2019.
- **Dữ liệu:** [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), bản đã căn chỉnh về $112 \times 112$.
- **Backbone:** ResNet-50 và MobileNetV3 Large của torchvision, khởi tạo từ ImageNet.
- **Embedding:** BN–Dropout–FC–BN, $512$ chiều.
- **Loss:** softmax và ArcFace với $s = 64$, $m = 0.5$.
- **Code:** [scripts/a2](https://github.com/tronglh241/co5085_dl4cv/tree/main/scripts/a2), [models/a2](https://github.com/tronglh241/co5085_dl4cv/tree/main/models/a2), [configs/a2](https://github.com/tronglh241/co5085_dl4cv/tree/main/configs/a2).
- **Báo cáo PDF:** [assignment_2.pdf](https://github.com/tronglh241/co5085_dl4cv/blob/main/reports/assignment_2.pdf).

Kết quả chính:

- ArcFace tốt hơn softmax ở cả hai backbone trên mọi metric đã đo. Với ResNet-50, accuracy tăng từ $92{,}42\%$ lên $94{,}10\%$, TAR@FAR=$10^{-2}$ tăng từ $73{,}53\%$ lên $86{,}14\%$, Rank-1 tăng từ $45{,}25\%$ lên $68{,}37\%$.
- Với MobileNetV3 Large, ArcFace cũng tăng accuracy từ $92{,}53\%$ lên $94{,}00\%$, TAR@FAR=$10^{-2}$ từ $77{,}18\%$ lên $84{,}42\%$, Rank-1 từ $52{,}05\%$ lên $66{,}48\%$.
- ResNet-50 và MobileNetV3 Large gần như bằng nhau về verification. MobileNetV3 có $3{,}47$ triệu tham số suy luận so với $24{,}56$ triệu của ResNet-50, tức nhỏ hơn khoảng $7{,}1$ lần.
- Chênh lệch giữa hai backbone ở identification không nhất quán: với ArcFace, ResNet-50 cao hơn ở Rank-1; với softmax, MobileNetV3 cao hơn.

## Bài toán và dữ liệu

### Nhiệm vụ

Gọi $f(x)$ là embedding của ảnh $x$. Độ tương đồng giữa hai ảnh được đo bằng cosine:

$$
s(x,y)=\frac{f(x)^\top f(y)}{\|f(x)\|\,\|f(y)\|}.
$$

Có hai nhiệm vụ:

- **Verification:** cho hai ảnh, quyết định chúng có cùng danh tính hay không. Nhiều cặp ảnh được tạo sẵn, mỗi cặp có nhãn $1$ hoặc $0$.
- **Identification:** cho một ảnh probe, tìm ảnh cùng danh tính trong một gallery đã biết trước. Mỗi probe được xếp hạng theo độ tương đồng với toàn bộ gallery.

### Tập dữ liệu

CelebA có $202.599$ ảnh và $10.177$ danh tính. Chúng tôi dùng bản đã căn chỉnh, chia thành ba tập không giao nhau về danh tính:

| Tập | Số ảnh | Số danh tính | Ảnh mỗi danh tính (trung vị) |
|---|---:|---:|---:|
| Train | 162.770 | 8.192 | 21 |
| Validation | 19.867 | 985 | 21 |
| Test | 19.962 | 1.000 | 21 |

*Bảng 1. Ba tập dữ liệu dùng trong A2.*

Việc kiểm tra không có danh tính chung giữa ba tập được thực hiện trong `scripts/a2/prepare_data.py`. Nếu phát hiện chồng lấn, script dừng lại.

![Phân bố số ảnh trên mỗi danh tính ở tập train](/assets/a2/eda/identity_hist.png)

*Hình 1. Số ảnh trên mỗi danh tính ở tập train. Đường nét đứt là trung vị, đường chấm là trung bình.*

Tập train có khoảng $20$ ảnh mỗi danh tính trên $8.192$ lớp. Đây là điểm khác biệt lớn so với paper: paper huấn luyện trên VGG2 và MS-Celeb-1M đã làm sạch, với hàng chục nghìn đến khoảng $85$ nghìn danh tính và hàng triệu ảnh.

### Tiền xử lý

Mỗi ảnh được căn chỉnh về $112 \times 112$ bằng một phép biến đổi đồng dạng dựa trên năm điểm mốc: hai mắt, mũi và hai khóe miệng. Ảnh được chuẩn hóa về khoảng $[-1,1]$ bằng:

$$
x \leftarrow \frac{x - 0{,}5}{0{,}5}.
$$

![Trước và sau căn chỉnh](/assets/a2/eda/alignment.png)

*Hình 2. Hàng trên là ảnh gốc trong `img_align_celeba`, hàng dưới là ảnh sau căn chỉnh về $112 \times 112$.*

### Tăng cường dữ liệu

Chỉ tập train được tăng cường:

- Lật ngang với xác suất $0{,}5$.
- ColorJitter với brightness, contrast và saturation trong $\pm 20\%$.
- Không dùng RandomResizedCrop, không xoay, không dịch.

![Trước và sau augmentation](/assets/a2/eda/augmentation.png)

*Hình 3. Hàng trên là ảnh trước augmentation, hàng dưới là ảnh sau augmentation.*

Validation và test không được tăng cường. Việc augmentation nhẹ phù hợp với dữ liệu đã căn chỉnh: ảnh đã ở cùng vị trí hai mắt, mũi và miệng, nên các phép biến đổi hình học mạnh dễ làm mất thông tin danh tính.

## Phương pháp ArcFace

### Từ softmax đến ArcFace

Softmax dùng logit:

$$
W_j^\top x_i + b_j.
$$

ArcFace biến đổi logit này theo ba bước:

1. Bỏ bias: $b_j = 0$.
2. Chuẩn hóa $\ell_2$ trọng số: $\|W_j\| = 1$.
3. Chuẩn hóa $\ell_2$ đặc trưng rồi đưa về bán kính $s$:

$$
W_j \leftarrow \frac{W_j}{\|W_j\|},
\qquad
x_i \leftarrow \frac{x_i}{\|x_i\|},
\qquad
\cos\theta_j = W_j^\top x_i,
\qquad
\|x_i\| = s.
$$

Khi đó logit chỉ phụ thuộc vào góc $\theta_j$ giữa đặc trưng và vector trọng số của lớp $j$. Paper chọn $s = 64$.

ArcFace chỉ thay logit của lớp đúng:

$$
s\cos(\theta_{y_i}+m),
$$

còn các lớp khác giữ nguyên $s\cos\theta_j$. Hàm mất mát là:

$$
L
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{
e^{s\cos(\theta_{y_i}+m)}
}{
e^{s\cos(\theta_{y_i}+m)}
+
\sum_{j\ne y_i}e^{s\cos\theta_j}
}.
$$

Paper chọn $m = 0{,}5$. Với $\theta \in [0,\pi-m]$, ta có $\cos(\theta+m) < \cos\theta$, nên để đạt cùng xác suất thì mẫu phải nằm gần tâm lớp đúng hơn so với softmax đã chuẩn hóa.

### Ý nghĩa hình học

Trên siêu cầu bán kính $s$, margin góc tương ứng với khoảng cách trắc địa (geodesic) giữa các lớp. Trong bài toán hai lớp, biên quyết định của ArcFace là:

$$
s\left(\cos(\theta_1+m)-\cos\theta_2\right)=0.
$$

So với:

- SphereFace: $\cos(m\theta_1)=\cos\theta_2$, margin nhân vào góc.
- CosineFace: $\cos\theta_1-m=\cos\theta_2$, margin trừ trong không gian cosine.
- ArcFace: $\cos(\theta_1+m)=\cos\theta_2$, margin cộng trong không gian góc.

Vì margin nằm trong không gian góc, nó có diễn giải hình học rõ hơn: khoảng cách trên bề mặt siêu cầu.

### Thí nghiệm của paper

Paper huấn luyện trên VGG2 và MS-Celeb-1M đã làm sạch. Các so sánh loss dùng LResNet100E-IR với MS1M. Kết quả trên LFW, CFP-FP và AgeDB-30:

| Loss | LFW | CFP-FP | AgeDB-30 |
|---|---:|---:|---:|
| Softmax | 99.70 | 91.40 | 95.56 |
| SphereFace ($m=4$) | 99.76 | 93.70 | 97.56 |
| CosineFace ($m=0.35$) | 99.80 | 94.40 | 97.91 |
| ArcFace ($m=0.5$) | 99.83 | 94.04 | 98.08 |

*Bảng 2. Kết quả so sánh loss trong paper (LResNet100E-IR, MS1M).*

Trên MegaFace bản gốc, ArcFace đạt rank-1 $83{,}27\%$ và TAR@FAR=$10^{-6}$ là $98{,}48\%$, so với $78{,}89\%$ và $94{,}95\%$ của softmax. Sau khi làm sạch distractors, rank-1 của ArcFace tăng lên $98{,}36\%$, còn softmax là $91{,}43\%$.

## Thiết lập thí nghiệm

### Bốn cấu hình

| Run | Backbone | Loss | Tham số suy luận |
|---|---|---:|---:|
| resnet50_softmax | ResNet-50 | softmax | 24,56M |
| resnet50_arcface | ResNet-50 | ArcFace | 24,56M |
| mobilenetv3_softmax | MobileNetV3 Large | softmax | 3,47M |
| mobilenetv3_arcface | MobileNetV3 Large | ArcFace | 3,47M |

*Bảng 3. Bốn cấu hình được so sánh.*

Cả bốn run dùng:

- SGD, momentum $0{,}9$, weight decay $5\times10^{-4}$.
- Batch size $128$.
- Learning rate khởi tạo $0{,}001$.
- Tối đa $100$ epoch, dừng sớm với patience $5$ theo ROC-AUC validation.
- ReduceLROnPlateau với factor $0{,}5$, patience $2$.
- FP32, không dùng mixed precision.
- Seed $42$.

### Tập pairs

Mỗi tập validation và test có $10.000$ cặp ảnh. Mỗi cặp được chọn với nhãn dương hoặc âm theo xác suất $1/2$:

- Cặp dương: chọn một danh tính có ít nhất hai ảnh, rồi chọn hai ảnh khác nhau của danh tính đó.
- Cặp âm: chọn hai danh tính khác nhau, rồi chọn một ảnh từ mỗi danh tính.

Số cặp dương và âm xấp xỉ bằng nhau, không đúng chính xác $5.000$ mỗi loại.

### Metric

Chúng tôi báo cáo bốn metric:

- **Accuracy** tại ngưỡng chọn trên validation.
- **TAR@FAR=$10^{-3}$** và **TAR@FAR=$10^{-2}$**.
- **Rank-1** và **Rank-5** trên identification.

ROC-AUC được dùng nội bộ để chọn checkpoint, giảm learning rate và dừng sớm, nhưng không nằm trong bảng kết quả chính.

Accuracy:

$$
\mathrm{Acc}(t)=\frac{TP(t)+TN(t)}{P+N}.
$$

TAR@FAR:

$$
\mathrm{TAR@FAR}
=
\mathrm{TPR}(t)
\quad\text{với}\quad
\mathrm{FPR}(t)=\mathrm{FAR}.
$$

Identification:

$$
\mathrm{Rank@1}
=
\frac{1}{Q}\sum_{q=1}^{Q}\mathbf{1}[r_q=1],
\qquad
\mathrm{Rank@5}
=
\frac{1}{Q}\sum_{q=1}^{Q}\mathbf{1}[r_q\le 5].
$$

Gallery có $997$ ảnh, probe có $18.962$ ảnh. Mỗi danh tính trong gallery có đúng một ảnh, nên mỗi probe có đúng một mục đúng.

## Kết quả

### Verification

| Run | Loss | Tham số suy luận | Acc (%) | TAR@FAR $10^{-3}$ (%) | TAR@FAR $10^{-2}$ (%) |
|---|---|---:|---:|---:|---:|
| mobilenetv3_arcface | ArcFace | 3,47M | 94,00 | 68,45 | 84,42 |
| mobilenetv3_softmax | Softmax | 3,47M | 92,53 | 55,93 | 77,18 |
| resnet50_arcface | ArcFace | 24,56M | 94,10 | 69,09 | 86,14 |
| resnet50_softmax | Softmax | 24,56M | 92,42 | 44,69 | 73,53 |

*Bảng 4. Kết quả verification trên tập test.*

![Đường ROC của verification](/assets/a2/roc_a2.png)

*Hình 4. Đường ROC trên tập test. ArcFace nằm trên softmax ở vùng FPR thấp.*

### Identification

| Run | Rank-1 (%) | Rank-5 (%) |
|---|---:|---:|
| mobilenetv3_arcface | 66,48 | 80,63 |
| mobilenetv3_softmax | 52,05 | 71,05 |
| resnet50_arcface | 68,37 | 82,31 |
| resnet50_softmax | 45,25 | 66,52 |

*Bảng 5. Kết quả identification trên tập test.*

![So sánh accuracy và Rank-1](/assets/a2/bars_a2.png)

*Hình 5. Accuracy và Rank-1 của bốn run.*

### Khoảng tin cậy bootstrap

Chúng tôi dùng paired bootstrap trên $10.000$ cặp test, $2.000$ lần lấy mẫu, seed $42$. Với mỗi lần lấy mẫu, cùng một chỉ số cặp được dùng cho mọi run, nên chênh lệch giữa hai run là ghép cặp.

| So sánh | Metric | Chênh lệch | KTC 95% |
|---|---|---:|---:|
| resnet50_arcface − resnet50_softmax | Accuracy | +1,68 | [+1,19, +2,19] |
| resnet50_arcface − resnet50_softmax | TAR@FAR $10^{-3}$ | +24,40 | [+14,12, +31,09] |
| resnet50_arcface − resnet50_softmax | TAR@FAR $10^{-2}$ | +12,61 | [+8,99, +16,13] |
| mobilenetv3_arcface − mobilenetv3_softmax | Accuracy | +1,47 | [+1,01, +1,94] |
| mobilenetv3_arcface − mobilenetv3_softmax | TAR@FAR $10^{-3}$ | +12,53 | [+4,21, +26,15] |
| mobilenetv3_arcface − mobilenetv3_softmax | TAR@FAR $10^{-2}$ | +7,25 | [+4,47, +9,70] |
| mobilenetv3_arcface − resnet50_arcface | Accuracy | −0,10 | [−0,52, +0,34] |
| mobilenetv3_arcface − resnet50_arcface | TAR@FAR $10^{-3}$ | −0,63 | [−8,71, +6,91] |
| mobilenetv3_arcface − resnet50_arcface | TAR@FAR $10^{-2}$ | −1,72 | [−3,99, +1,43] |
| mobilenetv3_softmax − resnet50_softmax | Accuracy | +0,11 | [−0,40, +0,63] |
| mobilenetv3_softmax − resnet50_softmax | TAR@FAR $10^{-3}$ | +11,24 | [−2,88, +19,88] |
| mobilenetv3_softmax − resnet50_softmax | TAR@FAR $10^{-2}$ | +3,64 | [+1,03, +7,82] |

*Bảng 6. Chênh lệch verification và khoảng tin cậy bootstrap 95%. Đơn vị: điểm phần trăm.*

### Đường cong huấn luyện

![Validation ROC-AUC theo epoch](/assets/a2/curves_a2.png)

*Hình 6. ROC-AUC validation theo epoch. ROC-AUC là metric chọn checkpoint, không phải metric báo cáo chính.*

![Loss huấn luyện theo epoch](/assets/a2/loss_a2.png)

*Hình 7. Loss huấn luyện trung bình mỗi epoch. Softmax và ArcFace có thang khác nhau, nên mỗi loss có một panel riêng.*

## Thảo luận

### ArcFace tốt hơn softmax

Ở cùng backbone và cùng learning rate $0{,}001$, ArcFace tốt hơn softmax trên mọi metric:

- ResNet-50: accuracy tăng $1{,}68$ điểm, TAR@FAR=$10^{-2}$ tăng $12{,}61$ điểm, Rank-1 tăng $23{,}12$ điểm.
- MobileNetV3 Large: accuracy tăng $1{,}47$ điểm, TAR@FAR=$10^{-2}$ tăng $7{,}25$ điểm, Rank-1 tăng $14{,}43$ điểm.

Khoảng tin cậy bootstrap của accuracy và TAR@FAR không chứa $0$ ở cả hai cặp so sánh. Chênh lệch ở identification lớn hơn nhiều so với verification. Điều này hợp lý: verification chỉ yêu cầu phân biệt một cặp đúng/sai, còn identification yêu cầu xếp hạng đúng ảnh trong $997$ ứng viên.

### Backbone không phải yếu tố quyết định trên verification

ResNet-50 và MobileNetV3 Large gần như bằng nhau về verification:

- ArcFace: accuracy $94{,}10\%$ so với $94{,}00\%$, chênh lệch $-0{,}10$ điểm, KTC $[-0{,}52, +0{,}34]$.
- Softmax: accuracy $92{,}42\%$ so với $92{,}53\%$, chênh lệch $+0{,}11$ điểm, KTC $[-0{,}40, +0{,}63]$.

Khoảng tin cậy chứa $0$, nên dữ liệu không cho thấy bằng chứng về một backbone tốt hơn trên verification. Tuy nhiên, MobileNetV3 có $3{,}47$ triệu tham số suy luận so với $24{,}56$ triệu của ResNet-50, tức nhỏ hơn khoảng $7{,}1$ lần.

Ở identification, hướng chênh lệch không nhất quán:

- ArcFace: ResNet-50 cao hơn MobileNetV3 ở Rank-1 ($68{,}37\%$ so với $66{,}48\%$).
- Softmax: MobileNetV3 cao hơn ResNet-50 ở Rank-1 ($52{,}05\%$ so với $45{,}25\%$).

Vì vậy, kết luận là: **MobileNetV3 là lựa chọn tốt về chi phí trong thiết lập này**, nhưng không có bằng chứng rằng nó luôn tốt hơn ResNet-50 ở identification.

### So với paper

Xu hướng chính của paper được tái hiện: ArcFace tốt hơn softmax trên cùng backbone. Paper cũng cho thấy khoảng cách giữa các loss nhỏ trên LFW nhưng lớn hơn trên CFP-FP và AgeDB-30. Ở thí nghiệm của chúng tôi, khoảng cách giữa ArcFace và softmax nhỏ trên verification nhưng lớn trên identification.

Không so sánh trực tiếp các con số tuyệt đối với paper vì:

- Dữ liệu khác: CelebA nhỏ hơn nhiều so với VGG2/MS1M.
- Backbone khác: paper dùng LResNet100E-IR; chúng tôi dùng ResNet-50 và MobileNetV3 Large của torchvision.
- Chế độ huấn luyện khác: paper huấn luyện từ đầu với batch lớn; chúng tôi fine-tune từ ImageNet với batch nhỏ.
- Paper chỉ quét margin trên LMobileNetE, không phải trên các backbone chúng tôi dùng.

## Kết luận

- **ArcFace tốt hơn softmax ở cả hai backbone.** Chênh lệch nhỏ trên verification nhưng lớn trên identification.
- **Backbone không phải yếu tố quyết định trên verification.** MobileNetV3 Large đạt kết quả gần bằng ResNet-50 với số tham số suy luận nhỏ hơn khoảng $7{,}1$ lần.
- **Identification phân biệt rõ hơn verification.** Đây là nơi margin góc thể hiện tác dụng rõ nhất.
- **Kết quả định tính giống paper, nhưng không so sánh trực tiếp số tuyệt đối.** Khác biệt về dữ liệu, backbone, khởi tạo và chế độ huấn luyện khiến các con số không cùng thang đo.
