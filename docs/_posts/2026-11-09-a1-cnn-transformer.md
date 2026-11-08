---
layout: post
title: "A1 CNN và Transformer trên tập ảnh lớn"
author: thien
date: 2026-11-09 11:00
toc: true
math: true
comments: false
---

## Giới thiệu

Bài A1 so sánh một mạng tích chập (CNN) và một Vision Transformer trên cùng một tập ảnh đủ lớn và đủ khó, với nhiều cách dùng trọng số pretrained. Chúng tôi trả lời bốn câu hỏi:

- **CNN hay Transformer tốt hơn?** So sánh ResNet-50 và DeiT-S ở từng chế độ huấn luyện.
- **Pretrain giúp được bao nhiêu?** So sánh train từ đầu với finetune toàn bộ, và so mức tăng đó giữa hai họ.
- **Đóng băng mất bao nhiêu accuracy?** So sánh chỉ train head, đóng băng nửa đầu mạng và finetune toàn bộ.
- **Chi phí là bao nhiêu?** Thời gian, số tham số được học và bộ nhớ GPU của từng cấu hình.

Thông tin chính của bài:

- **Tập dữ liệu:** [Food-101](https://huggingface.co/datasets/ethz/food101), 101 lớp món ăn, 101.000 ảnh.
- **CNN:** ResNet-50 (`microsoft/resnet-50`). **Transformer:** DeiT-S (`facebook/deit-small-patch16-224`). Cả hai có trọng số ImageNet-1k.
- **Bốn chế độ:** scratch, head, partial, full. Hai mô hình nhân bốn chế độ thành 8 run, mỗi run một lần với seed 42.
- **Code:** [thư mục `scripts/a1`, `models/a1` và `configs/a1`](https://github.com/tronglh241/co5085_dl4cv/tree/main/scripts/a1) trên GitHub.
- **Báo cáo PDF:** [assignment_1.pdf](https://github.com/tronglh241/co5085_dl4cv/blob/main/reports/assignment_1.pdf), có đầy đủ phần kiến trúc, siêu tham số và các hạn chế.

Kết quả chính:

- Mô hình tốt nhất là DeiT-S ở chế độ partial với 83,6% accuracy test, sau khoảng 20 phút train. DeiT-S full gần như bằng (83,5%), còn ResNet-50 partial và full đạt 81,7% và 81,8%.
- Pretrain quan trọng hơn nhiều với DeiT-S: full tăng test accuracy từ 47,7% lên 83,5%, trong khi ResNet-50 tăng từ 75,5% lên 81,8%.
- Đóng băng nửa đầu mạng cho test accuracy ngang full nhưng giảm 33% đến 44% thời gian mỗi epoch và 47% đến 67% bộ nhớ GPU. Chỉ train head làm test accuracy giảm từ 81,8% xuống 66,7% ở ResNet-50 và từ 83,5% xuống 66,3% ở DeiT-S.

## Dữ liệu

### Tập dữ liệu

- Food-101 gồm 101 lớp món ăn, mỗi lớp 1.000 ảnh: 750 ảnh train và 250 ảnh test theo split chính thức. Cạnh dài của ảnh luôn là 512 px; ảnh test đã được kiểm tra nhãn thủ công, còn ảnh train cố ý để nguyên nên còn nhiễu.
- Lí do chọn Food-101:
  - Đủ lớn và đủ khó: 101 lớp, nhiều món trông gần giống nhau.
  - Không nằm trong ImageNet, nên so sánh scratch với pretrained không bị lệch như khi dùng một tập con của ImageNet.
  - Ảnh tự nhiên, dùng được ở độ phân giải 224 px; CIFAR-100 ($32 \times 32$) phải phóng to gấp 7 lần nên bị loại.
  - Khoảng 68.000 ảnh train đủ nhỏ để train từ đầu trên một GPU A10G. Bài dùng toàn bộ dữ liệu, không lấy tập con.
- ImageNet-1k có sẵn vài lớp món ăn gần với Food-101, nên trọng số pretrained có lợi thế ngữ nghĩa ở các lớp này.
- Giấy phép: Food-101 không có giấy phép mở. Theo thẻ dữ liệu của nhóm ETH Zürich, ảnh thuộc về người đăng trên Foodspotting và dữ liệu chỉ dùng cho nghiên cứu khoa học. Bài chỉ dùng để học tập và không phân phối lại ảnh.

### Chia dữ liệu

- Split test chính thức (25.250 ảnh) được giữ nguyên. Validation tách từ split train chính thức: 10% mỗi lớp, phân tầng theo nhãn, seed 42.
- Split chỉ thực hiện một lần nên cả tám run dùng chung một tập dữ liệu.
- Test không tham gia quyết định nào: epoch được giữ lại và learning rate đều chọn trên validation, test chỉ được chấm sau khi train xong.

| Tập | Số ảnh | Ảnh mỗi lớp | Nguồn |
|---|---:|---:|---|
| Train | 68.175 | 675 | 90% split train chính thức |
| Validation | 7.575 | 75 | 10% split train chính thức |
| Test | 25.250 | 250 | split test chính thức |

*Bảng 1. Ba tập dữ liệu dùng trong A1.*

### Phân tích khám phá dữ liệu

![Số ảnh mỗi lớp trong train, validation và test](/assets/a1/eda/class_counts.png)

*Hình 1. Số ảnh mỗi lớp. Mọi lớp có đúng 675 ảnh train, 75 ảnh validation và 250 ảnh test, nên không cần trọng số lớp.*

- **Kích thước:** cạnh dài luôn là 512 px; cạnh ngắn từ 122 đến 512 px, trung vị 512 px. Chỉ 44 ảnh có cạnh ngắn dưới 256 px.
- **Chế độ màu:** 100.997 ảnh RGB và 3 ảnh xám, được chuyển sang RGB khi tiền xử lý.
- **Màu sắc:** mean RGB của train là $(0{,}557;\ 0{,}442;\ 0{,}327)$ và std khoảng 0,26, nghiêng về đỏ hơn thống kê ImageNet $(0{,}485;\ 0{,}456;\ 0{,}406)$.
- **Nhiễu nhãn và độ khó:** ảnh chụp trong nhà hàng nên ánh sáng, góc chụp và nền rất khác nhau, và có những lớp chứa nhiều món trong cùng một ảnh. Nhiều món gần nhau như prime rib và steak sẽ được đo bằng các cặp lớp nhầm ở phần kết quả.

![Ảnh train của một số lớp](/assets/a1/eda/samples.png)

*Hình 2. Ảnh train của một số lớp, sau bước cắt cố định.*

## Chuẩn bị dữ liệu huấn luyện

### Tiền xử lý

- Mọi ảnh của cả ba tập được xử lý đúng một lần: chuyển sang RGB, đổi cạnh ngắn về 256 px, cắt giữa $224 \times 224$ và lưu lại dạng JPEG.
- Ảnh được đưa về tensor trong $[0, 1]$ rồi chuẩn hóa bằng mean và std của ImageNet, cho mọi run kể cả scratch, để đầu vào của phần mạng bị đóng băng có cùng thang đo như lúc pretrain.

### Tăng cường dữ liệu

Chỉ tập train được tăng cường, theo thứ tự sau; validation và test chỉ được chuẩn hóa.

| Bước | Tham số |
|---|---|
| Cắt co giãn | Cắt một vùng chiếm 50% đến 100% diện tích ảnh $224 \times 224$, tỉ lệ cạnh từ $3/4$ đến $4/3$, rồi đổi lại về $224 \times 224$ |
| Lật | Lật ngang với xác suất 0,5 |
| Xoay | Ngẫu nhiên trong $\pm 15^\circ$ |
| Màu | Độ sáng, tương phản, độ bão hòa trong $\pm 20\%$; sắc độ trong $\pm 0{,}1$ vòng màu |
| Dịch | Tối đa 10% theo mỗi chiều |
| Chuẩn hóa | Trừ mean và chia std của ImageNet |

*Bảng 2. Pipeline tăng cường dữ liệu của tập train.*

![Minh họa tăng cường dữ liệu](/assets/a1/eda/augmentation.png)

*Hình 3. Cột đầu là ảnh sau bước cắt cố định, sáu cột sau là sáu lần tăng cường ngẫu nhiên của cùng một ảnh.*

### Cấu hình DataLoader

- **Train:** batch 128, xáo trộn mỗi epoch, 8 worker, bỏ batch cuối thiếu (`drop_last`), nên mỗi epoch có 532 batch.
- **Validation, test và đo accuracy trên train:** batch 256, không xáo trộn, 4 worker, giữ batch cuối để đánh giá đủ mọi ảnh.
- Bật `pin_memory` và `persistent_workers`. Dữ liệu được chép sang đĩa cục bộ của máy GPU trước khi train.

## Cài đặt thí nghiệm

### Các mô hình

Cả hai mô hình có cùng một khung: một bộ trích đặc trưng biến ảnh thành một vector, và một lớp phân loại biến vector đó thành 101 điểm số. ResNet-50 (23,7 triệu tham số) quét ảnh bằng các filter nhỏ và thu nhỏ ảnh dần qua từng stage. DeiT-S (21,7 triệu tham số) cắt ảnh thành các patch $16 \times 16$ rồi cho mọi patch trao đổi thông tin bằng self-attention.

### Các chế độ huấn luyện

| Chế độ | Khởi tạo | Phần được học | ResNet-50 học | DeiT-S học |
|---|---|---|---:|---:|
| scratch | ngẫu nhiên | toàn bộ mạng | 23.714.981 (100%) | 21.704.549 (100%) |
| head | pretrained | chỉ lớp phân loại | 206.949 (0,9%) | 38.885 (0,2%) |
| partial | pretrained | nửa sau của mạng | 22.270.053 (93,9%) | 10.686.437 (49,2%) |
| full | pretrained | toàn bộ mạng | 23.714.981 (100%) | 21.704.549 (100%) |

*Bảng 3. Bốn chế độ huấn luyện và số tham số được học.*

- **Partial** đóng băng nửa đầu mạng: stem, stage 1 và stage 2 với ResNet-50; patch embedding, token CLS, mã hóa vị trí và 6 block đầu với DeiT-S. ResNet-50 vẫn học 93,9% tham số vì phần lớn tham số của nó nằm ở hai stage cuối.

### Siêu tham số

Cả 8 run dùng chung một công thức:

- Tối ưu hóa bằng AdamW, weight decay 0,05 (không áp lên bias, tham số chuẩn hóa, token CLS và mã hóa vị trí).
- Cross-entropy với label smoothing 0,2.
- Warmup tuyến tính rồi cosine giảm về 0, cập nhật sau mỗi iteration, tính cho 100 epoch.
- Batch 128, bf16, seed 42.
- Dropout trên đặc trưng đã gộp, ngay trước lớp phân loại.
- Dừng sớm theo validation accuracy với patience 10; checkpoint giữ lại là epoch có validation accuracy cao nhất.

Riêng learning rate, warmup và dropout đổi theo chế độ:

| Chế độ | Learning rate | Warmup (epoch) | Dropout |
|---|---:|---:|---:|
| scratch | $5 \times 10^{-4}$ | 5 | 0,4 |
| head | $1 \times 10^{-3}$ | 1 | 0,2 |
| partial | $2 \times 10^{-4}$ | 1 | 0,3 |
| full | $1 \times 10^{-4}$ | 1 | 0,4 |

*Bảng 4. Siêu tham số theo chế độ, giống nhau ở ResNet-50 và DeiT-S.*

Lý do các tham số khác nhau giữa các chế độ:

- **Learning rate** giảm dần theo số trọng số pretrained được cập nhật: head chỉ học một lớp tuyến tính khởi tạo ngẫu nhiên nên chịu được bước lớn; partial và full cập nhật cả các trọng số đã học tốt từ ImageNet nên cần bước nhỏ để không phá chúng. Scratch khởi tạo ngẫu nhiên toàn bộ nên dùng giá trị lớn hơn.
- **Warmup** dài hơn ở scratch (5 epoch so với 1): khi mọi trọng số đều ngẫu nhiên, vài bước đầu với learning rate lớn dễ làm huấn luyện bất ổn, nhất là với Transformer.
- **Dropout** tăng theo số trọng số được học: 0,2 ở head, 0,3 ở partial, 0,4 ở full và scratch.

### Phương pháp đánh giá

- **Chọn checkpoint:** giữ epoch có validation accuracy cao nhất, rồi chấm checkpoint đó một lần trên test. Test không dùng cho bất kỳ quyết định nào.
- **Chất lượng:** test accuracy là chỉ số chính; kèm macro precision, recall, F1 và accuracy từng lớp.
- **Chi phí:** số tham số được học, thời gian mỗi epoch, bộ nhớ GPU đỉnh và tốc độ suy luận.
- **Phân tích lỗi:** confusion matrix, các cặp lớp bị nhầm nhiều nhất và các ảnh sai với độ tự tin cao, trên mô hình CNN và Transformer tốt nhất theo validation.
- **Các so sánh:** DeiT-S với ResNet-50 trong từng chế độ; full với scratch (pretrain giúp bao nhiêu); head và partial với full (mất bao nhiêu khi đóng băng); chi phí đặt cạnh accuracy.

Cách chạy lại thí nghiệm: làm theo [SETUP.md](https://github.com/tronglh241/co5085_dl4cv/blob/main/SETUP.md) để cài môi trường, sau đó mở shell trong container và chạy:

```bash
python scripts/a1/prepare_data.py --raw data/ --out data/food101_256
python scripts/a1/make_configs.py
python scripts/train.py configs/a1/train_resnet50_full.yml
python scripts/a1/analyze.py
```

Thay `train_resnet50_full.yml` bằng cấu hình tương ứng trong `configs/a1` cho bảy run còn lại. `analyze.py` đánh giá checkpoint tốt nhất của từng run trên test rồi sinh bảng và hình.

## Kết quả

### Accuracy và chi phí

| Mô hình | Chế độ | Tham số học | Val (%) | Test (%) | F1 (%) | Giây/epoch | GB | Ảnh/giây |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | scratch | 23,71 tr | 70,7 | 75,5 | 75,4 | 99 | 5,7 | 2.282 |
| | head | 0,21 tr | 62,7 | 66,7 | 66,4 | **46** | 0,9 | 2.279 |
| | partial | 22,27 tr | 77,1 | 81,7 | 81,7 | 55 | 1,9 | 2.279 |
| | full | 23,71 tr | 77,5 | 81,8 | 81,8 | 98 | 5,7 | 2.281 |
| DeiT-S | scratch | 21,70 tr | 43,6 | 47,7 | 47,5 | 80 | 4,9 | 2.856 |
| | head | 0,04 tr | 62,5 | 66,3 | 65,9 | 48 | **0,5** | **2.864** |
| | partial | 10,69 tr | **79,5** | **83,6** | **83,6** | 56 | 2,6 | 2.858 |
| | full | 21,70 tr | 79,4 | 83,5 | **83,6** | 84 | 4,9 | 2.859 |

*Bảng 5. Kết quả của 8 run (tr = triệu tham số, GB = bộ nhớ GPU đỉnh lúc train, tốc độ suy luận đo trên GPU A10G). In đậm: giá trị tốt nhất của mỗi cột. Test accuracy cao hơn validation accuracy ở cả 8 run.*

### Đường cong học

![Validation accuracy theo epoch của tám run](/assets/a1/curves_a1.png)

*Hình 4. Validation accuracy theo epoch. Mỗi đường kết thúc ở epoch run dừng do dừng sớm; không run nào đạt 100 epoch.*

| Mô hình | Chế độ | Epoch tốt nhất | Epoch dừng | Train (%) | Val (%) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | scratch | 85 | 97 | 99,8 | 70,7 |
| | head | 57 | 68 | 66,3 | 62,7 |
| | partial | 19 | 30 | 92,3 | 77,1 |
| | full | 43 | 54 | 94,7 | 77,5 |
| DeiT-S | scratch | 30 | 41 | 67,4 | 43,6 |
| | head | 9 | 20 | 63,2 | 62,5 |
| | partial | 10 | 21 | 93,3 | 79,5 |
| | full | 15 | 26 | 95,6 | 79,4 |

*Bảng 6. Epoch tốt nhất, epoch dừng, và accuracy trên train với validation tại epoch tốt nhất. Accuracy trên train đo trên 13.635 ảnh (20% mỗi lớp) không tăng cường.*

### So sánh giữa các chế độ

| Nội dung so sánh | Chuyển chế độ | ResNet-50 | DeiT-S |
|---|---|---|---|
| Pretrain giúp được bao nhiêu | scratch → full | 75,5% → 81,8% (+6,3) | 47,7% → 83,5% (+35,8) |
| Chỉ train head mất bao nhiêu | full → head | 81,8% → 66,7% (−15,1) | 83,5% → 66,3% (−17,2) |
| Đóng băng nửa đầu mất bao nhiêu | full → partial | 81,8% → 81,7% (−0,1) | 83,5% → 83,6% (+0,1) |

*Bảng 7. Test accuracy khi chuyển từ chế độ này sang chế độ khác; trong ngoặc là mức thay đổi, tính bằng điểm phần trăm, từ các số ở Bảng 5.*

### Chi phí

![Test accuracy theo thời gian train](/assets/a1/cost_a1.png)

*Hình 5. Test accuracy theo tổng thời gian train (phút, GPU A10G). Thời gian này gộp chi phí mỗi epoch với số epoch mà dừng sớm cho phép.*

### Phân tích lỗi

Phân tích lỗi dùng hai mô hình có validation accuracy cao nhất của mỗi họ: ResNet-50 full và DeiT-S partial.

![Mười lăm cặp lớp bị nhầm nhiều nhất của ResNet-50 full](/assets/a1/confused_resnet50_full.png)

*Hình 6. Mười lăm cặp lớp bị nhầm nhiều nhất của ResNet-50 full (nhãn thật → nhãn đoán).*

![Mười lăm cặp lớp bị nhầm nhiều nhất của DeiT-S partial](/assets/a1/confused_deit_s_partial.png)

*Hình 7. Mười lăm cặp lớp bị nhầm nhiều nhất của DeiT-S partial.*

- Hai mô hình nhầm các món có hình thức rất giống nhau: filet mignon và steak (hai chiều), chocolate cake và chocolate mousse, pork chop sang steak, beef tartare và tuna tartare.
- Cặp tartare lệch về một phía: DeiT-S đoán nhầm tuna tartare thành beef tartare ở 60 ảnh, tức 24% số ảnh test của lớp, còn ResNet-50 đoán nhầm chiều ngược lại ở 27 ảnh.
- Lớp dễ nhất với cả hai mô hình là edamame (99,6% ở ResNet-50 full, 100% ở DeiT-S partial); tiếp theo là pad thai và macarons ở ResNet-50, macarons và pho ở DeiT-S. Lớp khó nhất ở ResNet-50 full là steak (51,6%), pork chop (56,0%) và bread pudding (58,0%); ở DeiT-S partial là pork chop (54,0%), huevos rancheros (55,6%) và tuna tartare (57,2%).
- DeiT-S partial hơn ResNet-50 full nhiều nhất ở beef tartare, steak và risotto; ResNet-50 full hơn nhiều nhất ở tuna tartare và ceviche.

![Ảnh test bị DeiT-S partial đoán sai với độ tự tin cao nhất](/assets/a1/errors_deit_s_partial.png)

*Hình 8. 18 ảnh test bị DeiT-S partial đoán sai với độ tự tin cao nhất (nhãn thật → nhãn đoán, xác suất trong ngoặc).*

- Một phần các ảnh này có nhãn đáng ngờ hoặc chứa nhiều món (ví dụ ảnh gắn nhãn eggs benedict, huevos rancheros hay ceviche mà không giống tên món), một phần là lỗi thật của mô hình (ví dụ hamburger đoán thành falafel).

## Phân tích bổ sung

Phần này trả lời bốn câu hỏi về cả 8 run mà accuracy không trả lời được. Không phân tích nào được dùng để chọn mô hình hay siêu tham số.

### Xác suất của mô hình có đáng tin không

Một mô hình được calibrate tốt nếu trong các ảnh nó đoán với xác suất khoảng 0,8 thì khoảng 80% là đúng. Chúng tôi đo bằng ECE (chia độ tự tin thành 15 khoảng, lấy trung bình có trọng số của độ lệch giữa accuracy và độ tự tin của từng khoảng), cùng với AUROC khi dùng độ tự tin để tách ảnh đoán đúng khỏi ảnh đoán sai.

| Mô hình | Chế độ | Acc (%) | Độ tự tin TB (%) | ECE (%) | AUROC | Acc trên 80% ảnh chắc nhất (%) |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | scratch | 75,5 | 59,0 | 16,5 | 0,867 | 85,7 |
| | head | 66,7 | 40,7 | 26,0 | 0,834 | 76,2 |
| | partial | 81,7 | 68,3 | 13,5 | 0,872 | 91,5 |
| | full | 81,8 | 67,7 | 14,1 | 0,873 | 91,7 |
| DeiT-S | scratch | 47,7 | 45,1 | 2,6 | 0,811 | 55,6 |
| | head | 66,3 | 42,0 | 24,3 | 0,835 | 75,7 |
| | partial | 83,6 | 71,0 | 12,5 | 0,856 | 93,1 |
| | full | 83,5 | 71,8 | 11,8 | 0,861 | 93,3 |

*Bảng 8. Calibration trên 25.250 ảnh test.*

![Biểu đồ độ tin cậy của 8 run](/assets/a1/reliability_a1.png)

*Hình 9. Biểu đồ độ tin cậy: điểm nằm trên đường chéo nghĩa là mô hình thiếu tự tin. Đường chấm đứng là 0,802.*

- **Các mô hình thiếu tự tin một cách có hệ thống:** độ tự tin trung bình thấp hơn accuracy 12 đến 26 điểm ở bảy trên tám run. Nguyên nhân dễ thấy nhất là label smoothing 0,2: với 101 lớp, hàm mất mát nhỏ nhất khi lớp đúng nhận xác suất khoảng 0,802, nên mô hình được dạy là không tự tin quá mức đó, kể cả với ảnh dễ.
- **Chế độ head lệch nhiều nhất** (ECE 24 đến 26%): độ tự tin trung bình chỉ khoảng 41% trong khi accuracy khoảng 66%.
- **DeiT-S scratch gần như calibrate tốt** (ECE 2,6%), nhưng đó là vì nó khớp kém ngay cả tập train nên độ tự tin vốn đã thấp, không phải ưu điểm.
- **Độ tự tin vẫn xếp hạng tốt:** AUROC 0,81 đến 0,87. Nếu bỏ qua 20% ảnh kém chắc chắn nhất, DeiT-S full đạt 93,3% trên phần còn lại. Temperature scaling học trên validation có thể sửa độ lệch này; chúng tôi không làm vì tệp dự đoán chỉ lưu xác suất cao nhất, không lưu logit.

### CNN và Transformer có sai cùng chỗ không

| Chế độ | Cả hai đúng | Chỉ ResNet đúng | Chỉ DeiT đúng | Cả hai sai | Jaccard lỗi | Sai cùng lớp |
|---|---:|---:|---:|---:|---:|---:|
| scratch | 43,6 | 32,0 | 4,2 | 20,3 | 0,36 | 16,8 |
| head | 56,1 | 10,6 | 10,2 | 23,1 | 0,53 | 38,3 |
| partial | 75,4 | 6,4 | 8,2 | 10,1 | 0,41 | 41,7 |
| full | 75,5 | 6,3 | 8,1 | 10,1 | 0,41 | 41,8 |

*Bảng 9. ResNet-50 so với DeiT-S ở cùng chế độ trên 25.250 ảnh test (%, trừ cột Jaccard). Jaccard lỗi: số ảnh cả hai cùng sai chia cho số ảnh ít nhất một mô hình sai.*

![Mức trùng lỗi giữa 8 run](/assets/a1/error_overlap_a1.png)

*Hình 10. Jaccard của tập ảnh bị đoán sai giữa mọi cặp run.*

- **Hai họ bổ trợ nhau:** ở partial và full chỉ khoảng 10% ảnh bị cả hai cùng sai. Jaccard lỗi giữa hai họ là 0,41, thấp hơn giữa partial và full của cùng một họ (0,61 ở ResNet-50, 0,45 ở DeiT-S).
- **Khi cùng sai, hai mô hình thường sai giống nhau** (khoảng 42% đoán cùng một nhãn sai), phù hợp với các cặp món giống nhau và nhãn đáng ngờ ở phần phân tích lỗi.
- **Chế độ head có lỗi trùng nhiều nhất giữa hai họ** (Jaccard 0,53): cả hai cùng đọc đặc trưng ImageNet bằng một lớp tuyến tính, nên ảnh khó với đặc trưng đó thì khó với cả hai.

![Các lớp nghiêng về mỗi họ](/assets/a1/class_diff_a1.png)

*Hình 11. Các lớp nghiêng nhiều nhất về ResNet-50 (xanh) và DeiT-S (cam) ở chế độ full.*

Ở chế độ full, DeiT-S hơn ở 63 lớp và thua ở 32 lớp, trong đó nghiêng về DeiT-S nhiều nhất là risotto (+12,4 điểm) và pork chop (+11,2), nghiêng về ResNet-50 nhiều nhất là samosa (+10,4). Tuy vậy, chênh lệch theo lớp ở partial và full chỉ tương quan 0,15: lớp nào nghiêng về họ nào thay đổi nhiều giữa hai lần train gần giống nhau. Vì vậy chúng tôi không rút ra quy luật kiểu "Transformer giỏi món có hình dạng rõ, CNN giỏi món có kết cấu" từ các lớp này.

### Mô hình nhìn vào đâu

- **ResNet-50: Grad-CAM.** Đánh trọng số 2048 bản đồ đặc trưng 7×7 của stage cuối bằng gradient trung bình của điểm số lớp được đoán, cộng lại và giữ phần dương.
- **DeiT-S: attention rollout.** Lấy trung bình attention của 6 head ở mỗi block, trộn một nửa với ma trận đơn vị (cho kết nối tắt), nhân qua 12 block, rồi đọc hàng của class token trên 196 patch. Bản đồ này không phụ thuộc lớp được đoán.

Hai hình dưới dùng chung 8 ảnh test, hai ảnh từ mỗi nhóm: cả hai mô hình tốt nhất đều đúng, chỉ ResNet-50 đúng, chỉ DeiT-S đúng, cả hai sai. Mỗi cột là một chế độ.

![Grad-CAM của ResNet-50](/assets/a1/explain_resnet50.png)

*Hình 12. Grad-CAM của bốn chế độ ResNet-50. Tiêu đề đỏ là đoán sai.*

![Attention rollout của DeiT-S](/assets/a1/explain_deit_s.png)

*Hình 13. Attention rollout của bốn chế độ DeiT-S trên cùng 8 ảnh.*

Trên 8 ảnh này (quan sát bằng mắt trên mẫu nhỏ, không phải số liệu đo):

- ResNet-50 pretrained nhìn vào thân món ăn ở partial và full; ở chế độ head vùng nóng tản ra nhiều chỗ, đôi khi rơi vào đĩa hoặc góc ảnh.
- Đoán sai không có nghĩa là nhìn sai chỗ: ở ảnh ceviche, ResNet-50 full đoán greek salad nhưng vùng nóng vẫn nằm trên phần salad.
- Bản đồ rollout của DeiT-S scratch mờ và phẳng, chỉ có vài đốm sáng ở góc hoặc rìa ảnh, phù hợp với việc mô hình này chưa học được cấu trúc không gian.
- DeiT-S head, partial và full cho bản đồ gần giống nhau trên cùng một ảnh, vì dùng chung trọng số ImageNet ở nhiều block đầu.

Nhìn bản đồ bằng mắt dễ bị thiên kiến, nên chúng tôi kiểm tra thêm bằng phép xoá trên 2.000 ảnh test: chia ảnh thành lưới 14×14 ô 16 px, thay các ô mà bản đồ xếp cao nhất bằng màu trung bình, rồi đo lại accuracy; làm tương tự với các ô ngẫu nhiên. Nếu bản đồ chỉ đúng phần mô hình dùng, xoá ô nổi bật phải làm accuracy giảm nhiều hơn xoá ngẫu nhiên.

![Phép xoá](/assets/a1/deletion_a1.png)

*Hình 14. Accuracy theo tỉ lệ ô bị xoá. Nét liền: xoá ô nổi bật nhất; nét đứt: xoá ngẫu nhiên.*

### Độ bền với ảnh bị biến đổi

Chúng tôi đánh giá lại 8 run trên 5.050 ảnh test (50 ảnh mỗi lớp) bị làm xấu theo năm cách, mỗi cách ba mức: nhiễu Gauss, làm mờ, nén JPEG, chuyển ảnh xám, và xáo ô (cắt ảnh thành lưới 2×2, 4×4 hoặc 7×7 rồi xáo vị trí các ô, giữ kết cấu cục bộ nhưng phá hình dạng tổng thể). Kích thước ô là bội của patch 16 px nên phép xáo không cắt ngang token nào của DeiT-S. Mỗi ảnh bị biến đổi theo một cách cố định, nên 8 run thấy đúng cùng các điểm ảnh.

![Độ bền với ảnh bị biến đổi](/assets/a1/robustness_a1.png)

*Hình 15. Accuracy theo mức biến đổi. Nét liền: ResNet-50; nét đứt: DeiT-S; màu là chế độ huấn luyện.*

| Mô hình | Chế độ | Sạch | Nhiễu | Mờ | JPEG | Xám | Xáo ô | TB biến đổi | Tương đối |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | scratch | 75,6 | 41,8 | 45,4 | 44,3 | 18,5 | 40,4 | 38,1 | 50% |
| | head | 66,8 | 50,2 | 49,2 | 42,4 | 44,4 | 52,2 | 47,7 | 71% |
| | partial | 81,8 | 38,6 | 62,2 | 47,9 | 49,9 | 59,9 | 51,7 | 63% |
| | full | 81,6 | 43,4 | 61,3 | 42,2 | 46,1 | 61,2 | 50,8 | 62% |
| DeiT-S | scratch | 47,3 | 34,8 | 32,3 | 26,0 | 4,0 | 43,8 | 28,2 | 60% |
| | head | 65,8 | 55,0 | 52,8 | 42,3 | 44,1 | 51,5 | 49,1 | 75% |
| | partial | 83,7 | 74,0 | 69,0 | 54,7 | 56,4 | 69,1 | 64,6 | 77% |
| | full | 83,3 | 65,2 | 67,0 | 54,5 | 45,7 | 65,4 | 59,5 | 72% |

*Bảng 10. Accuracy (%) trên 5.050 ảnh test, trung bình qua các mức của mỗi loại biến đổi. Tương đối: accuracy trung bình khi bị biến đổi chia cho accuracy sạch.*

- **Grad-CAM của ResNet-50 pretrained chỉ đúng phần mô hình dùng:** xoá 25% ô nổi bật làm ResNet-50 full rơi xuống 40,0%, xoá 25% ô ngẫu nhiên chỉ xuống 70,8%. Attention rollout của DeiT-S chỉ trung thực một phần (full: 71,2% so với 77,7%), và ở chế độ head gần như không khác ngẫu nhiên.
- **DeiT-S chịu che ô tốt hơn hẳn:** xoá ngẫu nhiên 50% ô, DeiT-S partial còn 72,4% còn ResNet-50 partial chỉ 50,1%. Kết luận này không phụ thuộc phương pháp giải thích vì ô bị xoá được chọn ngẫu nhiên.
- **DeiT-S pretrained bền hơn ResNet-50 ở hầu hết các biến đổi** (ngoại lệ: ảnh xám ở chế độ full, hai mô hình ngang nhau): DeiT-S partial giữ 77% accuracy sạch, ResNet-50 partial 63%. Với nhiễu mạnh nhất, DeiT-S partial còn 63,6% còn ResNet-50 partial chỉ 8,3%.
- **Đóng băng một phần giúp DeiT-S bền hơn:** partial giữ 64,6% còn full giữ 59,5% trung bình dù accuracy sạch như nhau; ở ResNet-50 hai chế độ ngang nhau.
- **Mô hình train từ đầu phụ thuộc nhiều vào màu:** bỏ màu làm ResNet-50 scratch rơi xuống 18,5% và DeiT-S scratch xuống 4,0%, còn các mô hình pretrained giữ được 44 đến 56%.
- **DeiT-S scratch dựa vào kết cấu cục bộ:** xáo lưới 7×7 gần như không ảnh hưởng đến nó (46,3% so với 47,3% trên ảnh sạch), trong khi ResNet-50 scratch rơi từ 75,6% xuống 17,0%.

## Thảo luận

### DeiT-S với ResNet-50

- **Có pretrain và finetune:** DeiT-S đạt 83,5% ở full và 83,6% ở partial, cao hơn ResNet-50 (81,8% và 81,7%). Hai checkpoint pretrain được train bằng hai công thức khác nhau, nên phần chênh này gồm cả khác biệt về bộ trọng số, không chỉ kiến trúc.
- **Chỉ train head:** DeiT-S đạt 66,3% và ResNet-50 đạt 66,7%, dù DeiT-S chỉ học 38.885 tham số còn ResNet-50 học 206.949.
- **Train từ đầu:** ResNet-50 đạt 75,5% còn DeiT-S chỉ đạt 47,7%.

### Pretrain giúp bao nhiêu

- Full tăng test accuracy từ 47,7% lên 83,5% ở DeiT-S và từ 75,5% lên 81,8% ở ResNet-50. Validation accuracy của DeiT-S scratch đi ngang quanh 41% đến 43% từ epoch 25, trong khi ResNet-50 scratch còn tăng chậm đến epoch 85.
- Một cách giải thích thường gặp: phép tích chập vốn chỉ nhìn vùng ảnh lân cận và dùng cùng một filter ở mọi vị trí, nên ResNet-50 đã có sẵn hai giả định hợp với ảnh (đặc trưng nằm ở vùng gần nhau, và một vật ở đâu trong ảnh cũng nhận ra như nhau). DeiT-S không có hai giả định này mà phải học từ dữ liệu, nên cần nhiều ảnh hơn; 68.175 ảnh là ít đối với một Transformer. Bài không có thí nghiệm riêng cho cách giải thích này.

### Đóng băng

- **Head so với full:** chỉ train head làm test accuracy giảm từ 81,8% xuống 66,7% ở ResNet-50 và từ 83,5% xuống 66,3% ở DeiT-S. Giữ nguyên đặc trưng ImageNet và chỉ học lớp phân loại chưa đủ cho 101 món.
- **Partial so với full:** gần như ngang nhau (81,7% so với 81,8% ở ResNet-50, 83,6% so với 83,5% ở DeiT-S), nhưng rẻ hơn rõ: thời gian mỗi epoch giảm từ 98 s xuống 55 s (ResNet-50) và từ 84 s xuống 56 s (DeiT-S); bộ nhớ GPU giảm từ 5,7 GB xuống 1,9 GB và từ 4,9 GB xuống 2,6 GB.
- **Train và validation:** ở head, train và validation đều khoảng 62% đến 67%, nên giới hạn là khả năng biểu diễn của một lớp tuyến tính; ở partial và full, train đạt 92% đến 96% còn validation 77% đến 80%. ResNet-50 scratch khớp gần hết tập train (99,8%) mà validation chỉ 70,7%, còn DeiT-S scratch chưa khớp tốt ngay cả tập train (67,4%).

## Kết luận

- **DeiT-S tốt hơn ResNet-50 khi finetune từ ImageNet, còn ResNet-50 tốt hơn khi train từ đầu.** DeiT-S đạt 83,5% đến 83,6% test accuracy, ResNet-50 đạt 81,7% đến 81,8%; khi train từ đầu, ResNet-50 đạt 75,5% còn DeiT-S 47,7%.
- **Transformer cần pretrain trên tập cỡ này; CNN cũng được lợi nhưng ít hơn.** Full tăng test accuracy từ 47,7% lên 83,5% ở DeiT-S và từ 75,5% lên 81,8% ở ResNet-50.
- **Đóng băng nửa đầu mạng cho accuracy ngang finetune toàn bộ mà rẻ hơn; chỉ train head chỉ hợp khi thiếu tài nguyên.** Partial giảm 33% đến 44% thời gian mỗi epoch và 47% đến 67% bộ nhớ GPU so với full.
- **Hai mô hình chủ yếu nhầm các món trông giống nhau, và sai ở những ảnh khá khác nhau** (Jaccard lỗi 0,41).

## Hạn chế

- **Mỗi cấu hình chỉ train một lần.** Kết quả ứng với một seed duy nhất, nên các lần chạy khác có thể cho số hơi lệch.
- **Công thức huấn luyện chưa được tối ưu cho từng mô hình.** Hai họ dùng chung một công thức và siêu tham số được dò rất ít, nên kết quả, nhất là ở chế độ scratch, phản ánh cả công thức chứ không chỉ kiến trúc.
- **Mỗi họ chỉ có một mô hình.** Kết luận về CNN và Transformer chỉ chắc cho ResNet-50 và DeiT-S.
- **Chỉ một tập dữ liệu và một nguồn pretrain.** Food-101 có các lớp gần với ImageNet, nên lợi ích của pretrain có thể khác trên miền dữ liệu khác.
- **Các phân tích bổ sung ở quy mô nhỏ.** Phần giải thích, độ bền và calibration đo trên một tập con ảnh test hoặc dùng phương pháp xấp xỉ, nên mang tính tham khảo.
- **Chi phí chỉ đo trên một cấu hình phần cứng.** Thời gian và bộ nhớ đo trên một GPU A10G với bf16; ResNet-50 scratch được chạy làm hai đợt nên thời gian của nó là tổng hai đợt.
