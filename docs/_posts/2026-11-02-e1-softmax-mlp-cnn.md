---
layout: post
title: "E1 Phân loại ảnh: Softmax, MLP, CNN"
author: trong
date: 2026-11-02 10:00
toc: true
math: true
comments: false
---

## Giới thiệu

Bài E1 so sánh ba bộ phân loại trên MNIST:

- **Softmax:** bộ phân loại tuyến tính trên ảnh dạng vector phẳng.
- **MLP:** thêm các tầng ẩn phi tuyến nhưng vẫn coi ảnh là vector phẳng.
- **CNN:** nhận trực tiếp ảnh 2D, khai thác tính cục bộ và chia sẻ trọng số.

Mỗi mô hình được huấn luyện ở hai thiết lập: không tăng cường dữ liệu (base) và có tăng cường dữ liệu (aug).

Kết quả chính:

- CNN aug đạt accuracy kiểm thử cao nhất, 99,50%.
- CNN có ít tham số hơn MLP nhưng mắc khoảng một nửa số lỗi.
- Tăng cường dữ liệu giúp MLP và CNN, nhưng làm Softmax kém đi.

## Dữ liệu

### Tập dữ liệu

- MNIST gồm 70.000 ảnh chữ số viết tay, ảnh xám kích thước $28 \times 28$, thuộc 10 lớp từ 0 đến 9.
- Dữ liệu gốc chia sẵn thành 60.000 ảnh huấn luyện và 10.000 ảnh kiểm thử.
- Lí do chọn MNIST:
  - Là bộ dữ liệu tiêu chuẩn, phổ biến nên dễ đối chiếu kết quả.
  - Ảnh nhỏ, số mẫu vừa phải nên huấn luyện nhanh.
  - Các lớp tương đối cân bằng nên accuracy phản ánh đúng hiệu suất tổng thể.

![Một số ảnh trong MNIST](/assets/shared/mnist_samples.png)

*Hình 1. Một số ảnh mẫu của MNIST.*

### Chia dữ liệu

- 60.000 ảnh huấn luyện gốc được chia theo phương pháp phân tầng thành 50.000 ảnh huấn luyện và 10.000 ảnh kiểm định.
- Việc chia chỉ thực hiện một lần, nên mọi mô hình trong E1 đều dùng chung một tập dữ liệu.
- 10.000 ảnh kiểm thử chính thức được giữ nguyên và không tham gia huấn luyện hay chọn mô hình.

| Lớp | Huấn luyện | Kiểm định | Kiểm thử |
|---:|---:|---:|---:|
| 0 | 4.936 | 987 | 980 |
| 1 | 5.618 | 1.124 | 1.135 |
| 2 | 4.965 | 993 | 1.032 |
| 3 | 5.109 | 1.022 | 1.010 |
| 4 | 4.868 | 974 | 982 |
| 5 | 4.518 | 903 | 892 |
| 6 | 4.932 | 986 | 958 |
| 7 | 5.221 | 1.044 | 1.028 |
| 8 | 4.876 | 975 | 974 |
| 9 | 4.957 | 992 | 1.009 |
| **Tổng** | **50.000** | **10.000** | **10.000** |

*Bảng 1. Số mẫu từng lớp sau khi chia dữ liệu.*

![Phân bố số mẫu theo lớp](/assets/shared/class_balance.png)

*Hình 2. Phân bố số mẫu theo lớp trong ba tập, các lớp khá đồng đều.*

### Tiền xử lý và tăng cường dữ liệu

- Mỗi ảnh được chuyển thành tensor trong khoảng $[0, 1]$ rồi chuẩn hóa, với trung bình và độ lệch chuẩn tính trên 50.000 ảnh huấn luyện:

$$
x' = \frac{x - 0.1308}{0.3082}
$$

- Ở thiết lập aug, ảnh huấn luyện được biến đổi bằng `RandomAffine(10, translate=(0.1, 0.1))`:
  - Xoay ngẫu nhiên tối đa $\pm 10^\circ$.
  - Dịch ngẫu nhiên tối đa 10% mỗi chiều, tương đương khoảng 2,8 điểm ảnh.
  - Tham số được sinh lại mỗi lần nạp ảnh, nên mỗi epoch mô hình thấy một biến thể khác của cùng một mẫu.
- Tăng cường chỉ áp dụng cho các batch dùng để cập nhật trọng số. Mọi chỉ số huấn luyện, kiểm định và kiểm thử đều đo trên ảnh gốc.

![Ảnh gốc và các phiên bản sau khi tăng cường dữ liệu](/assets/shared/mnist_augmentation.png)

*Hình 3. Ảnh gốc và các biến thể sau tăng cường.*

### Phương pháp đánh giá

- Vai trò của ba tập dữ liệu:
  - **Huấn luyện:** cập nhật trọng số.
  - **Kiểm định:** điều chỉnh learning rate, dừng sớm và chọn checkpoint.
  - **Kiểm thử:** chỉ dùng một lần ở cuối, trên checkpoint đã chọn.
- Độ đo chính là accuracy, tức tỷ lệ mẫu được phân loại đúng:

$$
\mathrm{accuracy} = \frac{\sum_{c=1}^{K} C_{cc}}{\sum_{i=1}^{K}\sum_{j=1}^{K} C_{ij}},
$$

  với $K = 10$ và $C_{ij}$ là số mẫu thuộc lớp thật $i$ được dự đoán thành lớp $j$.
- Ngoài accuracy, bài phân tích tổng số ảnh bị phân loại sai, ma trận nhầm lẫn và các mẫu lỗi cụ thể.

## Mô hình

Hình 4 minh họa kiến trúc của ba mô hình. Cả ba đều xuất logits 10 chiều.

- **Softmax:** một tầng Linear $784 \to 10$.
- **MLP:** hai tầng ẩn $784 \to 256 \to 128 \to 10$ với ReLU.
- **CNN:** hai khối Conv $3 \times 3$ + ReLU + MaxPool $2 \times 2$, sau đó là Linear $1568 \to 128 \to 10$.

![Sơ đồ kiến trúc ba mô hình E1](/assets/e1/models_e1.png)

*Hình 4. Kiến trúc Softmax, MLP và CNN.*

## Thiết lập huấn luyện

Sáu thí nghiệm (ba mô hình, mỗi mô hình hai thiết lập) dùng chung một quy trình:

- Tối ưu hóa bằng Adam với learning rate $10^{-3}$.
- Batch size 64.
- Hàm mất mát cross-entropy.
- Tối đa 1000 epoch.
- Giảm learning rate 10 lần sau 10 epoch không cải thiện loss kiểm định.
- Dừng sớm sau 20 epoch không cải thiện loss kiểm định.
- Chọn checkpoint có loss kiểm định nhỏ nhất, và báo cáo accuracy kiểm thử tại checkpoint này.
- Toàn bộ thông số của mỗi thí nghiệm nằm trong một tệp cấu hình YAML riêng, và log huấn luyện được ghi bằng TensorBoard.

Vòng lặp huấn luyện do nhóm tự viết, gồm lần lượt các bước forward, tính loss, backward và cập nhật tham số, mà không dùng thư viện huấn luyện có sẵn:

```python
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        x, y = self.prepare_batch(batch, self.device)

        with self.autocast():
            loss = self.loss_fn(self.model(self.to_input(x)), y)

        loss.backward()

        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.optimizer.step()
        return {'loss': loss.item(), 'n': y.size(0)}
```
{: file='common/engine/trainer.py'}

Cách chạy lại thí nghiệm: làm theo [SETUP.md](https://github.com/tronglh241/co5085_dl4cv/blob/main/SETUP.md) để cài môi trường, sau đó mở shell trong container và chạy:

```bash
python scripts/train.py configs/e1/train_cnn_aug.yml
python scripts/evaluate.py configs/e1/test_cnn_aug.yml
```

Thay `train_cnn_aug.yml` và `test_cnn_aug.yml` bằng cấu hình tương ứng trong `configs/e1` cho các thí nghiệm còn lại.

## Kết quả

### Đường cong học

![Đường cong loss và accuracy của sáu thí nghiệm E1](/assets/e1/curves_e1.png)

*Hình 5. Đường cong loss và accuracy; nét đứt là loss, nét liền là accuracy, đỏ là huấn luyện, xanh là kiểm định.*

- **Base:**
  - MLP và CNN đạt accuracy cao hơn Softmax và dừng sớm hơn, với epoch tốt nhất lần lượt là 6 và 7.
  - Softmax cần 45 epoch để đạt checkpoint tốt nhất.
- **Aug:**
  - Quá trình cải thiện của MLP và CNN kéo dài hơn, với epoch tốt nhất là 102 và 72.
  - Khoảng cách giữa huấn luyện và kiểm định, cả về loss lẫn accuracy, thu hẹp lại ở MLP và CNN.
  - Loss của Softmax dao động zigzag liên tục và huấn luyện dừng sớm hơn vì mô hình không cải thiện thêm được.

### Accuracy

| Mô hình | Biến thể | Tham số | Epoch tốt nhất | Huấn luyện (%) | Kiểm định (%) | Kiểm thử (%) | Gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Softmax | base | 7.850 | 45 | 94,00 | 92,28 | 92,58 | 1,72 |
| Softmax | aug | 7.850 | 33 | 83,81 | 83,82 | 85,12 | −0,01 |
| MLP | base | 235.146 | 6 | 99,26 | 97,58 | 97,68 | 1,68 |
| MLP | aug | 235.146 | 102 | 99,69 | 98,99 | 99,07 | 0,70 |
| CNN | base | 206.922 | 7 | 99,68 | 98,73 | 98,85 | 0,95 |
| CNN | aug | 206.922 | 72 | **99,93** | **99,41** | **99,50** | 0,52 |

*Bảng 2. Kết quả E1 trên MNIST. Gap là chênh lệch accuracy giữa huấn luyện và kiểm định.*

- CNN aug đạt accuracy kiểm thử cao nhất, 99,50%.
- Ở thiết lập base, accuracy kiểm thử tăng dần từ Softmax (92,58%) sang MLP (97,68%) và CNN (98,85%).
- Tăng cường dữ liệu cải thiện MLP từ 97,68% lên 99,07% và CNN từ 98,85% lên 99,50%.
- Ngược lại, Softmax giảm từ 92,58% xuống 85,12%.

### Số lỗi và ma trận nhầm lẫn

| Mô hình | Base | Aug | Thay đổi |
|---|---:|---:|---:|
| Softmax | 742 | 1.488 | +100% |
| MLP | 232 | 93 | −60% |
| CNN | 115 | 50 | −57% |

*Bảng 3. Số ảnh kiểm thử bị phân loại sai trên tổng số 10.000 mẫu.*

![Ma trận nhầm lẫn của sáu thí nghiệm E1](/assets/e1/confusion_e1.png)

*Hình 6. Ma trận nhầm lẫn trên tập kiểm thử, ô chéo để trắng, màu đỏ đậm hơn là tỉ lệ lỗi cao hơn, số trong ô là số mẫu bị phân loại sai.*

- **Tổng quan:**
  - Theo thang màu, số lỗi giảm dần từ Softmax sang MLP rồi sang CNN.
  - Softmax base nhầm trên nhiều cặp chữ số có nét giống nhau như 2/8, 3/5, 4/9, 5/8 và 7/9.
- **Các cặp dễ nhầm lẫn:** $4 \to 9$, $5 \to 3$ và $7 \to 2$ đều xuất hiện ở cả ba mô hình, nên Bảng 4 so sánh số mẫu của ba cặp này trên từng mô hình.

| Cặp | Softmax base | Softmax aug | MLP base | MLP aug | CNN base | CNN aug |
|---|---:|---:|---:|---:|---:|---:|
| $4 \to 9$ | 31 | 37 | 22 | 8 | 3 | 3 |
| $5 \to 3$ | 36 | 80 | 10 | 5 | 7 | 4 |
| $7 \to 2$ | 21 | 27 | 13 | 6 | 6 | 3 |

*Bảng 4. Số mẫu kiểm thử bị nhầm của ba cặp theo dõi.*

- **Thiết lập base:**
  - Số mẫu của các cặp này giảm dần từ Softmax sang MLP rồi sang CNN, chẳng hạn $5 \to 3$ giảm từ 36 xuống 10 rồi 7 mẫu.
  - CNN base mắc tối đa 8 mẫu ở mỗi cặp, trong đó $9 \to 4$ và $9 \to 7$ là hai cặp lớn nhất.
- **Thiết lập aug:**
  - MLP và CNN tiếp tục giảm lỗi ở cả ba cặp, chẳng hạn $4 \to 9$ ở MLP giảm từ 22 xuống 8 mẫu và $7 \to 2$ ở CNN giảm từ 6 xuống 3 mẫu.
  - Softmax thì ngược lại, $5 \to 3$ tăng từ 36 lên 80 mẫu.

### Mẫu nhận diện sai

![Các mẫu CNN tăng cường phân loại sai với độ tự tin cao](/assets/e1/failures_cnn_aug.png)

*Hình 7. Các mẫu CNN aug phân loại sai với độ tự tin cao nhất, mỗi ảnh ghi nhãn thật → dự đoán và xác suất của lớp dự đoán sai.*

- Một số mẫu vốn khó phân biệt:
  - Chữ số 6 và 2 bị nhận thành 0 vì hình dạng khá giống chữ số 0.
  - Chữ số 9 viết thiếu nét nên trông giống chữ số 4.
- Một số lỗi cho thấy mô hình chưa tốt, chẳng hạn chữ số 1 bị nhận thành 2 dù hình dạng khá rõ ràng.

## Thảo luận

- **Kiến trúc quan trọng hơn số tham số:**
  - CNN có ít tham số hơn MLP khoảng 12% nhưng chỉ mắc khoảng một nửa số lỗi: 115 so với 232 ở base và 50 so với 93 ở aug.
  - Ưu thế của CNN đến từ tính cục bộ và chia sẻ trọng số.
- **Tăng cường dữ liệu có lợi cho mô hình đủ khả năng biểu diễn:**
  - Với MLP và CNN, tăng cường dữ liệu cải thiện accuracy kiểm thử từ 97,68% lên 99,07% ở MLP và từ 98,85% lên 99,50% ở CNN.
  - Gap thu hẹp từ 1,68 xuống 0,70 ở MLP và từ 0,95 xuống 0,52 ở CNN.
  - Epoch tốt nhất dời từ 6 lên 102 ở MLP và từ 7 lên 72 ở CNN. Quá trình cải thiện được kéo dài.

## Kết luận

- CNN là mô hình phù hợp nhất cho bài toán phân loại ảnh trong ba kiến trúc.
- Tăng cường dữ liệu có ích cho MLP và CNN, còn Softmax không đủ khả năng tận dụng nó.
