---
layout: post
title: "E2 Transformer: MSA và tokenization"
author: thien
date: 2026-11-02 11:00
toc: true
math: true
comments: false
---

## Giới thiệu

Bài E2 xây dựng bộ phân loại MNIST dựa trên cơ chế Multi-Head Self-Attention (MSA) và trả lời hai câu hỏi:

- **MSA tự cài đặt có đúng không?** Chúng tôi viết MSA từ đầu, không dùng `nn.MultiheadAttention`, rồi đối chiếu với bản của PyTorch về số học và về kết quả huấn luyện.
- **Cách biến ảnh thành chuỗi token ảnh hưởng thế nào?** Chúng tôi so sánh năm cách tạo token: hàng, patch $4 \times 4$, patch $7 \times 7$, patch $8 \times 8$ chồng lấn và CNN-stem.

Kết quả chính:

- MSA tự cài đặt khớp `nn.MultiheadAttention` đến sai số float32 và cho accuracy test tương đương ở cả năm cách tạo token.
- CNN-stem dẫn đầu với 99,33% (MSA tự cài đặt) và 99,49% (MSA PyTorch), nhưng cũng có nhiều tham số nhất.
- Cả mười mô hình đều đạt trên 98,6% với chưa tới 162.000 tham số, nên MNIST khó phân biệt các cách tạo token.

## Dữ liệu

### Tập dữ liệu

- MNIST gồm 70.000 ảnh chữ số viết tay, ảnh xám kích thước $28 \times 28$, thuộc 10 lớp từ 0 đến 9.
- Dữ liệu gốc chia sẵn thành 60.000 ảnh train và 10.000 ảnh test.
- Lí do chọn MNIST:
  - Là bộ dữ liệu tiêu chuẩn, phổ biến nên dễ đối chiếu kết quả.
  - Ảnh nhỏ, số mẫu vừa phải nên huấn luyện nhanh.
  - Các lớp tương đối cân bằng nên accuracy phản ánh đúng hiệu suất tổng thể.

![Một số ảnh trong MNIST](/assets/shared/mnist_samples.png)

*Hình 1. Một số ảnh mẫu của MNIST.*

### Chia dữ liệu

- 60.000 ảnh train gốc được chia theo phương pháp phân tầng thành 50.000 ảnh train và 10.000 ảnh val.
- Việc chia chỉ thực hiện một lần, với seed 42, nên mọi mô hình trong E1 và E2 đều dùng chung một tập dữ liệu.
- 10.000 ảnh test chính thức được giữ nguyên và không tham gia huấn luyện hay chọn mô hình.

| Lớp | Train | Val | Test |
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

### Tiền xử lý

- Mỗi ảnh được chuyển thành tensor trong khoảng $[0, 1]$ rồi chuẩn hóa, với trung bình và độ lệch chuẩn tính trên 50.000 ảnh train:

$$
x' = \frac{x - 0.1308}{0.3082}
$$

- E2 **không** dùng tăng cường dữ liệu, nên được so sánh với các biến thể base của E1.

### Phương pháp đánh giá

- Vai trò của ba tập dữ liệu:
  - **Train:** cập nhật trọng số.
  - **Val:** điều chỉnh learning rate, dừng sớm và chọn checkpoint.
  - **Test:** chỉ dùng một lần ở cuối, trên checkpoint đã chọn.
- Độ đo chính là accuracy, tức tỷ lệ mẫu được phân loại đúng:

$$
\mathrm{accuracy} = \frac{\sum_{c=1}^{K} C_{cc}}{\sum_{i=1}^{K}\sum_{j=1}^{K} C_{ij}},
$$

  với $K = 10$ và $C_{ij}$ là số mẫu thuộc lớp thật $i$ được dự đoán thành lớp $j$.
- Ngoài accuracy, bài còn xem tổng số ảnh bị phân loại sai, ma trận nhầm lẫn và thời gian huấn luyện mỗi epoch.

## Mô hình

### Multi-Head Self-Attention

Với chuỗi $N$ token chiều $d$, mỗi đầu $i$ chiếu đầu vào thành $Q_i, K_i, V_i$ rồi tính attention. Các đầu được ghép lại và chiếu ra bằng $W^O$:

$$
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i, \qquad
\mathrm{MSA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O
$$

Ma trận attention có kích thước $N \times N$ cho mỗi đầu, nên chi phí thời gian tăng theo $O(N^2 d)$. Điều này quan trọng khi so sánh các cách tạo token.

### Bản tự cài đặt

`ScratchMSA` chỉ dùng `Linear`, `einsum` và `softmax`. Một lớp `Linear` duy nhất ($d \to 3d$) tạo cả $Q, K, V$, với trọng số xếp chồng $[W^Q; W^K; W^V]$ cùng cách xếp với `in_proj_weight` của PyTorch:

```python
    def forward(self, x, return_attn=False):
        batch, length, _ = x.shape
        qkv = self.in_proj(x).view(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum('bhnd,bhmd->bhnm', query, key) / math.sqrt(self.head_dim)
        attn = self.attn_drop(scores.softmax(dim=-1))
        mixed = torch.einsum('bhnm,bhmd->bhnd', attn, value)
        mixed = mixed.transpose(1, 2).reshape(batch, length, self.embed_dim)
        out = self.out_proj(mixed)
        if return_attn:
            return out, attn
        return out, None
```
{: file='models/e2/attention.py'}

### Bản PyTorch

Cùng bộ phân loại nhưng thay MSA bằng `nn.MultiheadAttention` (`batch_first=True`, cùng số đầu và dropout). Hai bản dùng chung tokenizer, LayerNorm, FFN, token CLS và đầu phân loại, và với cùng seed thì khởi tạo trọng số giống hệt nhau.

### Kiểm tra tính đúng đắn

Chúng tôi sao chép trọng số từ module PyTorch sang module tự cài đặt và so sánh đầu ra, attention và gradient trên một số đầu vào ngẫu nhiên (`scripts/e2/check_msa.py`). Chênh lệch tối đa khoảng $10^{-7}$ ở float32 với đầu ra và $2 \times 10^{-5}$ với gradient.

### Kiến trúc bộ phân loại

- Mỗi cách tạo token đưa ảnh về chuỗi $N$ token và chiếu lên $d_{\text{model}} = 64$.
- Một token CLS được thêm vào đầu chuỗi, rồi cộng mã hóa vị trí học được kích thước $(N+1) \times d_{\text{model}}$.
- Chuỗi đi qua 4 khối encoder dạng pre-LayerNorm. Mỗi khối gồm MSA 4 đầu ($d_k = 16$) và FFN (`Linear`, GELU, `Linear`, chiều ẩn 128), mỗi phần có kết nối residual. Dropout $p = 0{,}1$ nằm trên ma trận attention và trên đầu ra của MSA trước khi cộng residual; ở FFN, dropout nằm giữa GELU và `Linear` thứ hai, không có dropout trên nhánh residual của FFN.
- Sau một LayerNorm cuối, đầu phân loại `Linear` $64 \to 10$ áp dụng lên token CLS.

### Các cách tạo token

- **Hàng:** mỗi hàng ảnh là một token.
- **Patch $4 \times 4$:** các ô lưới không chồng lấn.
- **Patch $7 \times 7$:** các ô lớn hơn, nên ít token hơn.
- **Patch $8 \times 8$ chồng lấn:** patch $8 \times 8$ với bước 4, hai patch liền kề chồng lên nhau một nửa.
- **CNN-stem:** hai khối Conv $3 \times 3$, BatchNorm, ReLU, MaxPool $2 \times 2$ (32 rồi 64 kênh) và một Conv $1 \times 1$ lên 64 kênh. Mỗi vị trí không gian của bản đồ đặc trưng $64 \times 7 \times 7$ là một token.

| Cách tạo token | Số token $N$ | Chiều token |
|---|---:|---:|
| Hàng | 28 | 28 |
| Patch $4 \times 4$ | 49 | 16 |
| Patch $7 \times 7$ | 16 | 49 |
| Patch $8 \times 8$ chồng lấn | 36 | 64 |
| CNN-stem | 49 | 64 |

*Bảng 2. Số token và chiều token trước khi chiếu lên $d_{\text{model}} = 64$. Riêng CNN-stem, Conv $1 \times 1$ nằm trong tokenizer nên token ra khỏi tokenizer đã có 64 chiều.*

### Sơ đồ và số tham số

![Sơ đồ bộ phân loại Transformer của E2](/assets/e2/model_e2.png)

*Hình 3. Sơ đồ bộ phân loại Transformer. $N$ là số token, tùy cách tạo token.*

| Thành phần | Hàng | Patch $4 \times 4$ | Patch $7 \times 7$ | Patch $8 \times 8$ chồng lấn | CNN-stem |
|---|---:|---:|---:|---:|---:|
| Tokenizer | 1.856 | 1.088 | 3.200 | 4.160 | 23.168 |
| Token CLS | 64 | 64 | 64 | 64 | 64 |
| Mã hóa vị trí $(N+1) \times 64$ | 1.856 | 3.200 | 1.088 | 2.368 | 3.200 |
| MSA (4 block) | 66.560 | 66.560 | 66.560 | 66.560 | 66.560 |
| FFN (4 block) | 66.304 | 66.304 | 66.304 | 66.304 | 66.304 |
| LayerNorm | 1.152 | 1.152 | 1.152 | 1.152 | 1.152 |
| Linear $64 \to 10$ | 650 | 650 | 650 | 650 | 650 |
| **Tổng** | **138.442** | **139.018** | **139.018** | **141.258** | **161.098** |

*Bảng 3. Số tham số của từng thành phần. Hai bản MSA có cùng số tham số.*

## Thiết lập huấn luyện

Mười thí nghiệm dùng chung một quy trình:

- Tối ưu hóa bằng AdamW với learning rate $10^{-3}$ và weight decay 0,05. Weight decay chỉ áp lên trọng số của `Linear` và `Conv2d`, không áp lên bias, tham số chuẩn hóa, token CLS và mã hóa vị trí.
- Batch size 128, hàm mất mát cross-entropy, tối đa 200 epoch.
- Giảm learning rate 10 lần sau 10 epoch không cải thiện loss val.
- Dừng sớm sau 20 epoch không cải thiện loss val.
- Chọn checkpoint có loss val nhỏ nhất, và báo cáo accuracy test tại checkpoint này.
- Mọi thí nghiệm dùng seed 42.
- Toàn bộ thông số của mỗi thí nghiệm nằm trong một tệp cấu hình YAML riêng, và log huấn luyện được ghi bằng TensorBoard.

So với E1, E2 dùng AdamW thay cho Adam, batch size 128 thay cho 64, tối đa 200 epoch thay cho 1.000, và không tăng cường dữ liệu.

Cách chạy lại thí nghiệm: làm theo [SETUP.md](https://github.com/tronglh241/co5085_dl4cv/blob/main/SETUP.md) để cài môi trường, sau đó mở shell trong container và chạy:

```bash
python scripts/train.py configs/e2/train_patch_scratch.yml
python scripts/evaluate.py configs/e2/test_patch_scratch.yml
```

Thay `train_patch_scratch.yml` và `test_patch_scratch.yml` bằng cấu hình tương ứng trong `configs/e2` cho các thí nghiệm còn lại.

## Kết quả

### Đường cong học

![Đường cong loss và accuracy của mười thí nghiệm E2](/assets/e2/curves_e2.png)

*Hình 4. Đường cong loss và accuracy; nét đứt là loss, nét liền là accuracy, đỏ là train, xanh là val, đường chấm tím là epoch của checkpoint được giữ. Mỗi hàng là một cách tạo token, cột trái là MSA tự cài đặt, cột phải là MSA PyTorch.*

- Hai bản MSA bám sát nhau ở cả năm cách tạo token.
- Epoch tốt nhất nằm trong khoảng 33 đến 53.
- Loss val gần như phẳng sau epoch tốt nhất, trong khi accuracy train đã gần 100%.

### Accuracy

| Cách tạo token | MSA | Tham số | Epoch tốt nhất | Train (%) | Val (%) | Test (%) | Gap | Giây/epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Patch $4 \times 4$ | Tự cài đặt | 139.018 | 38 | 99,95 | 98,73 | 98,85 | 1,22 | 11,08 |
|  | PyTorch | 139.018 | 33 | 99,96 | 98,60 | 98,67 | 1,36 | 15,83 |
| Patch $7 \times 7$ | Tự cài đặt | 139.018 | 47 | 99,99 | 98,60 | 98,70 | 1,39 | 11,76 |
|  | PyTorch | 139.018 | 48 | 99,98 | 98,55 | 98,80 | 1,43 | 15,81 |
| Patch $8 \times 8$ chồng lấn | Tự cài đặt | 141.258 | 52 | 99,99 | 98,94 | 98,98 | 1,05 | 16,17 |
|  | PyTorch | 141.258 | 53 | 100,00 | 99,06 | 99,07 | 0,94 | 15,55 |
| Hàng | Tự cài đặt | 138.442 | 43 | 99,96 | 98,87 | 99,13 | 1,09 | 16,03 |
|  | PyTorch | 138.442 | 34 | 99,89 | 98,82 | 98,94 | 1,07 | 10,69 |
| CNN-stem | Tự cài đặt | 161.098 | 33 | 99,99 | 99,32 | 99,33 | 0,67 | 16,73 |
|  | PyTorch | 161.098 | 34 | 99,99 | 99,29 | 99,49 | 0,70 | 15,94 |

*Bảng 4. Kết quả E2 trên MNIST. Gap là chênh lệch accuracy giữa train và val; giây/epoch chỉ tính các bước huấn luyện.*

- Với MSA tự cài đặt, accuracy test xếp từ cao xuống thấp là CNN-stem (99,33%), hàng (99,13%), patch $8 \times 8$ chồng lấn (98,98%), patch $4 \times 4$ (98,85%) và patch $7 \times 7$ (98,70%).
- Với MSA PyTorch, CNN-stem cũng dẫn đầu (99,49%).
- Hai bản MSA cho accuracy gần nhau ở cả năm cách tạo token, chênh lệch lớn nhất là 0,19 điểm phần trăm.
- Hướng chênh không nhất quán: bản tự cài đặt cao hơn ở patch $4 \times 4$ và hàng, thấp hơn ở patch $7 \times 7$, patch chồng lấn và CNN-stem. Điều này phù hợp với nhiễu tối ưu hơn là một khác biệt hệ thống.
- CNN-stem có nhiều tham số nhất (161.098, hơn các mô hình còn lại khoảng 16%), nên chưa tách được đóng góp của thiên kiến cục bộ khỏi đóng góp của dung lượng lớn hơn.
- Thời gian mỗi epoch dao động từ 10,69 s đến 16,73 s và không tăng theo số token. Con số này gồm cả thời gian nạp dữ liệu trên CPU và mỗi lượt chạy diễn ra trên một container riêng, nên chưa đủ để so sánh hai cách cài đặt hay kiểm chứng quan hệ $O(N^2 d)$.
- Cột ảnh/giây trong `results_e2.csv` đo riêng mô hình: patch $7 \times 7$ ($N = 16$) khoảng 8.400 đến 11.000 ảnh/s, patch $4 \times 4$ ($N = 49$) khoảng 3.800 đến 4.200 và CNN-stem ($N = 49$) khoảng 2.300 đến 2.600.
- Patch $7 \times 7$ nhanh hơn patch $4 \times 4$ khoảng 2,2 đến 2,6 lần, gần với tỉ lệ tuyến tính theo $N$ ($49/16 \approx 3{,}1$) hơn tỉ lệ bậc hai ($(49/16)^2 \approx 9{,}4$), nên ở $N \le 49$ và $d = 64$ phần $O(N d^2)$ của phép chiếu và FFN lấn át attention.
- Kết luận này còn hạn chế: ảnh/giây đo trên CPU của máy phân tích chứ không phải GPU lúc huấn luyện, dao động khoảng 30% giữa các lần đo, và gồm cả chi phí tokenizer (CNN-stem chậm hơn patch $4 \times 4$ dù cùng $N = 49$).

### Số lỗi và ma trận nhầm lẫn

| Cách tạo token | MSA tự cài đặt | MSA PyTorch |
|---|---:|---:|
| Patch $4 \times 4$ | 115 | 133 |
| Patch $7 \times 7$ | 130 | 120 |
| Patch $8 \times 8$ chồng lấn | 102 | 93 |
| Hàng | 87 | 106 |
| CNN-stem | 67 | 51 |

*Bảng 5. Số ảnh test bị phân loại sai trên tổng số 10.000 mẫu.*

![Ma trận nhầm lẫn của mười thí nghiệm E2](/assets/e2/confusion_e2.png)

*Hình 5. Ma trận nhầm lẫn trên tập test, hàng trên là MSA tự cài đặt, hàng dưới là MSA PyTorch. Ô chéo để trắng, màu đỏ đậm hơn là tỉ lệ lỗi cao hơn, số trong ô là số mẫu bị phân loại sai.*

- CNN-stem mắc ít lỗi nhất (67 với MSA tự cài đặt và 51 với MSA PyTorch), nên ma trận của nó thưa nhất.
- Cặp $4/9$ (cả hai chiều) nằm trong ba cặp bị nhầm nhiều nhất ở mọi mô hình, chẳng hạn $4 \to 9$ có 11 mẫu ở patch $4 \times 4$ với PyTorch.
- Cặp $5 \to 3$ xuất hiện ở mọi mô hình, từ 2 mẫu (CNN-stem, PyTorch) đến 9 mẫu (patch $7 \times 7$, tự cài đặt).

### Mẫu nhận diện sai

![Các mẫu CNN-stem với MSA tự cài đặt phân loại sai với độ tự tin cao](/assets/e2/failures_best.png)

*Hình 6. Các mẫu CNN-stem với MSA tự cài đặt phân loại sai với độ tự tin cao nhất, mỗi ảnh ghi nhãn thật → dự đoán và xác suất của lớp dự đoán sai.*

- Hơn một nửa trong số 20 lỗi này thuộc các cặp chữ số vốn dễ nhầm như $4/9$, $5/3$ và $7/1$.
- Nhiều ảnh viết khác thường, chẳng hạn một nét mảnh bị nhận thành chữ số 1 hoặc chữ số 4 khép kín phần đầu nên giống chữ số 9.

## Thảo luận

- **MSA tự cài đặt đủ tin cậy:** khớp `nn.MultiheadAttention` đến sai số float32 và cho accuracy test tương đương. Chênh lệch tối đa giữa hai bản là 0,19 điểm và đổi chiều giữa các cách tạo token.
- **Cách tạo token là một dạng thiên kiến quy nạp:**
  - CNN-stem đưa tính cục bộ và bất biến tịnh tiến vào trước khi attention nhìn toàn cục, và dẫn đầu.
  - Với MSA tự cài đặt, hàng đạt 99,13% trên test, cao hơn cả ba biến thể patch (98,70% đến 98,98%). Trên val của cùng bản này, patch chồng lấn cao hơn hàng (98,94% so với 98,87%). Ngược lại, với MSA PyTorch, patch chồng lấn cao hơn hàng trên test (99,07% so với 98,94%). Chênh lệch của hàng so với patch chồng lấn (0,15 điểm) và patch $4 \times 4$ (0,28 điểm) dưới 0,3 điểm; riêng patch $7 \times 7$ thấp hơn hàng 0,43 điểm nhưng đây chỉ là kết quả của một seed, và thứ hạng giữa hàng với patch chồng lấn đổi chiều theo cách cài đặt MSA nên chưa ổn định.
- **So sánh với E1:**
  - E2 không tăng cường dữ liệu nên được so với các biến thể base của E1.
  - Mọi mô hình Transformer (98,67% đến 99,49%) đều cao hơn MLP base (97,68%) và có ít tham số hơn (138.442 đến 161.098 so với 235.146). Khoảng cách khoảng 1 đến 1,8 điểm lớn hơn ngưỡng một seed. Phép so sánh này đi cùng khác biệt huấn luyện đã nêu ở phần thiết lập: E2 dùng AdamW, weight decay 0,05, batch 128, tối đa 200 epoch; E1 dùng Adam, batch 64, tối đa 1.000 epoch.
  - Bảy trên mười mô hình ngang hoặc vượt CNN base (98,85%, 206.922 tham số). CNN-stem tự cài đặt đạt 99,33% (67 lỗi), cao hơn CNN base (115 lỗi) 0,48 điểm, với ít tham số hơn (161.098 so với 206.922). Chỉ patch $4 \times 4$ với PyTorch (98,67%) và patch $7 \times 7$ (98,70% và 98,80%) thấp hơn một chút.
  - CNN-stem PyTorch đạt 99,49% (51 lỗi), sát CNN có tăng cường của E1 (99,50%, 50 lỗi). Lệch một ảnh test, nhỏ hơn nhiều so với ngưỡng khoảng 0,3 điểm, nên hai mức này chưa tách được.
- **Hạn chế:**
  - Mỗi thí nghiệm chỉ chạy một lần (seed 42), nên các chênh lệch dưới khoảng 0,3 điểm phần trăm chưa đủ để kết luận.
  - MNIST nhỏ và dễ: cả mười mô hình đều đạt trên 98,6% với chưa tới 162.000 tham số, nên khó phân biệt các cách tạo token. Muốn so sánh chắc chắn hơn cần nhiều seed hoặc một tập dữ liệu khó hơn.

## Kết luận

- MSA viết từ đầu cho kết quả tương đương `nn.MultiheadAttention` về cả số học lẫn accuracy: chênh accuracy tối đa 0,19 điểm và đổi chiều giữa các cách tạo token.
- CNN-stem cho accuracy test cao nhất trong năm cách tạo token: 99,33% với MSA tự cài đặt và 99,49% với MSA PyTorch. Phần chênh lệch có thể một phần đến từ số tham số lớn hơn. Bản tự cài đặt cao hơn CNN base của E1 0,48 điểm dù ít tham số hơn. Bản PyTorch lệch CNN có tăng cường một ảnh test, trong sai số của một seed.
- Mọi mô hình Transformer đều cao hơn MLP base của E1 và có ít tham số hơn. Khoảng cách này đi cùng khác biệt huấn luyện đã nêu ở phần thiết lập.
