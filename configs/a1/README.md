# A1 — CNN vs Transformer trên Food-101

So sánh một CNN (ResNet-50) và một Transformer (DeiT-S) trên Food-101, với 4 chế độ: train from-scratch, và 3 kiểu dùng pretrained (chỉ train head, freeze một phần, full finetune).

Mọi file `train_*.yml`, `test_*.yml` và `runs.json` trong thư mục này **được sinh** bởi `scripts/a1/make_configs.py`. Muốn đổi recipe thì sửa file đó rồi chạy lại, đừng sửa YAML bằng tay:

```bash
python scripts/a1/make_configs.py
```

## Dữ liệu

Food-101 (Bossard et al., ECCV 2014): 101 lớp, mỗi lớp 750 ảnh train và 250 ảnh test. Ảnh test đã được làm sạch thủ công; ảnh train **cố ý giữ nhãn nhiễu**, nên tập val (tách từ train) cũng nhiễu. Chỉ dùng cho mục đích nghiên cứu, phi thương mại.

| Split | Số ảnh | Nguồn |
|---|---|---|
| train | 68.175 | 90% train chính thức, chia phân tầng, seed 42 |
| val | 7.575 | 10% còn lại |
| test | 25.250 | test chính thức, chỉ chạy một lần ở cuối |

`datasets/food101.py` kế thừa `torchvision.datasets.Food101` (đọc split chính thức train/test từ `meta/*.json`), giống cách `datasets/mnist.py` kế thừa MNIST. Lớp con thêm hai thứ: tách val phân tầng từ train, và `fraction` lấy tập con phân tầng của train. `fraction` chỉ dùng cho evaluator `train_eval`, đọc 10% train mỗi epoch để vẽ đường accuracy trên train mà không tốn hết 68k ảnh. Mọi run đều train trên toàn bộ tập train. Danh sách ảnh của từng split nằm ở `data/food101_256/splits/`.

Chuẩn bị một lần: tải về, resize cạnh ngắn về 256px, ghi split.

```bash
python scripts/a1/prepare_data.py --raw data/ --out data/food101_256
```

EDA (số ảnh mỗi lớp, kích thước ảnh gốc, ảnh mẫu, ví dụ augmentation, mean/std theo kênh) ghi vào `docs/assets/a1/eda/`:

```bash
python scripts/a1/eda.py --raw data/ --root data/food101_256
```

## Mô hình

| Key | timm | Tham số | Pretrain |
|---|---|---|---|
| `resnet50` | `resnet50.a1_in1k` | 23,7M | ImageNet-1k |
| `deit_s` | `deit_small_patch16_224.fb_in1k` | 21,7M | ImageNet-1k |

Cả hai chỉ pretrain trên ImageNet-1k, để nguồn pretrain không làm lệch phép so sánh.

"Freeze một phần" là freeze nửa đầu theo **độ sâu**: ResNet freeze stem + layer1–2, DeiT freeze patch_embed + block 0–5. Vì tham số của ResNet dồn ở layer3–4, ResNet vẫn train khoảng 94% tham số, còn DeiT train khoảng 49%. Bảng kết quả luôn ghi số tham số trainable. BatchNorm trong phần bị freeze được giữ ở eval mode, nên running statistics không đổi.

## Xử lý dữ liệu (chung cho mọi run)

Giống MNIST ở E1/E2, mọi model và mọi chế độ dùng **cùng một** pipeline dữ liệu. Các run chỉ khác nhau ở model và cách train, không khác nhau ở dữ liệu đưa vào.

| Bước | Train | Eval (val, test, train_eval) |
|---|---|---|
| Preprocess offline | cạnh ngắn → 256px (`prepare_data.py`) | như train |
| Crop | `RandomResizedCrop(224, scale=(0.8, 1.0))` | `Resize(256)` + `CenterCrop(224)` |
| Lật | `RandomHorizontalFlip(p=0.5)` | — |
| Xoay | `RandomRotation(degrees=15)` | — |
| Màu | `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` | — |
| Dịch | `RandomAffine(degrees=0, translate=(0.1, 0.1))` | — |
| Chuẩn hoá | `ToTensor` + `Normalize(mean=[0.5]*3, std=[0.5]*3)` | như train |

Định nghĩa duy nhất nằm ở `TRAIN_STEPS`, `EVAL_STEPS`, `MEAN`, `STD` trong `scripts/a1/make_configs.py`.

**Lưu ý khi viết báo cáo:** weights pretrained (`resnet50.a1_in1k`, `deit_small_patch16_224.fb_in1k`) được train với mean/std của ImageNet. Normalize 0.5/0.5 làm lệch đầu vào so với lúc pretrain (std nhỏ hơn khoảng 2,2 lần). Full finetune tự thích nghi được, nhưng `head` và `partial` giữ nguyên phần backbone bị freeze, kể cả running stats của BatchNorm, nên có thể bị thiệt một phần. Đây là đánh đổi có chủ đích: đổi lấy một pipeline dữ liệu chung cho mọi run.

## Ma trận thí nghiệm

| Chế độ | Pretrained | Freeze | Epoch | LR (AdamW) |
|---|---|---|---|---|
| `scratch` | không | không | 40 | 5e-4, warmup 5 ep, clip 1.0 |
| `head` | có | trừ head | 10 | 1e-3, warmup 1 ep |
| `partial` | có | nửa đầu | 15 | 2e-4, warmup 1 ep |
| `full` | có | không | 15 | 1e-4, warmup 1 ep |

Chung cho mọi run:
- Batch 128, weight decay 0.05, label smoothing 0.1.
- Cosine theo từng iteration, bf16 autocast.
- Checkpoint tốt nhất được chọn theo **val accuracy**.

LR trong bảng là điểm xuất phát. Kiểm tra trên val (vài epoch) trước khi chạy thật, và tuyệt đối không tune trên test.

Có **8 run** (tập `core` trong `runs.json`): 2 model × 4 chế độ. Mỗi run train một lần với seed 42, trên toàn bộ tập train.

Chỉ có một seed, nên không có error bar. Hai run chênh nhau dưới khoảng 1% trên test (sai số ±0,44% với 25.250 ảnh, chưa tính dao động do quá trình train) thì không nên kết luận bên nào hơn.

## Lệnh

Đo tốc độ và ước tính chi phí trước, rồi mới chốt số epoch:

```bash
python scripts/a1/benchmark.py --usd-per-hour 1.5 --loader-root data/food101_256
```

Train một run:

```bash
python scripts/train.py configs/a1/train_resnet50_full.yml
```

Resume cục bộ: đặt `loader.path` thành file `backup_epoch_*.pt` của run đó. Log và checkpoint sẽ ghi tiếp vào đúng thư mục stamp cũ. Biến môi trường `CO5085_RUN_STAMP=yymmddHHMMSS` ép dùng một stamp cụ thể.

Phân tích: tự tìm `best_*.pt` mới nhất của từng run, đánh giá test, rồi ghi bảng và hình vào `docs/assets/a1/`:

```bash
python scripts/a1/analyze.py
python scripts/a1/analyze.py --runs resnet50_full,deit_s_full
```

Kết quả gồm:
- **Bảng:** `results_a1.csv` và `results_a1.tex`, có accuracy kèm khoảng tin cậy Wilson 95%, macro-F1, số tham số trainable, thời gian/epoch, peak memory và ảnh/s.
- **So sánh theo cặp:** `significance.csv` và `significance.tex`, 11 so sánh định trước (CNN so với ViT, giá trị của pretrain, giá của việc đóng băng) với chênh lệch, khoảng bootstrap theo cặp, p McNemar đã hiệu chỉnh Holm và kết luận.
- **Accuracy từng lớp:** `per_class.csv`.
- **Hình:** `curves_a1.png`, `cost_a1.png`, `confused_<run>.png`, `errors_<run>.png`.
- **Dự đoán từng ảnh:** `predictions/<run>.csv`.

Trên Modal, xem docstring của `modal_run/a1_modal.py`. Các action có sẵn: `prepare`, `eda`, `benchmark`, `smoke`, `train`, `analyze`.

Quy tắc của `--action train`:
- **Mặc định luôn train mới.** Mỗi lần chạy có một stamp riêng; run ghi vào `checkpoints/a1/train_<run>/<stamp>/` và `logs/a1/train_<run>/<stamp>/`. Một phiên mới không bao giờ nạp lại file của phiên cũ.
- **GPU là preemptible.** Nếu Modal tự thử lại một call bị ngắt, lần thử lại thấy backup mà lần trước để lại trong cùng stamp và chạy tiếp. Các bản sao lưu được ghi nguyên tử (ghi file tạm rồi đổi tên), nên không có file hỏng dở dang.
- **`--resume`** mới là cách chủ động chạy tiếp một phiên bị ngắt: mỗi run tiếp tục từ backup mới nhất của stamp mới nhất. Run chưa có backup thì báo lỗi, không tự bắt đầu lại.
- **`--max-epochs` và `--lr` bắt buộc đi kèm `--tag`.** Run có thẻ được ghi vào `train_<run>__<tag>`, `analyze.py` không thấy chúng. Lý do: trạng thái scheduler của một run ngắn mà bị nạp vào run dài cùng tên sẽ rút lịch learning rate về 0.
- `smoke` tự dùng thẻ `smoke`.

```bash
modal run modal_run/a1_modal.py --action smoke
modal run modal_run/a1_modal.py --action train --runs resnet50_full --max-epochs 3 --lr 2e-4 --tag lr2e-4
modal run --detach modal_run/a1_modal.py --action train --runs core
modal run --detach modal_run/a1_modal.py --action train --runs core --resume
```
