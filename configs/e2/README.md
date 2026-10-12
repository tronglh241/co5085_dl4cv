# E2 — MSA và tokenization

Cùng split MNIST với E1 (`datasets/mnist.py`, seed 42): 50.000 train / 10.000 val / 10.000 test. Ảnh không augment, chuẩn hóa `(0.1308,), (0.3082,)`.

Môi trường dùng chung với E1. Xem [SETUP.md](../../SETUP.md) và [DEV.md](../../DEV.md). `device: null` trong config: có CUDA thì dùng GPU, không thì CPU.

```bash
bash scripts/shell/dev.sh gpu   # hoặc cpu
```

## Mô hình

Một `TransformerClassifier`: tokenizer, token CLS, positional embedding, 4 block pre-norm, head 10 lớp. `D=64`, 4 head, `mlp_ratio=2`, dropout `0.1`. `attn_impl` chỉ đổi MSA tự viết (`scratch`) hoặc `nn.MultiheadAttention` (`torch`).

| Run | Token | N (28×28) | File train |
|---|---|---|---|
| patch | ô 4×4 | 49 | `train_patch_{scratch,torch}.yml` |
| row | từng hàng | 28 | `train_row_{scratch,torch}.yml` |
| cnnstem | CNN stem rồi 7×7 | 49 | `train_cnnstem_{scratch,torch}.yml` |
| patch7 | ô 7×7 | 16 | `train_patch7_scratch.yml` |
| patch overlap | kernel 8, stride 4 | 36 | `train_patch_overlap_scratch.yml` |

Token theo kênh màu không dùng: MNIST có `C=1`.

## Lệnh

Kiểm tra MSA tự viết khớp `nn.MultiheadAttention` (kỳ vọng float32 dưới `1e-5`):

```bash
python scripts/e2/check_msa.py
```

Train sáu run bắt buộc:

```bash
python scripts/train.py configs/e2/train_patch_scratch.yml
python scripts/train.py configs/e2/train_patch_torch.yml
python scripts/train.py configs/e2/train_row_scratch.yml
python scripts/train.py configs/e2/train_row_torch.yml
python scripts/train.py configs/e2/train_cnnstem_scratch.yml
python scripts/train.py configs/e2/train_cnnstem_torch.yml
```

Biến thể thêm, vẫn là MSA tự viết:

```bash
python scripts/train.py configs/e2/train_patch7_scratch.yml
python scripts/train.py configs/e2/train_patch_overlap_scratch.yml
```

Muốn mean±std cho cặp patch: copy `train_patch_scratch.yml` và `train_patch_torch.yml`, đổi `seed` thành 43 và 44, đồng thời đổi `log_dir` và `dirname` để không ghi đè.

Sau khi train, xuất scalar:

```bash
python scripts/tensorboard/extract.py --logdir logs/e2
```

Điền `loader.path` trong `configs/e2/test_*.yml` bằng file `best_*.pt` của đúng run (nằm trong thư mục có timestamp dưới `checkpoints/e2/train_<run>/`). Rồi:

```bash
python scripts/evaluate.py configs/e2/test_patch_scratch.yml
python scripts/e2/analyze.py --configs configs/e2/test_*.yml
python scripts/e2/attention_maps.py --config configs/e2/test_patch_scratch.yml
```

`analyze.py` đánh giá checkpoint trong `loader.path` của từng file test, rồi ghép với log TensorBoard được tạo cùng lúc với checkpoint đó (lệch timestamp tối đa 60 giây) và luôn xuất lại `metrics.csv` của log đó. Script ghi `docs/assets/e2/results_e2.csv`, `results_e2.tex` (bảng mà `reports/e2.tex` dùng), `curves_<tokenizer>.png` (train nét đứt, validation nét liền, mỗi bản MSA một màu), `acc_vs_time.png` và `confusion_best.png`.

## Kết quả

Bảng dưới được điền từ `docs/assets/e2/results_e2.csv` sau khi train xong. Accuracy để ở thang 0–1. `Best epoch` là epoch có loss validation thấp nhất, cũng là epoch của checkpoint `best_*.pt`; `Val acc` đọc tại epoch đó, nên `Val acc` và `Test acc` thuộc cùng một mô hình.

Số tham số đếm từ kiến trúc, không phụ thuộc lần train. Cột thời gian và accuracy điền sau khi train.

| Model | Variant | Params | Best epoch | Time/epoch (s) | Val acc | Test acc |
|---|---|---|---|---|---|---|
| patch | scratch | 139018 | 34 | 12.05 | 0.9863 | 0.9875 |
| patch | torch | 139018 | 31 | 11.25 | 0.9860 | 0.9883 |
| row | scratch | 138442 | 47 | 16.28 | 0.9898 | 0.9919 |
| row | torch | 138442 | 47 | 17.09 | 0.9896 | 0.9895 |
| cnnstem | scratch | 161098 | 30 | 17.97 | 0.9932 | 0.9936 |
| cnnstem | torch | 161098 | 29 | 16.59 | 0.9923 | 0.9935 |
| patch7 | scratch | 139018 | 25 | 11.27 | 0.9854 | 0.9867 |
| patch_overlap | scratch | 141258 | 47 | 16.14 | 0.9897 | 0.9911 |
