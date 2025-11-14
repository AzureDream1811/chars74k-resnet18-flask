# Handwritten English Character Classification (Chars74K + ResNet-18)

## Mô tả

- Phân loại ký tự tiếng Anh: **0–9, a–z, A–Z**.
- Dataset: **Chars74K - Digital English Font**  
  Link: https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont
- Mô hình: **CNN (ResNet-18)**.

## Chức năng / Demo

- Web demo (Flask):
  - Upload ảnh ký tự (PNG/JPG) → mô hình dự đoán ký tự tương ứng.
  - Hiển thị xác suất Top-k (ví dụ Top-3).
  - Khi đánh giá mô hình: hiển thị **confusion matrix**.

## Đơn vị đo hiệu suất

- Accuracy (%)
- F1-score (%)
- Top-3 Accuracy (%)
- Confusion matrix

## Cấu trúc thư mục (dự kiến)

- `app/` – Flask web demo (route upload ảnh, predict)
- `src/` – code train/evaluate model (dataset, model, train script)
- `src/dataset_chars74k.py` – Chứa class Dataset để đọc ảnh từ thư mục data/raw/English/Fnt - Chuyển ảnh PNG → tensor PyTorch, resize, normalize
- `src/model_resnet18.py.py`– Chứa hàm build model ResNet-18 (kiến trúc CNN) - Điều chỉnh cho ảnh input grayscale (1 kênh, 64×64).
- `src/train.py`
  Script chạy train từ A–Z:
  load dataset
  chia train/val
  tạo model, optimizer
  vòng lặp epoch
  tính Accuracy, F1, Top-3, Log loss
  lưu model_best.pth, confusion_matrix.npy, classes.txt.
- `data/raw/` – dữ liệu gốc tải từ Kaggle
- `data/processed/` – dữ liệu đã tiền xử lý (resize, normalize, split train/test)
- `notebooks/` – notebook (nếu cần) để EDA, vẽ biểu đồ
