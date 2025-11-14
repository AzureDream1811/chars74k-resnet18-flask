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
- `data/raw/` – dữ liệu gốc tải từ Kaggle
- `data/processed/` – dữ liệu đã tiền xử lý (resize, normalize, split train/test)
- `notebooks/` – notebook (nếu cần) để EDA, vẽ biểu đồ
