# Handwritten English Character Classification (Chars74K + ResNet-18)

## 📌 Mô tả

Bài toán phân loại ký tự tiếng Anh dạng viết in/viết thường gồm:  
**0–9, A–Z, a–z** (tổng tối đa 62 lớp).

- **Dataset:** Chars74K – Digital English Font  
  <https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont>
- **Model:** CNN – ResNet-18 (tùy chỉnh cho ảnh grayscale 64×64)

---

## 🎯 Chức năng / Demo (Flask)

- Upload ảnh (PNG/JPG) chứa 1 ký tự → mô hình dự đoán ký tự tương ứng.
- Hiển thị **Top-3 xác suất cao nhất**.
- Khi đánh giá mô hình sẽ hiển thị thêm **confusion matrix (heatmap)**.

---

## 📈 Đơn vị đo hiệu suất

- **Accuracy (%)**
- **F1-score (macro)**
- **Top-3 accuracy (%)**
- **Confusion matrix**

---

## 📚 Cấu trúc thư mục dự án

```text
chars74k-resnet18-flask/
│
├── app/                     # Flask web demo
│   └── app.py               # Routes upload/predict
│
├── src/                     # Code huấn luyện mô hình
│   ├── dataset_chars74k.py  # Đọc ảnh từ data/raw/English/Fnt
│   ├── model_resnet18.py    # Xây dựng model ResNet-18 (ảnh grayscale)
│   └── train.py             # Train model + tính metrics + lưu model
│
├── data/
│   ├── raw/                 # Dataset tải từ Kaggle (KHÔNG commit lên Git)
│   └── processed/           # Dữ liệu sau tiền xử lý (nếu cần)
│
├── model_best.pth           # Model tốt nhất (auto tạo sau khi train)
├── confusion_matrix.npy     # Lưu confusion matrix để vẽ heatmap
├── classes.txt              # Map index → tên class (Sample001 → A, ...)
│
├── requirements.txt
├── .gitignore
└── README.md
```
