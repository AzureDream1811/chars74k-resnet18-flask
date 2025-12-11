# Handwritten English Character Classification (Chars74K + ResNet-18)

## 📌 Mô tả

Bài toán phân loại ký tự tiếng Anh dạng viết in/viết thường gồm:  
**0–9, A–Z, a–z** (tổng tối đa 62 lớp).

- **Dataset:** Chars74K – Digital English Font  
  <https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont>
- **Model:** CNN – ResNet-18 (tùy chỉnh cho ảnh grayscale 64×64)

---

## Cài đặt

python3 -m venv .venv
.venv\scripts\activate.ps1
pip install -r requirements.txt

## Cài đặt pytorch có hỗ trợ gpu

pip3 install torch torchvision --index-url <https://download.pytorch.org/whl/cu126>

## 🎯 Chức năng / Demo (Flask)

- Upload ảnh (PNG/JPG) chứa 1 ký tự → mô hình dự đoán ký tự tương ứng.
- Hiển thị **Top-3 xác suất cao nhất**.
- Khi đánh giá mô hình sẽ hiển thị thêm **confusion matrix (heatmap)**.

---

## 📈 Đơn vị đo hiệu suất (cần fix)

- **Accuracy (%)**
- **F1-score (macro)**
- **Top-3 accuracy (%)**
- **Confusion matrix**

---

## Các quy trình

- **Dataset (Đọc dữ liệu)**
  - Thực hiện trong dataset_chars74k.py
  - Chỉ cần: load ảnh, trả về PIL image + label
- **Processing/Transform (xử lý ảnh đầu vào)**
  - Thực hiện trong image_transform.py
  - Tạo module xử lý ảnh đầu vào: resize, tensor, normalize
  - Tách ra file để dùng chung train + flask
- **Model (resNet18)**
  - Tạo file/class model
  - Chỉ cần forward run được
- **Training (lặp epoch + update)**
  - Viết function train()
  - lưu model .pth
- **Inference (load model + predict)**
  - Tạo inference module riêng
- **Flask**
  - Dùng function từ inference

## Mô hình truyền thống (baseline) để so sánh

- **hog_svm**
- **logreg_flatten**
