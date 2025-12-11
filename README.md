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

- **Dataset**
  - Thực hiện trong dataset_chars74k.py
  - Load ảnh - trả về:
    - PIL Image
    - label (0-61)
  - không xử lý ảnh, transform sẽ làm
- **Transform**
  - Thực hiện trong image_transform.py
  - Resize về 64x64
  - ToTensor(CHW)
  - Normalize theo ImageNet
- **Model (resNet18)**
  - Load Resnet18 (pretrained)
  - Thay đổi fc layer từ 1000 - 62 lớp
  - Forward trả về logits [batch, 62]
- **Training**
  - train.py
  - Load dataset
  - Lặp qua epoch
    - forward -> loss
    - backward -> cập nhật weight
      Lưu model:
      -> chars74k_resnet18.pth
- **Inference (load model + predict)**
  - Load .pth
  - Áp dụng transform inference
  - Trả về:
    - Top-1 prediction
    - Top-K probabilities
- **Flask**
  - Upload ảnh
  - Gọi inference module
  - Render kết quả dự đoán

## Mô hình truyền thống (baseline) để so sánh

- **hog_svm**
  - baseline/logreg_flatten.py
  - Ảnh -> grayscale -> resize 32x32 -> flatten 1024 chiều
  - Train Logistic Regression đa lớp
  - datasetL 18,600 ảnh (300/class)
  - Train/Test: 80%/20%
  - Accuracy thu được ~85%
- **logreg_flatten**

- **Ý nghĩa Baseline**
  - Cho thấy mô hình truyền thống không học được đặc trưng ảnh
  - ResNet18 học được cạnh, đường cong, stroke → độ chính xác cao hơn
