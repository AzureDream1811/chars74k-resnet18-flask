# Handwritten English Character Classification (Chars74K + ResNet-18)

## 📌 Mô tả

Bài toán phân loại ký tự tiếng Anh dạng viết in/viết thường gồm:  
**0–9, A–Z, a–z** (tổng tối đa 62 lớp).

- **Dataset:** Chars74K – Digital English Font  
  <https://www.kaggle.com/datasets/supreethrao/chars74kdigitalenglishfont>
- **Model:** CNN – ResNet-18 (tùy chỉnh cho ảnh grayscale 64×64)

---

## 📊 Kết quả Training

### Cấu hình thí nghiệm
- **Tổng số ảnh:** 62,992
- **Train/Val/Test:** 70%/20%/10% = 44,094 / 12,598 / 6,300
- **Batch size:** 64
- **Epochs:** 20
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Image size:** 64×64
- **Device:** CUDA (GPU)

### Hiệu suất mô hình

| Metric | Giá trị |
|--------|---------|
| **Final Test Accuracy** | **91.33%** |
| **Best Validation Accuracy** | 91.77% (Epoch 18) |
| **Final Training Loss** | 0.1106 |

### Training Progress

| Epoch | Train Loss | Val Accuracy |
|-------|-----------|--------------|
| 1/20  | 0.5799    | 84.93%       |
| 5/20  | 0.2547    | 88.93%       |
| 10/20 | 0.1846    | 90.10%       |
| 15/20 | 0.1409    | 91.73%       |
| 20/20 | 0.1106    | 90.78%       |

**Nhận xét:**
- Model hội tụ tốt sau 20 epochs
- Validation accuracy đạt đỉnh ~91.77% ở epoch 18
- Có dấu hiệu overfitting nhẹ (val acc giảm từ epoch 18→20)
- Training loss giảm đều đặn từ 0.5799 → 0.1106

---

## 🎯 Chức năng / Demo (Flask)

- Upload ảnh (PNG/JPG) chứa 1 ký tự → mô hình dự đoán ký tự tương ứng.
- Hiển thị **Top-3 xác suất cao nhất**.
- Khi đánh giá mô hình sẽ hiển thị thêm **confusion matrix (heatmap)**.

---

## 📈 Đơn vị đo hiệu suất

- ✅ **Accuracy (%)** - 91.33% trên test set
- 🔄 **F1-score (macro)** - (cần tính toán thêm)
- 🔄 **Top-3 accuracy (%)** - (cần tính toán thêm)
- 🔄 **Confusion matrix** - (cần tạo visualization)

---

## Cài đặt
```bash
python3 -m venv .venv
.venv\scripts\activate.ps1
pip install -r requirements.txt
```

## Cài đặt pytorch có hỗ trợ gpu
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

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
  
- **Model (ResNet18)**
  - Load ResNet18 (pretrained)
  - Thay đổi fc layer từ 1000 → 62 lớp
  - Forward trả về logits [batch, 62]
  
- **Training**
  - train.py
  - Load dataset (chia 70/20/10)
  - Lặp qua 20 epochs
    - forward → loss (CrossEntropyLoss)
    - backward → cập nhật weight (Adam optimizer)
  - Lưu model: `chars74k_resnet18.pth`
  
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

---

## Mô hình truyền thống (baseline) để so sánh

### Logistic Regression (Flatten)
- **File:** baseline/logreg_flatten.py
- **Phương pháp:** 
  - Ảnh → grayscale → resize 32×32 → flatten 1024 chiều
  - Train Logistic Regression đa lớp
- **Dataset:** 18,600 ảnh (300/class)
- **Train/Test:** 80%/20%
- **Accuracy:** ~85%

### So sánh ResNet18 vs Baseline

| Model | Accuracy | Tham số | Thời gian train |
|-------|----------|---------|----------------|
| Logistic Regression | ~85% | ~63K | Nhanh |
| **ResNet18** | **91.33%** | ~11M | ~20 epochs |

**Ý nghĩa Baseline:**
- Mô hình truyền thống không học được đặc trưng phức tạp từ ảnh
- ResNet18 học được cạnh, đường cong, stroke → độ chính xác cao hơn 6.33%
- Trade-off: ResNet18 phức tạp hơn nhưng cho kết quả tốt hơn đáng kể