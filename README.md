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

## 🚀 Flow & Giải thích nhanh (Training, Evaluation) 

Dưới đây là mô tả ngắn gọn từng bước để thầy hoặc người mới có thể hiểu quy trình mô hình hoạt động — bao gồm cả flow khi huấn luyện (train) và khi đánh giá (evaluate), kèm tóm tắt các đơn vị đo hiệu suất (Accuracy / Precision / Recall / F1) theo Why / When / How / What.

### A. Flow khi huấn luyện (training)
1. Chuẩn bị dữ liệu
   - Nguồn: `data/raw/EnglishFnt/English/Fnt` (thư mục `SampleXXX` tương ứng từng lớp).
   - File liên quan: `src/dataset/dataset_chars74k.py`.
2. Tiền xử lý (transform)
   - Resize → ToTensor → Normalize theo ImageNet mean/std (64×64).
   - File liên quan: `src/transform/image_transform.py`.
3. Chia tập và tạo DataLoader
   - Chia 70%/20%/10% (train/val/test) như trong `src/train/train.py`.
4. Xây dựng mô hình
   - ResNet‑18 từ `torchvision`, thay `fc` → 62 output (class). File: `src/model/model_resnet18.py`.
5. Huấn luyện
   - Loss: `CrossEntropyLoss`; Optimizer: `Adam` (lr=1e-3).
   - Vòng lặp: forward → loss → backward → optimizer.step.
6. Lưu checkpoint
   - Lưu weights: `chars74k_resnet18.pth`.

_Lệnh chạy training (PowerShell):_
```powershell
cd "D:\Coding\Projects\learning_1st_semester_2025\AI programming\chars74k-resnet18-flask"
.venv\scripts\activate.ps1
python -m src.train.train
```

### B. Flow khi đánh giá (evaluation)
1. Load model checkpoint và set `model.eval()`
   - File: `src/train/evaluate_metrics.py` (script đã có) hoặc load trong `app/app.py` để inference từng ảnh.
2. Tạo test DataLoader bằng transform test (giữ cùng split nếu muốn tái lập).
3. Forward qua toàn bộ test set (không grad): thu `y_true` và `y_pred`.
4. Tính metric với `sklearn.metrics` (accuracy, precision, recall, f1). Có thể in `classification_report` và vẽ `confusion_matrix` để visualization.

_Lệnh chạy đánh giá (PowerShell):_
```powershell
cd "D:\Coding\Projects\learning_1st_semester_2025\AI programming\chars74k-resnet18-flask"
.venv\scripts\activate.ps1
python -m src.train.evaluate_metrics
```

---

## 📏 Đơn vị đo hiệu suất — Why / When / How / What

1) Accuracy
- Why: đo tỉ lệ dự đoán đúng trên tổng mẫu.
- When: dùng để biết tổng quan khi các lớp tương đối cân bằng.
- How: accuracy = số dự đoán đúng / tổng mẫu.
- What: hàm dùng: `sklearn.metrics.accuracy_score(y_true, y_pred)`; report dưới dạng phần trăm.

2) Precision
- Why: đo độ chính xác của các dự đoán cho mỗi lớp (khi model nói "là X" thì có bao nhiêu là đúng).
- When: quan trọng khi false positives tốn kém.
- How: precision = TP / (TP + FP).
- What: hàm: `sklearn.metrics.precision_score(y_true, y_pred, average='macro')` (hoặc `weighted`).

3) Recall
- Why: đo khả năng tìm đủ các mẫu thực sự thuộc 1 lớp (không bỏ sót).
- When: quan trọng khi false negatives tốn kém.
- How: recall = TP / (TP + FN).
- What: hàm: `sklearn.metrics.recall_score(y_true, y_pred, average='macro')` (hoặc `weighted`).

4) F1‑score
- Why: là sự cân bằng giữa precision và recall — hữu ích khi cần trade‑off.
- When: dùng khi dataset không cân bằng hoặc cần 1 chỉ số tóm tắt hơn accuracy.
- How: F1 = 2 * (precision * recall) / (precision + recall).
- What: hàm: `sklearn.metrics.f1_score(y_true, y_pred, average='macro')` (hoặc `weighted`).

---

## 🔎 Ghi chú quan trọng
- Luôn nêu rõ kiểu average (`macro` / `weighted`) khi báo Precision/Recall/F1.
- Nếu muốn tái lập kết quả chính xác, set random seed trước khi chia dataset hoặc lưu indices split.
- Nên kèm `confusion_matrix` (heatmap) để minh hoạ các cặp class hay nhầm lẫn (ví dụ O ↔ 0, l ↔ 1).

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