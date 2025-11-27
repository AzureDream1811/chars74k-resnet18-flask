# ================== PHẦN IMPORT THƯ VIỆN ==================

# numpy: thư viện làm việc với mảng số, tính toán toán học
import numpy as np

import os

# torch: thư viện PyTorch, dùng để xây dựng và train mạng nơ-ron
import torch
import torch.nn as nn  # nn: nơi chứa các lớp mạng nơ-ron (Linear, Conv, Loss,...)
import torch.optim as optim  # optim: chứa các thuật toán tối ưu (Adam, SGD,...)

# Dùng để chia dữ liệu thành batch, shuffle,... khi train
from torch.utils.data import random_split, DataLoader

# torchvision.transforms: các phép biến đổi ảnh (chuyển sang tensor, chuẩn hóa,...)
import torchvision.transforms as transforms

# Các hàm đo lường hiệu suất từ scikit-learn
from sklearn.metrics import (
    accuracy_score,  # tính Accuracy
    f1_score,  # tính F1-score
    log_loss,  # tính Log loss
    confusion_matrix,  # tạo confusion matrix
    top_k_accuracy_score,  # tính Top-k accuracy (ở đây là Top-3)
)

# Import class Dataset tự viết, nằm trong cùng thư mục src
from dataset_chars74k import Chars74KDataset

# Import hàm build_model (tạo ResNet18) tự viết
from model_resnet18 import build_model

# ================== CÁC THAM SỐ CẤU HÌNH ==================

# Chọn thiết bị chạy: nếu có GPU (cuda) thì dùng, không có thì dùng CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Số lượng ảnh trong mỗi "lần nhét" vào mạng (batch)
BATCH_SIZE = 64

# Số vòng lặp qua toàn bộ dữ liệu train
EPOCHS = 5  # ban đầu để 5 cho nhẹ, sau này có thể tăng

# Tốc độ học (learning rate) - bước nhảy khi cập nhật trọng số
LR = 1e-3  # 0.001

# Đường dẫn đến thư mục chứa dataset gốc
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data", "raw", "English", "Fnt")


# ================== HÀM TRAIN 1 EPOCH ==================

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Train model trên toàn bộ tập train (1 epoch).

    model     : mạng nơ-ron (ResNet18)
    loader    : DataLoader chứa dữ liệu train chia batch
    criterion : hàm mất mát (loss function)
    optimizer : thuật toán tối ưu (Adam, SGD,...)
    """

    # Đưa model sang chế độ "train" (bật dropout, batchnorm, ...)
    model.train()

    # Biến lưu tổng loss (để sau tính trung bình)
    total_loss = 0.0

    # Các list để lưu nhãn thật, nhãn dự đoán, và xác suất
    all_labels = []
    all_preds = []
    all_probs = []

    # Vòng lặp qua từng batch dữ liệu trong loader
    for images, labels in loader:
        # Đưa ảnh và nhãn lên đúng DEVICE (CPU/GPU)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Xóa gradient cũ (tránh cộng dồn từ lần trước)
        optimizer.zero_grad()

        # Cho ảnh đi qua model để lấy output (logits)
        outputs = model(images)

        # Tính loss giữa output và labels
        loss = criterion(outputs, labels)

        # Tính gradient bằng backpropagation
        loss.backward()

        # Cập nhật trọng số model dựa trên gradient
        optimizer.step()

        # Cộng dồn loss để lát nữa tính trung bình
        total_loss += loss.item()

        # Chuyển output sang xác suất bằng softmax
        probs = torch.softmax(outputs, dim=1)

        # Lấy nhãn dự đoán là index có xác suất lớn nhất
        preds = torch.argmax(probs, dim=1)

        # Lưu nhãn thật, nhãn dự đoán và xác suất về CPU, chuyển sang numpy
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Tính loss trung bình trên tất cả batch
    avg_loss = total_loss / len(loader)

    # Tính Accuracy (độ chính xác)
    acc = accuracy_score(all_labels, all_preds)

    # Tính F1-score (trung bình macro: các lớp được coi ngang nhau)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # Tính Top-3 accuracy:
    #   Kiểm tra nhãn thật có nằm trong 3 xác suất cao nhất hay không
    top3 = top_k_accuracy_score(all_labels, np.array(all_probs), k=3)

    # Tính Log loss (càng nhỏ càng tốt)
    ll = log_loss(all_labels, np.array(all_probs))

    # Trả về các kết quả đo được
    return avg_loss, acc, f1, top3, ll


# ================== HÀM EVALUATE (ĐÁNH GIÁ) ==================

def evaluate(model, loader, criterion):
    """
    Đánh giá model trên tập validation (không cập nhật trọng số).
    Cách tính giống train_one_epoch nhưng không gọi backward / optimizer.step().
    """

    # Chuyển model sang chế độ đánh giá (tắt dropout, batchnorm dùng thống kê cố định)
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    # with torch.no_grad(): nói với PyTorch là không cần lưu gradient
    # → tiết kiệm RAM, nhanh hơn (vì không train, chỉ đánh giá)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Tính loss trung bình, Accuracy, F1, Top-3, Log loss giống train
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    top3 = top_k_accuracy_score(all_labels, np.array(all_probs), k=3)
    ll = log_loss(all_labels, np.array(all_probs))

    # Tính confusion matrix (ma trận nhầm lẫn)
    cm = confusion_matrix(all_labels, all_preds)

    # Trả về metrics + ma trận nhầm lẫn
    return avg_loss, acc, f1, top3, ll, cm


# ================== HÀM MAIN – GHÉP TẤT CẢ LẠI ==================

def main():
    """
    Hàm main sẽ được gọi khi chạy:
        python src/train.py

    Nhiệm vụ:
    - Tạo dataset + dataloader
    - Tạo model, loss, optimizer
    - Vòng lặp EPOCH: train + evaluate
    - Lưu model tốt nhất + confusion matrix + classes.txt
    """

    # 1. Tạo transform cho ảnh:
    #    - ToTensor(): chuyển ảnh PIL -> tensor (0..1)
    #    - Normalize((0.5,), (0.5,)): chuẩn hóa về khoảng [-1, 1]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # 2. Tạo dataset từ thư mục DATA_ROOT với transform ở trên
    dataset = Chars74KDataset(DATA_ROOT, transform=transform)

    # 3. Đếm số class (số loại ký tự) từ dataset
    num_classes = len(dataset.classes)

    # 4. Chia dataset thành train / val theo tỉ lệ 80 / 20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # 5. Tạo DataLoader cho train & validation
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Tạo model ResNet18 với num_classes lớp đầu ra, đưa lên DEVICE
    model = build_model(num_classes).to(DEVICE)

    # 7. Chọn hàm loss: CrossEntropyLoss dùng cho phân loại nhiều lớp
    criterion = nn.CrossEntropyLoss()

    # 8. Chọn optimizer: Adam với learning rate LR
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 9. In thông tin cơ bản
    print(f"Using device: {DEVICE}")
    print(f"Num classes: {num_classes}")
    print(f"Train size: {train_size}, Val size: {val_size}")

    # best_val_acc: dùng để lưu lại model nào có độ chính xác validation tốt nhất
    best_val_acc = 0.0
    best_cm = None  # lưu confusion matrix tương ứng

    # 10. Vòng lặp qua từng epoch
    for epoch in range(1, EPOCHS + 1):
        # Train trên tập train
        train_loss, train_acc, train_f1, train_top3, train_ll = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # Đánh giá trên tập validation
        val_loss, val_acc, val_f1, val_top3, val_ll, cm = evaluate(
            model, val_loader, criterion
        )

        # In kết quả của epoch này
        print(
            f"Epoch {epoch}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% "
            f"F1: {train_f1 * 100:.2f}% Top3: {train_top3 * 100:.2f}% LogLoss: {train_ll:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% "
            f"F1: {val_f1 * 100:.2f}% Top3: {val_top3 * 100:.2f}% LogLoss: {val_ll:.4f}"
        )

        # Nếu accuracy trên validation tốt hơn model trước đó → lưu lại model này
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cm = cm
            torch.save(model.state_dict(), "model_best.pth")
            print(f"  -> Saved new best model (val_acc={val_acc * 100:.2f}%)")

    # 11. Sau khi train xong, nếu có confusion matrix tốt nhất thì lưu ra file .npy
    if best_cm is not None:
        np.save("confusion_matrix.npy", best_cm)
        print("Saved confusion_matrix.npy")

    # 12. Lưu mapping index -> tên class (Sample001, Sample002, ...)
    #     Sau này Flask sẽ dùng để biết: index 0 là ký tự nào, index 1 là ký tự nào,...
    with open("classes.txt", "w", encoding="utf-8") as f:
        for idx, name in enumerate(dataset.classes):
            f.write(f"{idx}\t{name}\n")

    print("Training done.")


# Đoạn này để đảm bảo chỉ chạy main khi file được chạy trực tiếp,
# không chạy khi file được import bởi file khác.
if __name__ == "__main__":
    main()
