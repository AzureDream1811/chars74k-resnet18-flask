import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset.dataset_chars74k import Chars74KDataset
from src.transform.image_transform import get_train_transform
from src.model.model_resnet18 import BuildResnet18


def train(
    data_root="data/raw/EnglishFnt/English/Fnt",
    num_epochs=10,
    batch_size=64,
    lr=1e-3,
    train_ratio=0.8,
    model_path="models/resnet18_chars74k.pth",
):
    """
    Hàm train chính:
    - Load dataset Chars74K
    - Chia train/val
    - Huấn luyện ResNet18
    - Lưu model tốt nhất (.pth)
    """
    # 1. Device: GPU nếu có, không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. Transform cho ảnh (resize + tensor + normalize)
    transform = get_train_transform(image_size=64)

    # 3. Dataset đầy đủ
    full_dataset = Chars74KDataset(root_dir=data_root, transform=transform)
    print("Tổng số ảnh:", len(full_dataset))

    # 4. Chia train/val
    train_size = int(len(full_dataset) * train_ratio)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Số ảnh train: {len(train_dataset)}")
    print(f"Số ảnh val  : {len(val_dataset)}")

    # 5. DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    # 6. Khởi tạo model
    model = BuildResnet18(num_classes=62, pretrained=True, requires_grad=True)
    model.to(device)

    # 7. Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 8. Thư mục lưu model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    best_val_acc = 0.0

    # 9. Vòng lặp epoch
    for epoch in range(num_epochs):
        print(f"\n===== Epoch [{epoch + 1}/{num_epochs}] =====")

        # --- TRAIN MODE ---
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # a) Reset gradient
            optimizer.zero_grad()

            # b) Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # c) Backward
            loss.backward()

            # d) Update weight
            optimizer.step()

            # e) Thống kê
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total

        # --- EVAL MODE (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}"
        )

        # 10. Lưu model tốt nhất (theo val_acc)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_path)
            print(f"👉 Saved best model to '{model_path}' (Val Acc: {best_val_acc:.4f})")

    print("\nHoàn thành training.")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    # Có thể chỉnh tham số ở đây
    train(
        data_root="data/raw/EnglishFnt/English/Fnt",
        num_epochs=1,
        batch_size=64,
        lr=1e-3,
        train_ratio=0.8,
        model_path="models/resnet18_chars74k.pth",
    )
