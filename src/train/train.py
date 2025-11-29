import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

from src.dataset.dataset_chars74k import Chars74KDataset
from src.transform.image_transform import get_train_transform, get_test_transform
from src.model.model_resnet18 import BuildResnet18


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 3
    learning_rate = 1e-3
    root_dir = "data/raw/EnglishFnt/English/Fnt"

    train_tf = get_train_transform(image_size=64)
    test_tf = get_test_transform(image_size=64)

    full_dataset = Chars74KDataset(root_dir=root_dir, transform=train_tf)
    print(f"Tổng số ảnh trong Dataset: {len(full_dataset)}")

    val_ratio = 0.1
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train size: {train_size}, Val size {val_size}")

    val_dataset.dataset.transform = test_tf
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    model = BuildResnet18(num_classes=62, pretrained=True, requires_grad=True)
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            
        avg_train_loss = running_loss/ len(train_loader)

        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():  # tắt gradient khi test
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)          # [batch, 62]
                _, predicted = torch.max(outputs, dim=1)  # lấy index lớn nhất

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}]  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

    # =======================
    # 7. Lưu model
    # =======================
    torch.save(model.state_dict(), "chars74k_resnet18.pth")
    print("Đã lưu model vào chars74k_resnet18.pth")


if __name__ == "__main__":
    main()
