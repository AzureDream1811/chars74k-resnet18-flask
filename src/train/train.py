import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset.dataset_chars74k import Chars74KDataset
from src.transform.image_transform import get_train_transform, get_test_transform
from src.model.model_resnet18 import BuildResnet18
from src.train.evaluate_metrics import evaluate_metrics, print_metrics


def get_device():
    """
    Return the device to use (either cpu or cuda).
    Print the device used.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def create_dataloaders(root_dir, batch_size=64, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, img_size=64):
    """
    Create train, validation and test dataloaders from Chars74KDataset.

    Parameters:
        root_dir (str): path to dataset root directory
        batch_size (int): batch size for dataloader (default: 64)
        train_ratio (float): proportion for training (default: 0.7)
        val_ratio (float): proportion for validation (default: 0.2)
        test_ratio (float): proportion for testing (default: 0.1)
        img_size (int): image size for transform (default: 64)

    Returns:
        tuple of train, validation and test dataloaders
    """
    base_dataset = Chars74KDataset(root_dir=root_dir, transform=None)
    num_samples = len(base_dataset)
    print(f"Total image in dataset: {num_samples}")

    indices = torch.randperm(num_samples).tolist()

    # Tính kích thước train
    train_size = int(num_samples * train_ratio)
    # Tính kích thước val
    val_size = int(num_samples * val_ratio)
    # Tính kích trước test
    test_size = num_samples - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    print(f"train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")

    train_transform = get_train_transform(image_size=img_size)
    test_transform = get_test_transform(image_size=img_size)

    train_base = Chars74KDataset(root_dir=root_dir, transform=train_transform)
    val_base = Chars74KDataset(root_dir=root_dir, transform=test_transform)
    test_base = Chars74KDataset(root_dir=root_dir, transform=test_transform)

    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)
    test_dataset = Subset(test_base, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_model(
        num_classes=62, lr=1e-3, pretrained=True, requires_grad=True, device=None
):
    """
    Build a model, criterion, and optimizer for training.

    Parameters:
        num_classes (int): number of classes in classification dataset (default: 62)
        lr (float): learning rate for optimizer (default: 1e-3)
        pretrained (bool): whether to use pretrained weights (default: True)
        requires_grad (bool): whether to require gradient for model parameters (default: True)
        device (torch.device): device to use (default: None)

    Returns:
        tuple of model, criterion, and optimizer
    """
    if device is None:
        device = get_device()

    model = BuildResnet18(
        num_classes=num_classes, pretrained=pretrained, requires_grad=requires_grad
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train one epoch of a model on a given dataset.

    Parameters:
        model (nn.Module): model to train
        train_loader (DataLoader): data loader for training set
        criterion (nn.Module): loss function to use
        optimizer (torch.optim.Optimizer): optimizer to use
        device (torch.device): device to use for training

    Returns:
        float: average loss of the epoch
    """
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

    avg_loss = running_loss / len(train_loader)
    return avg_loss


def evaluate(model, val_loader, device):
    """
    Evaluate the accuracy of a model on a given validation set.

    Parameters:
        model (nn.Module): model to evaluate
        val_loader (DataLoader): data loader for validation set
        device (torch.device): device to use for evaluation

    Returns:
        float: accuracy of the model on the validation set
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predict = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predict == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc


def main(
        root_dir="data/raw/EnglishFnt/English/Fnt",
        num_classes=62,
        batch_size=64,
        num_epochs=20,
        lr=1e-3,
        image_size=64,
        pretrained=True,
        requires_grad=True,
        save_path="chars74k_resnet18.pth",
):
    """
    Main function to train a ResNet18 model on the Chars74K dataset.

    Parameters:
        root_dir (str): path to dataset root directory (default: "data/raw/EnglishFnt/English/Fnt")
        num_classes (int): number of classes in the dataset (default: 62)
        batch_size (int): batch size for dataloader (default: 64)
        num_epochs (int): number of epochs to train (default: 3)
        lr (float): learning rate for optimizer (default: 1e-3)
        image_size (int): image size for transform (default: 64)
        pretrained (bool): whether to use pretrained weights (default: True)
        requires_grad (bool): whether to require gradient for model parameters (default: True)
        save_path (str): path to save the model (default: "chars74k_resnet18.pth")
    """
    device = get_device()

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        img_size=image_size
    )

    model, criterion, optimizer = build_model(
        num_classes,
        lr,
        pretrained,
        requires_grad,
        device,
    )

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)  # Dùng val_loader thay vì test_loader

        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

    # Sau khi training xong, evaluate trên test set:
    test_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Đã lưu model vào {save_path}")

    # Run full evaluation on test set and print detailed metrics
    try:
        metrics, labels, preds = evaluate_metrics(model, test_loader, device)
        print_metrics(metrics)
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
