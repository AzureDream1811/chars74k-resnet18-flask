"""
Evaluate model metrics: Accuracy, Precision, Recall, F1-score
"""
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.dataset.dataset_chars74k import Chars74KDataset
from src.transform.image_transform import get_test_transform
from src.model.model_resnet18 import BuildResnet18


# Character set for labels
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def get_device():
    """Return the device to use (either cpu or cuda)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def load_model(model_path, num_classes=62, device=None):
    """
    Load trained model from checkpoint.

    Parameters:
        model_path (str): path to model checkpoint
        num_classes (int): number of classes
        device: torch device

    Returns:
        model: loaded model in eval mode
    """
    if device is None:
        device = get_device()

    model = BuildResnet18(num_classes=num_classes, pretrained=False)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    return model


def create_test_loader(root_dir, batch_size=64, train_ratio=0.7, val_ratio=0.2, img_size=64, seed=None):
    """
    Create test dataloader (same split as training).

    Parameters:
        root_dir (str): path to dataset root directory
        batch_size (int): batch size for dataloader
        train_ratio (float): proportion for training
        val_ratio (float): proportion for validation
        img_size (int): image size for transform
        seed (int): random seed for reproducibility

    Returns:
        test_loader: DataLoader for test set
    """
    base_dataset = Chars74KDataset(root_dir=root_dir, transform=None)
    num_samples = len(base_dataset)

    # Use same random split as training
    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randperm(num_samples).tolist()

    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_indices = indices[train_size + val_size:]

    print(f"Total samples: {num_samples}")
    print(f"Test samples: {len(test_indices)}")

    test_transform = get_test_transform(image_size=img_size)
    test_base = Chars74KDataset(root_dir=root_dir, transform=test_transform)
    test_dataset = Subset(test_base, test_indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader


def evaluate_metrics(model, test_loader, device):
    """
    Evaluate model and compute metrics.

    Parameters:
        model: trained model
        test_loader: DataLoader for test set
        device: torch device

    Returns:
        dict: dictionary containing all metrics
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    return metrics, all_labels, all_preds


def print_metrics(metrics):
    """Print metrics in a formatted table."""
    print("\n" + "=" * 50)
    print("📊 MODEL EVALUATION METRICS")
    print("=" * 50)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Accuracy':<25} {metrics['accuracy']*100:>14.2f}%")
    print("-" * 42)
    print(f"{'Precision (macro)':<25} {metrics['precision_macro']*100:>14.2f}%")
    print(f"{'Recall (macro)':<25} {metrics['recall_macro']*100:>14.2f}%")
    print(f"{'F1-score (macro)':<25} {metrics['f1_macro']*100:>14.2f}%")
    print("-" * 42)
    print(f"{'Precision (weighted)':<25} {metrics['precision_weighted']*100:>14.2f}%")
    print(f"{'Recall (weighted)':<25} {metrics['recall_weighted']*100:>14.2f}%")
    print(f"{'F1-score (weighted)':<25} {metrics['f1_weighted']*100:>14.2f}%")
    print("=" * 50)


def print_classification_report(all_labels, all_preds):
    """Print detailed classification report."""
    print("\n📋 DETAILED CLASSIFICATION REPORT")
    print("=" * 70)

    target_names = [CHARSET[i] for i in range(62)]
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report)


def main(
    root_dir="data/raw/EnglishFnt/English/Fnt",
    model_path="chars74k_resnet18.pth",
    batch_size=64,
    img_size=64,
    show_detailed_report=False,
):
    """
    Main function to evaluate model metrics.

    Parameters:
        root_dir (str): path to dataset root directory
        model_path (str): path to model checkpoint
        batch_size (int): batch size for evaluation
        img_size (int): image size
        show_detailed_report (bool): whether to show per-class report
    """
    device = get_device()

    # Load model
    model = load_model(model_path, num_classes=62, device=device)

    # Create test loader
    test_loader = create_test_loader(
        root_dir=root_dir,
        batch_size=batch_size,
        img_size=img_size,
    )

    # Evaluate
    metrics, all_labels, all_preds = evaluate_metrics(model, test_loader, device)

    # Print results
    print_metrics(metrics)

    if show_detailed_report:
        print_classification_report(all_labels, all_preds)

    return metrics


if __name__ == "__main__":
    main(show_detailed_report=True)
