import os
import numpy as np
import PIL.Image as Image

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT_DIR = "data/raw/EnglishFnt/English/Fnt"
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def load_charset74k_flatten(root_dir=ROOT_DIR, max_per_class=300, img_size=32):
    """
    Load Chars74K dataset and flatten images into feature vectors.

    Parameters:
        root_dir (str): path to dataset root directory (default: "data/raw/EnglishFnt/English/Fnt")
        max_per_class (int): maximum number of samples to load per class (default: 300)
        img_size (int): image size to resize to (default: 32)

    Returns:
        tuple of feature vectors and labels
    """
    features = []
    labels = []
    count = 0

    for folder in sorted(os.listdir(root_dir)):
        if not folder.startswith("Sample"):
            continue

        class_dir = os.path.join(root_dir, folder)
        if not os.path.isdir(class_dir):
            continue

        sample_num = int(folder.replace("Sample", ""))
        label = sample_num - 1

        print(f"Loading class {sample_num}...")

        class_count = 0
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            img_path = os.path.join(class_dir, fname)

            # Open image in grayscale mode
            img = Image.open(img_path).convert("L")

            # Resize image to img_size x img_size
            img = img.resize((img_size, img_size))

            # Convert image to numpy array with float32 data type
            img_np = np.array(img, dtype=np.float32)

            # Normalize image by dividing all pixels by 255.0
            img_np = img_np / 255.0

            # Flatten image into a feature vector
            feat = img_np.flatten()

            # Append feature vector and label to lists
            features.append(feat)
            labels.append(label)

            # Increment counters
            count += 1
            class_count += 1

            # limit per class (fix previous global-only limit)
            if max_per_class is not None and class_count >= max_per_class:
                break

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    num_classes = len(np.unique(labels)) if labels.size > 0 else 0
    print(f"Loaded {count} samples across {num_classes} classes.")
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)

    return features, labels


def main():
    features, labels = load_charset74k_flatten(
        root_dir=ROOT_DIR, max_per_class=300, img_size=32
    )

    if features.shape[0] == 0:
        print("No data loaded. Exiting.")
        return

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Train size: ", train_features.shape[0])
    print("Test size: ", test_features.shape[0])

    # Create a Logistic Regression classifier
    # We use the lbfgs solver, which is a type of optimization algorithm
    # that is efficient for large datasets
    # We also set the maximum number of iterations to 1000
    # and the number of jobs to -1, which means that all available CPU cores will be used
    clf = LogisticRegression(
        solver="lbfgs",  # optimization algorithm
        max_iter=1000,  # maximum number of iterations
        n_jobs=-1,  # use all available CPU cores
    )

    print("Training Logistic Regression...")
    clf.fit(train_features, train_labels)

    y_pred = clf.predict(test_features)

    acc = accuracy_score(test_labels, y_pred)

    # Detailed metrics
    precision_macro = precision_score(test_labels, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(test_labels, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(test_labels, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(test_labels, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(test_labels, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(test_labels, y_pred, average="weighted", zero_division=0)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nMetrics summary:")
    print(f"  Precision (macro):   {precision_macro:.4f}")
    print(f"  Recall (macro):      {recall_macro:.4f}")
    print(f"  F1 (macro):          {f1_macro:.4f}")
    print(f"  Precision (weighted):{precision_weighted:.4f}")
    print(f"  Recall (weighted):   {recall_weighted:.4f}")
    print(f"  F1 (weighted):       {f1_weighted:.4f}")

    # Per-class report (use CHARSET for readable class names when available)
    num_classes = len(np.unique(test_labels))
    if num_classes <= len(CHARSET):
        target_names = [CHARSET[i] for i in range(num_classes)]
        print("\nClassification Report:")
        print(classification_report(test_labels, y_pred, target_names=target_names, zero_division=0))
    else:
        print("\nClassification Report:")
        print(classification_report(test_labels, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
