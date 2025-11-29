import os
from PIL import Image
from torch.utils.data import Dataset

<<<<<<< HEAD
=======
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

>>>>>>> origin/main
class Chars74KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for folder in sorted(os.listdir(root_dir)):
            if not folder.startswith("Sample"):
                continue

            class_dir = os.path.join(root_dir, folder)
            if not os.path.isdir(class_dir):
                continue

            sample_num = int(folder.replace("Sample", ""))
            label_idx = sample_num - 1

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(class_dir, fname)
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

<<<<<<< HEAD
        return img, label
=======
        return img, label
>>>>>>> origin/main
