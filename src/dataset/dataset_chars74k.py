import os
from torch.utils.data import Dataset
from PIL import Image

class Chars74KDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for label in range(62):
            folder = f"Sample{label+1:03d}"
            class_dir = os.path.join(self.root, folder)
            if not os.path.isdir(class_dir):
                continue

            for file in os.listdir(class_dir):
                if file.endswith(".png"):
                    img_path = os.path.join(class_dir, file)
                    samples.append((img_path, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
