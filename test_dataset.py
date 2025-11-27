from src.dataset.dataset_chars74k import Chars74KDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    # pipeline transform dùng cho dataset (tối thiểu phải convert to tensor)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = Chars74KDataset(
        root_dir="data/raw/EnglishFnt/English/Fnt",
        transform=transform
    )

    print("Tổng số mẫu:", len(dataset))

    img, label = dataset[0]
    print("Kiểu ảnh:", type(img))  # lúc này là Tensor
    print("Label id:", label)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(loader))
    print("Batch tensor shape:", images.shape)
    print("Batch labels:", labels[:10])

if __name__ == "__main__":
    main()
