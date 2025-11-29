from .dataset_chars74k import Chars74KDataset
import matplotlib.pyplot as plt

def main():
    dataset = Chars74KDataset(root_dir="data/raw/EnglishFnt/English/Fnt")

    print("Tổng số ảnh:", len(dataset))

    # Lấy ảnh đầu tiên
    img, label = dataset[0]

    print("Label:", label)

    # Show ảnh bằng matplotlib
    plt.imshow(img)
    plt.title(f"Label = {label}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()