from .dataset_chars74k import Chars74KDataset

def main():
    dataset = Chars74KDataset(root="data/raw/EnglishFnt/English/Fnt")
    print("Tổng số ảnh:", len(dataset))

    img, label = dataset[0]
    print(type(img))
    print(label)
    img.show()

if __name__ == "__main__":
    main()
