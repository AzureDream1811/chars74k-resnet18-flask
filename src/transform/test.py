import torch
import matplotlib.pyplot as plt
import numpy as np

# Import Dataset (Yêu cầu 1)
# Giả định cấu trúc module cho phép import từ 'dataset'
from src.dataset.dataset_chars74k import Chars74KDataset

# Import Transform (Yêu cầu 2)
from image_transform import get_test_transform, IMAGENET_MEAN, IMAGENET_STD


# Hàm đảo ngược chuẩn hóa để hiển thị ảnh
def denormalize(tensor, mean, std):
    """
    Đảo ngược phép chuẩn hóa: (tensor * std) + mean.
    """
    # Tạo tensor mean và std có shape (C, 1, 1) để broadcast
    mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)

    # Thực hiện (tensor * std) + mean
    tensor = tensor * std + mean

    # Kẹp giá trị về [0, 1]
    return torch.clamp(tensor, 0, 1)


def main():
    # 1. Khởi tạo Transform
    transform_func = get_test_transform(image_size=64)

    # 2. Khởi tạo Dataset, truyền Transform vào
    # LƯU Ý: Thay đổi đường dẫn đến dữ liệu của bạn
    root_dir = "D:/EMNIST/chars74k-resnet18-flask/data/raw/EnglishFnt/English/Fnt"

    try:
        # Sự liên kết: Dataset.__init__ nhận transform_func
        dataset = Chars74KDataset(root_dir=root_dir, transform=transform_func)

        print(f"Tổng số ảnh trong Dataset: {len(dataset)}")

        # 3. Lấy ảnh đầu tiên
        # Đầu ra (img) sẽ là một PyTorch Tensor (không phải PIL Image nữa)
        tensor_img, label = dataset[0]

        print(f"--- Kết quả sau Transform ---")
        print(f"Label: {label}")
        print(f"Kích thước Tensor: {tensor_img.shape} (C, H, W)")
        print(f"Kiểu dữ liệu: {tensor_img.dtype}")
        print(f"Giá trị Min/Max: {tensor_img.min():.4f} / {tensor_img.max():.4f}")

        # 4. Hiển thị ảnh kiểm tra

        # a) Đảo ngược chuẩn hóa
        denorm_img_tensor = denormalize(tensor_img, IMAGENET_MEAN, IMAGENET_STD)

        # b) Chuyển Tensor (C, H, W) sang NumPy Array (H, W, C)
        img_np = denorm_img_tensor.permute(1, 2, 0).numpy()

        # c) Hiển thị
        plt.imshow(img_np)
        plt.title(f"Ảnh đã được Transform và Denormalized (Size 64x64)\nLabel: {label}")
        plt.axis("off")
        plt.show()

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại '{root_dir}'.")
    except ImportError:
        print("Lỗi Import: Không thể import 'Chars74KDataset'. Đảm bảo cấu trúc thư mục và import là chính xác.")


if __name__ == "__main__":
    main()