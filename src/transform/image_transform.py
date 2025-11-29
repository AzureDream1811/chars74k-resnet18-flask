from torchvision import transforms
import torch

# Hằng số chuẩn hóa (Mean và Std của ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform(image_size=64):
    """
    Tạo transform cho quá trình huấn luyện: Resize -> ToTensor -> Normalize.
    """
    return transforms.Compose([
        # 1. Resize: Đảm bảo ảnh có cùng kích thước (liên kết với đầu ra PIL Image của Dataset)
        transforms.Resize((image_size, image_size)),
        # 2. ToTensor: Chuyển PIL Image sang Tensor, chuẩn hóa về [0, 1]
        transforms.ToTensor(),
        # 3. Normalize: Chuẩn hóa Tensor với Mean/Std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_test_transform(image_size=64):
    """
    Tạo transform cho quá trình kiểm tra/suy luận.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_inference_transform(image_size=64):
    """
    Transform dùng cho Flask (suy luận), tương tự test transform.
    """
    return get_test_transform(image_size)

if __name__ == "__main__":
    print("Module image_transform.py đã sẵn sàng.")