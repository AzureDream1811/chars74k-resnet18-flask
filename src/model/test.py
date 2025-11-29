import torch
from .model_resnet18 import BuildResnet18


def main():
    """
    1. tạo model Resnet18
    2. model.eval(): chuyển model sang chế độ dự đoán (không học)
    3. tạo ảnh giả: randn(batch_size, channels (RGB), height, weight)
    """
    model = BuildResnet18(num_classes=62, pretrained=True, requires_grad=True)
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    output = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)


if __name__ == "__main__":
    main()
