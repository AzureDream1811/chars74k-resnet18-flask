import torch
from src.model.model_resnet18 import BuildResnet18


def main():
    """
    1. Tạo model Resnet18 (pretrained, thay fc từ 1000 -> 62 class)
    2. model.eval(): chuyển model sang chế độ dự đoán (không học)
    3. Tạo ảnh giả: randn(batch_size, channels (RGB), height, weight)
    4. Nhận biết test thành công:
        - Không báo lỗi khi output = model(x)
        - Output shape: (1, 62)
            -> fc được thay từ đúng 1000 -> 62
        - Output là 1 tensor có 62 số thực (logits) ví dụ:
            tensor([[..., 0.5, -0.2, 1.3, ...]])
    5. pred (class dự đoán): 
        - trong 62 số đó, số lớn nhất nằm ở index nào -> là class model dự đoán
    """
    model = BuildResnet18(num_classes=62, pretrained=True, requires_grad=True)
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    output = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    pred = torch.argmax(output, dim=1)
    print("Predicted class:", pred.item())


if __name__ == "__main__":
    main()
