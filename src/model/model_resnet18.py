import torch.nn as nn
from torchvision import models


class BuildResnet18(nn.Module):
    """
    1. resnet gốc có output 1000 -> thay bằng output 62
    2. weight: kiến thức có sẵn -> nhanh hơn, chính xác hơn
    3. self.model.parameters(): danh sách weight + bias để model học
    4. requires_rad: bật/tắt việc học trọng số
    5. for param in ...: lặp param để cho weight + bias để bật/tắt học
    6. self.model.fc: lớp cuối cùng của resnet (fully connected layer)
    7. self.model.fc.in_features: số input của lớp fc (thường 512)
    8. nn.Linear(in_feature, num_classes): thay fc mới (512 -> 62)
    """

    def __init__(self, num_classes=62, pretrained=True, requires_grad=True):
        super(BuildResnet18, self).__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        in_feature = self.model.fc.in_features

        self.model.fc = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
