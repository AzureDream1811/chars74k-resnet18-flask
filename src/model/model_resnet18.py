import torch.nn as nn
from torchvision import models


class BuildResnet18(nn.Module):
    def __init__(self, num_classes=62, pretrained=True, requires_grad=True):
        """
        Initialize a ResNet18 model with specified number of classes, pretrained weights and require gradient flag.

        Parameters:
            num_classes (int): number of classes in the output layer (default: 62)
            pretrained (bool): whether to use pretrained weights (default: True)
            requires_grad (bool): whether to require gradient for model parameters (default: True)
        """
        super(BuildResnet18, self).__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        for param in self.model.parameters():
            param.requires_grad = requires_grad

        in_feature = self.model.fc.in_features

        self.model.fc = nn.Linear(in_feature, num_classes)

    def forward(self, images):
        images = self.model(images)
        return images
