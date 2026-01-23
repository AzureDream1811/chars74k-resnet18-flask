import torch
from PIL import Image
import torchvision

from src.model.model_resnet18 import BuildResnet18
from src.transform.image_transform import get_inference_transform

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class ResnetInference:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = BuildResnet18(num_classes=62, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = get_inference_transform(image_size=64)

    def predict(self, img_path):
        pil_img = Image.open(img_path).convert("RGB")
        img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)

        pred_idx = torch.argmax(output, dim=1).item()
        pred_char = CHARSET[int(pred_idx)]

        return pred_char
