from flask import Flask, request, jsonify
from pathlib import Path
from PIL import Image
from numpy import argmax
import torch

from src.model.model_resnet18 import BuildResnet18
from src.transform.image_transform import get_inference_transform

# Paths and device
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT.parent / "chars74k_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = BuildResnet18(num_classes=62, pretrained=False)
state = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()


# Transforms and charset
transforms = get_inference_transform(image_size=64)
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


app = Flask(__name__)


@app.get("/")
def index():
    return "chars74k ResNet18 inference"


@app.post("/predict")
def predict():
    """
    Predict a character from an image.

    The API endpoint accepts a multipart/form-data request with an image file.

    Returns a JSON response with the predicted character and its index in the character set.

    Example response:
    {
        "prediction": "A",
        "index": 0
    }

    If no image is provided, returns a 400 error response with a JSON error message.
    Example response:
    {
        "error": "No image uploaded"
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file)

    tensor = transforms(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    output = model(tensor)
    index = int(output.argmax(dim=1).item())
    char = CHARSET[index]

    return jsonify({"prediction": char, "index": index}), 200


if __name__ == "__main__":
    app.run(debug=True)
