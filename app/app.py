from io import BytesIO
from flask import Flask, request, jsonify
from pathlib import Path
from PIL import Image
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
transform = get_inference_transform(image_size=64)
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


app = Flask(__name__)


@app.get("/")
def index():
    return "chars74k ResNet18 inference"


@app.post("/predict")
def predict():
    """
    Predicts the character from an image.

    Returns a JSON object with the predicted character and its index in the CHARSET.

    :statuscode 200: Prediction successful
    :statuscode 400: No image uploaded
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_data = image_file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")

    tensor = transform(image)
    assert isinstance(tensor, torch.Tensor)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    output = model(tensor)
    index = int(output.argmax(dim=1).item())
    char = CHARSET[index]

    return jsonify({"prediction": char, "index": index}), 200


if __name__ == "__main__":
    app.run(debug=True)
