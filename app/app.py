from flask import Flask, request, jsonify
from pathlib import Path
import torch
from PIL import Image

from src.model.model_resnet18 import BuildResnet18
from src.transform.image_transform import get_inference_transform

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "chars74k_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model and load weights
model = BuildResnet18(num_classes=62, pretrained=False)
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    # handle checkpoints that wrap state_dict
    model.load_state_dict(state["state_dict"])
else:
    model.load_state_dict(state)
model.to(device)
model.eval()

app = Flask(__name__)

transforms = get_inference_transform(image_size=64)

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@app.route("/")
def index():
    return "Hello, World!"


@app.post("/predict")
def predict():
    """
    Predict the character from an image.

    Request Body:
        image (file): an image file (jpg/png)

    Response:
        {
            "prediction": str,  # predicted character
            "index": int   # index of the predicted character in CHARSET
        }

    Errors:
        - "no image uploaded" (400): if no image is provided in the request body
        - "invalid image file" (400): if the image file is invalid
    """
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    image_file = request.files["image"]
    try:
        from io import BytesIO

        image = Image.open(BytesIO(image_file.read())).convert("RGB")
        # reset the file pointer in case the FileStorage is reused later
        try:
            image_file.stream.seek(0)
        except Exception:
            pass
    except Exception:
        return jsonify({"error": "invalid image file"}), 400

    img = transforms(image)
    # Ensure we have a torch.Tensor; if not, convert with torchvision.transforms.ToTensor
    if not isinstance(img, torch.Tensor):
        from torchvision import transforms as _tv_transforms

        img = _tv_transforms.ToTensor()(img)

    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    pred_idx = int(torch.argmax(output, dim=1).item())
    pred_char = CHARSET[pred_idx]

    return jsonify({"prediction": pred_char, "index": pred_idx})


if __name__ == "__main__":
    app.run(debug=True)
