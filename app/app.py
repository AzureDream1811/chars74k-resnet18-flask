from io import BytesIO
from flask import Flask, render_template, request, jsonify
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
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# Transforms and charset
transform = get_inference_transform(image_size=64)
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

app = Flask(__name__)


@app.route("/")
def index():
    """Render modern homepage with tabs"""
    return render_template("index.html")


@app.route("/upload")
def upload():
    """Render legacy upload page (optional)"""
    return render_template("upload.html")


@app.route("/draw")
def draw():
    """Render legacy draw page (optional)"""
    return render_template("draw.html")


@app.post("/predict")
def predict():
    """
    Predicts the character from an image.
    Returns Top-3 predictions with confidence scores.

    Returns:
        JSON object with:
        - top_prediction: The most likely character
        - top3_results: List of top 3 predictions with confidence scores
    
    Status codes:
        200: Prediction successful
        400: No image uploaded
        500: Server error
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files["image"]
        image_data = image_file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Transform and predict
        tensor = transform(image)
        assert isinstance(tensor, torch.Tensor)
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get Top-3 predictions
            top3_prob, top3_indices = probabilities.topk(3)
            
            results = []
            for prob, idx in zip(top3_prob, top3_indices):
                results.append({
                    "char": CHARSET[int(idx)],
                    "confidence": f"{prob.item() * 100:.2f}%",
                    "probability": float(prob.item())
                })

        return jsonify({
            "prediction": results[0]["char"],  # For backward compatibility
            "top_prediction": results[0]["char"],
            "top3_results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Model loaded on device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    app.run(debug=True, host="0.0.0.0", port=5000)