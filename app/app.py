from flask import Flask
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent[0]
MODEL_PATH = ROOT_DIR / "chars74k_resnet18.pth"

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello, World!"


if __name__ == "__main__":
    app.run(debug=True)
