from pathlib import Path

from src.inference.Inference import ResnetInference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "chars74k_resnet18.pth"
# IMG_PATH = PROJECT_ROOT / "data" / "raw" / "EnglishFnt" / "English" / "Fnt" / "Sample042" / "img042-00001.png"
IMG_PATH = PROJECT_ROOT / "test" / "test_img.png"

def main():
    infer = ResnetInference(str(MODEL_PATH))
    pred = infer.predict(str(IMG_PATH))
    print("Image: ", str(IMG_PATH))
    print("Prediction: ", pred)

if __name__ == '__main__':
    main()