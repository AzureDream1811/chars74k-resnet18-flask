import torch

def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("num devices:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device name:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()