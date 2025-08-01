from ultralytics import YOLO

def main():

    # Load a pretrained YOLO11s model
    model = YOLO("yolo11s.pt")

    train_results = model.train(
        data="config.yaml",
        epochs=100,
        batch = 32,
        device=0, 
    )

if __name__ == "__main__":
    main()
