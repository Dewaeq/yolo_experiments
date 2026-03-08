import argparse
from ultralytics.models import YOLO


def train_model(args):
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.resolution, project=args.project
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "train a YOLO model via the awesome (!!) ultralytics library"
    )
    parser.add_argument("--model", type=str, default="yolo26n-seg")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--data",
        type=str,
        default="./yolo_fsoco/fsoco.yaml",
        help="Path to data.yaml file",
    )
    parser.add_argument(
        "--project", type=str, default="train_runs", help="Project name (folder)"
    )

    args = parser.parse_args()
    train_model(args)
