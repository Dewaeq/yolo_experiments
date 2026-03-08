import sys
from ultralytics.models import YOLO


# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="lvis.yaml", epochs=100, imgsz=640)

# model = YOLO("yolo11n-seg.pt")
# model = YOLO("yolo26x-seg.pt")
#
# results = model(sys.argv[-1])
#
# for r in results:
#     r.show()
