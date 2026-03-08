from ultralytics.models import YOLO

model = YOLO("yolo26n-seg")
results = model.train(data="./yolo_fsoco/fsoco.yaml", epochs=50, imgsz=640)
