from ultralytics import YOLO

model=YOLO("yolov8l.pt")

results=model.train(data="coco128.yaml", epochs=3)