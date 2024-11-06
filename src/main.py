from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.tune(data="xView.yaml", epochs=1, iterations=200, plots=False, save=False, val=False)

model.export(format="onnx")
