from ultralytics import YOLO
import resource
import time, datetime


class Timer:
    def __init__(self):
        return
    def set(self):
        self.start = time.perf_counter()
        return
    def get(self):
        return datetime.timedelta(seconds=(time.perf_counter() - self.start))


MEMORY_LIMIT_KB = int(2.5e10)
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_KB, MEMORY_LIMIT_KB))

timer = Timer()    
timer.set()

model = YOLO("yolo11n.pt")

print(f"//// Model load time: {timer.get()} ////")

# model.tune(data="xView.yaml", epochs=1, imgsz=1024, batch=1, iterations=100, plots=False, save=False, val=False)

timer.set()

model.train(
    data="config/xView.yaml", epochs=100, imgsz=1280, batch=1, workers=0,
    hsv_h=0.5, hsv_s=0.5, hsv_v=0.5, degrees=90, erasing=0.5, crop_fraction=0.5)

print(f"//// Model train time: {timer.get()} ////")

# model.export(format="onnx")
