from ultralytics import YOLO
import resource
import time, datetime
import shutil


class Timer:
    def __init__(self):
        return

    def set(self):
        self.start = time.perf_counter()
        return

    def get(self):
        return datetime.timedelta(seconds=(time.perf_counter() - self.start))


MEMORY_LIMIT_KB = int(2.5e10)  # 25 GB
resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_KB, MEMORY_LIMIT_KB))

timer = Timer()
timer.set()

RESUME = False

if RESUME:
    model = YOLO("latest/train/weights/last.pt")
else:
    model = YOLO("yolo11n.pt")

print(f"//// Model load time: {timer.get()} ////")

# timer.set()
# model.tune(
#     data="config/xView.yaml", epochs=1, imgsz=512, batch=4, workers=1, fraction=0.2, project="latest", exist_ok=True,
#     hsv_h=0.2, hsv_s=0.2, hsv_v=0.2, degrees=45, erasing=0.2, crop_fraction=0.8,
#     use_ray=False, iterations=100, plots=False, save=False, val=False
# )
# print(f"//// Model tune time: {timer.get()} ////")


classes = [
    #     0, # Fixed-wing Aircraft
    #     1, # Small Aircraft
    #     2, # Cargo Plane
    #     3, # Helicopter
    #     4, # Passenger Vehicle
    #     5, # Small Car
    #     6, # Bus
    #     7, # Pickup Truck
    #     8, # Utility Truck
    #     9, # Truck
    #     10, # Cargo Truck
    #     11, # Truck w/Box
    #     12, # Truck Tractor
    #     13, # Trailer
    #     14, # Truck w/Flatbed
    #     15, # Truck w/Liquid
    #     16, # Crane Truck
    #     17, # Railway Vehicle
    #     18, # Passenger Car
    #     19, # Cargo Car
    #     20, # Flat Car
    #     21, # Tank car
    #     22, # Locomotive
    #     23, # Maritime Vessel
    #     24, # Motorboat
    #     25, # Sailboat
    #     26, # Tugboat
    #     27, # Barge
    #     28, # Fishing Vessel
    #     29, # Ferry
    #     30, # Yacht
    #     31, # Container Ship
    #     32, # Oil Tanker
    #     33, # Engineering Vehicle
    #     34, # Tower crane
    #     35, # Container Crane
    #     36, # Reach Stacker
    #     37, # Straddle Carrier
    #     38, # Mobile Crane
    #     39, # Dump Truck
    #     40, # Haul Truck
    #     41, # Scraper/Tractor
    #     42, # Front loader/Bulldozer
    #     43, # Excavator
    #     44, # Cement Mixer
    #     45, # Ground Grader
    46,  # Hut/Tent
    47,  # Shed
    48,  # Building
    #     49, # Aircraft Hangar
    50,  # Damaged Building
    51,  # Facility
    52,  # Construction Site
    #     53, # Vehicle Lot
    #     54, # Helipad
    55,  # Storage Tank
    #     56, # Shipping container lot
    #     57, # Shipping Container
    #     58, # Pylon
    #     59, # Tower
]


"""
model, imgsz (px), batch, memory (GB), time (minutes/epoch)
y11m, 128, 64, 
y11m, 256, 16, 5
y11m, 512, 4, 6, SUCCESS, 35
"""
timer.set()
if RESUME:
    model.train(resume=True)
else:
    try:
        shutil.rmtree(path="/app/latest/train")
    except FileNotFoundError:
        pass
    model.train(
        data="/datasets/xview_chipped/xview.yaml",
        epochs=100,
        imgsz=128,
        batch=64,
        workers=2,
        fraction=1.0,
        project="latest",
        exist_ok=False,
        # classes=classes,
        hsv_h=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=45,
        erasing=0.2,
        crop_fraction=0.8,
        multi_scale=True,
        optimizer="SGD",
        momentum=0.9,
        lr0=0.001,
        lrf=0.001,
    )
print(f"//// Model train time: {timer.get()} ////")

model.export(format="engine")
