import time
from ultralytics import YOLO

print("Testing batch=32, workers=2 on 100 batches...")
model = YOLO("yolo11s.pt")
t0 = time.time()
results = model.train(
    data="baseline_work/data.yaml",
    epochs=1,
    batch=32,
    imgsz=640,
    device=0,
    workers=2,
    plots=False,
    save=False,
    project="throughput_test",
    name="b32_test",
    exist_ok=True,
    fraction=0.1
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
