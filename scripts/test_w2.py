import time
from ultralytics import YOLO

def main():
    print("Testing workers=2, batch=16 on 20 batches...")
    model = YOLO("yolo11s.pt")
    t0 = time.time()
    results = model.train(
        data="baseline_work/data.yaml",
        epochs=1,
        batch=16,
        imgsz=640,
        device=0,
        workers=2,
        amp=True,
        plots=False,
        save=False,
        project="throughput_test",
        name="w2_test",
        exist_ok=True,
        fraction=0.03
    )
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
