import time
from ultralytics import YOLO

def test_speed(workers, batch, cache):
    print(f"\n--- Testing workers={workers}, batch={batch}, cache={cache} ---")
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    # Train 1 epoch on 500 images or 1 quick epoch to measure it/s
    results = model.train(
        data="baseline_work/data.yaml",
        epochs=1,
        batch=batch,
        imgsz=640,
        device=0,
        workers=workers,
        cache=cache,
        plots=False,
        save=False,
        val=False,
        project="throughput_test",
        name=f"w{workers}_b{batch}",
        exist_ok=True,
        fraction=0.1  # 10% of dataset (1469 images, ~90 batches)
    )
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s for 10% epoch (extrapolated full epoch: {elapsed*10/60:.1f} min)")

if __name__ == "__main__":
    test_speed(workers=4, batch=16, cache=False)
    test_speed(workers=4, batch=32, cache=False)
