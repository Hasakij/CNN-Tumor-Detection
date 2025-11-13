import comet_ml
from ultralytics import YOLO

model = YOLO('yolo11m.pt') 

results = model.train(
    data='mri_config.yaml',
    name='yolo11m-mri-run',
    project="comet-yolo11-MRI",
    batch=32,
    save_json=True,
    epochs=50,
    imgsz=128
)

print("Training ended.")
print(results)