from ultralytics import YOLO
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# path to yaml
dataset_yaml = "../../../configs/yolov11_dataset.yaml"

# model version
model = YOLO("yolov8n")

# training model
model.train(
    data = dataset_yaml,
    epochs = 50,
    imgsz = 640,
    batch = 16,
    project = os.path.join(script_dir,"../../outputs/yolo"),
    name = "esophagous_opening"
)

print("Model Trained Successfully.")