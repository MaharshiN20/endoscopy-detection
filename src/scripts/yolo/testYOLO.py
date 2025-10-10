from ultralytics import YOLO 
import os 


# dirs for relevant stuff
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

val_dir = "../../../data/dataset/data/val/images"
weights_path = "../../outputs/yolo/esophagous_opening/weights/best.pt"
output_dir = "../../outputs/yolo/val_predictions"

os.makedirs(output_dir,exist_ok=True)

# using weights from weights dir
model = YOLO(weights_path)

# model predict
results = model.predict(
    source=val_dir,
    save = True,
    project=output_dir,
    name = "predictions",
    exist_ok=True
)

print("Inference Complete.")