import os
import cv2
import numpy as np

# path setup
script_dir = os.getcwd()
dataset_dir = os.path.join(script_dir, "../../data/dataset/data")

# train & val subdirectories
sets = ["train", "val"]
for s in sets:
    image_dir = os.path.join(dataset_dir, s, "images")
    mask_dir = os.path.join(dataset_dir, s, "masks")
    label_dir = os.path.join(dataset_dir, s, "labels")
    
    print(f"Processing {s} set: {len(os.listdir(image_dir))} images found.")

    for mask_file in os.listdir(mask_dir):
        if not mask_file.endswith((".png", ".jpg")):
            continue
            
        # load mask as grayscale
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, 0)

        if mask is None:
            print(f"Warning: could not read {mask_path}")
            continue
        
        # find object pixels
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len (ys) == 0:
            # no object in mask -> create empty label
            label_path = os.path.join(label_dir, mask_file.rsplit('.',1)[0] + ".txt")
            open(label_path, "w").close()
            continue
        
        # bounding box coordinates
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        h, w = mask.shape

        # normalize for yolo
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        width = (x_max - x_min) / w
        height = (y_max - y_min) / h

        # save label
        label_path = os.path.join(label_dir, mask_file.rsplit('.',1)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("Label generation complete for train & val sets.")