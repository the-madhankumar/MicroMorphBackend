import os
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

YOLO_MODEL_PATH = r"D:\projects\MicroMorph AI\Models\YOLO\yolo_model.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.fuse()

def brute_force_statistics(values):
    N = len(values)
    if N == 0:
        return {"mean": 0, "std": 0, "se": 0}
    mean = sum(values) / N
    sq_diff = sum((v - mean) ** 2 for v in values)
    std = (sq_diff / (N - 1)) ** 0.5 if N > 1 else 0.0
    se = std / (N ** 0.5)
    return {"mean": mean, "std": std, "se": se}

def reliability_score(mean, std, se):
    raw = mean / (1 + std + se)
    return max(0.0, min(1.0, raw))

def yolo_search_file(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"classes": [], "conf": 0.0, "stats": {}, "reliability": 0.0}
    
    results = yolo_model(img)[0]
    confidences = []

    if results.boxes is not None:
        for box in results.boxes:
            conf = float(box.conf)
            confidences.append(conf)
    
    conf_array = np.array(confidences[1:]) if len(confidences) > 1 else np.array([])
    conf = float(np.max(conf_array)) if conf_array.size > 0 else 0.0
    stats = brute_force_statistics(conf_array.tolist() if conf_array.size > 0 else [])
    R = reliability_score(stats["mean"], stats["std"], stats["se"])
    
    classes = [results.names.get(int(box.cls), "Unknown") for box in results.boxes] if results.boxes is not None else []
    
    return {"classes": classes, "conf": conf, "stats": stats, "reliability": R}

confidence_scores = []
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
train = r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\Dataset\Species-3\train"

for root, dirs, files in os.walk(train):
    for filename in tqdm(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue
        img_path = os.path.join(root, filename)
        result = yolo_search_file(img_path)
        confidence_scores.append(result["conf"])
        print(f"[OK] {filename} → conf: {result['conf']:.4f}")

stats = brute_force_statistics(confidence_scores)
R = reliability_score(stats["mean"], stats["std"], stats["se"])

print("\n==============================")
print(" YOLO MODEL RELIABILITY")
print("==============================")
print("Mean Confidence     :", stats["mean"])
print("Std Deviation       :", stats["std"])
print("Standard Error      :", stats["se"])
print("Reliability Score   :", R)
print("==============================")
