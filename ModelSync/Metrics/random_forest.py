import joblib
from ultralytics import YOLO
import cv2
import numpy as np
from Polygons.Extract import PolygonFeatureExtractor
import pandas as pd
import os
from tqdm import tqdm 

RANDOMFOREST_MODEL = r"D:\projects\MicroMorph AI\Models\RandomForest\random_forest_model.pkl"
YOLOSEG_RF = r"D:\projects\MicroMorph AI\Models\YOLOSeg\best.pt"

confidence_scores = []

train = r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\Dataset\Species-3\train"

def random_forest_inference(img_np):
    model = YOLO(YOLOSEG_RF)
    rf_loaded = joblib.load(RANDOMFOREST_MODEL)
    feature_columns = rf_loaded.feature_names_in_
    results = model(img_np)[0]
    if not results:
        return None
    h, w = results.orig_shape
    coco_polygons = []
    for mask_tensor in results.masks.data:
        mask = mask_tensor.cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0].reshape(-1, 2)
            coco_polygon = contour.flatten().tolist()
            coco_polygons.append(coco_polygon)
    extractor = PolygonFeatureExtractor(polygon_points_list=coco_polygons)
    features_all = extractor.compute_features()
    features_vec = features_all[0] if features_all else {}
    if "centroid" in features_vec and isinstance(features_vec["centroid"], tuple):
        features_vec["centroid_x"] = float(features_vec["centroid"][0])
        features_vec["centroid_y"] = float(features_vec["centroid"][1])
        del features_vec["centroid"]
    clean_features = {}
    for k in feature_columns:
        clean_features[k] = float(features_vec.get(k, 0.0))
    feature_df = pd.DataFrame([clean_features], columns=feature_columns)
    prediction = rf_loaded.predict(feature_df)

    conf = np.max(rf_loaded.predict_proba(feature_df))

    return {"predicted_value": prediction[0], "features": clean_features, "conf": conf}

def brute_force_statistics(conf_list):
    N = len(conf_list)
    if N == 0:
        return {"mean": 0, "std": 0, "se": 0}

    total = 0.0
    for c in conf_list:
        total += c
    mean = total / N

    sq_diff_sum = 0.0
    for c in conf_list:
        sq_diff_sum += (c - mean) ** 2
    std = (sq_diff_sum / (N - 1)) ** 0.5 if N > 1 else 0.0

    se = std / (N ** 0.5)

    return {"mean": mean, "std": std, "se": se}


image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for root, dirs, files in os.walk(train):
    for filename in tqdm(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue  

        image_path = os.path.join(root, filename)

        img = cv2.imread(image_path)
        if img is None:
            print("Could not load:", image_path)
            continue

        output = random_forest_inference(img)
        if not output:
            continue

        confidence_scores.append(output["conf"])

        print(f"[OK] {filename} → conf: {output['conf']:.4f}")

def reliability_score(mean, std, se):
    raw = mean / (1 + std + se)
    return max(0.0, min(1.0, raw))

stats = brute_force_statistics(confidence_scores)

R = reliability_score(stats["mean"], stats["std"], stats["se"])

print("\n==============================")
print(" RANDOM FOREST RELIABILITY")
print("==============================")
print("Mean Confidence     :", stats["mean"])
print("Std Deviation       :", stats["std"])
print("Standard Error      :", stats["se"])
print("Reliability Score   :", R)
print("==============================")