import pandas as pd
from Polygons.Extract import PolygonFeatureExtractor
from ultralytics import YOLO
import cv2
import numpy as np
import joblib

# Load YOLO segmentation model
model = YOLO("D:/projects/MicroMorph AI/Models/YOLOSeg/best.pt")

# Load trained Random Forest model
rf_loaded = joblib.load("D:/projects/MicroMorph AI/Models/RandomForest/random_forest_model.pkl")
feature_columns = rf_loaded.feature_names_in_  # Columns used during training

# Load and process image
img_path = "D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/Dataset/Species-3/test/Alexandrium_17_png.rf.ff1f0e8a67ff8ecaa4eaca9c4fe83618.jpg"
results = model(img_path)[0]

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

for i, poly in enumerate(coco_polygons, 1):
    print(f"Object {i} polygon (COCO segmentation):")
    print(poly)

# Extract polygon features
extractor = PolygonFeatureExtractor(polygon_points_list=coco_polygons)
features_all = extractor.compute_features()
features_vec = features_all[0]

# Convert centroid tuple into separate features
if "centroid" in features_vec and isinstance(features_vec["centroid"], tuple):
    features_vec["centroid_x"] = float(features_vec["centroid"][0])
    features_vec["centroid_y"] = float(features_vec["centroid"][1])
    del features_vec["centroid"]

# Clean features: convert all to float
clean_features = {}
for k in feature_columns:
    clean_features[k] = float(features_vec.get(k, 0.0))  # fill missing with 0.0

feature_df = pd.DataFrame([clean_features], columns=rf_loaded.feature_names_in_)
prediction = rf_loaded.predict(feature_df)
print("Prediction:", prediction[0])
