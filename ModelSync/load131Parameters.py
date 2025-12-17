import os
import json
import pandas as pd
from tqdm import tqdm

from Polygons.Extract import PolygonFeatureExtractor

# --------------------------------------------------
# COCO JSON files
# --------------------------------------------------
json_files = [
    r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/Dataset/Microorganisms--2/train/_annotations.coco.json",
    r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/Dataset/Microorganisms--2/test/_annotations.coco.json",
    r"D:/projects/MicroMorph AI/Project MicroMorph AI/ModelSync/Dataset/Microorganisms--2/valid/_annotations.coco.json"
]

# --------------------------------------------------
# Function to process a single json file
# --------------------------------------------------
def process_coco_json(json_path):
    print(f"\nProcessing: {json_path}")

    # Load JSON
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Class ID → Name
    class_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Collect polygon annotations
    polygon_items = []
    for ann in coco["annotations"]:
        cat_name = class_names[ann["category_id"]]

        # segmentation is a list of polygon arrays
        for poly in ann["segmentation"]:
            polygon_items.append({
                "class": cat_name,
                "polygon": poly
            })

    print("Total polygons:", len(polygon_items))

    # Extract polygon features
    feature_rows = []
    for item in tqdm(polygon_items, desc="Extracting features"):
        extractor = PolygonFeatureExtractor(
            polygon_points_list=[item["polygon"]]
        )
        features = extractor.compute_features()[0]

        features["target"] = item["class"]
        feature_rows.append(features)

    df = pd.DataFrame(feature_rows)
    return df


# --------------------------------------------------
# PROCESS ALL THREE FILES
# --------------------------------------------------
all_dfs = []

for jf in json_files:
    df = process_coco_json(jf)
    all_dfs.append(df)

# Merge all
final_df = pd.concat(all_dfs, ignore_index=True)

print("\nFINAL DataFrame Shape:", final_df.shape)
print(final_df.head())

# Save to CSV
os.makedirs("./polygon_metadata", exist_ok=True)
save_path = "./polygon_metadata/microorganism.csv"
final_df.to_csv(save_path, index=False)

print(f"\nSaved → {save_path}")
