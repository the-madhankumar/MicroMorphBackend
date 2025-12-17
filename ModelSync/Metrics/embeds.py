import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb

from uvision.embeddings import ImageEmbeddingEngine

client = chromadb.PersistentClient(path="../chroma_storage")
collection = client.get_or_create_collection("species_embeddings")
engine = ImageEmbeddingEngine()

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

def embedding_search_file(image_path, n_results=5):
    try:
        Image.open(image_path).convert("RGB")
    except:
        return {
            "detections": {},
            "best_one": "Unknown",
            "nearest_one": "Unknown",
            "conf": 0.0,
            "stats": {},
            "reliability": 0.0
        }

    emb = engine.generate_embeddings_from_image(image_path)
    response = collection.query(query_embeddings=[emb], n_results=n_results)
    metadatas = response["metadatas"][0]
    distances = response["distances"][0]

    compressedDict = {}
    for meta, dist in zip(metadatas, distances):
        cls = meta.get("class_name", "Unknown")
        if cls not in compressedDict:
            compressedDict[cls] = {"count": 1, "distances": [dist]}
        else:
            compressedDict[cls]["count"] += 1
            compressedDict[cls]["distances"].append(dist)

    if len(compressedDict) == 0:
        return {
            "detections": {},
            "best_one": "Unknown",
            "nearest_one": "Unknown",
            "conf": 0.0,
            "stats": {},
            "reliability": 0.0
        }

    best_one = max(compressedDict, key=lambda c: compressedDict[c]["count"])
    nearest_one = min(compressedDict, key=lambda c: sum(compressedDict[c]["distances"]) / len(compressedDict[c]["distances"]))

    dist = np.array(distances[1:])
    if len(dist) > 0:
        probs = np.exp(-dist) / np.sum(np.exp(-dist))
        conf = float(np.max(probs))
    else:
        conf = 0.0

    stats = brute_force_statistics(dist.tolist() if len(dist) > 0 else [])
    R = reliability_score(stats["mean"], stats["std"], stats["se"])

    return {
        "detections": compressedDict,
        "best_one": best_one,
        "nearest_one": nearest_one,
        "conf": conf,
        "stats": stats,
        "reliability": R
    }

confidence_scores = []
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

train = r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\Dataset\Species-3\train"

for root, dirs, files in os.walk(train):
    for filename in tqdm(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue

        img_path = os.path.join(root, filename)
        result = embedding_search_file(img_path, n_results=5)
        conf = result["conf"]
        confidence_scores.append(conf)
        print(f"[OK] {filename} → conf: {conf:.4f}")

stats = brute_force_statistics(confidence_scores)
R = reliability_score(stats["mean"], stats["std"], stats["se"])

print("\n==============================")
print(" EMBEDDING MODEL RELIABILITY")
print("==============================")
print("Mean Confidence     :", stats["mean"])
print("Std Deviation       :", stats["std"])
print("Standard Error      :", stats["se"])
print("Reliability Score   :", R)
print("==============================")
