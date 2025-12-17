import json
import os
import io
import shutil
import base64
import tempfile
import time
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms, models
from ultralytics import YOLO
from uvision.embeddings import ImageEmbeddingEngine
from Polygons.Extract import PolygonFeatureExtractor
import chromadb
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile as StarletteUploadFile
import firebase_admin
from firebase_admin import credentials, db, storage
from urllib.parse import quote_plus
from Bio import SeqIO
import torch.nn as nn
from io import BytesIO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GCRequest(BaseModel):
    sequence: str | None = None
    file_path: str | None = None
    window_size: int = 1000

def gc_sliding_window(sequence: str, window_size: int = 1000):
    sequence = sequence.upper().replace("\n", "").replace(" ", "")

    positions = []
    gc_values = []

    for start in range(0, len(sequence) - window_size + 1, window_size):
        window = sequence[start:start + window_size]

        gc = window.count("G") + window.count("C")
        percent = (gc / window_size) * 100

        positions.append(start)
        gc_values.append(percent)

    return positions, gc_values

def kmer_counts(sequence, k=6):
    from collections import Counter
    from itertools import product
    bases = 'ACGT'
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    counts = Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))
    return [counts[kmer] for kmer in kmers]

def compute_gc_summary(seq: str):
    length = len(seq)

    counts = {
        "A": seq.count("A"),
        "C": seq.count("C"),
        "G": seq.count("G"),
        "T": seq.count("T"),
    }

    other = length - sum(counts.values())
    counts["Other"] = other

    percentages = {base: round((count / length) * 100, 2) for base, count in counts.items()}

    gc_content = round(((counts["G"] + counts["C"]) / length) * 100, 2)

    return {
        "gc_content": gc_content,
        "length": length,
        "counts": counts,
        "percentages": percentages
    }


public_inference_folder = r"D:\projects\MicroMorph AI\Project MicroMorph AI\frontend\public\inference_images"
if not os.path.exists(public_inference_folder):
    os.makedirs(public_inference_folder, exist_ok=True)
app.mount("/inference_images", StaticFiles(directory=public_inference_folder), name="inference_images")

RANDOMFOREST_MODEL = r"D:\projects\MicroMorph AI\Models\RandomForest\random_forest_model.pkl"
YOLOSEG_RF = r"D:\projects\MicroMorph AI\Models\updated\YOLOv11nseg\best.pt"
YOLO_MODEL = r"D:\projects\MicroMorph AI\Models\updated\YOLOv11nseg\best.pt"
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("species_embeddings")
engine = ImageEmbeddingEngine()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
NUM_CLASSES = 12
CLASS_NAMES = [
    "Alexandrium","Cerataulina","Ceratium","Entomoneis",
    "Guinardia","Hemiaulus","Nitzschia","Pinnularia",
    "Pleurosigma","Prorocentrum","UnknownClass","Other"
]
MASK_R_CNN_PATH = r"D:\projects\MicroMorph AI\Models\MaskRCNN\model_epoch_10.pth"
MaskRCNN_model = models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES)
state_dict = torch.load(MASK_R_CNN_PATH, map_location=DEVICE)
MaskRCNN_model.load_state_dict(state_dict)
MaskRCNN_model.to(DEVICE)
MaskRCNN_model.eval()
transform = transforms.Compose([transforms.ToTensor()])

# RESNET

num_classes = 10  

resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

resnet_model.load_state_dict(torch.load(r"D:\projects\MicroMorph AI\Models\ResNet\resnet_model.pth", map_location=DEVICE))

resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()

def np_to_bytes(np_img):
    pil_img = Image.fromarray(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return buf.getvalue()

def img_to_base64(img: np.ndarray):
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def read_image(upload_file: UploadFile):
    upload_file.file.seek(0)
    contents = upload_file.file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    return img

def mask_rcnn_inference(img: np.ndarray, threshold: float = 0.4):
    image_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predictions = MaskRCNN_model(image_tensor)
    masks = predictions[0].get('masks', torch.tensor([]))
    boxes = predictions[0].get('boxes', torch.tensor([]))
    labels = predictions[0].get('labels', torch.tensor([]))
    scores = predictions[0].get('scores', torch.tensor([]))
    overlay = img.copy()
    best_class = ""
    best_score = 0.0
    outputs = []
    count = masks.shape[0] if masks is not None and masks.ndim > 0 else 0
    for i in range(count):
        score = float(scores[i].cpu().item()) if scores is not None else 0.0
        if score < threshold:
            continue
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        mask_bool = mask > 127
        box = boxes[i].cpu().numpy().astype(int).tolist()
        class_index = int(labels[i].cpu().item()) if labels is not None else -1
        class_name = CLASS_NAMES[class_index - 1] if 1 <= class_index <= len(CLASS_NAMES) else "UnknownClass"
        if score > best_score:
            best_class = class_name
            best_score = score
        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
        colored_mask = np.zeros_like(overlay)
        for c in range(3):
            colored_mask[:, :, c] = mask_bool * color[c]
        alpha = 0.4
        overlay = np.where(
            mask_bool[:, :, None],
            ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8),
            overlay
        )
        x1, y1, x2, y2 = box
        cv2.putText(
            overlay,
            f"{class_name}: {score:.2f}",
            (max(0, x1), max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
        outputs.append({
            "confidence": score,
            "class_name": class_name,
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        })
    return overlay, best_class, best_score, outputs

firebase_key_path = r"./firebaseCred/micromorphai-firebase-adminsdk-fbsvc-b75052a815.json"
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://micromorphai-default-rtdb.firebaseio.com/",
        "storageBucket": "micromorphai.appspot.com"
    })
bucket = storage.bucket()
yolo_model = YOLO(YOLO_MODEL)
yolo_model.fuse()


def save_yolo_result_to_firebase(results):
    # 1. Save plotted YOLO image
    plotted_img = results[0].plot(conf=False, labels=False)

    ok1, buffer_plot = cv2.imencode(".jpg", plotted_img)
    if not ok1:
        raise RuntimeError("Failed to encode plotted image")

    encoded_plot = base64.b64encode(buffer_plot).decode("utf-8")
    db.reference("/main_image_/yolo_output").set(encoded_plot)

    # 2. Save raw YOLO image (no plot)
    raw_img = results[0].orig_img  # << THIS is what you asked for

    ok2, buffer_raw = cv2.imencode(".jpg", raw_img)
    if not ok2:
        raise RuntimeError("Failed to encode raw YOLO image")

    encoded_raw = base64.b64encode(buffer_raw).decode("utf-8")
    db.reference("/main_image_/orig_image").set(encoded_raw)

    return {
        "plotted": encoded_plot,
        "raw": encoded_raw
    }

def save_crop_to_firebase_realtime(crop_np):
    ok, buffer = cv2.imencode(".jpg", crop_np)
    if not ok:
        raise RuntimeError("Failed to encode image")

    img_base64 = base64.b64encode(buffer).decode("utf-8")

    file_id = str(uuid.uuid4())

    ref = db.reference(f"inference_images/{file_id}")
    ref.set({
        "image": img_base64
    })

    return img_base64, file_id


SAVE_PATH = r"D:\projects\MicroMorph AI\Project MicroMorph AI\frontend\src\components\Unknown\Unknowns\u2"

def savetofolder():
    ref = db.reference("/main_image_/orig_image")
    encoded_img = ref.get()

    if not encoded_img:
        print("No original image found in Firebase")
        return

    img_data = base64.b64decode(encoded_img)
    img_array = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    os.makedirs(SAVE_PATH, exist_ok=True)

    filename = os.path.join(SAVE_PATH, f"unknown_{int(time.time())}.jpg")
    cv2.imwrite(filename, img)

    print(f"Image saved to: {filename}")


def store_inference_to_firebase(data, file_id):
    root_ref = db.reference("/inference_images")

    if not hasattr(store_inference_to_firebase, "_deleted"):
        root_ref.delete()
        store_inference_to_firebase._deleted = True  
    root_ref.child(file_id).set(data)

def update_class_count(class_name):
    ref = db.reference("/class_counts").child(class_name)
    current = ref.get() or 0
    ref.set(current + 1)

def yolo_get_crops(img):
    results = yolo_model.predict(img)
    save_yolo_result_to_firebase(results)
    h, w = img.shape[:2]
    crops = []
    expand_w, expand_h = 40, 40

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            try:
                xy = box.xyxy[0].cpu().numpy()
            except Exception:
                continue

            x1, y1, x2, y2 = map(int, xy.tolist())

            x1 = max(0, x1 - expand_w)
            y1 = max(0, y1 - expand_h)
            x2 = min(w, x2 + expand_w)
            y2 = min(h, y2 + expand_h)

            cls = int(box.cls[0]) if hasattr(box, "cls") and len(box.cls) > 0 else -1
            name = yolo_model.names.get(cls, "Unknown") if isinstance(yolo_model.names, dict) else yolo_model.names[int(cls)] if cls >= 0 else "Unknown"
            conf = float(box.conf[0]) if hasattr(box, "conf") and len(box.conf) > 0 else None

            if x2 > x1 and y2 > y1:
                crop_np = img[y1:y2, x1:x2].copy()
                crops.append({
                    "crop_np": crop_np,
                    "class_name": name,
                    "confidence": conf
                })

    return crops

def random_forest_inference(img_np):
    model = YOLO(YOLOSEG_RF)
    rf_loaded = joblib.load(RANDOMFOREST_MODEL)
    feature_columns = rf_loaded.feature_names_in_
    results = model(img_np)[0]
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

def embedding_search_file(upload_file: UploadFile, n_results: int = 5):
    upload_file.file.seek(0)
    try:
        image = Image.open(upload_file.file).convert("RGB")
        emb = engine.generate_embeddings_from_image(image)
    except Exception:
        upload_file.file.seek(0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(upload_file.file.read())
        tmp.flush()
        tmp.close()
        emb = engine.generate_embeddings_from_image(tmp.name)
        os.unlink(tmp.name)
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
        return {"detections": {}, "best_one": "Unknown", "nearest_one": "Unknown"}
    best_one = max(compressedDict, key=lambda c: compressedDict[c]["count"])
    nearest_one = min(compressedDict, key=lambda c: sum(compressedDict[c]["distances"]) / len(compressedDict[c]["distances"]))
    
    dist = np.array(distances[1:])
    probs = np.exp(-dist) / np.sum(np.exp(-dist))
    conf = np.max(probs)

    if len(compressedDict) == 1:
        only_class = list(compressedDict.keys())[0]
        best_one = only_class
        nearest_one = only_class
    return {"detections": compressedDict, "best_one": best_one, "nearest_one": nearest_one, "conf": conf}

def combined_route_frontend(file: UploadFile = File(...)):
    img = read_image(file)
    crops = yolo_get_crops(img)
    if not crops:
        return {"error": "No organisms detected"}
    results = []
    for crop in crops:
        crop_np = crop["crop_np"]
        _, mask_class, mask_conf, _ = mask_rcnn_inference(crop_np)
        try:
            rf_res = random_forest_inference(crop_np)
            rf_label = rf_res.get("predicted_value")
        except Exception:
            rf_label = "Unknown"
        file.file.seek(0)
        emb = embedding_search_file(file, n_results=5)
        emb_label = emb.get("best_one", "Unknown")
        yolo_label = crop.get("class_name", "Unknown")
        vote = {}
        for c in [mask_class or "Unknown", yolo_label or "Unknown", rf_label or "Unknown", emb_label or "Unknown"]:
            vote[c] = vote.get(c, 0) + 1
        final_label = max(vote, key=vote.get)
        try:
            crop_url, file_id = save_crop_to_firebase_realtime(crop_np)
        except Exception as e:
            crop_url = None
            file_id = str(uuid.uuid4())
        result = {
            "final_class": final_label,
            "crop_image_url": crop_url,
            "votes": vote,
            "model_outputs": {
                "mask_r_cnn": mask_class,
                "yolo": yolo_label,
                "random_forest": rf_label,
                "embedding": emb_label
            },
            "confidence": {
                "mask_r_cnn": float(mask_conf) if mask_conf is not None else None,
                "yolo": float(crop.get("confidence")) if crop.get("confidence") is not None else None,
                "random_forest": None,
                "embedding": None
            }
        }
        try:
            store_inference_to_firebase(result, file_id)
        except Exception:
            pass
        try:
            update_class_count(final_label)
        except Exception:
            pass
        results.append(result)
    return {"total_crops": len(results), "results": results}

def yolo_inference(img_np):
    model = YOLO(YOLO_MODEL)
    model.fuse()
    results = model(img_np)[0]
    results.show()
    predictions = []
    if hasattr(results, "boxes") and results.boxes is not None:
        for cls_idx in results.boxes.cls:
            try:
                class_name = results.names[int(cls_idx)]
            except Exception:
                class_name = "Unknown"
            predictions.append(class_name)
    return predictions

def resnet_inference(model, image_bytes, device, class_names):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)                     
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    predicted_class = class_names[pred.item()]
    confidence_score = float(confidence.item())   

    print("\n--- PREDICTION RESULT ---")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence     : {confidence_score:.4f}")

    return predicted_class, confidence_score

def microEggsReference(img, threshold=0.5):
    MASK_R_CNN_PATH = r"D:\projects\MicroMorph AI\Models\MicroEggs\microeggs.pth"
    MaskRCNN_model = models.detection.maskrcnn_resnet50_fpn(weights=None, num_classes=3)
    state_dict = torch.load(MASK_R_CNN_PATH, map_location=DEVICE)
    MaskRCNN_model.load_state_dict(state_dict)
    MaskRCNN_model.to(DEVICE)
    MaskRCNN_model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = MaskRCNN_model(image_tensor)

    masks = predictions[0].get("masks", torch.tensor([]))
    boxes = predictions[0].get("boxes", torch.tensor([]))
    labels = predictions[0].get("labels", torch.tensor([]))
    scores = predictions[0].get("scores", torch.tensor([]))

    overlay = img.copy()
    best_class = ""
    best_score = 0.0
    outputs = []

    count = masks.shape[0] if masks is not None and masks.ndim > 0 else 0

    for i in range(count):
        score = float(scores[i].cpu().item()) if scores is not None else 0.0
        if score < threshold:
            continue

        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        mask_bool = mask > 127
        box = boxes[i].cpu().numpy().astype(int).tolist()

        class_index = int(labels[i].cpu().item()) if labels is not None else -1
        class_name = ["micro-eggs"]

        if score > best_score:
            best_class = class_name
            best_score = score

        color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
        colored_mask = np.zeros_like(overlay)
        for c in range(3):
            colored_mask[:, :, c] = mask_bool * color[c]

        alpha = 0.4
        overlay = np.where(
            mask_bool[:, :, None],
            ((1 - alpha) * overlay + alpha * colored_mask).astype(np.uint8),
            overlay
        )

        x1, y1, x2, y2 = box
        # cv2.putText(
        #     overlay,
        #     f"{class_name}: {score:.2f}",
        #     (max(0, x1), max(0, y1 - 10)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     color,
        #     2
        # )

        outputs.append({
            "confidence": score,
            "class_name": class_name,
            "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
        })

    return overlay, best_class, best_score, outputs


@app.get("/")
async def root():
    return {"message": "API working successfully"}

@app.post("/micro-eggs")
async def micro_eggs_api(file: UploadFile = File(...)):
    content = await file.read()
    file_bytes = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    overlay, best_class, best_score, outputs = microEggsReference(img)

    pil_img = Image.fromarray(overlay)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    base64_image = base64.b64encode(buff.getvalue()).decode()

    return {
        "detected_class": best_class,
        "confidence": best_score,
        "count": len(outputs),
        "detections": outputs,
        "masked_image": base64_image
    }

@app.post("/embedding")
async def embed_similarity(file: UploadFile = File(...), n_results: int = 5):
    result = embedding_search_file(file, n_results=n_results)
    return result

@app.post("/random_forest")
async def random_forest_route(uploadfile: UploadFile):
    contents = await uploadfile.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(image)
    result = random_forest_inference(img_np)
    return result

@app.post("/mask_r_cnn")
async def mask_r_cnn_route(file: UploadFile = File(...)):
    img = read_image(file)
    annotated_img, detected_class, confidence, outputs = mask_rcnn_inference(img)
    return {"image_base64": img_to_base64(annotated_img), "detected_class": detected_class, "confidence": confidence, "detections": outputs}

from collections import Counter
import math

def aggregate(conf_list, reliability_list, model_preds, class_names=None):
    """
    Aggregate model outputs into a single class and confidence.

    Args:
        conf_list (list[float]): per-model reported confidence (0..1).
        reliability_list (list[float]): per-model reliability weights (>=0).
        model_preds (list[str]): per-model predicted labels (strings). Same length as conf_list.
        class_names (list[str], optional): list of possible class names (not required).

    Returns:
        final_conf (float): normalized confidence for the chosen class (0..1).
        final_class (str): chosen class label (or "Unknown" if none).
    """

    # Basic validation
    if not (len(conf_list) == len(reliability_list) == len(model_preds)):
        raise ValueError("conf_list, reliability_list and model_preds must have the same length")

    # Convert inputs to floats and sanitize
    confs = [float(c) if c is not None else 0.0 for c in conf_list]
    rels  = [float(r) if r is not None else 0.0 for r in reliability_list]
    preds = [p if p is not None else "Unknown" for p in model_preds]

    # Compute per-model weight = conf * reliability
    weights = [c * r for c, r in zip(confs, rels)]
    total_weight = sum(weights)

    # If nothing contributes, fallback to majority vote -> highest single confidence -> Unknown
    if total_weight <= 0:
        # majority vote among non-Unknown preds
        non_unknown = [p for p in preds if p and p != "Unknown"]
        if not non_unknown:
            return 0.0, "Unknown"
        majority = Counter(non_unknown).most_common()
        best_label = majority[0][0]
        # approximate confidence: best model's conf among predictors of best_label
        best_conf = 0.0
        for p, c in zip(preds, confs):
            if p == best_label:
                best_conf = max(best_conf, c)
        return float(best_conf), best_label

    # Sum weighted contributions per class
    class_score = {}
    for pred, w in zip(preds, weights):
        if pred is None or pred == "Unknown":
            continue
        class_score[pred] = class_score.get(pred, 0.0) + w

    # If no class has a score (all Unknown), return Unknown
    if not class_score:
        return 0.0, "Unknown"

    # Find top scoring class(es)
    max_score = max(class_score.values())
    top_classes = [cls for cls, s in class_score.items() if math.isclose(s, max_score, rel_tol=1e-9) or s == max_score]

    # Tie-breaking
    if len(top_classes) == 1:
        final_class = top_classes[0]
    else:
        # 1) majority vote among tied classes
        votes = Counter([p for p in preds if p in top_classes])
        if votes:
            most_common = votes.most_common()
            # if clear winner
            if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
                final_class = most_common[0][0]
            else:
                # 2) pick class with highest average weighted confidence among tied
                avg_scores = {}
                for cls in top_classes:
                    # sum of weights from models that predicted this class
                    sum_w = sum(w for p, w in zip(preds, weights) if p == cls)
                    cnt  = sum(1 for p in preds if p == cls)
                    avg_scores[cls] = (sum_w / cnt) if cnt > 0 else 0.0
                final_class = max(avg_scores, key=avg_scores.get)
        else:
            # fallback: choose lexicographically first (rare)
            final_class = sorted(top_classes)[0]

    # final confidence: normalized score for chosen class
    final_conf = class_score.get(final_class, 0.0) / total_weight
    final_conf = max(0.0, min(1.0, float(final_conf)))  # clamp

    return final_conf, final_class


@app.post("/yolo")
async def yolo_route(file: UploadFile = File(...)):
    img = read_image(file)
    crops = yolo_get_crops(img)
    if not crops:
        return {"error": "No organisms detected"}

    # Clear old Firebase entries
    db.reference("/inference_images").delete()
    db.reference("/class_counts").delete()
    db.reference("/main_image/yolo_output").delete()

    results = []

    for crop in crops:
        crop_np = crop["crop_np"]

        # ===== MASK R-CNN =====
        _, mask_class, mask_conf, _ = mask_rcnn_inference(crop_np)

        # ===== RESNET =====
        crop_bytes = np_to_bytes(crop_np)
        resnet_label, resnet_conf = resnet_inference(
            resnet_model,
            crop_bytes,
            DEVICE,
            CLASS_NAMES
        )

        # ===== RANDOM FOREST =====
        try:
            rf_res = random_forest_inference(crop_np)
            rf_label = rf_res.get("predicted_value")
        except Exception:
            rf_res = {"conf": 0}
            rf_label = "Unknown"

        # ===== EMBEDDINGS =====
        file.file.seek(0)
        emb = embedding_search_file(file, n_results=5)
        emb_label = emb.get("best_one", "Unknown")

        # ===== YOLO =====
        yolo_label = crop.get("class_name", "Unknown")

        # ===== VOTING =====
        vote = {}
        for c in [
            mask_class or "Unknown",
            yolo_label or "Unknown",
            rf_label or "Unknown",
            emb_label or "Unknown",
            resnet_label or "Unknown"
        ]:
            vote[c] = vote.get(c, 0) + 1

        final_label = max(vote, key=vote.get)

        # ===== CONFIDENCE LIST =====
        conf_list = [
            float(mask_conf) if mask_conf is not None else 0,
            float(crop.get("confidence")) if crop.get("confidence") is not None else 0,
            float(rf_res.get("conf")) if rf_res.get("conf") is not None else 0,
            float(emb.get("conf")) if emb.get("conf") is not None else 0,
            float(resnet_conf) if resnet_conf is not None else 0
        ]

        reliable = [
            0.706,  # mask rcnn
            0.026,  # yolo
            0.489,  # random forest
            0.49,   # embedding
            0.61    # resnet
        ]

        model_preds = [
            mask_class or "Unknown",
            yolo_label or "Unknown",
            rf_label or "Unknown",
            emb_label or "Unknown",
            resnet_label or "Unknown"
        ]

        final_conf, final_class = aggregate(conf_list, reliable, model_preds, CLASS_NAMES)

        # ===== SAVE TO FIREBASE =====
        try:
            crop_url, file_id = save_crop_to_firebase_realtime(crop_np)
        except:
            crop_url = None
            file_id = str(uuid.uuid4())

        # ===== RESULT JSON =====
        result = {
            "final_class": final_class,
            "final_conf": final_conf,
            "crop_image_url": crop_url,

            "votes": vote,

            "model_outputs": {
                "mask_r_cnn": mask_class,
                "yolo": yolo_label,
                "random_forest": rf_label,
                "embedding": emb_label,
                "resnet": resnet_label,
            },

            "confidence": {
                "mask_r_cnn": float(mask_conf) if mask_conf is not None else None,
                "yolo": float(crop.get("confidence")) if crop.get("confidence") is not None else None,
                "random_forest": float(rf_res.get("conf")) if rf_res.get("conf") is not None else None,
                "embedding": float(emb.get("conf")) if emb.get("conf") is not None else None,
                "resnet": float(resnet_conf) if resnet_conf is not None else None
            }
        }

        # Save to DB (safe)
        try:
            store_inference_to_firebase(result, file_id)
        except:
            pass

        try:
            update_class_count(final_label)
        except:
            pass

        results.append(result)

    return {"total_crops": len(results), "results": results}


@app.post("/yolo_seperate")
async def yolo_sep_route(file: UploadFile = File(...)):
    img = read_image(file)
    detections = yolo_inference(img)
    return {"detections": detections}

@app.post("/combined")
async def combined_route(file: UploadFile = File(...)):
    img = read_image(file)
    _, mask_class, mask_conf, mask_outputs = mask_rcnn_inference(img)
    yolo_predictions = yolo_inference(img)
    file.file.seek(0)
    embedding_results = embedding_search_file(file, n_results=5)
    rf_output = random_forest_inference(img)
    resultDict = {
        "mask_r_cnn": {"best_class": mask_class, "confidence": mask_conf, "detections": mask_outputs},
        "yolo_predictions": yolo_predictions[0] if isinstance(yolo_predictions, list) and yolo_predictions else "Unknown",
        "random_forest_output": rf_output,
        "embedding_results": embedding_results.get("best_one", "Unknown")
    }
    maxVotingDict = {}
    for key in [
        resultDict["mask_r_cnn"]["best_class"],
        resultDict["yolo_predictions"],
        resultDict["random_forest_output"].get("predicted_value") if isinstance(resultDict["random_forest_output"], dict) else "Unknown",
        resultDict["embedding_results"]
    ]:
        maxVotingDict[key] = maxVotingDict.get(key, 0) + 1
    max_key = max(maxVotingDict, key=maxVotingDict.get)
    return max_key

@app.post("/resnet")
async def computeResNet(file: UploadFile = File(...)):
    image_bytes = await file.read()

    prediction, confidence = resnet_inference(
        resnet_model,
        image_bytes,
        DEVICE,
        CLASS_NAMES
    )

    return {
        "predicted_value": prediction,
        "confidence": confidence
    }

@app.get("/save")
async def savefolder():
    savetofolder()
    return {"message": f"saved to the folder : {SAVE_PATH}"}

@app.post("/compute-gc")
async def compute_gc(
    file: UploadFile = File(...),
    window_size: int = 1000
):
    rf_model_path = r"D:\projects\MicroMorph AI\Models\Genes\rf_model_gene.pkl"
    clf = joblib.load(rf_model_path)

    content = await file.read()
    temp_path = "temp.fna"

    with open(temp_path, "wb") as f:
        f.write(content)

    record = next(SeqIO.parse(temp_path, "fasta"))
    seq = str(record.seq).upper()

    positions, gc_values = gc_sliding_window(seq, window_size)
    gc_summary = compute_gc_summary(seq)

    features = kmer_counts(seq)  
    prediction = clf.predict([features])[0]

    return {
        "positions": positions,
        "gc_values": gc_values,
        "prediction": str(prediction),
        "gc_summary": gc_summary
    }