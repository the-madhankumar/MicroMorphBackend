from torchvision import transforms, models
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

# -------------------------------
# MASK RCNN INFERENCE
# -------------------------------
def mask_rcnn_inference(img: np.ndarray, threshold: float = 0.4):
    image_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = MaskRCNN_model(image_tensor)

    masks = predictions[0].get("masks", torch.tensor([]))
    boxes = predictions[0].get("boxes", torch.tensor([]))
    labels = predictions[0].get("labels", torch.tensor([]))
    scores = predictions[0].get("scores", torch.tensor([]))

    best_class = ""
    best_score = 0.0
    outputs = []

    count = masks.shape[0] if masks is not None and masks.ndim > 0 else 0

    for i in range(count):
        score = float(scores[i].cpu().item())
        if score < threshold:
            continue

        class_index = int(labels[i].cpu().item())
        class_name = CLASS_NAMES[class_index - 1] if 1 <= class_index <= len(CLASS_NAMES) else "UnknownClass"

        if score > best_score:
            best_class = class_name
            best_score = score

        outputs.append({
            "confidence": score,
            "class_name": class_name
        })

    return best_class, best_score, outputs


# -------------------------------
# STATISTICS
# -------------------------------
def brute_force_statistics(conf_list):
    N = len(conf_list)
    if N == 0:
        return {"mean": 0, "std": 0, "se": 0}

    total = sum(conf_list)
    mean = total / N

    sq_diff_sum = sum((c - mean)**2 for c in conf_list)
    std = (sq_diff_sum / (N - 1))**0.5 if N > 1 else 0.0
    se = std / (N**0.5)

    return {"mean": mean, "std": std, "se": se}

def reliability_score(mean, std, se):
    raw = mean / (1 + std + se)
    return max(0.0, min(1.0, raw))


# -------------------------------
# TRAVERSE TRAIN FOLDER
# -------------------------------
confidence_scores = []
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

train = r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\Dataset\Species-3\train"

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

        best_class, best_score, outputs = mask_rcnn_inference(img)
        confidence_scores.append(best_score)

        print(f"[OK] {filename} → score: {best_score:.4f}")

# -------------------------------
# GET RELIABILITY FOR FIRST 100
# -------------------------------

stats = brute_force_statistics(confidence_scores)
R = reliability_score(stats["mean"], stats["std"], stats["se"])

print("\n==============================")
print(" MASK R-CNN RELIABILITY (100 samples)")
print("==============================")
print("Mean Confidence     :", stats["mean"])
print("Std Deviation       :", stats["std"])
print("Standard Error      :", stats["se"])
print("Reliability Score   :", R)
print("==============================")
