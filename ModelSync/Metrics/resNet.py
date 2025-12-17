import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 10
CLASS_NAMES = [
    "Alexandrium","Cerataulina","Ceratium","Entomoneis",
    "Guinardia","Hemiaulus","Nitzschia","Pinnularia",
    "Pleurosigma","Prorocentrum","UnknownClass","Other"
]

resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
resnet_model.load_state_dict(torch.load(r"D:\projects\MicroMorph AI\Models\ResNet\resnet_model.pth", map_location=DEVICE))
resnet_model = resnet_model.to(DEVICE)
resnet_model.eval()


# ===============================
#  RESNET INFERENCE (MATCHES RF STYLE)
# ===============================
def resnet_inference_resnet18(image_bytes):
    """
    Returns:
        {
          "predicted_value": str,
          "conf": float
        }
    """

    # Transform same as training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load image from byte-stream
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = resnet_model(img_t)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_label = CLASS_NAMES[pred.item()]
    confidence = float(conf.item())

    print(f"[RESNET] predicted={predicted_label}, conf={confidence:.4f}")

    return {
        "predicted_value": predicted_label,
        "conf": confidence
    }

# -------------------------
# SAME IMAGE EXTENSIONS
# -------------------------
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -------------------------
# TRAIN FOLDER
# -------------------------
train_folder = r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\Dataset\Species-3\train"

resnet_confidence_scores = []


# -------------------------
# LOOP THROUGH TRAIN IMAGES
# -------------------------
for root, dirs, files in os.walk(train_folder):
    for filename in tqdm(files):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue  

        image_path = os.path.join(root, filename)

        img = cv2.imread(image_path)
        if img is None:
            print("Could not load:", image_path)
            continue

        # Encode image back to bytes for ResNet fn
        success, encoded = cv2.imencode(".jpg", img)
        if not success:
            print("Encoding failed:", image_path)
            continue

        output = resnet_inference_resnet18(encoded.tobytes())

        if not output:
            continue

        resnet_confidence_scores.append(output["conf"])

        print(f"[RESNET OK] {filename} → conf: {output['conf']:.4f}")

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
def reliability_score(mean, std, se):
    raw = mean / (1 + std + se)
    return max(0.0, min(1.0, raw))

# -------------------------
# COMPUTE STATISTICS
# -------------------------
stats_resnet = brute_force_statistics(resnet_confidence_scores)
R_resnet = reliability_score(stats_resnet["mean"], stats_resnet["std"], stats_resnet["se"])


# -------------------------
# PRINT SUMMARY
# -------------------------
print("\n==============================")
print("      RESNET RELIABILITY")
print("==============================")
print("Mean Confidence     :", stats_resnet["mean"])
print("Std Deviation       :", stats_resnet["std"])
print("Standard Error      :", stats_resnet["se"])
print("Reliability Score   :", R_resnet)
print("==============================")
