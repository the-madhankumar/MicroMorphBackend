import os

YOLO_MODEL = r"D:\projects\MicroMorph AI\Models\YOLO\yolo_model.pt"

if not os.path.exists(YOLO_MODEL):
    raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL}")
else:
    print(f"File found at {YOLO_MODEL}")