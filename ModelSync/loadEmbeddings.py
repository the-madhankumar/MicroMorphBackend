import os
import json
import chromadb
from tqdm import tqdm
from uvision.embeddings import ImageEmbeddingEngine

folders = [
    "D:/projects/MicroMorph AI/Project MicroMorph AI/ContourFromPolygon/Species-3/train/_annotations.coco.json",
    "D:/projects/MicroMorph AI/Project MicroMorph AI/ContourFromPolygon/Species-3/test/_annotations.coco.json",
    "D:/projects/MicroMorph AI/Project MicroMorph AI/ContourFromPolygon/Species-3/valid/_annotations.coco.json"
]

client = chromadb.PersistentClient(
    path=r"D:\projects\MicroMorph AI\Project MicroMorph AI\ModelSync\chroma_storage"
)

collection = client.get_or_create_collection("species_embeddings")

engine = ImageEmbeddingEngine()

for json_file in folders:

    with open(json_file, "r") as f:
        coco_data = json.load(f)

    class_names = {c["id"]: c["name"] for c in coco_data["categories"]}
    image_map = {img["id"]: img["file_name"] for img in coco_data["images"]}

    items = []

    for ann in coco_data["annotations"]:
        img_file = image_map[ann["image_id"]]
        img_path = os.path.join(os.path.dirname(json_file), img_file)
        filename = os.path.basename(img_path)
        image_name = filename.split("_")[0]

        items.append({
            "image_path": img_path,
            "image_name": image_name,
            "class_name": class_names[ann["category_id"]]
        })

    for item in tqdm(items, desc=f"Processing {os.path.basename(json_file)}"):

        embed_id = f"{item['image_path']}:{item['class_name']}"

        embedding = engine.generate_embeddings_from_image(item["image_path"])

        metadata = {
            "image_path": item["image_path"],
            "image_name": item["image_name"],
            "class_name": item["class_name"]
        }

        collection.add(
            ids=[embed_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[item["image_path"]]
        )

print("Vector store updated successfully.")
