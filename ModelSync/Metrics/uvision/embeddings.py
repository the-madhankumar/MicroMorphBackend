import os
import glob
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbeddingEngine:
    """
    Handles CLIP-based embedding generation for images or text queries.
    Includes:
        - Generating embeddings from a folder
        - Generating embedding for a single image
        - Saving embeddings
        - Loading embeddings
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        """
        Initialize the SentenceTransformer CLIP model.
        """
        self.model = SentenceTransformer(model_name)

    def _load_image(self, image_path):
        """
        Internal helper: loads an image safely and converts to RGB.
        Prevents grayscale/P mode issues.
        """
        print("[INFO] Image Loading...")
        return Image.open(image_path).convert("RGB")

    def generate_embeddings_from_image(self, image_path):
        """
        Generate embedding for a single image.
        """
        image = self._load_image(image_path)
        print("[INFO] Image Loaded...")
        return self.model.encode(image)

    def generate_embeddings_from_folder(self, folder_path):
        """
        Generate embeddings for all images in a folder (all extensions).

        Args:
            folder_path (str): Path to image dataset.

        Returns:
            embeddings (list): List of embedding vectors.
            image_paths (list): File paths corresponding to each embedding.
        """
        supported_ext = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        image_paths = []

        for ext in supported_ext:
            image_paths.extend(
                glob.glob(os.path.join(folder_path, "**", ext), recursive=True)
            )

        embeddings = []
        for path in image_paths:
            image = self._load_image(path)
            emb = self.model.encode(image)
            embeddings.append(emb)

        return embeddings, image_paths

    def save_embeddings(self, embeddings, image_paths, output_file):
        """
        Save embeddings and image paths to a compressed .npz file.
        """
        np.savez_compressed(
            output_file,
            embeddings=np.array(embeddings, dtype=np.float32),
            image_paths=np.array(image_paths)
        )
        print(f"Embeddings saved to {output_file}")

    def load_embeddings(self, file_path):
        """
        Load embeddings and image paths from a .npz file.

        Returns:
            embeddings (array): Loaded embeddings.
            image_paths (list): Loaded image paths.
        """
        data = np.load(file_path, allow_pickle=True)
        embeddings = data["embeddings"]
        image_paths = data["image_paths"].tolist()

        print(f"Loaded embeddings from {file_path}")
        return embeddings, image_paths