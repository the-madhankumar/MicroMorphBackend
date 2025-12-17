
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class FaissIndex:
    """
    Handles creation, saving, loading, and searching of FAISS vector index.
    """

    def __init__(self, dimension):
        """
        Initialize a FAISS index with Inner Product similarity.
        """
        index = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIDMap(index)

    def build(self, embeddings):
        """
        Add embeddings to the FAISS index.
        """
        vectors = np.array(embeddings).astype(np.float32)
        ids = np.arange(len(vectors))
        self.index.add_with_ids(vectors, ids)

    def save(self, index_path, image_paths):
        """
        Save FAISS index + corresponding image paths.
        """
        faiss.write_index(self.index, index_path)

        with open(index_path + ".paths", 'w') as f:
            for path in image_paths:
                f.write(path + "\n")

        print(f"Index saved at: {index_path}")

    @staticmethod
    def load(index_path):
        """
        Load a FAISS index and image paths from disk.
        """
        index = faiss.read_index(index_path)

        with open(index_path + ".paths", 'r') as f:
            image_paths = [line.strip() for line in f]

        print(f"Index loaded: {index_path}")
        return index, image_paths

    def search(self, query_vector, top_k=3):
        """
        Search FAISS index and return top-k similar image indices and distances.
        """
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices