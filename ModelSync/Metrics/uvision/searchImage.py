from UnSeenVision.uvision.embeddings import ImageEmbeddingEngine
from uvision.faissIndex import FaissIndex

class ImageSearchEngine:
    """
    High-level wrapper that combines embedding generation and FAISS search.
    """

    def __init__(self, model_name="clip-ViT-B-32"):
        self.embedder = ImageEmbeddingEngine(model_name)

    def create_index(self, image_folder, output_index_path):
        """
        Generate embeddings → Build FAISS index → Save index.
        """
        embeddings, image_paths = self.embedder.generate_embeddings_from_folder(image_folder)

        dimension = len(embeddings[0])
        faiss_index = FaissIndex(dimension)

        faiss_index.build(embeddings)
        faiss_index.save(output_index_path, image_paths)

        return faiss_index.index, image_paths

    def load_index(self, index_path):
        """
        Load previously created FAISS index.
        """
        return FaissIndex.load(index_path)

    def search(self, query, index, image_paths, top_k=3):
        """
        Given a query (text or image), retrieve similar images.
        """
        query_vector = self.embedder.embed_query(query)
        _, indices = index.search(query_vector, top_k)

        retrieved_images = [image_paths[i] for i in indices[0]]
        return retrieved_images