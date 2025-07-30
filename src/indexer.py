# SRI_Multimodal/src/indexer.py 

import faiss
import numpy as np
import pickle
import os

# Importación relativa para config
from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, IMAGE_INFO_PATH

class FaissVectorDB:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        # Usar IndexFlatL2 para búsqueda de distancia euclidiana L2
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.image_info = [] # Lista para almacenar metadatos de las imágenes

    def add_vectors(self, embeddings, image_info):
        """
        Añade vectores al índice FAISS y sus metadatos asociados.
        embeddings: numpy array de forma (num_vectors, embedding_dim)
        image_info: Lista de diccionarios con metadatos para cada vector
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Las dimensiones del embedding ({embeddings.shape[1]}) no coinciden con la dimensión del índice ({self.embedding_dim}).")
        
        self.index.add(embeddings)
        self.image_info.extend(image_info)
        print(f"Añadidos {len(embeddings)} vectores. Total de vectores en el índice: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        """
        Realiza una búsqueda en el índice FAISS.
        query_embedding: numpy array de forma (1, embedding_dim)
        k: Número de resultados a recuperar
        Retorna: distancias y índices de los k vectores más cercanos
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1) # Asegurarse de que sea 2D
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Las dimensiones del embedding de consulta ({query_embedding.shape[1]}) no coinciden con la dimensión del índice ({self.embedding_dim}).")

        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self):
        """Guarda el índice FAISS y los metadatos de las imágenes."""
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(IMAGE_INFO_PATH, 'wb') as f:
            pickle.dump(self.image_info, f)
        print(f"Índice FAISS guardado en: {FAISS_INDEX_PATH}")
        print(f"Metadatos de imágenes guardados en: {IMAGE_INFO_PATH}")

    @classmethod
    def load_index(cls):
        """Carga el índice FAISS y los metadatos de las imágenes."""
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(IMAGE_INFO_PATH):
            print(f"Advertencia: No se encontraron los archivos del índice FAISS o metadatos en '{FAISS_INDEX_PATH}' y '{IMAGE_INFO_PATH}'.")
            print("Por favor, ejecuta 'python src/build_index.py' primero.")
            return None
        
        try:
            instance = cls() # Crea una instancia vacía
            instance.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(IMAGE_INFO_PATH, 'rb') as f:
                instance.image_info = pickle.load(f)
            print(f"Índice FAISS y metadatos cargados. Total de elementos: {instance.index.ntotal}")
            return instance
        except Exception as e:
            print(f"Error al cargar el índice FAISS o los metadatos: {e}")
            return None