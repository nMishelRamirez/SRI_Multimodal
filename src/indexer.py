# SRI_Multimodal/src/indexer.py 

import faiss
import numpy as np
import pickle
import os

# Importación relativa para config
from src.config import EMBEDDING_DIM, FAISS_INDEX_PATH, IMAGE_INFO_PATH

class FaissVectorDB:
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        """
        Inicializa la clase FaissVectorDB para gestionar un índice FAISS de vectores de embeddings.

        Args:
            embedding_dim (int): Dimensión de los vectores de embedding (por defecto se toma el valor de EMBEDDING_DIM desde config).
        """
        self.embedding_dim = embedding_dim
        # Usar IndexFlatL2 para búsqueda de distancia euclidiana L2
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.image_info = []  # Lista para almacenar metadatos de las imágenes

    def add_vectors(self, embeddings, image_info):
        """
        Añade vectores al índice FAISS y sus metadatos asociados.

        Args:
            embeddings (np.ndarray): Array de numpy con los embeddings a agregar al índice (forma: num_vectors x embedding_dim).
            image_info (List[dict]): Lista de diccionarios que contienen los metadatos de las imágenes asociadas a los vectores.
        
        Raises:
            ValueError: Si las dimensiones del embedding no coinciden con la dimensión definida en el índice.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Las dimensiones del embedding ({embeddings.shape[1]}) no coinciden con la dimensión del índice ({self.embedding_dim}).")
        
        self.index.add(embeddings)  # Añadir embeddings al índice FAISS
        self.image_info.extend(image_info)  # Añadir los metadatos de las imágenes
        print(f"Añadidos {len(embeddings)} vectores. Total de vectores en el índice: {self.index.ntotal}")

    def search(self, query_embedding, k=5):
        """
        Realiza una búsqueda en el índice FAISS utilizando un embedding de consulta.

        Args:
            query_embedding (np.ndarray): El embedding de la consulta (forma: 1 x embedding_dim).
            k (int): Número de resultados (vectores) a recuperar (por defecto 5).

        Returns:
            Tuple: Una tupla con dos elementos:
                - distancias (np.ndarray): Las distancias a los vectores más cercanos.
                - indices (np.ndarray): Los índices de los vectores más cercanos.
        
        Raises:
            ValueError: Si las dimensiones del embedding de la consulta no coinciden con las dimensiones del índice.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)  
        
        if query_embedding.shape[1] != self.embedding_dim:
            raise ValueError(f"Las dimensiones del embedding de consulta ({query_embedding.shape[1]}) no coinciden con la dimensión del índice ({self.embedding_dim}).")

        # Realizar la búsqueda en el índice FAISS
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self):
        """
        Guarda el índice FAISS y los metadatos de las imágenes a archivos.
        
        Guarda el índice FAISS en un archivo binario y los metadatos de las imágenes en un archivo pickle.
        """
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)  # Crear directorio si no existe
        faiss.write_index(self.index, FAISS_INDEX_PATH)  # Guardar el índice en el archivo especificado
        with open(IMAGE_INFO_PATH, 'wb') as f:
            pickle.dump(self.image_info, f)  # Guardar los metadatos en un archivo pickle
        print(f"Índice FAISS guardado en: {FAISS_INDEX_PATH}")
        print(f"Metadatos de imágenes guardados en: {IMAGE_INFO_PATH}")

    @classmethod
    def load_index(cls):
        """
        Carga el índice FAISS y los metadatos de las imágenes desde los archivos guardados.

        Returns:
            FaissVectorDB: Una instancia de FaissVectorDB con el índice y los metadatos cargados.
            None: Si no se pueden cargar los archivos de índice o metadatos.

        Raises:
            Exception: Si hay un error al cargar los archivos del índice o los metadatos.
        """
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(IMAGE_INFO_PATH):
            print(f"Advertencia: No se encontraron los archivos del índice FAISS o metadatos en '{FAISS_INDEX_PATH}' y '{IMAGE_INFO_PATH}'.")
            print("Por favor, ejecuta 'python src/build_index.py' primero.")
            return None
        
        try:
            instance = cls()  # Crear una instancia vacía de la clase
            instance.index = faiss.read_index(FAISS_INDEX_PATH)  # Cargar el índice desde el archivo
            with open(IMAGE_INFO_PATH, 'rb') as f:
                instance.image_info = pickle.load(f)  # Cargar los metadatos desde el archivo pickle
            print(f"Índice FAISS y metadatos cargados. Total de elementos: {instance.index.ntotal}")
            return instance
        except Exception as e:
            print(f"Error al cargar el índice FAISS o los metadatos: {e}")
            return None
