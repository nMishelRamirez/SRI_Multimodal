# SRI_Multimodal/src/retriever.py 

import numpy as np
import os
from PIL import Image

# Importaciones relativas
from src.indexer import FaissVectorDB 
from src.embedding import generate_text_embeddings, generate_image_embedding, combine_embeddings
from src.config import EMBEDDING_MODEL_NAME

def retrieve_by_text(query_text: str, faiss_db: FaissVectorDB, k: int = 5):
    """
    Busca imágenes en la base de datos FAISS utilizando una consulta de texto.
    """
    if faiss_db is None:
        print("Error: La base de datos FAISS no está inicializada.")
        return None, None

    # Generar embedding para el texto de consulta
    text_embedding = generate_text_embeddings([query_text], model_name=EMBEDDING_MODEL_NAME)[0]
    
    # Realizar búsqueda en el índice FAISS
    distances, indices = faiss_db.search(text_embedding, k=k)
    return distances, indices

def retrieve_by_image(query_image_path: str, faiss_db: FaissVectorDB, k: int = 5):
    """
    Busca imágenes en la base de datos FAISS utilizando una imagen de consulta.
    """
    # if faiss_db is None:
    #     print("Error: La base de datos FAISS no está inicializada.")
    #     return None, None

    # if not os.path.exists(query_image_path):
    #     print(f"Error: La ruta de la imagen de consulta no existe: {query_image_path}")
    #     return None, None
    
    try:
        image_embedding = generate_image_embedding(query_image_path, model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error al generar embedding para la imagen '{query_image_path}': {e}")
        return None, None
    
    distances, indices = faiss_db.search(image_embedding, k=k)
    return distances, indices