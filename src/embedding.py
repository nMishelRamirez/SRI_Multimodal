# SRI_Multimodal/src/embedding.py

from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import torch
from typing import List, Union

# Importación relativa para config
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, get_device 

# Cargar el modelo una sola vez para eficiencia
try:
    _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=get_device()) 
    print(f"Modelo de embedding '{EMBEDDING_MODEL_NAME}' cargado exitosamente en {get_device()}.")
except Exception as e:
    print(f"Error al cargar el modelo de embedding '{EMBEDDING_MODEL_NAME}': {e}")
    _model = None 


def get_embedding_model():
    """
    Retorna la instancia del modelo de embedding cargado.

    Este modelo se carga solo una vez para mejorar la eficiencia. Si no se ha cargado correctamente,
    se lanza un error.

    Returns:
        SentenceTransformer: Instancia del modelo cargado de embeddings.
        
    Raises:
        RuntimeError: Si el modelo no se ha cargado correctamente.
    """

    if _model is None:
        raise RuntimeError(f"El modelo de embedding '{EMBEDDING_MODEL_NAME}' no pudo ser cargado.")
    return _model


def generate_text_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Genera embeddings para una lista de textos utilizando el modelo de SentenceTransformers.
    
    Args:
        texts (List[str]): Lista de textos a los que se les generarán los embeddings.
        model_name (str): Nombre del modelo de embeddings a utilizar (por defecto, `EMBEDDING_MODEL_NAME`).
        
    Returns:
        np.ndarray: Embeddings de los textos, como un array de numpy con dimensiones (número de textos, EMBEDDING_DIM).
        
    Raises:
        ValueError: Si la entrada 'texts' no es una lista de strings.
    """
    model = get_embedding_model()
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("La entrada 'texts' debe ser una lista de strings.")
        
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # Verificar si la dimensión del embedding generado es la esperada
    if embeddings.shape[1] != EMBEDDING_DIM:
        print(f"Advertencia: La dimensión del embedding generado ({embeddings.shape[1]}) no coincide con EMBEDDING_DIM en config ({EMBEDDING_DIM}).")
    
    return embeddings.astype('float32')

def generate_image_embedding(image_path: Union[str, Image.Image], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Genera un embedding para una única imagen utilizando el modelo de SentenceTransformers.
    
    Args:
        image_path (Union[str, Image.Image]): Ruta del archivo de imagen o un objeto PIL Image.
        model_name (str): Nombre del modelo de embeddings a utilizar (por defecto, `EMBEDDING_MODEL_NAME`).
        
    Returns:
        np.ndarray: El embedding de la imagen como un array de numpy con dimensiones (1, EMBEDDING_DIM).
        
    Raises:
        FileNotFoundError: Si la imagen no se encuentra en la ruta especificada.
        ValueError: Si la imagen no puede ser procesada correctamente.
        TypeError: Si la entrada no es una ruta de archivo o un objeto PIL Image.
    """
    
    model = get_embedding_model()
    
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"La imagen no se encontró en la ruta: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se pudo abrir o procesar la imagen desde la ruta '{image_path}': {e}")
    elif isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        raise TypeError("image_path debe ser una ruta de archivo (str) o un objeto PIL Image.")

    embedding = model.encode([image], convert_to_numpy=True, show_progress_bar=False)[0]
    
    # Verificar si la dimensión del embedding de la imagen es la esperada
    if embedding.shape[0] != EMBEDDING_DIM:
        print(f"Advertencia: La dimensión del embedding de imagen generado ({embedding.shape[0]}) no coincide con EMBEDDING_DIM en config ({EMBEDDING_DIM}).")
    
    return embedding.reshape(1, -1).astype('float32')

def generate_image_embeddings(image_paths: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Genera embeddings para una lista de imágenes.

    Args:
        image_paths (List[str]): Lista de rutas a las imágenes.
        model_name (str): Nombre del modelo de embeddings a utilizar (por defecto, `EMBEDDING_MODEL_NAME`).

    Returns:
        np.ndarray: Embeddings de las imágenes, como un array de numpy con dimensiones (número de imágenes, EMBEDDING_DIM).
        
    Raises:
        ValueError: Si las imágenes no pueden ser cargadas correctamente.
    """
    model = get_embedding_model()
    images = []
    
    # Cargar las imágenes y verificar que existan
    for path in tqdm(image_paths, desc="Cargando imágenes para embeddings"):
        if os.path.exists(path):
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as e:
                print(f"Advertencia: No se pudo cargar la imagen '{path}'. Error: {e}")
        else:
            print(f"Advertencia: Imagen no encontrada en la ruta '{path}'. Saltando.")
            
    if not images:
        print("No se encontraron imágenes válidas para generar embeddings.")
        return np.array([]).reshape(0, EMBEDDING_DIM).astype('float32')

    embeddings = model.encode(images, convert_to_numpy=True, show_progress_bar=True)
    
    # Verificar si la dimensión del embedding generado es la esperada
    if embeddings.shape[1] != EMBEDDING_DIM:
        print(f"Advertencia: La dimensión de los embeddings de imagen generados ({embeddings.shape[1]}) no coincide con EMBEDDING_DIM en config ({EMBEDDING_DIM}).")

    return embeddings.astype('float32')

def combine_embeddings(text_embeddings: np.ndarray, image_embeddings: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Combina embeddings de texto e imagen usando un enfoque ponderado.

    Args:
        text_embeddings (np.ndarray): Embeddings de texto, un array numpy de forma (num_textos, EMBEDDING_DIM).
        image_embeddings (np.ndarray): Embeddings de imagen, un array numpy de forma (num_imagenes, EMBEDDING_DIM).
        alpha (float): Peso de los embeddings de texto. El valor debe estar entre 0.0 y 1.0. Los embeddings de imagen se ponderarán con (1 - alpha).

    Returns:
        np.ndarray: Embeddings combinados, como un array numpy con dimensiones (num_elementos, EMBEDDING_DIM).
        
    Raises:
        ValueError: Si los embeddings de texto e imagen no tienen la misma forma.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha debe estar entre 0.0 y 1.0.")
    
    if text_embeddings.shape != image_embeddings.shape:
        raise ValueError("Los arrays de embeddings de texto e imagen deben tener la misma forma.")
        
    combined_embeddings = alpha * text_embeddings + (1 - alpha) * image_embeddings
    return combined_embeddings.astype('float32')
