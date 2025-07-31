# SRI_Multimodal/src/config.py

import os
import json

# Directorio base del proyecto 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Directorio base del proyecto: {BASE_DIR}")

# Directorio donde se encuentran los datos (dataset)
DATA_DIR = os.path.join(BASE_DIR, 'data')
print(f"Directorio de datos: {DATA_DIR}")

# Rutas específicas para el dataset de Flickr8k
FLICKR8K_DATASET_DIR = os.path.join(DATA_DIR, 'Flicker8k_Dataset')  # Imágenes de Flickr8k
print(f"Directorio de Flickr8k: {FLICKR8K_DATASET_DIR}")

# Archivo de descripciones (captions) de Flickr8k. Este archivo contiene los tokens necesarios para las descripciones.
FLICKR8K_TOKEN_FILE = os.path.join(DATA_DIR, 'Flickr8k_text', 'Flickr8k.token.txt')  # Descripciones de Flickr8k
print(f"Archivo de token Flickr8k: {FLICKR8K_TOKEN_FILE}")

# Número de muestras que se deben usar para el entrenamiento o pruebas. 
NUM_SAMPLES_TO_USE = None  

# Configuración del modelo de embeddings
EMBEDDING_MODEL_NAME = 'clip-ViT-B-32'

# Dimensión del embedding, en este caso, 512 para el modelo 'clip-ViT-B-32'.
EMBEDDING_DIM = 512

# Rutas para guardar y cargar el índice FAISS y la información de los embeddings.
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')  # Ruta para guardar el índice FAISS
IMAGE_INFO_PATH = os.path.join(BASE_DIR, 'image_info.pkl')  # Ruta para guardar la información de las imágenes (metadatos)

# Configuración del modelo generativo
GENERATIVE_MODEL_NAME = "gpt-4.1"

# Número máximo de tokens a generar en cada respuesta del modelo generativo.
GENERATION_MAX_NEW_TOKENS = 150

# Número de documentos relevantes a recuperar por cada consulta.
TOP_K_RETRIEVAL = 10

# --- Cargar configuración adicional desde config.json ---
CONFIG_JSON_PATH = os.path.join(BASE_DIR, 'config.json')
_additional_config = {}
if os.path.exists(CONFIG_JSON_PATH):
    try:
        with open(CONFIG_JSON_PATH, 'r') as f:
            _additional_config = json.load(f)  
        print(f"Configuración cargada desde {CONFIG_JSON_PATH}")
    except json.JSONDecodeError as e:
        print(f"Error al leer config.json: {e}")
else:
    print(f"No se encontró config.json en {CONFIG_JSON_PATH}. Usando valores predeterminados.")

# Función para obtener el dispositivo (CPU/GPU)
def get_device():
    """
    Función para determinar el dispositivo en el que se ejecutará el modelo (GPU si está disponible, de lo contrario, CPU).
    Utiliza PyTorch para comprobar si hay una GPU disponible.
    
    Returns:
        str: 'cuda' si hay una GPU disponible, 'cpu' si solo está disponible la CPU.
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
