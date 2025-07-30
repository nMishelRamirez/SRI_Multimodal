# SRI_Multimodal/src/config.py

import os
import json

# Directorio base del proyecto (SRI_Multimodal)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Directorio base del proyecto: {BASE_DIR}")
# Rutas a tus datasets de Flickr8k (o COCO, si es lo que estás usando con tus archivos)
# Adaptado para el formato de tus archivos proporcionados (main.py y dataset_loader.py)
DATA_DIR = os.path.join(BASE_DIR, 'data')
print(f"Directorio de datos: {DATA_DIR}")
# Si estás usando Flickr8k, estas rutas son correctas:
FLICKR8K_DATASET_DIR = os.path.join(DATA_DIR, 'Flicker8k_Dataset') # Imágenes Flickr8k
print(f"Directorio de Flickr8k: {FLICKR8K_DATASET_DIR}")
FLICKR8K_TOKEN_FILE = os.path.join(DATA_DIR, 'Flickr8k_text', 'Flickr8k.token.txt') # Descripciones Flickr8k
print(f"Archivo de token Flickr8k: {FLICKR8K_TOKEN_FILE}")

# Si estás usando COCO, ajusta estas rutas según tu main.py original:
# COCO_IMG_DIR = os.path.join(DATA_DIR, 'test') # O 'val2014' si es el directorio completo
# COCO_CAPTION_FILE = os.path.join(DATA_DIR, 'annotations', 'captions_val2014.json')

# Puedes reducir el corpus para desarrollo/pruebas rápidas
NUM_SAMPLES_TO_USE = None # Usar None para el corpus completo
# NUM_SAMPLES_TO_USE = 1000 # Para usar solo 1000 imágenes para pruebas

# Configuración del modelo de embeddings (SentenceTransformer con CLIP)
EMBEDDING_MODEL_NAME = 'clip-ViT-B-32'
# Dimensión del embedding (para clip-ViT-B-32)
EMBEDDING_DIM = 512

# Rutas para guardar/cargar el índice FAISS y la información de los embeddings
# Estos archivos se guardarán en la raíz del proyecto para fácil acceso
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')
IMAGE_INFO_PATH = os.path.join(BASE_DIR, 'image_info.pkl') # Para mapear embeddings a imágenes y descripciones

# Configuración del modelo generativo (basado en el rag.py proporcionado)
GENERATIVE_MODEL_NAME = "gpt-4.1" # El rag.py original usa "gpt-4.1", ajústalo si usas otro
# MAX_NEW_TOKENS para la generación
GENERATION_MAX_NEW_TOKENS = 150

# Número de documentos relevantes a recuperar por consulta
TOP_K_RETRIEVAL = 5

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

# API Keys (preferiblemente cargadas desde variables de entorno para producción)
OPENAI_API_KEY = _additional_config.get("OPENAI_API_KEY", None)

# Utilidad para obtener el dispositivo (CPU/GPU)
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"