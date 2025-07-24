import os

# Rutas
DATA_DIR = "data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "captions.csv")

INDEX_FILE = os.path.join(DATA_DIR, "clip_index.faiss")
PATHS_FILE = os.path.join(DATA_DIR, "image_paths.npy")
DESCRIPTIONS_FILE = os.path.join(DATA_DIR, "descriptions.npy")
CONCEPTS_FILE = os.path.join(DATA_DIR, "concept_descriptions.csv")

# Modelo Generativo
OPENAI_API_KEY = "tu_clave_openai"
