import numpy as np
import faiss
from encoder import encode_image, encode_text
from config import INDEX_FILE, PATHS_FILE, DESCRIPTIONS_FILE

index = faiss.read_index(INDEX_FILE)
paths = np.load(PATHS_FILE, allow_pickle=True)
descs = np.load(DESCRIPTIONS_FILE, allow_pickle=True)

def get_labels_from_indices(indices):
    # Devuelve las etiquetas cortas (columna caption del CSV original)
    return [descs[i] for i in indices[0]]  # descs ya contiene los labels


def retrieve_by_image(image_path, k=5):
    q_emb = encode_image(image_path).cpu().numpy()
    D, I = index.search(q_emb, k)
    return paths[I[0]], descs[I[0]]

def retrieve_by_text(text_query, k=5):
    q_emb = encode_text(text_query).cpu().numpy()
    D, I = index.search(q_emb, k)
    return paths[I[0]], descs[I[0]]
