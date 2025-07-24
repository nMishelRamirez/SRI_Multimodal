import pandas as pd
import numpy as np
import faiss
import os
from app.encoder import encode_image
from app.config import IMAGES_DIR, CAPTIONS_FILE, INDEX_FILE, PATHS_FILE, DESCRIPTIONS_FILE

def build_index():
    df = pd.read_csv(CAPTIONS_FILE)
    image_embeddings = []
    paths = []
    captions = []

    for _, row in df.iterrows():
        img_path = os.path.join(IMAGES_DIR, row['filename'])
        caption = row['caption']
        try:
            emb = encode_image(img_path).cpu().numpy()
            image_embeddings.append(emb)
            paths.append(img_path)
            captions.append(caption)
        except:
            continue

    image_embeddings_np = np.vstack(image_embeddings)
    index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
    index.add(image_embeddings_np)

    faiss.write_index(index, INDEX_FILE)
    np.save(PATHS_FILE, paths)
    np.save(DESCRIPTIONS_FILE, captions)

    print("Índice construido y guardado.")
