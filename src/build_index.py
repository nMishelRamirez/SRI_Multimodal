# SRI_Multimodal/src/build_index.py

import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

# Importaciones relativas dentro del paquete 'src'
from src.data_loader import load_flickr8k_data_adapted
from src.preprocessing import preprocess_documents, merge_captions_by_image
from src.embedding import generate_text_embeddings, generate_image_embeddings, combine_embeddings
from src.indexer import FaissVectorDB # Renombrado de vector_db
from src.config import (
    FLICKR8K_DATASET_DIR, FLICKR8K_TOKEN_FILE,
    NUM_SAMPLES_TO_USE, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    FAISS_INDEX_PATH, IMAGE_INFO_PATH
)

def build_faiss_index_main():
    print("--- Iniciando el proceso de construcción del índice FAISS ---")
    
    # Asegúrate de que las rutas FLICKR8K_DATASET_DIR y FLICKR8K_TOKEN_FILE estén correctas en src/config.py
    df_raw = load_flickr8k_data_adapted(FLICKR8K_DATASET_DIR, FLICKR8K_TOKEN_FILE, NUM_SAMPLES_TO_USE)
    
    if df_raw.empty:
        print("No hay datos para indexar. Abortando.")
        return

    df = merge_captions_by_image(df_raw, image_col='file_name', caption_col='caption')
    print(f"Número de imágenes únicas con captions combinados: {len(df)}")
    
    print("Preprocesando textos...")
    df_processed = preprocess_documents(df['combined_caption'].tolist())
    df['prep_doc'] = df_processed['prep_doc']
    del df_processed

    print(f"Generando embeddings de texto con {EMBEDDING_MODEL_NAME}...")
    txt_embeddings = generate_text_embeddings(df['prep_doc'].tolist(), model_name=EMBEDDING_MODEL_NAME)
    df['txt_embedding'] = [vec for vec in txt_embeddings]

    print(f"Generando embeddings de imagen con {EMBEDDING_MODEL_NAME}...")
    image_full_paths = [os.path.join(FLICKR8K_DATASET_DIR, f_name) for f_name in df['file_name'].tolist()]
    
    img_embeddings = generate_image_embeddings(image_full_paths, model_name=EMBEDDING_MODEL_NAME)
    df['img_embedding'] = [vec for vec in img_embeddings]

    print("Combinando embeddings de texto e imagen...")
    df_with_embeddings = df.dropna(subset=['txt_embedding', 'img_embedding']).copy()
    if df_with_embeddings.empty:
        print("No hay suficientes datos con embeddings válidos para construir el índice. Abortando.")
        return

    txt_embs_np = np.array(df_with_embeddings['txt_embedding'].tolist()).astype('float32')
    img_embs_np = np.array(df_with_embeddings['img_embedding'].tolist()).astype('float32')
    
    combined_embeddings = combine_embeddings(txt_embs_np, img_embs_np, alpha=0.5)
    combined_embeddings_np = np.array(combined_embeddings).astype('float32')

    print("Construyendo índice FAISS...")
    faiss_db = FaissVectorDB(embedding_dim=EMBEDDING_DIM)
    
    info_to_save = []
    for index, row in df_with_embeddings.iterrows():
        info_to_save.append({
            'file_name': row['file_name'],
            'image_path': os.path.join(FLICKR8K_DATASET_DIR, row['file_name']),
            'caption': row['combined_caption']
        })

    faiss_db.add_vectors(combined_embeddings_np, info_to_save)
    faiss_db.save_index()
    
    print(f"Índice FAISS y metadatos guardados. Total de elementos indexados: {faiss_db.index.ntotal}")
    print("--- Proceso de construcción del índice FAISS finalizado ---")


if __name__ == "__main__":
    build_faiss_index_main()