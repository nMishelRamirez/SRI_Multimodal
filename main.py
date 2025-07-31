import os
import sys
import pandas as pd
import numpy as np
import pickle
import faiss
import nltk
from PIL import Image

# Añadir el directorio 'src' al path para permitir importaciones absolutas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Importar módulos desde el paquete src
from src.config import (
    FLICKR8K_DATASET_DIR, FLICKR8K_TOKEN_FILE,
    NUM_SAMPLES_TO_USE, EMBEDDING_MODEL_NAME, EMBEDDING_DIM,
    FAISS_INDEX_PATH, IMAGE_INFO_PATH, TOP_K_RETRIEVAL
)
from src.data_loader import load_flickr8k_data_adapted
from src.preprocessing import preprocess_documents, concatenar_captions_by_image
from src.embedding import generate_text_embeddings, generate_image_embeddings, generate_image_embedding
from src.indexer import FaissVectorDB
from src.retriever import retrieve_by_text, retrieve_by_image, retrieve_by_text_and_image
    

def calculate_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula la distancia euclidiana entre dos vectores."""
    return np.linalg.norm(vec1 - vec2)

def main():
    print("===============================================")
    print("         PRUEBAS SRI MULTIMODAL                ")
    print("===============================================")

    # 1. Carga de Datos
    print("\n--- 1: Carga de Datos ---")
    df_raw = load_flickr8k_data_adapted(FLICKR8K_DATASET_DIR, FLICKR8K_TOKEN_FILE, NUM_SAMPLES_TO_USE)
    if df_raw.empty:
        print("Error: DataFrame de datos brutos vacío. Asegúrate de que las rutas del dataset sean correctas en config.py.")
        return
    print(f"Cargado {len(df_raw)} entradas de captions.")
    print("Ejemplo de datos brutos:")
    print(df_raw.head())

    # 2. Preprocesamiento
    print("\n--- 2: Preprocesamiento de Textos ---")
    df_combined_captions = concatenar_captions_by_image(df_raw, image_col='file_name', caption_col='caption')
    print(f"Número de imágenes únicas con captions combinados: {len(df_combined_captions)}")
    
    df_processed_texts = preprocess_documents(df_combined_captions['combined_caption'].tolist())
    df_combined_captions['prep_doc'] = df_processed_texts['prep_doc']
    
    print("Ejemplo de captions preprocesados:")
    print(df_combined_captions[['file_name', 'combined_caption', 'prep_doc']].head())

    # 3. Generación de Embeddings
    print("\n--- 3: Generación de Embeddings ---")
    print(f"Usando modelo de embedding: {EMBEDDING_MODEL_NAME}")

    text_embeddings = generate_text_embeddings(df_combined_captions['prep_doc'].tolist(), model_name=EMBEDDING_MODEL_NAME)
    print(f"Forma de los embeddings de texto: {text_embeddings.shape} (Esperado: N, {EMBEDDING_DIM})")
    print(f"Primer embedding de texto (fragmento): {text_embeddings[0][:5]}...")

    image_full_paths = [os.path.join(FLICKR8K_DATASET_DIR, f_name) for f_name in df_combined_captions['file_name'].tolist()]
    image_embeddings = generate_image_embeddings(image_full_paths, model_name=EMBEDDING_MODEL_NAME)
    print(f"Forma de los embeddings de imagen: {image_embeddings.shape} (Esperado: N, {EMBEDDING_DIM})")
    print(f"Primer embedding de imagen (fragmento): {image_embeddings[0][:5]}...")

    # 4. Construcción/Carga del Índice FAISS
    print("\n--- 4: Construcción/Carga del Índice FAISS ---")
    
    # Primero, intentar cargar el índice. Si no existe o hay error, construirlo.
    faiss_db = FaissVectorDB.load_index()
    if faiss_db is None or faiss_db.index.ntotal == 0:
        print("Índice FAISS no encontrado o vacío. Procediendo a construirlo...")
        # Llama a la función principal de construcción del índice si es necesario
        # Esto es un poco redundante con el código de build_index.py pero asegura la validación aquí
        
        # Combinar embeddings para el índice
        df_with_embeddings = df_combined_captions.copy()
        df_with_embeddings['txt_embedding'] = [vec for vec in text_embeddings]
        df_with_embeddings['img_embedding'] = [vec for vec in image_embeddings]

        df_valid_embeddings = df_with_embeddings.dropna(subset=['txt_embedding', 'img_embedding']).copy()
        if df_valid_embeddings.empty:
            print("No hay suficientes datos con embeddings válidos para construir el índice. Abortando.")
            return

        txt_embs_np = np.array(df_valid_embeddings['txt_embedding'].tolist()).astype('float32')
        img_embs_np = np.array(df_valid_embeddings['img_embedding'].tolist()).astype('float32')
        combined_embeddings = (0.5 * txt_embs_np + 0.5 * img_embs_np).astype('float32') # Combinación simple para el índice

        faiss_db = FaissVectorDB(embedding_dim=EMBEDDING_DIM)
        info_to_save = []
        for index, row in df_valid_embeddings.iterrows():
            info_to_save.append({
                'file_name': row['file_name'],
                'image_path': os.path.join(FLICKR8K_DATASET_DIR, row['file_name']),
                'caption': row['combined_caption']
            })
        faiss_db.add_vectors(combined_embeddings, info_to_save)
        faiss_db.save_index()
        print(f"Índice FAISS construido y guardado. Elementos indexados: {faiss_db.index.ntotal}")
    else:
        print(f"Índice FAISS cargado exitosamente con {faiss_db.index.ntotal} elementos.")

    # 5. Pruebas de Recuperación (Retriever)
    print("\n--- 5: Pruebas de Recuperación (Retriever) ---")

    # Prueba 5.1: Búsqueda por Texto
    print("\n--- 5.1: Búsqueda por Texto ---")
    query_text = "A dog jumping over a hurdle"
    print(f"Consulta de texto: '{query_text}'")
    
    distances_text, indices_text = retrieve_by_text(query_text, faiss_db, k=TOP_K_RETRIEVAL)
    
    if indices_text is not None and len(indices_text[0]) > 0:
        print(f"\nTop-{TOP_K_RETRIEVAL} resultados recuperados por texto:")
        retrieved_text_info = [faiss_db.image_info[idx] for idx in indices_text[0] if idx != -1]
        
        # Generar embedding de la consulta de texto para la validación de distancia
        query_text_emb = generate_text_embeddings([query_text], model_name=EMBEDDING_MODEL_NAME)[0]

        for i, (idx, dist) in enumerate(zip(indices_text[0], distances_text[0])):
            if idx == -1: 
                continue
            item_info = faiss_db.image_info[idx]

            print(f"--- Resultado {i+1} ---")
            print(f"  Imagen: {item_info['file_name']}")
            print(f"  Caption: {item_info['caption']}")
            print(f"  Distancia FAISS (L2): {dist:.4f}")
    else:
        print("No se encontraron resultados para la búsqueda de texto.")


    # Prueba 5.2: Búsqueda por Imagen
    print("\n--- 5.2: Búsqueda por Imagen ---")
    test_image_path = os.path.join(FLICKR8K_DATASET_DIR, "2218743570_9d6614c51c.jpg")
    if not os.path.exists(test_image_path):
        print(f"Advertencia: Imagen de prueba no encontrada en '{test_image_path}'. Saltando prueba por imagen.")
    else:
        print(f"Consulta de imagen: '{os.path.basename(test_image_path)}'")
        
        distances_image, indices_image = retrieve_by_image(test_image_path, faiss_db, k=TOP_K_RETRIEVAL)

        if indices_image is not None and len(indices_image[0]) > 0:
            print(f"\nTop-{TOP_K_RETRIEVAL} resultados recuperados por imagen:")
            
            # Generar embedding de la imagen de consulta para la validación de distancia
            query_image_emb = generate_image_embedding(test_image_path, model_name=EMBEDDING_MODEL_NAME)[0]

            for i, (idx, dist) in enumerate(zip(indices_image[0], distances_image[0])):
                if idx == -1:
                    continue
                item_info = faiss_db.image_info[idx]
                print(f"--- Resultado {i+1} ---")
                print(f"  Imagen: {item_info['file_name']}")
                print(f"  Caption: {item_info['caption']}")
                print(f"  Distancia FAISS (L2): {dist:.4f}")

        else:
            print("No se encontraron resultados para la búsqueda de imagen.")

    print("\n===============================================")
    print("         PRUEBAS FINALIZADAS         ")
    print("===============================================")

if __name__ == "__main__":
    main()