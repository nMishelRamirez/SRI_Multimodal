# SRI_Multimodal/src/data_loader.py

import os
import pandas as pd
from typing import Optional

def load_captions(file_path: str) -> pd.DataFrame:
    """
    Carga los captions desde un archivo de texto (como Flickr8k.token.txt).
    El archivo debe tener el formato "image_name#index\tcaption_text".
    
    Args:
        file_path (str): Ruta del archivo de captions (por ejemplo, 'Flickr8k.token.txt').
        
    Returns:
        pd.DataFrame: DataFrame con dos columnas, 'file_name' (nombre de la imagen) y 'caption' (texto de la descripción).
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                img_id_full, caption = parts[0], parts[1]
                # Separar el nombre del archivo de imagen del índice del caption
                img_name_parts = img_id_full.split('#')
                if len(img_name_parts) == 2:
                    image_name = img_name_parts[0]
                    data.append({'file_name': image_name, 'caption': caption})
    
    # Si no se encuentran datos válidos en el archivo, se retorna un DataFrame vacío
    if not data:
        print(f"Advertencia: No se encontraron datos válidos en '{file_path}'.")
        return pd.DataFrame(columns=['file_name', 'caption'])

    df = pd.DataFrame(data)
    return df

def load_flickr8k_data_adapted(dataset_dir: str, token_file: str, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Carga el dataset Flickr8k, combinando la información de imágenes y captions.
    Filtra las imágenes que no existen en el directorio de imágenes y permite cargar solo una cantidad 
    determinada de muestras para pruebas.

    Args:
        dataset_dir (str): Ruta al directorio principal de Flickr8k (donde están las imágenes, por ejemplo, 'data/Flicker8k_Dataset').
        token_file (str): Nombre del archivo de tokens (por ejemplo, 'Flickr8k.token.txt'), asumiendo que está en 'data/Flickr8k_text'.
        num_samples (Optional[int]): Número de muestras a cargar para pruebas. Si es None, se cargan todas las muestras.

    Returns:
        pd.DataFrame: DataFrame con dos columnas, 'file_name' (nombre de la imagen) y 'caption' (texto de la descripción) después de aplicar los filtros y selección.
    """
    # Construir la ruta completa del archivo de tokens
    full_token_path = os.path.join(os.path.dirname(dataset_dir), 'Flickr8k_text', token_file)
    
    # Verificar si el archivo de tokens existe
    if not os.path.exists(full_token_path):
        print(f"Error: Archivo de tokens no encontrado en '{full_token_path}'.")
        print("Asegúrate de que el archivo Flickr8k.token.txt esté en 'data/Flickr8k_text'.")
        return pd.DataFrame()

    # Cargar los captions desde el archivo de tokens
    df_captions = load_captions(full_token_path)
    
    # Verificar si se pudieron cargar captions
    if df_captions.empty:
        print("No se pudieron cargar captions. Asegúrate de que el archivo de tokens tenga el formato correcto.")
        return pd.DataFrame()

    # Filtrar para asegurarse de que solo se incluyan imágenes que realmente existen en el directorio de imágenes
    image_dir_files = set(os.listdir(dataset_dir))
    df_captions = df_captions[df_captions['file_name'].isin(image_dir_files)].copy()
    
    # Si el DataFrame está vacío después de filtrar por las imágenes existentes, retornar un DataFrame vacío
    if df_captions.empty:
        print("Después de verificar la existencia de imágenes, el DataFrame de captions está vacío.")
        print("Asegúrate de que las imágenes del archivo de tokens existan en el directorio del dataset.")
        return pd.DataFrame()

    # Si se proporciona un número de muestras, se seleccionan aleatoriamente ese número de imágenes únicas
    if num_samples is not None and num_samples > 0:
        # Obtener las imágenes únicas y seleccionar aleatoriamente 'num_samples' imágenes
        unique_images = df_captions['file_name'].unique()
        if len(unique_images) > num_samples:
            sampled_images = pd.Series(unique_images).sample(n=num_samples, random_state=42).tolist()
            df_captions = df_captions[df_captions['file_name'].isin(sampled_images)].copy()
        print(f"Cargadas {len(df_captions['file_name'].unique())} imágenes únicas con {len(df_captions)} captions (muestras: {num_samples if num_samples > 0 else 'todas'}).")
    else:
        print(f"Cargadas {len(df_captions['file_name'].unique())} imágenes únicas con {len(df_captions)} captions.")

    return df_captions
