import sys
import os
import random
import streamlit as st
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importaciones absolutas desde el paquete src
from src.config import FLICKR8K_DATASET_DIR, TOP_K_RETRIEVAL, GENERATIVE_MODEL_NAME
from src.embedding import generate_image_embedding, generate_text_embeddings
from src.indexer import FaissVectorDB 
from src.retriever import retrieve_by_text, retrieve_by_image
from src.generator import build_prompt, generate_response,get_openai_client

@st.cache_resource
def load_all_resources():
    """
    Carga todos los recursos necesarios para la aplicación, incluyendo la base de datos FAISS.
    Se utiliza st.cache_resource para que se cargue una única vez.
    """
    st.write("Cargando base de datos FAISS...")
    faiss_db_instance = FaissVectorDB.load_index() # load_index ya usa las rutas de config
    if faiss_db_instance is None:
        st.error("Error al cargar el índice FAISS. Por favor, asegúrate de haber ejecutado 'python src/build_index.py' primero.")
        st.stop() # Detiene la ejecución de la app Streamlit
        
    return faiss_db_instance

# Cargar todos los recursos
faiss_db = load_all_resources()

# Configuración de la página Streamlit
st.set_page_config(layout="wide", page_title="SRI Multimodal RAG")
st.title("🖼️ Sistema de Búsqueda Multimodal con RAG")
st.markdown("Busca imágenes con texto o sube una imagen para encontrar similares y obtener una descripción generada por IA.")

tab1, tab2 = st.tabs(["🔍 Búsqueda por Texto", "📷 Búsqueda por Imagen"])

with tab1: 
    st.header("Búsqueda por Texto")
    query_input_text = st.text_input("Escribe tu consulta de texto:", key="text_search_input")
    if st.button("Buscar por Texto", key="text_search_button"):
        if query_input_text:
            st.info(f"Realizando búsqueda por texto para: '{query_input_text}'...")
            distances, indices = retrieve_by_text(query_input_text, faiss_db, k=TOP_K_RETRIEVAL)
        else:
            st.warning("Por favor, ingresa una consulta de texto o sube una imagen para buscar.")
            indices = None # Asegurar que indices sea None para no entrar al bucle si no hay consulta
    
        st.header("Resultados de la Búsqueda")
        retrieved_results = []

        if indices is not None and len(indices[0]) > 0:
            # Filtrar índices -1 y obtener información de los resultados válidos
            valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(faiss_db.image_info)]
            # Ordenar los resultados por distancia (FAISS ya los devuelve ordenados, pero es una buena práctica)
            # Asegurarse de que las distancias también se correspondan con los índices válidos
            valid_distances = [d for i, d in enumerate(distances[0]) if indices[0][i] != -1 and indices[0][i] < len(faiss_db.image_info)]
            
            # Crear la lista de resultados con sus metadatos
            for i, idx in enumerate(valid_indices):
                info = faiss_db.image_info[idx]
                retrieved_results.append({
                    'file_name': info['file_name'],
                    'image_path': info['image_path'],
                    'caption': info['caption'],
                    'distance': valid_distances[i]
                })
            
            # Mostrar resultados con la nueva disposición
            if retrieved_results:
                st.write(f"Se recuperaron {len(retrieved_results)} resultados.")

                # Separar el Top-1 del resto
                top_1_result = retrieved_results[0]
                other_results = retrieved_results[1:]

                # Columna principal para el Top-1
                col1, col2 = st.columns([0.45, 0.55]) 

                with col1:
                    st.subheader("Top-1 Resultado")
                    try:
                        # Mostrar imágenes más pequeñas
                        caption = top_1_result['caption']
                        if isinstance(caption, list):
                            caption = random.choice(caption)
                        caption = caption.split('.')[0].strip() + '.' if caption else ""

                        st.image(Image.open(top_1_result['image_path']).convert("RGB"), width=400, caption=caption)

                    except Exception as e:
                        st.error(f"Error al cargar la imagen principal {top_1_result['image_path']}: {e}")

                with col2:
                    st.subheader("Otros Resultados")
                    # Organizar las otras imágenes en filas, 2 por fila
                    num_other_results = len(other_results)
                    for i in range(0, num_other_results, 3):
                        row_cols = st.columns(3)
                        for j in range(3):
                            if (i + j) < num_other_results:
                                res = other_results[i + j]
                                with row_cols[j]:
                                    try:
                                        caption = info['caption']
                                        if isinstance(caption, list):
                                            caption = random.choice(caption)
                                        caption = caption.split('.')[0].strip() + '.' if caption else ""
                                        st.image(Image.open(res['image_path']).convert("RGB"), width=150,caption=caption)
                                    except Exception as e:
                                        st.error(f"Error al cargar imagen {res['image_path']}: {e}")
                
                # --- Generación Aumentada por Recuperación (RAG) ---
                st.header("Respuesta Generada (RAG)")
                # Construir el prompt para generar la respuesta
                prompt = build_prompt('Haz lo que dice el prompt, y no menciones "claro" ni "aquí está..."', [result['caption'] for result in retrieved_results])
                generated_response = generate_response(prompt)
                st.write(generated_response)
            else:
                st.write("No se encontraron resultados para tu consulta.")
        elif st.button:
            st.write("No se encontraron resultados para tu consulta.")

with tab2:
    st.header("Búsqueda por Imagen")
    uploaded_image = st.file_uploader("O sube una imagen aquí", type=["jpg", "jpeg", "png"])
    search_button = st.button("Realizar Búsqueda", key="image_search")

    if uploaded_image:
        st.info(f"Realizando búsqueda por imagen para: '{uploaded_image.name}'...")
        # Guardar la imagen temporalmente
        temp_image_path = os.path.join("temp_uploaded_image.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
            
        distances, indices = retrieve_by_image(temp_image_path, faiss_db, k=TOP_K_RETRIEVAL)
        os.remove(temp_image_path) 
    else:
        st.warning("Por favor, ingresa una consulta de texto o sube una imagen para buscar.")
        indices = None 
    
    st.header("Resultados de la Búsqueda")
    retrieved_results = []

    if indices is not None and len(indices[0]) > 0:
        # Filtrar índices -1 y obtener información de los resultados válidos
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(faiss_db.image_info)]
        # Ordenar los resultados por distancia (FAISS ya los devuelve ordenados, pero es una buena práctica)
        # Asegurarse de que las distancias también se correspondan con los índices válidos
        valid_distances = [d for i, d in enumerate(distances[0]) if indices[0][i] != -1 and indices[0][i] < len(faiss_db.image_info)]
        
        # Crear la lista de resultados con sus metadatos
        for i, idx in enumerate(valid_indices):
            info = faiss_db.image_info[idx]
            retrieved_results.append({
                'file_name': info['file_name'],
                'image_path': info['image_path'],
                'caption': info['caption'],
                'distance': valid_distances[i]
            })
        
        # Mostrar resultados con la nueva disposición
        if retrieved_results:
            st.write(f"Se recuperaron {len(retrieved_results)} resultados.")

            # Separar el Top-1 del resto
            top_1_result = retrieved_results[0]
            other_results = retrieved_results[1:]

            # Columna principal para el Top-1
            col1, col2 = st.columns([0.45, 0.55]) 

            with col1:
                st.subheader("Top-1 Resultado")
                try:
                    # Mostrar imágenes más pequeñas
                    caption = top_1_result['caption']
                    if isinstance(caption, list):
                        caption = random.choice(caption)
                    caption = caption.split('.')[0].strip() + '.' if caption else ""

                    st.image(Image.open(top_1_result['image_path']).convert("RGB"), width=400, caption=caption)
                except Exception as e:
                    st.error(f"Error al cargar la imagen principal {top_1_result['image_path']}: {e}")

            with col2:
                st.subheader("Otros Resultados")
                # Organizar las otras imágenes en filas, 2 por fila
                num_other_results = len(other_results)
                for i in range(0, num_other_results, 3):
                    row_cols = st.columns(3)
                    for j in range(3):
                        if (i + j) < num_other_results:
                            res = other_results[i + j]
                            with row_cols[j]:
                                try:
                                    caption = info['caption']
                                    if isinstance(caption, list):
                                        caption = random.choice(caption)
                                    caption = caption.split('.')[0].strip() + '.' if caption else ""
                                    st.image(Image.open(res['image_path']).convert("RGB"), width=150,caption=caption)
                                except Exception as e:
                                    st.error(f"Error al cargar imagen {res['image_path']}: {e}")
            
            # --- Generación Aumentada por Recuperación (RAG) ---
            st.header("Respuesta Generada (RAG)")
            # Construir el prompt para generar la respuesta
            prompt = build_prompt('Haz lo que dice el prompt, y no menciones "claro" ni "aquí está..."', [result['caption'] for result in retrieved_results])
            generated_response = generate_response(prompt)
            st.write(generated_response)
            
        else:
            st.write("No se encontraron resultados para tu consulta.")
    elif search_button: # Solo mostrar si se apretó el botón pero no hubo resultados
        st.write("No se encontraron resultados para tu consulta.")